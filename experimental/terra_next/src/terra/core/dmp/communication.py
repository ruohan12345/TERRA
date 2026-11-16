"""Differentiable native collectives and the adjoint of H halo exchange."""
import torch
import torch.distributed as dist
from torch.nn import functional as F
from terra.runtime.profiling import profile_range
from terra.runtime.collectives import all_to_all_single, batch_p2p
from terra.runtime.tensor_layout import contiguous_for_device, has_channels_last_layout


def _wire(tensor, mesh):
    mesh.check_open()
    if tensor.device.type not in ("cpu", "cuda", "npu") or not tensor.is_floating_point():
        raise TypeError("DMP communication requires floating CPU/CUDA/NPU tensors")
    if tensor.device.type == "npu" and tensor.dtype == torch.float64:
        raise TypeError("Ascend 910B transport does not support FP64")
    dtype = tensor.dtype
    wire_dtype = torch.float32 if tensor.device.type == "cpu" and dtype in (torch.float16, torch.bfloat16) else dtype
    mesh.stats["logical_dtypes"].add(str(dtype).removeprefix("torch."))
    mesh.stats["wire_dtypes"].add(str(wire_dtype).removeprefix("torch."))
    return tensor.to(dtype=wire_dtype).contiguous()


def _restore_layout(tensor, reference):
    channels_last = tensor.ndim == 4 and has_channels_last_layout(reference)
    return contiguous_for_device(tensor, channels_last=channels_last)


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, mesh):
        ctx.mesh = mesh
        ctx.dtype = packed.dtype
        mesh.stats["all_to_all_calls"] += 1
        wire = _wire(packed, mesh)
        received = torch.empty_like(wire)
        with profile_range("terra/comm/dmp/conv_repartition",direction="forward",
                           group_size=mesh.tp_size,tensor_bytes=wire.numel()*wire.element_size()):
            all_to_all_single(received, wire, group=mesh.tp_group)
        return received.to(dtype=packed.dtype)

    @staticmethod
    def backward(ctx, grad_received):
        mesh = ctx.mesh
        mesh.stats["all_to_all_backward_calls"] += 1
        wire = _wire(grad_received, mesh)
        returned = torch.empty_like(wire)
        # Equal-sized all-to-all is its own adjoint. Tensor packing/concats
        # outside this Function apply the source/destination axis inverses.
        with profile_range("terra/comm/dmp/conv_repartition",direction="backward",
                           group_size=mesh.tp_size,tensor_bytes=wire.numel()*wire.element_size()):
            all_to_all_single(returned, wire, group=mesh.tp_group)
        return returned.to(dtype=ctx.dtype), None


def all_to_all(packed, mesh):
    """Exchange dimension-0 destination blocks with equal-sized TP peers."""
    mesh.check_open()
    if packed.ndim < 1 or packed.shape[0] != mesh.tp_size:
        raise ValueError("The first packed dimension must equal the TP group size")
    mesh.peak("max_packed_exchange_numel", packed.numel())
    if mesh.tp_size == 1:
        return packed
    return _AllToAll.apply(packed, mesh)


class _AllToAllV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, send_counts, receive_counts, mesh):
        ctx.send_counts, ctx.receive_counts, ctx.mesh = send_counts, receive_counts, mesh
        ctx.dtype = packed.dtype
        wire = _wire(packed, mesh)
        output = wire.new_empty(sum(receive_counts))
        mesh.stats['all_to_all_calls'] += 1
        with profile_range('terra/comm/dmp/conv_repartition', direction='forward', group_size=mesh.tp_size):
            all_to_all_single(output, wire, output_split_sizes=list(receive_counts),
                                   input_split_sizes=list(send_counts), group=mesh.tp_group)
        return output.to(packed.dtype)

    @staticmethod
    def backward(ctx, gradient):
        wire = _wire(gradient, ctx.mesh)
        output = wire.new_empty(sum(ctx.send_counts))
        ctx.mesh.stats['all_to_all_backward_calls'] += 1
        with profile_range('terra/comm/dmp/conv_repartition', direction='backward', group_size=ctx.mesh.tp_size):
            all_to_all_single(output, wire, output_split_sizes=list(ctx.send_counts),
                                   input_split_sizes=list(ctx.receive_counts), group=ctx.mesh.tp_group)
        return output.to(ctx.dtype), None, None, None


def exchange_pieces(pieces, receive_counts, mesh):
    """Source-major variable payloads; reverse counts define the adjoint."""
    mesh.check_open()
    if len(pieces) != mesh.tp_size or len(receive_counts) != mesh.tp_size:
        raise ValueError('One payload and receive count is required per channel-team rank')
    send_counts = tuple(piece.numel() for piece in pieces)
    packed = torch.cat([piece.reshape(-1) for piece in pieces])
    mesh.peak('max_packed_exchange_numel', packed.numel())
    if mesh.tp_size == 1:
        return packed
    return _AllToAllV.apply(packed, send_counts, tuple(receive_counts), mesh)


def _p2p(send_prev, send_next, recv_prev, recv_next, mesh):
    """Matched global-rank neighbors inside a fixed channel-coordinate group."""
    previous = mesh.spatial_ranks[mesh.spatial_rank - 1] if mesh.spatial_rank else None
    following = (mesh.spatial_ranks[mesh.spatial_rank + 1]
                 if mesh.spatial_rank + 1 < mesh.spatial_size else None)
    ops = []
    if previous is not None:
        ops.extend((dist.P2POp(dist.isend, send_prev, previous, group=mesh.spatial_group),
                    dist.P2POp(dist.irecv, recv_prev, previous, group=mesh.spatial_group)))
    if following is not None:
        ops.extend((dist.P2POp(dist.isend, send_next, following, group=mesh.spatial_group),
                    dist.P2POp(dist.irecv, recv_next, following, group=mesh.spatial_group)))
    if ops:
        for request in batch_p2p(ops):
            request.wait()
    return previous is not None, following is not None


class _Halo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, width, mesh):
        ctx.width, ctx.mesh, ctx.height = width, mesh, tensor.shape[2]
        ctx.dtype = tensor.dtype
        wire = _wire(tensor, mesh)
        first, last = wire[:, :, :width, :].contiguous(), wire[:, :, -width:, :].contiguous()
        previous, following = torch.zeros_like(first), torch.zeros_like(last)
        with profile_range("terra/comm/dmp/halo",direction="forward",halo_rows=width,
                           group_size=mesh.spatial_size,
                           tensor_bytes=(first.numel()+last.numel())*first.element_size()):
            _p2p(first, last, previous, following, mesh)
        output = torch.cat((previous, wire, following), dim=2).to(dtype=tensor.dtype)
        return _restore_layout(output, tensor)

    @staticmethod
    def backward(ctx, grad_output):
        width, mesh, height = ctx.width, ctx.mesh, ctx.height
        mesh.stats["halo_backward_calls"] += 1
        wire = _wire(grad_output, mesh)
        # Forward's left halo came from previous's bottom edge, and conversely
        # for the right halo. Its adjoint adds both contributions to core edges.
        send_prev = wire[:, :, :width, :].contiguous()
        send_next = wire[:, :, width + height:, :].contiguous()
        recv_prev, recv_next = torch.zeros_like(send_prev), torch.zeros_like(send_next)
        with profile_range("terra/comm/dmp/halo",direction="backward",halo_rows=width,
                           group_size=mesh.spatial_size,
                           tensor_bytes=(send_prev.numel()+send_next.numel())*send_prev.element_size()):
            has_prev, has_next = _p2p(send_prev, send_next, recv_prev, recv_next, mesh)
        core = wire[:, :, width:width + height, :].clone().contiguous()
        if has_prev:
            core[:, :, :width, :].add_(recv_prev)
        if has_next:
            core[:, :, -width:, :].add_(recv_next)
        return core.to(dtype=ctx.dtype), None, None


def halo_exchange(tensor, width, mesh):
    """Add zero/neighbor halo rows on the temporary macro-spatial layout."""
    mesh.check_open()
    if tensor.ndim != 4:
        raise ValueError("Halo input must be NCHW")
    if isinstance(width, bool) or not isinstance(width, int) or width < 0:
        raise ValueError("Halo width must be a nonnegative integer")
    if width == 0:
        return tensor
    if tensor.shape[2] < width:
        raise ValueError("A macro-spatial strip must contain at least halo_width rows")
    mesh.stats["halo_calls"] += 1
    shape = list(tensor.shape)
    shape[2] += 2 * width
    numel = tensor.numel() + 2 * tensor.shape[0] * tensor.shape[1] * width * tensor.shape[3]
    mesh.peak("max_halo_output_numel", numel)
    mesh.record("halo", input_shape=list(tensor.shape), output_shape=shape,
                width=width, spatial_size=mesh.spatial_size,
                logical_dtype=str(tensor.dtype).removeprefix("torch."))
    if mesh.spatial_size == 1:
        return _restore_layout(F.pad(tensor, (0, 0, width, width)), tensor)
    return _Halo.apply(tensor, width, mesh)
