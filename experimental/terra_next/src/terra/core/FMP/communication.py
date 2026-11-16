"""Autograd collectives with explicit sequence ownership and SUM adjoints.

An all-gather used by differently sharded operators is not an identity in
backward: contributions from every consumer must be summed for its owner.
"""
import torch
import torch.distributed as dist
from terra.runtime.profiling import profile_range
from terra.runtime.collectives import all_to_all_single, batch_p2p


def _collective(kind,operation,*args,direction,**kwargs):
    # CPU enqueue annotation; the profiler's GPU activity supplies the actual
    # communication intervals. This wrapper adds no waits or synchronizations.
    with profile_range("terra/comm/fmp/"+kind,direction=direction):
        return operation(*args,**kwargs)


def _wire(x):
    if x.device.type == "npu" and x.dtype == torch.float64:
        raise TypeError("Ascend 910B transport does not support FP64")
    dtype = torch.float32 if x.device.type == "cpu" and x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    return x.to(dtype=dtype).contiguous()


def _validate_route_transport(value):
    if not isinstance(value, str) or value not in ("all_to_all", "p2p"):
        raise ValueError("Token route transport_implementation must be 'all_to_all' or 'p2p'")
    return value


def _p2p_route_exchange(received, packed, send_counts, receive_counts, group, size):
    """Exchange the existing destination-major wire into source-major storage.

    Keep the same packed/received buffers and peer order as all-to-all. This
    backend changes no token ownership, local output placement or wire dtype.
    """
    if (isinstance(size, bool) or not isinstance(size, int) or size < 1
            or len(send_counts) != size or len(receive_counts) != size
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 0
                   for n in (*send_counts, *receive_counts))):
        raise ValueError("P2P route split sizes must contain one nonnegative integer per group peer")
    if (packed.ndim != 2 or received.ndim != 2 or packed.shape[1:] != received.shape[1:]
            or packed.shape[0] != sum(send_counts) or received.shape[0] != sum(receive_counts)
            or packed.dtype != received.dtype or packed.device != received.device
            or not packed.is_contiguous() or not received.is_contiguous()):
        raise ValueError("P2P route buffers must match their contiguous wire split sizes, dtype and device")
    if size == 1:
        if send_counts[0] != receive_counts[0]:
            raise ValueError("P2P route self send/receive counts must match")
        received.copy_(packed)
        return
    if dist.get_world_size(group) != size:
        raise ValueError("P2P route size differs from its process group")
    rank = dist.get_rank(group)
    if rank < 0 or rank >= size:
        raise ValueError("P2P route caller must belong to its process group")
    if send_counts[rank] != receive_counts[rank]:
        raise ValueError("P2P route self send/receive counts must match")

    sends, receives = [], []
    send_start = receive_start = 0
    for peer, (send_count, receive_count) in enumerate(zip(send_counts, receive_counts)):
        send_view = packed.narrow(0, send_start, send_count)
        receive_view = received.narrow(0, receive_start, receive_count)
        send_start += send_count
        receive_start += receive_count
        if peer == rank:
            if send_count:
                receive_view.copy_(send_view)
            continue
        if send_count or receive_count:
            # P2POp.peer uses a global rank, even when group is a subgroup.
            # The existing native HCCL adapter translates it back to group rank.
            global_peer = peer if group is None else dist.get_global_rank(group, peer)
            if send_count:
                sends.append(dist.P2POp(dist.isend, send_view, global_peer, group=group))
            if receive_count:
                receives.append(dist.P2POp(dist.irecv, receive_view, global_peer, group=group))
    ops = sends + receives
    # ops owns every tensor view; packed/received and all work handles remain
    # live through the last wait. There are no zero-sized remote messages.
    requests = batch_p2p(ops) if ops else []
    for request in requests:
        request.wait()


def _exchange_token_route(received, packed, send_counts, receive_counts, group, size,
                          transport, *, direction):
    if transport == "all_to_all":
        return _collective("token_route", all_to_all_single, received, packed,
                           output_split_sizes=receive_counts, input_split_sizes=send_counts,
                           group=group, direction=direction)
    with profile_range("terra/comm/fmp/token_route", direction=direction,
                       transport="p2p", group_size=size):
        return _p2p_route_exchange(received, packed, send_counts, receive_counts, group, size)


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, size, rank, dim):
        ctx.group, ctx.size, ctx.rank, ctx.dim, ctx.dtype = group, size, rank, dim, x.dtype
        wire = _wire(x)
        parts = [torch.empty_like(wire) for _ in range(size)]
        _collective("sequence_gather",dist.all_gather,parts,wire,group=group,direction="forward")
        return torch.cat(parts, dim=dim).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, gradient):
        # Reduce the complete cotangent BEFORE selecting this owner's slice.
        # Selecting first would mix distinct source slices in an all-reduce.
        wire = _wire(gradient)
        _collective("sequence_gather_grad_sum",dist.all_reduce,wire,op=dist.ReduceOp.SUM,group=ctx.group,direction="backward")
        local = wire.chunk(ctx.size, dim=ctx.dim)[ctx.rank].contiguous()
        return local.to(dtype=ctx.dtype), None, None, None, None


def gather_sequence(x, group, size, rank, dim=-2):
    return x if size == 1 else _Gather.apply(x, group, size, rank, dim)


class _ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, size, rank, dim):
        ctx.group, ctx.size, ctx.rank, ctx.dim, ctx.dtype = group, size, rank, dim, x.dtype
        if x.shape[dim] % size:
            raise ValueError("Reduce-scatter sequence length must divide TP size")
        # Wire dim 0 consists of consecutive destination blocks. Accumulating
        # partial half/bfloat products in FP32 avoids an extra low precision SUM.
        dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        pieces = [p.contiguous() for p in x.to(dtype).chunk(size, dim=dim)]
        wire = torch.cat(pieces, dim=0).contiguous()
        out = torch.empty_like(pieces[0])
        _collective("sequence_reduce_scatter",dist.reduce_scatter_tensor,out,wire,
                    op=dist.ReduceOp.SUM,group=group,direction="forward")
        return out.to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, gradient):
        wire = _wire(gradient)
        pieces = [torch.empty_like(wire) for _ in range(ctx.size)]
        _collective("sequence_reduce_scatter_grad_gather",dist.all_gather,pieces,wire,
                    group=ctx.group,direction="backward")
        return torch.cat(pieces, dim=ctx.dim).to(dtype=ctx.dtype), None, None, None, None


def reduce_scatter_sequence(x, group, size, rank, dim=-2):
    return x if size == 1 else _ReduceScatter.apply(x, group, size, rank, dim)


class _Permute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, send_order, receive_order, send_counts, receive_counts, destination_rows, group):
        ctx.save_for_backward(send_order, receive_order)
        ctx.send_counts, ctx.receive_counts, ctx.group, ctx.dtype = send_counts, receive_counts, group, x.dtype
        ctx.source_rows=x.shape[0]
        packed = _wire(x.index_select(0, send_order))
        received = packed.new_empty((sum(receive_counts), x.shape[1]))
        _collective("token_route",all_to_all_single,received,packed,
                    output_split_sizes=receive_counts,input_split_sizes=send_counts,
                    group=group,direction="forward")
        # receive_order maps source-major wire positions to output positions.
        output = received.new_zeros((destination_rows,x.shape[1]))
        output.index_copy_(0, receive_order, received)
        return output.to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, gradient):
        send_order, receive_order = ctx.saved_tensors
        packed = _wire(gradient.index_select(0, receive_order))
        received = packed.new_empty((sum(ctx.send_counts), gradient.shape[1]))
        _collective("token_route",all_to_all_single,received,packed,
                    output_split_sizes=ctx.send_counts,input_split_sizes=ctx.receive_counts,
                    group=ctx.group,direction="backward")
        output = received.new_zeros((ctx.source_rows,gradient.shape[1]))
        output.index_copy_(0, send_order, received)
        return output.to(dtype=ctx.dtype), None, None, None, None, None, None


def _inverse_full_permutation(order, rows):
    """Plan an inverse gather once on CPU; partial/invalid orders stay scatter.

    A route may insert stage padding, crop it, or carry excess SP capacity.
    Those directions are not permutations of every output row and must keep
    the established zero-fill/scatter implementation.
    """
    if order.device.type != "cpu" or order.ndim != 1 or order.dtype != torch.int64:
        raise ValueError("Inverse route planning requires a one-dimensional CPU int64 order")
    if order.numel() != rows:
        return None
    sorted_order, inverse = torch.sort(order)
    if not torch.equal(sorted_order, torch.arange(rows, dtype=torch.int64)):
        return None
    return inverse


def _restore_route_rows(received, order, rows, inverse):
    if inverse is not None:
        # Empty owners still participate in the collective, but need no
        # zero-element gather kernel after receiving their empty payload.
        return received if rows == 0 else received.index_select(0, inverse)
    output = received.new_zeros((rows, received.shape[1]))
    return output.index_copy_(0, order, received)


class _InversePermute(torch.autograd.Function):
    """The same route/adjoint, with separately proven complete-row gathers."""

    @staticmethod
    def forward(ctx, x, send_order, receive_order, send_counts, receive_counts,
                destination_rows, group, send_inverse, receive_inverse, size, transport):
        ctx.save_for_backward(send_order, receive_order, send_inverse, receive_inverse)
        ctx.send_counts, ctx.receive_counts = send_counts, receive_counts
        ctx.group, ctx.dtype, ctx.size = group, x.dtype, size
        ctx.transport = transport
        ctx.source_rows = x.shape[0]
        packed = _wire(x.index_select(0, send_order))
        if size == 1:
            received = packed
        else:
            received = packed.new_empty((sum(receive_counts), x.shape[1]))
            _exchange_token_route(received, packed, send_counts, receive_counts, group, size,
                                  transport, direction="forward")
        output = _restore_route_rows(received, receive_order, destination_rows, receive_inverse)
        return output.to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, gradient):
        send_order, receive_order, send_inverse, receive_inverse = ctx.saved_tensors
        packed = _wire(gradient.index_select(0, receive_order))
        if ctx.size == 1:
            received = packed
        else:
            received = packed.new_empty((sum(ctx.send_counts), gradient.shape[1]))
            _exchange_token_route(received, packed, ctx.receive_counts, ctx.send_counts,
                                  ctx.group, ctx.size, ctx.transport, direction="backward")
        output = _restore_route_rows(received, send_order, ctx.source_rows, send_inverse)
        return output.to(dtype=ctx.dtype), None, None, None, None, None, None, None, None, None, None


def permute_tokens(x, plan, group, size, *, inverse_orders=None, transport_implementation="all_to_all"):
    transport = _validate_route_transport(transport_implementation)
    send_order, receive_order, send_counts, receive_counts,destination_rows = plan
    send_order, receive_order = send_order.to(x.device), receive_order.to(x.device)
    if inverse_orders is not None or transport == "p2p":
        send_inverse, receive_inverse = (None, None) if inverse_orders is None else inverse_orders
        return _InversePermute.apply(x, send_order, receive_order, send_counts, receive_counts,
                                     destination_rows, group, send_inverse, receive_inverse, size, transport)
    if size == 1:
        packed = x.index_select(0, send_order)
        output=x.new_zeros((destination_rows,x.shape[1]))
        return output.index_copy(0,receive_order,packed)
    return _Permute.apply(x, send_order, receive_order, send_counts, receive_counts,destination_rows, group)


class _EqualAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,group):
        ctx.group,ctx.dtype=group,x.dtype
        wire=_wire(x);out=torch.empty_like(wire)
        _collective("sp_sequence_head_exchange",all_to_all_single,out,wire,group=group,direction="forward")
        return out.to(dtype=x.dtype)

    @staticmethod
    def backward(ctx,gradient):
        wire=_wire(gradient);out=torch.empty_like(wire)
        _collective("sp_sequence_head_exchange",all_to_all_single,out,wire,group=ctx.group,direction="backward")
        return out.to(dtype=ctx.dtype),None


def sequence_to_heads(x,group,size):
    """Ulysses: [B,W,N/SP,H/TP,d] -> [B,W,H/(TP*SP),N,d]."""
    b,w,q,h,d=x.shape
    if size==1:
        return x.permute(0,1,3,2,4)
    packed=torch.stack(x.chunk(size,dim=3),dim=0).contiguous()
    received=_EqualAllToAll.apply(packed,group)
    return received.permute(1,2,4,0,3,5).reshape(b,w,h//size,q*size,d)


def heads_to_sequence(x,group,size):
    """Inverse Ulysses permutation, including its exact autograd transpose."""
    b,w,h,n,d=x.shape
    if size==1:
        return x.permute(0,1,3,2,4).reshape(b,w,n,h*d)
    packed=torch.stack(x.chunk(size,dim=3),dim=0).contiguous()
    received=_EqualAllToAll.apply(packed,group)
    return received.permute(1,2,4,0,3,5).reshape(b,w,n//size,h*size*d)


class _WindowGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,counts,rank,group):
        ctx.counts,ctx.rank,ctx.group,ctx.dtype=counts,rank,group,x.dtype
        maximum=max(counts)
        shape=list(x.shape);shape[1]=maximum
        padded=x.new_zeros(shape);padded[:,:x.shape[1]].copy_(x)
        wire=_wire(padded);parts=[torch.empty_like(wire) for _ in counts]
        _collective("tp_window_gather",dist.all_gather,parts,wire,group=group,direction="forward")
        return torch.cat([p[:,:n] for p,n in zip(parts,counts)],dim=1).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx,gradient):
        wire=_wire(gradient)
        _collective("tp_window_gather_grad_sum",dist.all_reduce,wire,op=dist.ReduceOp.SUM,
                    group=ctx.group,direction="backward")
        start=sum(ctx.counts[:ctx.rank])
        return wire[:,start:start+ctx.counts[ctx.rank]].contiguous().to(dtype=ctx.dtype),None,None,None


class _WindowScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,counts,rank,group):
        ctx.counts,ctx.rank,ctx.group,ctx.dtype=counts,rank,group,x.dtype
        maximum=max(counts);parts=[];start=0
        dtype=torch.float32 if x.dtype in (torch.float16,torch.bfloat16) else x.dtype
        shape=list(x.shape);shape[1]=maximum
        for n in counts:
            part=x.new_zeros(shape,dtype=dtype);part[:,:n].copy_(x[:,start:start+n]);parts.append(part);start+=n
        wire=torch.cat(parts,dim=0).contiguous();out=torch.empty_like(parts[0])
        _collective("tp_window_reduce_scatter",dist.reduce_scatter_tensor,out,wire,
                    op=dist.ReduceOp.SUM,group=group,direction="forward")
        return out[:,:counts[rank]].contiguous().to(dtype=x.dtype)

    @staticmethod
    def backward(ctx,gradient):
        maximum=max(ctx.counts);shape=list(gradient.shape);shape[1]=maximum
        padded=gradient.new_zeros(shape);padded[:,:gradient.shape[1]].copy_(gradient)
        wire=_wire(padded);parts=[torch.empty_like(wire) for _ in ctx.counts]
        _collective("tp_window_reduce_scatter_grad_gather",dist.all_gather,parts,wire,
                    group=ctx.group,direction="backward")
        return torch.cat([p[:,:n] for p,n in zip(parts,ctx.counts)],dim=1).to(dtype=ctx.dtype),None,None,None


def gather_windows(x,counts,rank,group):
    return x if len(counts)==1 or max(counts)==0 else _WindowGather.apply(x,tuple(counts),rank,group)


def scatter_windows(x,counts,rank,group):
    return x if len(counts)==1 or max(counts)==0 else _WindowScatter.apply(x,tuple(counts),rank,group)
