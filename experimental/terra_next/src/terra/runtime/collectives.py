"""Native collectives with an explicit empty-payload HCCL transport adapter."""
import math
import torch
import torch.distributed as dist


def batch_p2p(ops):
    """Return P2P work handles, using HCCL's native batch transport on NPU.

    TorchNPU 2.7.1/CANN 9.2 selects a coalescing/hcclGroupEnd path for 910B;
    that path failed the cross-node halo gate while the plugin's native
    batch interface passed without global launch blocking. Keep tensors and
    this list alive until the caller has waited on every returned handle.
    """
    if not ops:
        return []
    device = ops[0].tensor.device
    if device.type != 'npu':
        return dist.batch_isend_irecv(ops)
    group = ops[0].group if ops[0].group is not None else dist.group.WORLD
    tensors, types, peers = [], [], []
    for op in ops:
        op_group = op.group if op.group is not None else dist.group.WORLD
        if op_group is not group or op.tensor.device != device:
            raise ValueError('HCCL batch P2P requires one group and one local NPU')
        if op.tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError('Ascend halo P2P supports FP16/BF16/FP32')
        if op.op.__name__ not in ('isend', 'irecv') or op.tag != 0:
            raise NotImplementedError('HCCL halo P2P requires isend/irecv with tag=0')
        tensors.append(op.tensor)
        types.append(op.op.__name__)
        # P2POp.peer is global; native ProcessGroupHCCL expects group rank,
        # matching TorchNPU's own non-coalesced fallback implementation.
        peers.append(dist.get_group_rank(group, op.peer))
    backend = group._get_backend(device)
    native = getattr(backend, 'batch_isend_irecv', None)
    if native is None:
        raise NotImplementedError('Installed TorchNPU lacks native HCCL batch P2P; run the platform capability gate')
    return [native(types, tensors, peers)]


def all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None):
    if input.device.type != 'npu':
        return dist.all_to_all_single(output, input, output_split_sizes=output_split_sizes,
                                      input_split_sizes=input_split_sizes, group=group)
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError('Ascend DMP/FMP floating transport supports FP16/BF16/FP32')
    size = dist.get_world_size(group)
    def counts(tensor, given):
        if given is None:
            if tensor.shape[0] % size:
                raise ValueError('Equal all-to-all first dimension must divide group size')
            given = [tensor.shape[0]//size] * size
        if len(given) != size or any(n < 0 for n in given) or sum(given) != tensor.shape[0]:
            raise ValueError('Invalid all-to-all split sizes')
        width = math.prod(tensor.shape[1:])
        return [int(n)*width for n in given]
    send, receive = counts(input, input_split_sizes), counts(output, output_split_sizes)
    if all(send) and all(receive):
        # Common nonempty path uses flat views, not extra full-size buffers.
        # Peer counts are expressed in elements on every NPU rank, including
        # ranks that require the padding path below.
        return dist.all_to_all_single(output.view(-1), input.reshape(-1),
                                      output_split_sizes=receive,
                                      input_split_sizes=send, group=group)
    # One dummy element per empty peer keeps every rank in the collective,
    # including legitimate empty ownership. The dummy has no logical value.
    padded_send, padded_receive = [max(n, 1) for n in send], [max(n, 1) for n in receive]
    flat = input.reshape(-1)
    packed = input.new_zeros(sum(padded_send))
    start = destination = 0
    for count, padded in zip(send, padded_send):
        packed[destination:destination+count].copy_(flat[start:start+count])
        start += count
        destination += padded
    wire = output.new_empty(sum(padded_receive))
    dist.all_to_all_single(wire, packed, output_split_sizes=padded_receive,
                          input_split_sizes=padded_send, group=group)
    pieces, start = [], 0
    for count, padded in zip(receive, padded_receive):
        pieces.append(wire[start:start+count])
        start += padded
    output.copy_(torch.cat(pieces).reshape_as(output))
