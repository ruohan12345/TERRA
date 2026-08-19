import torch
import torch.distributed as dist
from typing import List, Sequence
from torch.autograd import Function

use_global_buffer = True
from core.buffer import get_preallocated_buffer


def _reduce(input_, manager):
    """All-reduce the input tensor across model parallel group."""

    '''
    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size()==1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())
    '''

    torch.distributed.all_reduce(input_, group = manager.model_parallel_group)

    return input_

class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""
    #@staticmethod


    @staticmethod
    def forward(ctx, input_, manager):
        ctx.manager = manager
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        manager = ctx.manager
        return _reduce(grad_output, manager), None

class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    #@staticmethod


    @staticmethod
    def forward(ctx, input_, manager):
        return _reduce(input_, manager)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """ Split a tensor along its last dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def _gather_along_last_dim(input_, group, rank):
    """Gather tensors and concatinate along the last dimension."""

    group_size = dist.get_world_size(group=group)

    # Bypass the function if we are using only 1 GPU.
    if group_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1

    tensor_list = [torch.empty_like(input_) for _ in range(group_size)]
    tensor_list[rank] = input_

    torch.distributed.all_gather(tensor_list, input_, group = group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output

def _split_along_last_dim(input_, manager):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    # Bypass the function if we are using only 1 GPU.
    if manager.mp_group_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, manager.mp_group_size)

    # Note: torch.split does not create contiguous tensors by default.
    output = input_list[manager.mp_rank].contiguous()

    return output

class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    #@staticmethod


    @staticmethod
    def forward(ctx, input_, group, rank):


        ctx.group = group
        ctx.rank = rank


        return _gather_along_last_dim(input_, group, rank)


    @staticmethod
    def backward(ctx, grad_output):
        manager = ctx.manager


        # tensor(-1.2405e-05, device='cuda:0')  tensor(-1.1120e-05, device='cuda:1')

        # tensor(-2.3525e-05, device='cuda:0')


        return _split_along_last_dim(grad_output, manager), None


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    #@staticmethod


    @staticmethod
    def forward(ctx, input_, manager):
        ctx.manager = manager
        return _split_along_last_dim(input_, manager)


    @staticmethod
    def backward(ctx, grad_output):
        manager = ctx.manager
        return _gather_along_last_dim(grad_output, manager), None


class GenericAllToAllFunction(Function):
    @staticmethod
    def forward(ctx, x, manager, split_dim, concat_dim):
        """Perform all-to-all conversion between tensor sharding dimensions."""


        tp_size = manager.get_mp_group_size()
        ctx.manager = manager
        ctx.split_dim = split_dim
        ctx.concat_dim = concat_dim
        ctx.tp_size = tp_size
        ctx.original_shape = x.shape


        batch, seq_len, hidden = x.shape


        if split_dim == 1:
            assert seq_len % tp_size == 0, f"seq_len {seq_len} must be divisible by tp_size {tp_size}"
            split_size = seq_len // tp_size
            concat_size = hidden
        elif split_dim == 2:
            assert hidden % tp_size == 0, f"hidden {hidden} must be divisible by tp_size {tp_size}"
            split_size = hidden // tp_size
            concat_size = seq_len
        else:
            raise ValueError("split_dim must be 1 or 2 for 3D tensors")


        if split_dim == 1 and concat_dim == 2:


            # x: [batch, seq_len, hidden] -> [batch, tp_size, seq_per_rank, hidden]
            send = x.view(batch, tp_size, split_size, hidden).permute(1, 0, 2, 3).contiguous()
            send_flat = send.view(tp_size * batch * split_size, hidden)

            if use_global_buffer:
                recv_flat = get_preallocated_buffer(send_flat.shape,
                                    send_flat.dtype,
                                    send_flat.device,
                                    manager.rank, # global_rank
                                    )
            else:
                recv_flat = torch.empty_like(send_flat)

            split_sizes = [batch * split_size] * tp_size

            dist.all_to_all_single(
                recv_flat, send_flat,
                input_split_sizes=split_sizes,
                output_split_sizes=split_sizes,
                group=manager.model_parallel_group,
            )

            # recv_flat: [tp_size * batch * split_size, hidden] -> [tp_size, batch, split_size, hidden]
            recv = recv_flat.view(tp_size, batch, split_size, hidden)
            # -> [batch, split_size, tp_size, hidden] -> [batch, split_size, tp_size * hidden]
            out = recv.permute(1, 2, 0, 3).contiguous().view(batch, split_size, tp_size * hidden)


        elif split_dim == 2 and concat_dim == 1:

            # x: [batch, seq_len, hidden] -> [batch, seq_len, tp_size, hidden_per_rank]
            send = x.view(batch, seq_len, tp_size, split_size).permute(2, 0, 1, 3).contiguous()
            send_flat = send.view(tp_size * batch * seq_len, split_size)


            if use_global_buffer:
                recv_flat = get_preallocated_buffer(send_flat.shape,
                                    send_flat.dtype,
                                    send_flat.device,
                                    manager.rank, # global_rank
                                    )
            else:
                recv_flat = torch.empty_like(send_flat)


            split_sizes = [batch * seq_len] * tp_size

            dist.all_to_all_single(
                recv_flat, send_flat,
                input_split_sizes=split_sizes,
                output_split_sizes=split_sizes,
                group=manager.model_parallel_group,
            )

            # recv_flat: [tp_size * batch * seq_len, split_size] -> [tp_size, batch, seq_len, split_size]
            recv = recv_flat.view(tp_size, batch, seq_len, split_size)
            # -> [batch, tp_size, seq_len, split_size] -> [batch, tp_size * seq_len, split_size]
            out = recv.permute(1, 0, 2, 3).contiguous().view(batch, tp_size * seq_len, split_size)

        else:
            raise ValueError(f"Unsupported combination: split_dim={split_dim}, concat_dim={concat_dim}")

        return out

    @staticmethod
    def backward(ctx, grad_output):


        return GenericAllToAllFunction.apply(grad_output, ctx.manager, ctx.concat_dim, ctx.split_dim), None, None, None


def generic_all_to_all(x, manager, split_dim, concat_dim): # 1 2
    return GenericAllToAllFunction.apply(x, manager, split_dim, concat_dim)


class GenericAllToAllGroupFunction(Function):
    @staticmethod
    def forward(ctx, x, group, group_size, split_dim, concat_dim):
        ctx.group = group
        ctx.group_size = group_size
        ctx.split_dim = split_dim
        ctx.concat_dim = concat_dim

        batch, seq_len, hidden = x.shape
        if split_dim == 1:
            assert seq_len % group_size == 0, f"seq_len {seq_len} must be divisible by group_size {group_size}"
            split_size = seq_len // group_size
        elif split_dim == 2:
            assert hidden % group_size == 0, f"hidden {hidden} must be divisible by group_size {group_size}"
            split_size = hidden // group_size
        else:
            raise ValueError("split_dim must be 1 or 2 for 3D tensors")

        if split_dim == 1 and concat_dim == 2:
            send = x.view(batch, group_size, split_size, hidden).permute(1, 0, 2, 3).contiguous()
            send_flat = send.view(group_size * batch * split_size, hidden)
            recv_flat = torch.empty_like(send_flat)
            split_sizes = [batch * split_size] * group_size
            dist.all_to_all_single(
                recv_flat,
                send_flat,
                input_split_sizes=split_sizes,
                output_split_sizes=split_sizes,
                group=group,
            )
            recv = recv_flat.view(group_size, batch, split_size, hidden)
            return recv.permute(1, 2, 0, 3).contiguous().view(batch, split_size, group_size * hidden)

        if split_dim == 2 and concat_dim == 1:
            send = x.view(batch, seq_len, group_size, split_size).permute(2, 0, 1, 3).contiguous()
            send_flat = send.view(group_size * batch * seq_len, split_size)
            recv_flat = torch.empty_like(send_flat)
            split_sizes = [batch * seq_len] * group_size
            dist.all_to_all_single(
                recv_flat,
                send_flat,
                input_split_sizes=split_sizes,
                output_split_sizes=split_sizes,
                group=group,
            )
            recv = recv_flat.view(group_size, batch, seq_len, split_size)
            return recv.permute(1, 0, 2, 3).contiguous().view(batch, group_size * seq_len, split_size)

        raise ValueError(f"Unsupported combination: split_dim={split_dim}, concat_dim={concat_dim}")

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = GenericAllToAllGroupFunction.apply(
            grad_output,
            ctx.group,
            ctx.group_size,
            ctx.concat_dim,
            ctx.split_dim,
        )
        return grad_input, None, None, None, None


def generic_all_to_all_group(x, group, group_size, split_dim, concat_dim):
    if group_size == 1:
        return x
    return GenericAllToAllGroupFunction.apply(x, group, group_size, split_dim, concat_dim)


class AllGatherReduceScatter(Function):
    @staticmethod
    def forward(ctx, x, manager, dim):

        group = manager.model_parallel_group

        ctx.manager = manager
        ctx.dim = dim
        ctx.world_size = dist.get_world_size(group)
        ctx.input_shape = x.shape


        if dim != 0:
            perm = [dim] + [i for i in range(x.dim()) if i != dim]
            x_t = x.permute(perm).contiguous()
        else:
            perm = None
            x_t = x.contiguous()


        out_shape = [ctx.world_size * x_t.shape[0]] + list(x_t.shape[1:])

        if use_global_buffer:
            gathered = get_preallocated_buffer(out_shape,
                                    x.dtype,
                                    x.device,
                                    manager.rank, # global_rank
                                    )
        else:
            gathered = torch.empty(out_shape, dtype=x.dtype, device=x.device)

        dist.all_gather_into_tensor(gathered, x_t, group=group)


        if perm is not None:
            inv_perm = [perm.index(i) for i in range(len(perm))]
            y = gathered.permute(inv_perm).contiguous()
        else:
            y = gathered

        return y


    @staticmethod
    def backward(ctx, grad_output):
        world_size = ctx.world_size
        dim = ctx.dim

        manager = ctx.manager
        group = manager.model_parallel_group


        if dim != 0:
            perm = [dim] + [i for i in range(grad_output.dim()) if i != dim]
            grad_t = grad_output.permute(perm).contiguous()
        else:
            perm = None
            grad_t = grad_output.contiguous()


        chunk_shape = list(grad_t.shape)
        chunk_shape[0] //= world_size

        if use_global_buffer:
            grad_input_t = get_preallocated_buffer(chunk_shape,
                                    grad_output.dtype,
                                    grad_output.device,
                                    manager.rank, # global_rank
                                    )
        else:
            grad_input_t = torch.empty(chunk_shape, dtype=grad_output.dtype, device=grad_output.device)


        dist.reduce_scatter_tensor(grad_input_t, grad_t, group=group, op=dist.ReduceOp.SUM)


        if perm is not None:
            inv_perm = [perm.index(i) for i in range(len(perm))]
            grad_input = grad_input_t.permute(inv_perm).contiguous()
        else:
            grad_input = grad_input_t

        return grad_input, None, None

class ReduceScatterAllGatherDim(Function):


    @staticmethod
    def forward(ctx, x, group, dim):
        if group is None:
            group = dist.group.WORLD

        dim = dim if dim >= 0 else x.dim() + dim

        world_size = dist.get_world_size(group)
        ctx.group = group
        ctx.dim = dim
        ctx.world_size = world_size
        ctx.input_shape = x.shape
        ctx.device = x.device
        ctx.dtype = x.dtype

        # flatten the dim to 0 for continuous memory if needed
        if dim != 0:

            perm = [dim] + [i for i in range(x.dim()) if i != dim]
            x_flat = x.permute(perm).contiguous()
        else:
            perm = None
            x_flat = x.contiguous()


        dim_size = x_flat.shape[0]
        chunk_size = dim_size // world_size
        out_shape = (chunk_size, *x_flat.shape[1:])
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

        # reduce_scatter_tensor
        dist.reduce_scatter_tensor(out, x_flat, group=group, op=dist.ReduceOp.SUM)


        if perm is not None:

            inv_perm = [perm.index(i) for i in range(len(perm))]
            out = out.permute(inv_perm).contiguous()

        return out


    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        dim = ctx.dim
        world_size = ctx.world_size
        input_shape = ctx.input_shape


        if dim != 0:
            perm = [dim] + [i for i in range(grad_output.dim()) if i != dim]
            grad_flat = grad_output.permute(perm).contiguous()
        else:
            perm = None
            grad_flat = grad_output.contiguous()


        dim_size = grad_flat.shape[0]
        gathered_shape = (world_size * dim_size, *grad_flat.shape[1:])
        gathered = torch.empty(gathered_shape, dtype=grad_flat.dtype, device=grad_flat.device)

        # all_gather_into_tensor
        dist.all_gather_into_tensor(gathered, grad_flat, group=group)


        grad_input = gathered


        if perm is not None:
            inv_perm = [perm.index(i) for i in range(len(perm))]
            grad_input = grad_input.permute(inv_perm).contiguous()

        return grad_input, None, None


#'''


class AllGatherReduceScatter_forMLP(Function):
    @staticmethod
    def forward(ctx, x, group, dim):
        ctx.group = group
        ctx.dim = dim
        ctx.world_size = dist.get_world_size(group)
        ctx.input_shape = x.shape


        gathered = torch.empty(
            [ctx.world_size * s if i == ctx.dim else s for i, s in enumerate(x.shape)],
            dtype=x.dtype, device=x.device
        )


        dist.all_gather_into_tensor(gathered, x.contiguous(), group=group)
        return gathered

    @staticmethod
    def backward(ctx, grad_output):
        world_size = ctx.world_size
        dim = ctx.dim
        group = ctx.group


        grad_input = torch.empty(ctx.input_shape, dtype=grad_output.dtype, device=grad_output.device)


        dist.reduce_scatter_tensor(
            grad_input,
            grad_output,
            group=group,
            op=dist.ReduceOp.SUM
        )

        return grad_input, None, None
#'''

#'''
class ReduceScatterAllGatherDim_forMLP(Function):
    @staticmethod
    def forward(ctx, x, group, dim):
        if group is None:
            group = dist.group.WORLD

        world_size = dist.get_world_size(group)
        ctx.group = group
        ctx.dim = dim
        ctx.world_size = world_size
        ctx.input_shape = x.shape


        output_shape = list(x.shape)
        output_shape[dim] = output_shape[dim] // world_size
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)


        dist.reduce_scatter_tensor(
            out,
            x.contiguous(),
            group=group,
            op=dist.ReduceOp.SUM
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        dim = ctx.dim
        world_size = ctx.world_size


        grad_input = torch.empty(
            ctx.input_shape,
            dtype=grad_output.dtype,
            device=grad_output.device
        )

        dist.all_gather_into_tensor(
            grad_input,
            grad_output.contiguous(),
            group=group
        )

        return grad_input, None, None
#'''
