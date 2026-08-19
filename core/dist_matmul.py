import torch
import torch.distributed as dist
from core.buffer import get_preallocated_buffer


USE_BROADCAST_COMM = True

def dist_XWT(X, W, manager, device,
    input_split,
    output_split,
    split_param,
):
    dp_rank = manager.get_dp_rank()
    mp_rank = manager.get_mp_rank()
    wp_rank = manager.get_wp_rank()

    global_rank = manager.get_global_rank(
        dp_rank = dp_rank,
        mp_rank = mp_rank,
        wp_rank = wp_rank,
        )

    mp_group = manager.model_parallel_group
    mp_group_size = manager.mp_group_size


    if input_split == '(1,n)':
        if output_split == '(m,1)': # done


            X_list = torch.split(X, int(X.shape[-2]//mp_group_size), dim=-2) # [2, 14400, 1674]
            ret = None
            for i in range(0, mp_group_size):
                XW = torch.matmul(X_list[i], W.mT) # [2, 14400, 768]
                dist.reduce(XW,
                        dst=manager.get_global_rank(
                                    dp_rank = dp_rank,
                                    mp_rank = i,
                                    wp_rank = wp_rank,
                                    ),
                        group=mp_group,
                        op=dist.ReduceOp.SUM)
                if mp_rank == i:
                    ret = XW
        elif output_split == '(1,n)': # done

            if True:


                W_list = torch.split(W, int(W.shape[-2]//mp_group_size), dim=-2) # [384, 1674]
                ret = None
                for i in range(0, mp_group_size):
                    XWT = torch.matmul(X, W_list[i].mT) # [2, 28800, 384]
                    dist.reduce(XWT,
                            dst=manager.get_global_rank(
                                        dp_rank = dp_rank,
                                        mp_rank = i,
                                        wp_rank = wp_rank,
                                        ),
                            group=mp_group,
                            op=dist.ReduceOp.SUM)
                    if mp_rank == i:
                        ret = XWT # [2, 28800, 384]
    elif input_split == '(m,1)':
        if output_split == '(m,1)':

            if USE_BROADCAST_COMM:
                buffer = get_preallocated_buffer(W.shape, W.dtype, device, global_rank)
                ret = None
                for source_rank in range(0, mp_group_size):
                    if mp_rank == source_rank:
                        buffer.copy_(W)
                    torch.distributed.broadcast(buffer,
                                                src=manager.get_global_rank(
                                                            dp_rank = dp_rank,
                                                            mp_rank = source_rank,
                                                            wp_rank = wp_rank,
                                                            ),
                                                group=mp_group)

                    if source_rank == 0:
                        ret = torch.matmul(X, buffer.mT)
                    else:
                        ret = torch.concat([ret, torch.matmul(X, buffer.mT)], dim=-1) # [2, 7200, 768]
            else:
                buffer_allW = get_preallocated_buffer((W.shape[0]*mp_group_size, W.shape[1]), W.dtype, device, global_rank)

                dist.all_gather_into_tensor(buffer_allW, W, group=mp_group)
                ret = torch.matmul(X, buffer_allW.mT) # [2, 14400, 768]
        elif output_split == '(1,n)':  # done


            if USE_BROADCAST_COMM:
                buffer = get_preallocated_buffer(X.shape, X.dtype, device, global_rank)
                ret = None
                for source_rank in range(0, mp_group_size):
                    if mp_rank == source_rank:

                        buffer.copy_(X)
                    torch.distributed.broadcast(buffer,
                                                src=manager.get_global_rank(
                                                            dp_rank = dp_rank,
                                                            mp_rank = source_rank,
                                                            wp_rank = wp_rank,
                                                            ),
                                                group=mp_group)

                    if source_rank == 0:
                        ret = torch.matmul(buffer, W.mT) # [2, 3348, 384]
                    else:
                        ret = torch.concat([ret, torch.matmul(buffer, W.mT)], dim=-2)
            else:

                X = X.permute(1, 0, 2).contiguous() # [14400, 2, 768]
                buffer_allX = get_preallocated_buffer((X.shape[0]*mp_group_size, X.shape[1], X.shape[2]), X.dtype, device, global_rank) # [28800, 2, 768]

                dist.all_gather_into_tensor(buffer_allX, X, group=mp_group)
                ret = torch.matmul(buffer_allX, W.mT)
                ret = ret.permute(1, 0, 2).contiguous()

    return ret


def dist_XW(X, W, manager, device,
    input_split,
    output_split,
    split_param,
):
    dp_rank = manager.get_dp_rank()
    mp_rank = manager.get_mp_rank()
    wp_rank = manager.get_wp_rank()

    global_rank = manager.get_global_rank(
        dp_rank = dp_rank,
        mp_rank = mp_rank,
        wp_rank = wp_rank,
        )

    mp_group = manager.model_parallel_group
    mp_group_size = manager.mp_group_size


    if X.dtype != W.dtype:
        W = W.to(X.dtype)


    if input_split == '(1,n)':
        if output_split == '(m,1)':


            if USE_BROADCAST_COMM:

                buffer = get_preallocated_buffer(X.shape, X.dtype, device, global_rank)
                ret = None
                for source_rank in range(0, mp_group_size):
                    if mp_rank == source_rank:

                        buffer.copy_(X)
                    torch.distributed.broadcast(buffer,
                                                src=manager.get_global_rank(
                                                            dp_rank = dp_rank,
                                                            mp_rank = source_rank,
                                                            wp_rank = wp_rank,
                                                            ),
                                                group=mp_group)

                    if source_rank == 0:
                        ret = torch.matmul(buffer, W) # [2, 28800, 837]
                    else:
                        ret = torch.concat([ret, torch.matmul(buffer, W)], dim=-2)
            else:

                X = X.permute(1, 0, 2).contiguous() # [14400, 2, 768]
                buffer_allX = get_preallocated_buffer((X.shape[0]*mp_group_size, X.shape[1], X.shape[2]), X.dtype, device, global_rank)
                dist.all_gather_into_tensor(buffer_allX, X, group=mp_group)
                ret = torch.matmul(buffer_allX, W) # [28800, 2, 1674]
                ret = ret.permute(1, 0, 2).contiguous()
        elif output_split == '(1,n)':


            if USE_BROADCAST_COMM:
                weight_tuple = torch.split(W, W.shape[-2]//mp_group_size, dim=-2) # [1674, 384]
                buffer = get_preallocated_buffer(X.shape, X.dtype, device, global_rank)
                ret = None
                for source_rank in range(0, mp_group_size):
                    if mp_rank == source_rank:

                        buffer.copy_(X)
                    torch.distributed.broadcast(buffer,

                                                src=manager.get_global_rank(
                                                            dp_rank = dp_rank,
                                                            mp_rank = source_rank,
                                                            wp_rank = wp_rank,
                                                            ),
                                                group=mp_group)
                    if source_rank == 0:
                        ret = torch.matmul(buffer, weight_tuple[source_rank]) # [2, 28800, 384]
                    else:
                        ret = ret + torch.matmul(buffer, weight_tuple[source_rank])
            else:
                X = X.permute(2, 0, 1).contiguous()
                buffer_allX = get_preallocated_buffer((X.shape[0]*mp_group_size, X.shape[1], X.shape[2]), X.dtype, device, global_rank)
                dist.all_gather_into_tensor(buffer_allX, X, group=mp_group) # [3348, 2, 28800]
                buffer_allX = buffer_allX.permute(1, 2, 0).contiguous()
                ret = torch.matmul(buffer_allX, W) # [2, 28800, 384]
    elif input_split == '(m,1)':
        if output_split == '(m,1)':


            if USE_BROADCAST_COMM:
                X_tuple = torch.split(X, X.shape[-1]//mp_group_size, dim=-1)
                buffer = get_preallocated_buffer(W.shape, W.dtype, device, global_rank)
                ret = None
                for source_rank in range(0, mp_group_size):
                    if mp_rank == source_rank:
                        buffer.copy_(W)
                    torch.distributed.broadcast(buffer,
                                                src=manager.get_global_rank(
                                                            dp_rank = dp_rank,
                                                            mp_rank = source_rank,
                                                            wp_rank = wp_rank,
                                                            ),
                                                group=mp_group)
                    if source_rank == 0:
                        ret = torch.matmul(X_tuple[source_rank], buffer) # [2, 7200, 768]
                    else:
                        ret = ret + torch.matmul(X_tuple[source_rank], buffer)
            else:
                buffer_allW = get_preallocated_buffer((W.shape[0]*mp_group_size, W.shape[1]), W.dtype, device, global_rank) # [3348, 768]
                dist.all_gather_into_tensor(buffer_allW, W, group=mp_group)
                ret = torch.matmul(X, buffer_allW) # [2, 14400, 768]
        elif output_split == '(1,n)':

            if True:
                X_list = torch.split(X, int(X.shape[-2]//mp_group_size), dim=-2) # [2, 14400, 1674]
                ret = None
                for i in range(0, mp_group_size):
                    XW = torch.matmul(X_list[i], W) # [2, 14400, 768]
                    dist.reduce(XW,
                            dst=manager.get_global_rank(
                                        dp_rank = dp_rank,
                                        mp_rank = i,
                                        wp_rank = wp_rank,
                                        ),
                            group=mp_group,
                            op=dist.ReduceOp.SUM)
                    if mp_rank == i:
                        ret = XW
    return ret


def dist_XTW(X, W, manager, device,
    input_split,
    output_split,
    split_param,
):
    dp_rank = manager.get_dp_rank()
    mp_rank = manager.get_mp_rank()
    wp_rank = manager.get_wp_rank()

    global_rank = manager.get_global_rank(
        dp_rank = dp_rank,
        mp_rank = mp_rank,
        wp_rank = wp_rank,
        )

    mp_group = manager.model_parallel_group
    mp_group_size = manager.mp_group_size


    if input_split == '(1,n)':
        if output_split == '(m,1)':


            if USE_BROADCAST_COMM:
                weight_tuple = torch.split(W, W.shape[-2]//mp_group_size, dim=-2)
                buffer = get_preallocated_buffer(X.shape, X.dtype, device, global_rank)
                ret = None
                for source_rank in range(0, mp_group_size):
                    if mp_rank == source_rank:
                        buffer.copy_(X)
                    torch.distributed.broadcast(buffer,
                                                src=manager.get_global_rank(
                                                            dp_rank = dp_rank,
                                                            mp_rank = source_rank,
                                                            wp_rank = wp_rank,
                                                            ),
                                                group=mp_group)
                    if source_rank == 0:
                        ret = torch.matmul(buffer.mT, weight_tuple[source_rank]) # [2, 768, 837]
                    else:
                        ret = ret + torch.matmul(buffer.mT, weight_tuple[source_rank])
            else:
                X = X.permute(1, 0, 2).contiguous()
                buffer_allX = get_preallocated_buffer((X.shape[0]*mp_group_size, X.shape[1], X.shape[2]), X.dtype, device, global_rank)
                dist.all_gather_into_tensor(buffer_allX, X, group=mp_group) # [28800, 2, 768]
                buffer_allX = buffer_allX.permute(1, 2, 0).contiguous() # [2, 768, 28800]
                ret = torch.matmul(buffer_allX, W) # [2, 768, 1674]
        elif output_split == '(1,n)':


            if USE_BROADCAST_COMM:
                buffer = get_preallocated_buffer(X.shape, X.dtype, device, global_rank)
                ret = None
                for source_rank in range(0, mp_group_size):
                    if mp_rank == source_rank:

                        buffer.copy_(X)
                    torch.distributed.broadcast(buffer,
                                                src=manager.get_global_rank(
                                                            dp_rank = dp_rank,
                                                            mp_rank = source_rank,
                                                            wp_rank = wp_rank,
                                                            ),
                                                group=mp_group)

                    if source_rank == 0:
                        ret = torch.matmul(buffer.mT, W) # [2, 3348, 384]
                    else:
                        ret = torch.concat([ret, torch.matmul(buffer.mT, W)], dim=-2)
            else:
                X = X.permute(2, 0, 1).contiguous()
                buffer_allX = get_preallocated_buffer((X.shape[0]*mp_group_size, X.shape[1], X.shape[2]), X.dtype, device, global_rank)
                dist.all_gather_into_tensor(buffer_allX, X, group=mp_group)
                buffer_allX = buffer_allX.permute(1, 2, 0).contiguous()

                ret = torch.matmul(buffer_allX.mT, W)
    elif input_split == '(m,1)':
        if output_split == '(m,1)':


            X_list = torch.split(X, int(X.shape[-1]//mp_group_size), dim=-1) # [2, 14400, 1674]

            ret = None
            for i in range(0, mp_group_size):
                XW = torch.matmul(X_list[i].mT, W) # [2, 837, 768]
                dist.reduce(XW,
                            dst=manager.get_global_rank(
                                        dp_rank = dp_rank,
                                        mp_rank = i,
                                        wp_rank = wp_rank,
                                        ),
                            group=mp_group,
                            op=dist.ReduceOp.SUM)
                if mp_rank == i:
                    ret = XW # [2, 1674, 768]
        elif output_split == '(1,n)':


            if USE_BROADCAST_COMM:
                X_tuple = torch.split(X, X.shape[-2]//mp_group_size, dim=-2) # [2, 7200, 837]
                buffer = get_preallocated_buffer(W.shape, W.dtype, device, global_rank)
                ret = None
                for source_rank in range(0, mp_group_size):
                    if mp_rank == source_rank:
                        buffer.copy_(W)
                    torch.distributed.broadcast(buffer,
                                                src=manager.get_global_rank(
                                                            dp_rank = dp_rank,
                                                            mp_rank = source_rank,
                                                            wp_rank = wp_rank,
                                                            ),
                                                group=mp_group)
                    if source_rank == 0:
                        ret = torch.matmul(X_tuple[source_rank].mT, buffer) # [2, 837, 768]
                    else:
                        ret = ret + torch.matmul(X_tuple[source_rank].mT, buffer)
            else:

                W = W.permute(1, 0, 2).contiguous() # [14400, 2, 768]
                buffer_allW = get_preallocated_buffer((W.shape[0]*mp_group_size, W.shape[1], W.shape[2]), W.dtype, device, global_rank) # [28800, 2, 768]

                dist.all_gather_into_tensor(buffer_allW, W, group=mp_group)
                buffer_allW = buffer_allW.permute(1, 0, 2).contiguous()

                ret = torch.matmul(X.mT, buffer_allW) # [2, 1674, 768]

    return ret


class XWT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, bias, manager, device,
        input_split,
        output_split,
        split_param,
        use_bias = True):
        ctx.save_for_backward(x, W)
        ctx.manager = manager

        ctx.device = device
        ctx.use_bias = use_bias

        ctx.input_split = input_split
        ctx.output_split = output_split
        ctx.split_param = split_param


        xwt = dist_XWT(x, W, manager, device,
            input_split,
            output_split,
            split_param
            ) # [2, 14400, 768]

        if use_bias:
            bias = bias.to(xwt.dtype)
            xwt = xwt.add(bias)


        return xwt


    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        manager = ctx.manager


        device = ctx.device
        use_bias = ctx.use_bias

        input_split = ctx.input_split
        output_split = ctx.output_split
        split_param = ctx.split_param


        grad_input  = dist_XW(grad_output, weight, manager, device,
            input_split,
            output_split,
            split_param,
            )


        grad_weight = dist_XTW(grad_output, input, manager, device,
            input_split,
            output_split,
            split_param,
            )


        if use_bias:
            grad_b = grad_output
            return grad_input, grad_weight, grad_b, None, None, None, None, None, None, None, None
        else:
            return grad_input, grad_weight, None, None, None, None, None, None, None, None, None


def XWT_apply(x, weight, bias, manager, device,
    input_split = None,
    output_split = None,
    split_param = True,
    use_bias = True):

    return XWT.apply(x, weight, bias, manager, device,
        input_split,
        output_split,
        split_param,
        use_bias)
