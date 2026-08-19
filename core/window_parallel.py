import torch
from torch.autograd import Function


'''
input_split_sizes = [
    0,              # to rank0
    chunk_numel,    # to rank1
    chunk_numel,    # to rank2
    chunk_numel,    # to rank3
]

output_split_sizes = [
    0,              # from rank0
    chunk_numel,    # from rank1
    chunk_numel,    # from rank2
    chunk_numel,    # from rank3
]
'''


from torch.distributed import P2POp
from core.parallel.window_assignment import get_window_assignment_mode, get_window_indices


def _get_aeris_shift_source(ii, jj, corner, shift_direction, num_windows_h, num_windows_w):
    if shift_direction == 'upper_left':
        if corner == 0:
            return ii, jj, 3
        if corner == 1:
            return ii, (jj + 1) % num_windows_w, 2
        if corner == 2:
            return (ii + 1) % num_windows_h, jj, 1
        if corner == 3:
            return (ii + 1) % num_windows_h, (jj + 1) % num_windows_w, 0
    elif shift_direction == 'lower_right':
        if corner == 0:
            return (ii - 1) % num_windows_h, (jj - 1) % num_windows_w, 3
        if corner == 1:
            return (ii - 1) % num_windows_h, jj, 2
        if corner == 2:
            return ii, (jj - 1) % num_windows_w, 1
        if corner == 3:
            return ii, jj, 0

    raise RuntimeError(f"unsupported Aeris shift: corner={corner}, shift_direction={shift_direction}")


def _get_generic_aeris_route(manager, shift_direction, device):
    wp_group_h = manager.xfmr_wp_group_h
    wp_group_w = manager.xfmr_wp_group_w


    wp_group_size = manager.get_wp_group_size()
    wp_rank = manager.get_wp_rank()

    if not hasattr(manager, "_generic_aeris_route_cache"):
        manager._generic_aeris_route_cache = {}

    cache_key = (
        manager.num_windows_h,
        manager.num_windows_w,
        wp_group_h,
        wp_group_w,
        wp_rank,
        shift_direction,
        get_window_assignment_mode(manager),
        str(device),
    )
    if cache_key in manager._generic_aeris_route_cache:
        return manager._generic_aeris_route_cache[cache_key]

    rank_to_indices, global_to_local = get_window_indices(
        manager.num_windows_h,
        manager.num_windows_w,
        wp_group_h,
        wp_group_w,
        mode=get_window_assignment_mode(manager),
        debug_rank=getattr(manager, "xfmr_window_group_rank", wp_rank),
        debug_global_rank=getattr(manager, "rank", None),
    )
    local_indices = rank_to_indices[wp_rank]

    send_src_local_idx_by_dst = []
    send_src_corner_by_dst = []
    input_split_sizes = []
    for dst_rank in range(wp_group_size):
        src_local_idx_list = []
        src_corner_list = []
        for _, (ii, jj) in enumerate(rank_to_indices[dst_rank]):
            for dst_corner in range(4):
                src_i, src_j, src_corner = _get_aeris_shift_source(
                    ii,
                    jj,
                    dst_corner,
                    shift_direction,
                    manager.num_windows_h,
                    manager.num_windows_w,
                )
                src_rank, src_local_idx = global_to_local[(src_i, src_j)]
                if src_rank == wp_rank:
                    src_local_idx_list.append(src_local_idx)
                    src_corner_list.append(src_corner)

        input_split_sizes.append(len(src_local_idx_list))
        send_src_local_idx_by_dst.append(
            torch.tensor(src_local_idx_list, dtype=torch.long, device=device)
        )
        send_src_corner_by_dst.append(
            torch.tensor(src_corner_list, dtype=torch.long, device=device)
        )

    recv_dst_local_idx_by_src = []
    recv_dst_corner_by_src = []
    output_split_sizes = []
    for src_rank in range(wp_group_size):
        dst_local_idx_list = []
        dst_corner_list = []
        for dst_local_idx, (ii, jj) in enumerate(local_indices):
            for dst_corner in range(4):
                src_i, src_j, _ = _get_aeris_shift_source(
                    ii,
                    jj,
                    dst_corner,
                    shift_direction,
                    manager.num_windows_h,
                    manager.num_windows_w,
                )
                cur_src_rank, _ = global_to_local[(src_i, src_j)]
                if cur_src_rank == src_rank:
                    dst_local_idx_list.append(dst_local_idx)
                    dst_corner_list.append(dst_corner)

        output_split_sizes.append(len(dst_local_idx_list))
        recv_dst_local_idx_by_src.append(
            torch.tensor(dst_local_idx_list, dtype=torch.long, device=device)
        )
        recv_dst_corner_by_src.append(
            torch.tensor(dst_corner_list, dtype=torch.long, device=device)
        )

    route = {
        "rank_to_indices": rank_to_indices,
        "local_window_count": len(local_indices),
        "send_src_local_idx_by_dst": send_src_local_idx_by_dst,
        "send_src_corner_by_dst": send_src_corner_by_dst,
        "input_split_sizes": input_split_sizes,
        "recv_dst_local_idx_by_src": recv_dst_local_idx_by_src,
        "recv_dst_corner_by_src": recv_dst_corner_by_src,
        "output_split_sizes": output_split_sizes,
    }
    manager._generic_aeris_route_cache[cache_key] = route
    return route


def generic_aeris_2d_p2p_window_shift_with_direction(x, manager, shift_direction):
    assert x.dim() == 5

    wp_group_h = manager.xfmr_wp_group_h
    wp_group_w = manager.xfmr_wp_group_w


    wp_group_size = manager.get_wp_group_size()
    wp_rank = manager.get_wp_rank()
    wp_group = manager.window_parallel_group

    if wp_group_h * wp_group_w != wp_group_size:
        raise RuntimeError(
            f"Invalid wp topology: ({wp_group_h}, {wp_group_w}) vs wp_group_size={wp_group_size}"
        )

    #    raise RuntimeError("generic Aeris P2P shift is for 2D/window-embedding topology, not (m, 1)")


    route = _get_generic_aeris_route(manager, shift_direction, x.device)
    local_window_count = route["local_window_count"]
    if local_window_count != x.shape[1]:
        raise RuntimeError(
            f"local window count mismatch: expected {local_window_count}, got {x.shape[1]}"
        )

    B, _, _, sub_tokens, hidden_dim = x.shape
    x_flat = x.view(B, local_window_count * 4, sub_tokens, hidden_dim)
    out = torch.empty_like(x)
    out_flat = out.view(B, local_window_count * 4, sub_tokens, hidden_dim)

    send_tensors = {}
    recv_tensors = {}
    ops = []

    for peer_rank in range(wp_group_size):
        src_local_idx = route["send_src_local_idx_by_dst"][peer_rank]
        src_corner = route["send_src_corner_by_dst"][peer_rank]
        send_count = src_local_idx.numel()
        if send_count > 0:
            send_flat_idx = src_local_idx * 4 + src_corner
            send_tensor = torch.index_select(
                x_flat,
                dim=1,
                index=send_flat_idx,
            ).permute(1, 0, 2, 3).contiguous()
            send_tensors[peer_rank] = send_tensor

        recv_count = route["output_split_sizes"][peer_rank]
        if recv_count > 0:
            recv_tensors[peer_rank] = torch.empty(
                (recv_count, B, sub_tokens, hidden_dim),
                dtype=x.dtype,
                device=x.device,
            )

    for peer_rank, send_tensor in send_tensors.items():
        if peer_rank == wp_rank:
            continue
        peer_global_rank = manager.get_global_rank(
            dp_rank=manager.get_dp_rank(),
            mp_rank=manager.get_mp_rank(),
            wp_rank=peer_rank,
        )
        ops.append(
            P2POp(
                torch.distributed.isend,
                send_tensor,
                peer_global_rank,
                wp_group,
            )
        )

    for peer_rank, recv_tensor in recv_tensors.items():
        if peer_rank == wp_rank:
            continue
        peer_global_rank = manager.get_global_rank(
            dp_rank=manager.get_dp_rank(),
            mp_rank=manager.get_mp_rank(),
            wp_rank=peer_rank,
        )
        ops.append(
            P2POp(
                torch.distributed.irecv,
                recv_tensor,
                peer_global_rank,
                wp_group,
            )
        )

    if ops:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    if wp_rank in send_tensors:
        recv_tensors[wp_rank] = send_tensors[wp_rank]

    for peer_rank, recv_tensor in recv_tensors.items():
        dst_local_idx = route["recv_dst_local_idx_by_src"][peer_rank]
        dst_corner = route["recv_dst_corner_by_src"][peer_rank]
        if dst_local_idx.numel() == 0:
            continue
        dst_flat_idx = dst_local_idx * 4 + dst_corner
        out_flat.index_copy_(
            1,
            dst_flat_idx,
            recv_tensor.permute(1, 0, 2, 3).contiguous(),
        )

    return out

def window_shift_with_direction(x,
                                manager,
                                shift_direction,
                                ):
    assert x.dim() ==5
    return generic_aeris_2d_p2p_window_shift_with_direction(
        x,
        manager,
        shift_direction,
    )


class Dist_Window_Shift(Function):
    @staticmethod
    def forward(ctx,
                x,
                manager,
                shift_direction_to_perm_list,
                shift_direction = 'upper_left',
                ): # [2, 200, 4, 9, 768], wp_group_size = 4

        ctx.manager = manager
        #ctx.shift_direction_to_perm_list = shift_direction_to_perm_list
        ctx.shift_direction = shift_direction

        x = window_shift_with_direction(x,
                                manager,
                                #shift_direction_to_perm_list,
                                shift_direction,
                                )


        return x # [2, 200, 4, 9, 768]


    @staticmethod
    def backward(ctx, grad_output): # [B, 200, 4, 9, 768]

        manager = ctx.manager

        shift_direction = ctx.shift_direction

        if shift_direction == 'upper_left':
            reverse_shift_direction = 'lower_right'
        elif shift_direction == 'lower_right':
            reverse_shift_direction = 'upper_left'

        grad_input = window_shift_with_direction(grad_output,
                                manager,
                                #shift_direction_to_perm_list,
                                reverse_shift_direction,
                                )

        return grad_input, None, None, None, None
