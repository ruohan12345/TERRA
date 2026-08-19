import torch
from core.buffer import get_preallocated_buffer
from core.p2p_comm import ring_transfer


def m_1_window_shift_with_direction(x, # [1, 990, 4, 9, 768]
                                manager,
                                shift_direction_to_perm_list,
                                shift_direction,
                                ):
    assert x.dim() ==5
    wp_group_size = manager.get_wp_group_size()

    assert manager.xfmr_wp_group_w == 1, f"m_1_window_shift only supports (m, 1), got ({manager.xfmr_wp_group_h}, {manager.xfmr_wp_group_w})"
    assert wp_group_size == manager.xfmr_wp_group_h


    wp_rank = manager.get_wp_rank()
    dp_rank = manager.get_dp_rank()
    mp_rank = manager.get_mp_rank()

    wp_group = manager.window_parallel_group

    global_rank = manager.get_global_rank(
        dp_rank = manager.get_dp_rank(),
        mp_rank = manager.get_mp_rank(),
        wp_rank = wp_rank,
        )
    device = x.device

    perm_list = shift_direction_to_perm_list[shift_direction]
    perm1 = perm_list[0]
    perm2 = perm_list[1]
    perm3 = perm_list[2]

    num_windows_h = manager.num_windows_h # 44, 90
    num_windows_w = manager.num_windows_w

    assert num_windows_h % wp_group_size == 0, (
        f"num_windows_h={num_windows_h} must be divisible by wp_group_size={wp_group_size}. "
        "Increase padding_scale or choose a compatible wp_topo."
    )

    num_window_per_rank = (num_windows_h*num_windows_w)//wp_group_size
    assert num_window_per_rank == x.shape[1] # 990

    recv_buffer = get_preallocated_buffer(x[:, 0:num_windows_w, 0].shape, x.dtype, device, global_rank, index = 0) # [1, 90, 9, 768]
    tmp_buffer = get_preallocated_buffer(x[:, num_windows_w:num_window_per_rank, 0].shape, x.dtype, device, global_rank, index = 1) # [1, 900, 9, 768]


    '''
    if wp_group_size == 4:
        if shift_direction == 'upper_left':
            rank_mapping_src_2_dst = {0: 3, 1: 0, 2: 1, 3: 2}
        elif shift_direction == 'lower_right':
            rank_mapping_src_2_dst = {0: 1, 1: 2, 2: 3, 3: 0}
    '''
    if shift_direction == 'upper_left':
        rank_mapping_src_2_dst = {
            src_rank: (src_rank - 1) % wp_group_size
            for src_rank in range(wp_group_size)
        }
    elif shift_direction == 'lower_right':
        rank_mapping_src_2_dst = {
            src_rank: (src_rank + 1) % wp_group_size
            for src_rank in range(wp_group_size)
        }
    else:
        raise RuntimeError(f"unsupported shift_direction: {shift_direction}")


    if shift_direction == 'upper_left':


        send_data = x[:, 0:num_windows_w, 0].contiguous()

        ring_transfer(send_data, recv_buffer, ring_topo = rank_mapping_src_2_dst, manager=manager)

        tmp_buffer.copy_(x[:, num_windows_w:num_window_per_rank, 0])


        x[:, :, 0] = x[:, :, 3]


        #'''
        #x[:, 0:num_windows_w, 3] = recv_buffer
        #x[:, num_windows_w:num_window_per_rank, 3] = tmp_buffer

        #x[:, 0:(num_window_per_rank-num_windows_w), 3] = tmp_buffer
        #x[:, (num_window_per_rank-num_windows_w):num_window_per_rank, 3] = recv_buffer
        x[:, 0:(num_window_per_rank-num_windows_w), 3].copy_(tmp_buffer)
        x[:, (num_window_per_rank-num_windows_w):num_window_per_rank, 3].copy_(recv_buffer)


        x[:, :, 3] = torch.index_select(x[:, :, 3], dim=1, index=perm3)

        send_data = x[:, 0:num_windows_w, 1].contiguous()
        ring_transfer(send_data, recv_buffer, ring_topo = rank_mapping_src_2_dst, manager=manager)

        tmp_buffer.copy_(x[:, num_windows_w:num_window_per_rank, 1])


        x[:, :, 1] = torch.index_select(x[:, :, 2], dim=1, index=perm2)


        # x[:, 0:num_windows_w, 2] = recv_buffer
        # x[:, num_windows_w:num_window_per_rank, 2] = tmp_buffer

        # x[:, 0:(num_window_per_rank-num_windows_w), 2] = tmp_buffer
        # x[:, (num_window_per_rank-num_windows_w):num_window_per_rank, 2] = recv_buffer

        x[:, 0:(num_window_per_rank-num_windows_w), 2].copy_(tmp_buffer)
        x[:, (num_window_per_rank-num_windows_w):num_window_per_rank, 2].copy_(recv_buffer)

    elif shift_direction == 'lower_right':

        send_data = x[:, (num_window_per_rank-num_windows_w):num_window_per_rank, 3].contiguous() # [1, 90, 9, 768]

        ring_transfer(send_data, recv_buffer, ring_topo = rank_mapping_src_2_dst, manager=manager)


        tmp_buffer.copy_(x[:, 0:(num_window_per_rank-num_windows_w), 3])


        if True:
            x[:, :, 3].copy_(x[:, :, 0].clone())
        else:
            x[:, :, 3] = x[:, :, 0]


        # x[:, 0:num_windows_w, 0] = recv_buffer
        # x[:, num_windows_w:num_window_per_rank, 0] = tmp_buffer

        x[:, 0:num_windows_w, 0].copy_(recv_buffer)


        x[:, num_windows_w:num_window_per_rank, 0].copy_(tmp_buffer)

        x[:, :, 0] = torch.index_select(x[:, :, 0], dim=1, index=perm2)


        send_data = x[:, (num_window_per_rank-num_windows_w):num_window_per_rank, 2].contiguous() # [1, 90, 9, 768]
        ring_transfer(send_data, recv_buffer, ring_topo = rank_mapping_src_2_dst, manager=manager)

        tmp_buffer.copy_(x[:, 0:(num_window_per_rank-num_windows_w), 2])

        #x[:, :, 2] = x[:, :, 1]
        x[:, :, 2] = torch.index_select(x[:, :, 1], dim=1, index=perm3)


        # x[:, 0:num_windows_w, 1] = recv_buffer
        # x[:, num_windows_w:num_window_per_rank, 1] = tmp_buffer
        x[:, 0:num_windows_w, 1].copy_(recv_buffer)
        x[:, num_windows_w:num_window_per_rank, 1].copy_(tmp_buffer)


    return x
