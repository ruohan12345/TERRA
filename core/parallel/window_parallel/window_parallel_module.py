import torch
from torch.autograd import Function

from core.parallel.window_parallel.m_1_window_split import m_1_window_shift_with_direction


class unified_dist_window_shift(Function):
    @staticmethod
    def forward(ctx,
                x,
                manager,
                shift_direction_to_perm_list,
                shift_direction = 'upper_left',
                ):

        wp_group_h = manager.xfmr_wp_group_h
        wp_group_w = manager.xfmr_wp_group_w

        ctx.manager = manager
        ctx.shift_direction_to_perm_list = shift_direction_to_perm_list
        ctx.shift_direction = shift_direction

        if wp_group_h>0 and wp_group_w==1:
            x = m_1_window_shift_with_direction(
                                x,
                                manager,
                                shift_direction_to_perm_list,
                                shift_direction,
                                )
        else:
            print(' 2, 2 or m n is to be done')
            exit(0)

        return x


    @staticmethod
    def backward(ctx, grad_output):
        manager = ctx.manager
        shift_direction_to_perm_list = ctx.shift_direction_to_perm_list
        shift_direction = ctx.shift_direction

        if shift_direction == 'upper_left':
            reverse_shift_direction = 'lower_right'
        elif shift_direction == 'lower_right':
            reverse_shift_direction = 'upper_left'

        wp_group_h = manager.xfmr_wp_group_h
        wp_group_w = manager.xfmr_wp_group_w

        if wp_group_h>0 and wp_group_w==1:
            grad_input = m_1_window_shift_with_direction(
                                grad_output,
                                manager,
                                shift_direction_to_perm_list,
                                reverse_shift_direction,
                                )
        else:
            print(' 2, 2 or m n is to be done')
            exit(0)

        return grad_input, None, None, None
