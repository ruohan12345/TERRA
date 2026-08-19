import os
import numpy as np
import torch
import torch.distributed as dist

from timm.models.layers import to_2tuple
from models.utils import get_pad2d


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def init_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    #torch.cuda.set_device(rank % torch.cuda.device_count())
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank = rank,

        init_method="env://"
        )
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()

    return rank, local_rank, device, world_size


def get_task_partition(rank, world_size, all_days=331, div_10 = False):
    if div_10:
        total_numbers = all_days // 10 + 1  # 34
    else:
        total_numbers = all_days

    chunk_size = total_numbers // world_size  # 4
    remainder = total_numbers % world_size   # 2


    if rank < (world_size - remainder):

        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size
    else:

        offset = (world_size - remainder) * chunk_size
        start_idx = offset + (rank - (world_size - remainder)) * (chunk_size + 1)
        end_idx = start_idx + (chunk_size + 1)

    return start_idx, end_idx


def sort_key(file_path):
    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    return int(name_without_ext)


def all_reduce_and_print_rank0(x,
                            rank,
                            group,
                            description):

    tensor = x.clone().detach()

    torch.distributed.all_reduce(tensor,
                                op=torch.distributed.ReduceOp.SUM,
                                group=group)
    if rank==0:
        print(description,
              #tensor.sum(),
              f'{tensor.sum().item():.20f}',
              tensor.dtype)
    # f'{grad_sum.item():.20f}'


def get_criterion(loss_func):
    if loss_func == 'L1':
        criterion = torch.nn.L1Loss(reduction = 'mean')
    elif loss_func == 'L2' or loss_func == 'MSE':
        criterion = torch.nn.MSELoss(reduction='mean')
    else:
        print('unsupported loss_func', loss_func)
        exit(0)

    return criterion


def get_padded_shape(height, width, patch_size, window_size, padding_scale=2, padded_shape=None):
    image_size = (height, width)
    if padded_shape is not None:
        padded_h, padded_w = tuple(padded_shape)
        if padded_h < height or padded_w < width:
            raise ValueError(
                f"padded_shape={padded_shape} must cover input shape={(height, width)}"
            )
        pad_h = padded_h - height
        pad_w = padded_w - width
        padding_top = pad_h // 2
        padding_bottom = pad_h - padding_top
        padding_left = pad_w // 2
        padding_right = pad_w - padding_left
        initial_padding = (padding_left, padding_right, padding_top, padding_bottom)
    else:

        padding_size=patch_size*(padding_scale*window_size)

        tp_padding_size = to_2tuple(padding_size)
        initial_padding = get_pad2d(image_size, tp_padding_size)
    padding_left, padding_right, padding_top, padding_bottom = initial_padding
    padded_shape = (height+padding_top+padding_bottom, width+padding_left+padding_right)
    need_padding = (padding_left+padding_right+padding_top+padding_bottom)>0

    return need_padding, initial_padding, padded_shape

def check_parallel_config(
    dp_size,
    mp_size,
    wp_topo,
    domain_topo, # (1, 2)
    world_size,
    height,
    width,
    patch_size,
    window_size,
    padding_scale = 1,
):
    wp_group_h, wp_group_w = wp_topo
    wp_size = wp_group_h*wp_group_w


    assert dp_size * mp_size * wp_size == world_size, f"dp_size * mp_size * wp_size ({dp_size * mp_size * wp_size}) != world_size ({world_size})"


    need_padding, initial_padding, padded_shape = get_padded_shape(height, width, patch_size, window_size, padding_scale=padding_scale)
    assert ((padded_shape[0]//patch_size)//window_size)%wp_group_h == 0  and  ((padded_shape[1]//patch_size)//window_size)%wp_group_w == 0, f'window error'


    assert domain_topo[0]*domain_topo[1] == mp_size
