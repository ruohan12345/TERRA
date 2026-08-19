import torch

from core.global_env_config import USE_FAKE_INPUT
from dataloader.glorys_paths import resolve_glorys_mask_path


def _selected_glorys_channels(num_ocean_layers=23):
    if num_ocean_layers != 23:
        raise ValueError("Only the public 23-level GLORYS layout is supported")
    depth_indices = [
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    ]
    channels = [variable * 40 + depth for variable in range(4) for depth in depth_indices]
    channels.append(160)
    return channels


def get_land_sea_mask():
    if USE_FAKE_INPUT:
        return torch.ones((93, 2041, 4320), dtype=torch.bool)

    mask = torch.load(
        resolve_glorys_mask_path(),
        weights_only=True,
        map_location="cpu",
    )
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 3:
        raise RuntimeError(
            f"Expected GLORYS mask with shape [C,H,W] or [1,C,H,W], got {tuple(mask.shape)}"
        )
    if mask.shape[0] != 93:
        mask = mask[_selected_glorys_channels()]
    if mask.shape[0] != 93:
        raise RuntimeError(f"Expected 93-channel GLORYS mask, got {tuple(mask.shape)}")
    return mask.contiguous()


def get_wp_slice_from_bhwc(
    x,
    num_channel,
    patch_size,
    window_size,
    wp_topo,
    wp_rank=-1,
):
    if x.ndim != 4:
        raise ValueError(f"Expected BHWC input, got shape={tuple(x.shape)}")
    batch, height, width, channels = x.shape
    if channels != num_channel:
        raise ValueError(f"Expected {num_channel} channels, got {channels}")
    token_h, token_w = height // patch_size, width // patch_size
    if height % patch_size or width % patch_size:
        raise ValueError("Input resolution must be divisible by patch_size")
    if token_h % window_size or token_w % window_size:
        raise ValueError("Token resolution must form complete attention windows")

    hidden = patch_size * patch_size * num_channel
    patches = x.reshape(
        batch, token_h, patch_size, token_w, patch_size, num_channel
    )
    patches = patches.permute(0, 1, 3, 2, 4, 5).contiguous()
    patches = patches.reshape(batch, token_h, token_w, hidden)

    num_windows_h = token_h // window_size
    num_windows_w = token_w // window_size
    windows = patches.reshape(
        batch,
        num_windows_h,
        window_size,
        num_windows_w,
        window_size,
        hidden,
    )
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.reshape(
        batch,
        num_windows_h,
        num_windows_w,
        window_size * window_size,
        hidden,
    )

    wp_group_h, wp_group_w = wp_topo
    if not 0 <= wp_rank < wp_group_h * wp_group_w:
        raise ValueError(f"Invalid wp_rank={wp_rank} for wp_topo={wp_topo}")

    assigned = []
    if wp_group_w == 1:
        if num_windows_h % wp_group_h:
            raise ValueError(
                f"num_windows_h={num_windows_h} must be divisible by wp_group_h={wp_group_h}"
            )
        rows_per_rank = num_windows_h // wp_group_h
        row_start = wp_rank * rows_per_rank
        row_end = row_start + rows_per_rank
        assigned = [
            (row, col)
            for row in range(row_start, row_end)
            for col in range(num_windows_w)
        ]
    else:
        assigned = [
            (row, col)
            for row in range(num_windows_h)
            for col in range(num_windows_w)
            if (col % wp_group_w) + (row % wp_group_h) * wp_group_w == wp_rank
        ]

    if not assigned:
        raise RuntimeError(f"wp_rank={wp_rank} owns no windows")
    return torch.stack([windows[:, row, col] for row, col in assigned], dim=1).clone()
