import torch
from torch import nn

def get_pad3d(input_resolution, window_size):
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back

def get_pad2d(input_resolution, window_size):
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[: 4]


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def window_partition_HWBC(x, window_size):
    H, W, B, C = x.shape
    x = x.view(H // window_size, window_size, W // window_size, window_size, B, C)

    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, window_size, window_size, C) # [num_windows*B, 6, 6, 384]
    return windows


def get_swin_attention_mask(
    model_type,
    input_resolution,
    window_size,
    shift_size,
    manager = None,
    ):


    # calculate attention mask for SW-MSA
    H, W = input_resolution
    img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)) # [800, 36, 36]


    if model_type=='parallel':


        mask_tuple = torch.split(attn_mask, attn_mask.shape[0]//manager.get_mp_group_size(), dim = 0) # dim is not important for bias
        attn_mask = mask_tuple[manager.get_mp_rank()] # [400, 36, 36]


    return attn_mask


# Adapted from ``credit/models/fuxi.py`` in NCAR/miles-credit (Apache-2.0),
# reviewed at commit ac83d0a0d67e029af5e57babb100b7bbd0ace78e.
# TERRA applies checkpointing around these public reference blocks rather than
# embedding checkpoint behavior in the blocks themselves.
class CreditDownBlock(nn.Module):
    """CREDIT FuXi-style factor-two convolutional down-sampling block."""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        num_groups: int,
        num_residuals: int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chans,
            out_chans,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
        )
        residual_layers = []
        for _ in range(num_residuals):
            residual_layers.extend(
                [
                    nn.Conv2d(
                        out_chans,
                        out_chans,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GroupNorm(num_groups, out_chans),
                    nn.SiLU(),
                ]
            )
        self.b = nn.Sequential(*residual_layers)

    def forward(self, x):
        x = self.conv(x)
        return self.b(x) + x


class CreditUpBlock(nn.Module):
    """CREDIT FuXi-style factor-two transposed-convolution up-sampling block."""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        num_groups: int,
        num_residuals: int = 2,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_chans,
            out_chans,
            kernel_size=2,
            stride=2,
        )
        residual_layers = []
        for _ in range(num_residuals):
            residual_layers.extend(
                [
                    nn.Conv2d(
                        out_chans,
                        out_chans,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GroupNorm(num_groups, out_chans),
                    nn.SiLU(),
                ]
            )
        self.b = nn.Sequential(*residual_layers)

    def forward(self, x):
        x = self.conv(x)
        return self.b(x) + x
