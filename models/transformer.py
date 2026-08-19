import torch
from timm.models.layers import to_2tuple
from einops import rearrange

from models.utils import window_partition, window_reverse, get_swin_attention_mask, window_partition_HWBC
from models.attention import WindowAttention, ParallelWindowAttention, UlyssesWindowAttention, WPUlyssesWindowAttention

from core.tensor_parallel import AllGatherReduceScatter
from core.tensor_parallel import generic_all_to_all

from core.window_parallel import Dist_Window_Shift


from core.parallel.window_parallel.window_parallel_module import unified_dist_window_shift


from core.global_env_config import use_layernorm, use_attn_mask as default_use_attn_mask
from core.global_env_config import use_relative_position_bias as default_use_relative_position_bias
from core.global_env_config import USE_FLASH_ATTENTION as default_use_flash_attention


from core.global_env_config import use_shift_size_0

from core.parallel.layout_transform import stripe_grid_to_ulysses_windows, ulysses_windows_to_stripe_grid, ulysses_windows_to_ulysses_windows
from core.parallel.window_assignment import get_window_assignment_mode, get_window_indices, is_terra_m1_assignment_mode


import copy


WP_DEBUG_RANK = 0

def get_wp_window_slice_from_full(x,
                                  manager,
                                  window_h, # 120
                                  window_w, # 240
                                  window_size,
                                  ): # [800, 36, 36]

    assert x.dim()==3 # # [800, 36, 36]

    if manager==None:


        wp_group_h = 2
        wp_group_w = 1
        wp_rank = WP_DEBUG_RANK
    else:
        wp_group_h = manager.xfmr_wp_group_h # 2
        wp_group_w = manager.xfmr_wp_group_w # 2
        wp_rank = manager.get_wp_rank()


    x = x.view(window_h//window_size, window_w//window_size, x.shape[-2], x.shape[-1])


    assigned_indices = []
    for ii in range(0, window_h // window_size):# 0~20
        for jj in range(0, window_w // window_size):# 0~40
            cur_rank = (jj % wp_group_w) + (ii % wp_group_h) * wp_group_w

            if cur_rank == wp_rank:
                assigned_indices.append((ii, jj))

    x = [x[i, j] for (i, j) in assigned_indices]
    x = torch.stack(x, dim=0) # [200, 36, 36]

    return x


class SequentialSwinTransformer(torch.nn.Module):
    def __init__(self,
        layer_idx = -1,
        height = 720//6,
        width = 1440//6,
        embedding_dim = 4320,
        norm_layer = torch.nn.LayerNorm,
        window_size = 6,
        num_heads = 4320//72,
        use_attn_mask=default_use_attn_mask,
        use_relative_position_bias=default_use_relative_position_bias,
        use_flash_attention=default_use_flash_attention,
        init_seed_base=None,
        ):
        super().__init__()
        self.layer_idx = layer_idx
        self.init_seed_base = None if init_seed_base is None else int(init_seed_base) + int(layer_idx) * 100
        shift_size=0 if (self.layer_idx % 2 == 0) else window_size // 2

        if use_shift_size_0:
            shift_size = 0

        self.shift_size = shift_size
        self.window_size = window_size
        self.use_attn_mask = use_attn_mask

        if use_layernorm:
            self.norm1 = norm_layer(embedding_dim)


            # 'layers.0.blocks.transformer.norm1.weight',
            # 'layers.0.blocks.transformer.norm1.bias'

        self.input_resolution = (height, width) # (120, 240)

        self.attn = WindowAttention(
            embedding_dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=True,
            qk_scale=None,
            use_attn_mask=use_attn_mask,
            use_relative_position_bias=use_relative_position_bias,
            use_flash_attention=use_flash_attention,
            init_seed_base=self.init_seed_base,
            )


        #--atten mask
        if self.shift_size > 0:
            if self.use_attn_mask:
                # calculate attention mask for SW-MSA
                attn_mask = get_swin_attention_mask(
                    'sequential',
                    self.input_resolution,
                    self.window_size,
                    self.shift_size,
                    )
            else:
                attn_mask = None
        else:
            attn_mask = None


        self.register_buffer("attn_mask", attn_mask)


    def forward(self, x):


        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x

        if use_layernorm:


            x = self.norm1(x) # [1, 28800, 4320]


        x = x.view(B, H, W, C)


        # cyclic shift
        if self.shift_size > 0: # this does not affect the hidden dimension
            if False:

                s_x = x.detach().clone()
                s_x = window_partition(s_x, self.window_size) # [1600, 6, 6, 768]
                s_x = rearrange(s_x, 'b (gh ph) (gw pw) d -> b (gh gw) (ph pw) d',
                        gh=2, gw=2, ph=self.shift_size, pw=self.shift_size) # [1600, 4, 9, 768]


                s_x = s_x.view(B, -1, self.window_size * self.window_size, C) # [B, 800, 36, 768]


                s_x = get_wp_window_slice_from_full(s_x[0],
                                                    manager=None,
                                                    window_h = self.input_resolution[0],
                                                    window_w = self.input_resolution[1],
                                                    window_size = self.window_size,
                                                )

                s_x = s_x.view(200, 4, 9, 768)


            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) # [2, 120, 240, 768], [1, 264, 540, 768]


            x_windows = window_partition(shifted_x, self.window_size)# [800*B, 6, 6, 768], [3960, 6, 6, 768]


            if False and self.layer_idx==1:
                if x_windows.device.type == 'cuda' and x_windows.device.index == 0:
                    print('we are here before save cuda 0')

                    target_save_pth = './debug_dumps/'
                    torch.save(x_windows[0:990], target_save_pth+'rk0.pth')
                    torch.save(x_windows[990*1:990*2], target_save_pth+'rk1.pth')
                    torch.save(x_windows[990*2:990*3], target_save_pth+'rk2.pth')
                    torch.save(x_windows[990*3:990*4], target_save_pth+'rk3.pth')
                exit(0)


            if False:
                s_x = x_windows.detach().clone()# [800*B, 6, 6, 768]
                s_x = rearrange(s_x, 'b (gh ph) (gw pw) d -> b (gh gw) (ph pw) d',
                        gh=2, gw=2, ph=self.shift_size, pw=self.shift_size) # [800*B, 4, 9, 768]
                s_x = s_x.view(B, -1, self.window_size * self.window_size, C) # [2, 800, 36, 768]

                s_x = get_wp_window_slice_from_full(s_x[0],
                                                    manager=None,
                                                    window_h = self.input_resolution[0],
                                                    window_w = self.input_resolution[1],
                                                    window_size = self.window_size,
                                                )


                s_x = s_x.view(400, 4, 9, 768)


                target_save_pth = './debug_dumps/'
                torch.save(s_x[:, 0], target_save_pth+'seq_upper_left.pth')
                torch.save(s_x[:, 1], target_save_pth+'seq_upper_right.pth')
                torch.save(s_x[:, 2], target_save_pth+'seq_lower_left.pth')
                torch.save(s_x[:, 3], target_save_pth+'seq_lower_right.pth')
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C


        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA


        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C  # [1600, 36, 768]


        if False:

            target_save_pth = './debug_dumps/'
            torch.save(attn_windows, target_save_pth+'x_windows.pth')


        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C) # [800, 6, 6, 4320]


         # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) # [1, 120, 240, 768]


            if False and self.layer_idx==1:
                s_x = x.detach().clone()
                s_x = window_partition(s_x, self.window_size) # [3960, 6, 6, 768]


                if s_x.device.type == 'cuda' and s_x.device.index == 0:
                    print('we are here after save cuda 0')

                    target_save_pth = './debug_dumps/'
                    torch.save(s_x[0:990], target_save_pth+'after_rk0.pth')
                    torch.save(s_x[990*1:990*2], target_save_pth+'after_rk1.pth')
                    torch.save(s_x[990*2:990*3], target_save_pth+'after_rk2.pth')
                    torch.save(s_x[990*3:990*4], target_save_pth+'after_rk3.pth')
                exit(0)


            if False:
                s_x = x.detach().clone()# [800*B, 6, 6, 768]

                s_x = window_partition(s_x, self.window_size) # [800*B, 6, 6, 768]

                s_x = rearrange(s_x, 'b (gh ph) (gw pw) d -> b (gh gw) (ph pw) d',
                        gh=2, gw=2, ph=self.shift_size, pw=self.shift_size) # [800*B, 4, 9, 768]
                s_x = s_x.view(B, -1, self.window_size * self.window_size, C) # [2, 800, 36, 768]

                s_x = get_wp_window_slice_from_full(s_x[0],
                                                    manager=None,
                                                    window_h = self.input_resolution[0],
                                                    window_w = self.input_resolution[1],
                                                    window_size = self.window_size,
                                                )

                s_x = s_x.view(400, 4, 9, 768)

                print('seq s_x 0 upper left', s_x[:, 0].shape, s_x[:, 0].sum()) #  [200, 9, 768]  9322.6064


                target_save_pth = './debug_dumps/'
                torch.save(s_x[:, 0], target_save_pth+'seq_upper_left.pth')
                torch.save(s_x[:, 1], target_save_pth+'seq_upper_right.pth')
                torch.save(s_x[:, 2], target_save_pth+'seq_lower_left.pth')
                torch.save(s_x[:, 3], target_save_pth+'seq_lower_right.pth')

        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x

        x = x.view(B, H * W, C)

        x = shortcut + x

        if False:

            target_save_pth = './debug_dumps/'
            torch.save(x, target_save_pth+'x_windows.pth')


        if False and self.layer_idx==1:
            if x_windows.device.type == 'cuda' and x_windows.device.index == 0:# [1, 142560, 768]
                x = x.view(B, H, W, C)

                x = window_partition(x, self.window_size)  # [3960, 6, 6, 768]

                print('we are here save final', x.shape) # [1, 142560, 768]


                target_save_pth = './debug_dumps/'
                torch.save(x[0:990], target_save_pth+'final_rk0.pth')
                torch.save(x[990*1:990*2], target_save_pth+'final_rk1.pth')
                torch.save(x[990*2:990*3], target_save_pth+'final_rk2.pth')
                torch.save(x[990*3:990*4], target_save_pth+'final_rk3.pth')


            exit(0)


        if False:
            if self.shift_size > 0:
                s_x = x.detach().clone()
                s_x = s_x.view(B, H ,W, C)
                s_x = window_partition(s_x, self.window_size) # [B*800, 6, 6, 768]
                s_x = rearrange(s_x, 'b (gh ph) (gw pw) d -> b (gh gw) (ph pw) d',
                        gh=2, gw=2, ph=self.shift_size, pw=self.shift_size) # [1600, 4, 9, 768]


                s_x = s_x.view(B, -1, self.window_size * self.window_size, C) # [B, 800, 36, 768]

                s_x = get_wp_window_slice_from_full(s_x[0],
                                                    manager=None,
                                                    window_h = self.input_resolution[0],
                                                    window_w = self.input_resolution[1],
                                                    window_size = self.window_size,
                                                )

                s_x = s_x.view(200, 4, 9, 768) # [200, 4, 9, 768]


                s_x = rearrange(s_x, 'n (gh gw) (ph pw) d -> n (gh ph) (gw pw) d',
                        gh=2, gw=2, ph=self.shift_size, pw=self.shift_size) # [200, 6, 6, 768]

                s_x = s_x.view(200, 36, 768)

                target_save_pth = './debug_dumps/'
                torch.save(s_x, target_save_pth+'final_sum.pth')
                print('seq x sum', x.sum())


        return x

class TensorParallelSwinTransformer(torch.nn.Module):
    def __init__(self,
        kaiming_init,
        manager = None,


        layer_idx = -1,
        height = 720//6,
        width = 1440//6,

        embedding_dim = 4320,

        norm_layer = torch.nn.LayerNorm,
        window_size = 6,

        num_heads = -1,
        ):
        super().__init__()

        self.layer_idx = layer_idx
        shift_size=0 if (self.layer_idx % 2 == 0) else window_size // 2

        if use_shift_size_0:
            shift_size = 0

        self.shift_size = shift_size
        self.window_size = window_size

        if use_layernorm:
            self.norm1 = norm_layer(embedding_dim)


        self.input_resolution = (height, width) # (120, 240)

        self.attn = ParallelWindowAttention(
            kaiming_init,
            manager = manager,
            tp_size = manager.mp_group_size,
            mp_rank = manager.get_mp_rank(),
            embedding_dim = embedding_dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=True,
            qk_scale=None,
            )

        #--atten mask
        if self.shift_size > 0:
            if use_attn_mask:
                # calculate attention mask for SW-MSA
                attn_mask = get_swin_attention_mask(
                    'sequential',
                    self.input_resolution,
                    self.window_size,
                    self.shift_size,
                    )
            else:
                attn_mask = None
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)


    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x

        if use_layernorm:
            x = self.norm1(x) # [1, 28800, 4320]


        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) # this does not affect the hidden dimension
            x_windows = window_partition(shifted_x, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C  [800, 6, 6, 4320]


        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)


        #    exit(0)


        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C) # [800, 6, 6, 4320], and this is also fp16


        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x


        x = x.view(B, H * W, C)

        x = shortcut + x


        return x

class MegatronSequenceParallelSwinTransformer(torch.nn.Module):
    def __init__(self,
                kaiming_init,
                manager = None,
                layer_idx = -1,
                height = 720//6,
                width = 1440//6,
                embedding_dim = 4320,
                norm_layer = torch.nn.LayerNorm,
                window_size = 6,
                num_heads = -1,
        ):
        super().__init__()

        self.manager = manager
        self.layer_idx = layer_idx
        shift_size=0 if (self.layer_idx % 2 == 0) else window_size // 2

        if use_shift_size_0:
            shift_size = 0

        self.shift_size = shift_size
        self.window_size = window_size

        if use_layernorm:
            self.norm1 = norm_layer(embedding_dim)

        self.input_resolution = (height, width) # (120, 240)

        self.tp_size = manager.get_mp_group_size()
        self.mp_rank = manager.get_mp_rank()


        self.attn = ParallelWindowAttention(
            kaiming_init,
            manager = manager,
            tp_size = self.tp_size,
            mp_rank = self.mp_rank,
            embedding_dim = embedding_dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            parallel_model_type='sequence_parallel',
            qkv_bias=True,
            qk_scale=None,
        )

        #--atten mask
        if self.shift_size > 0:
            if use_attn_mask:
                attn_mask = get_swin_attention_mask(
                        'sequential',
                        self.input_resolution,
                        self.window_size,
                        self.shift_size,
                        )
            else:
                attn_mask = None
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        self.all_gather_dim = AllGatherReduceScatter().apply

    def forward(self, x):

        shortcut = x


        '''
        def print_grad(name):
            def hook(grad):
                print(f"{name} grad sum: {grad.sum().item()}")
            return hook

        x.register_hook(print_grad("x after all_to_all"))
        '''

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == (H * W)//self.tp_size, "input feature has wrong size" # the assertion for sequence parallel is different

        if use_layernorm:


            x = self.norm1(x)


            #exit(0)

        x = self.all_gather_dim(x, self.manager, 1) # [1, 28800, 768]

        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) # this does not affect the hidden dimension
            x_windows = window_partition(shifted_x, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C  [800, 6, 6, 4320]


        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask) # # [800, 36, 384]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C//self.tp_size) # [800, 6, 6, 4320], and this is also fp16

        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x


        x = x.view(B, H * W, C//self.tp_size) # [1, 28800, 384]


        x = generic_all_to_all(x, self.manager, split_dim=1, concat_dim=2) # [1, 14400, 768]

        x = shortcut + x

        return x


from core.buffer import get_preallocated_buffer


class WindowParallelSwinTransformer(torch.nn.Module):
    def _get_contiguous_m1_window_slice_from_full(self, x):
        assert x.dim() == 3

        num_windows_h = self.window_h // self.window_size
        num_windows_w = self.window_w // self.window_size
        wp_group_size = self.manager.get_wp_group_size()
        wp_rank = self.manager.get_wp_rank()

        assert self.manager.xfmr_wp_group_w == 1
        assert self.manager.xfmr_wp_group_h == wp_group_size
        assert num_windows_h % wp_group_size == 0, (
            f"num_windows_h={num_windows_h} must be divisible by wp_group_size={wp_group_size} "
            "for contiguous (m, 1) window mask slicing."
        )

        rows_per_rank = num_windows_h // wp_group_size
        h_start = wp_rank * rows_per_rank
        h_end = (wp_rank + 1) * rows_per_rank

        x = x.view(num_windows_h, num_windows_w, x.shape[-2], x.shape[-1])
        x = x[h_start:h_end].contiguous()
        return x.view(rows_per_rank * num_windows_w, x.shape[-2], x.shape[-1])

    def _get_local_attention_mask_from_full(self, full_attn_mask):
        if self.use_wp_ulysses_attention:
            num_windows_h = self.window_h // self.window_size
            num_windows_w = self.window_w // self.window_size
            rank_to_indices, _ = get_window_indices(
                num_windows_h,
                num_windows_w,
                self.manager.xfmr_wp_group_h,
                self.manager.xfmr_wp_group_w,
                mode=get_window_assignment_mode(self.manager),
                debug_rank=self.manager.xfmr_window_group_rank,
                debug_global_rank=getattr(self.manager, "rank", None),
            )
            local_indices = [
                ii * num_windows_w + jj
                for ii, jj in rank_to_indices[self.manager.xfmr_window_group_rank]
            ]
            return full_attn_mask[local_indices].contiguous()

        if self.manager is None:
            return get_wp_window_slice_from_full(
                full_attn_mask,
                self.manager,
                window_h=self.window_h,
                window_w=self.window_w,
                window_size=self.window_size,
            )


        assignment_mode = get_window_assignment_mode(self.manager)
        if assignment_mode == "ragged_round_robin" or is_terra_m1_assignment_mode(assignment_mode):
            num_windows_h = self.window_h // self.window_size
            num_windows_w = self.window_w // self.window_size
            rank_to_indices, _ = get_window_indices(
                num_windows_h,
                num_windows_w,
                self.manager.xfmr_wp_group_h,
                self.manager.xfmr_wp_group_w,
                mode=assignment_mode,
                debug_rank=getattr(self.manager, "xfmr_window_group_rank", self.manager.get_wp_rank()),
                debug_global_rank=getattr(self.manager, "rank", None),
            )
            local_indices = [
                ii * num_windows_w + jj
                for ii, jj in rank_to_indices[self.manager.get_wp_rank()]
            ]
            return full_attn_mask[local_indices].contiguous()


        if self.manager.xfmr_wp_group_w == 1:
            return self._get_contiguous_m1_window_slice_from_full(full_attn_mask)


        return get_wp_window_slice_from_full(
            full_attn_mask,
            self.manager,
            window_h=self.window_h,
            window_w=self.window_w,
            window_size=self.window_size,
        )

    def __init__(self,
                kaiming_init,
                manager = None,

                layer_idx = -1,
                height = 720//6,
                width = 1440//6,

                embedding_dim = 4320,
                norm_layer = torch.nn.LayerNorm,
                window_size = -1,
                num_heads = -1,
                use_attn_mask=default_use_attn_mask,
                use_relative_position_bias=default_use_relative_position_bias,
                use_flash_attention=default_use_flash_attention,
                init_seed_base=None,
        ):
        super().__init__()
        self.manager = manager
        self.use_wp_ulysses_attention = (
            self.manager is not None
            and (
                int(getattr(self.manager, "xfmr_sp_size", self.manager.get_wp_group_size())) > 1
                or int(getattr(self.manager, "xfmr_tp_size", 1)) > 1
            )
        )

        self.layer_idx = layer_idx
        self.init_seed_base = None if init_seed_base is None else int(init_seed_base) + int(layer_idx) * 100
        shift_size=0 if (self.layer_idx % 2 == 0) else window_size // 2

        if use_shift_size_0:
            shift_size = 0

        self.shift_size = shift_size
        self.window_size = window_size
        self.use_attn_mask = use_attn_mask

        self.window_h = height # 264
        self.window_w = width  # 540


        if use_layernorm:
            self.norm1 = norm_layer(embedding_dim)

        self.input_resolution = (height, width) # (120, 240)


        if self.use_wp_ulysses_attention:
            self.attn = WPUlyssesWindowAttention(
                manager=self.manager,
                embedding_dim=embedding_dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                kaiming_init=kaiming_init,
                use_attn_mask=use_attn_mask,
                use_relative_position_bias=use_relative_position_bias,
                use_flash_attention=use_flash_attention,
                init_seed_base=self.init_seed_base,
            )


        else:
            self.attn = WindowAttention(
                embedding_dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=True,
                qk_scale=None,
                use_attn_mask=use_attn_mask,
                use_relative_position_bias=use_relative_position_bias,
                use_flash_attention=use_flash_attention,
                init_seed_base=self.init_seed_base,
                )

        #--atten mask
        if self.shift_size > 0:
            if self.use_attn_mask:
                # Build the full Swin mask first, then slice it according to
                # the actual local window layout.  The optimized (m, 1) path
                # uses contiguous window stripes, while generic (m, n) keeps
                # the old round-robin ownership.
                full_attn_mask = get_swin_attention_mask(
                    'sequential',
                    self.input_resolution,
                    self.window_size,
                    self.shift_size,
                    manager = self.manager,
                    ) # [3960, 36, 36]
                attn_mask = self._get_local_attention_mask_from_full(full_attn_mask)
            else:
                attn_mask = None
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask) # [800, 36, 36]


        wp_group_h = self.manager.xfmr_wp_group_h
        wp_group_w = self.manager.xfmr_wp_group_w


        if wp_group_h>0 and wp_group_w==1 and get_window_assignment_mode(self.manager) == "regular":
            self.dist_window_shift = unified_dist_window_shift().apply

            if self.layer_idx==0 and self.manager.rank==0:
                print('ours (m, 1) window_shift stra')

        else:
            self.dist_window_shift = Dist_Window_Shift().apply

            if self.layer_idx==0 and self.manager.rank==0:
                print('aeris (m, n) window_shift stra')


        self.wp_rank_slice_to_shifted_batch_idx = {}


        rank_cnt = [0] * (wp_group_h*wp_group_w)
        num_windows_h = self.window_h // self.window_size
        num_windows_w = self.window_w // self.window_size

        self.manager.update_window_info(num_windows_h, num_windows_w)


        if wp_group_h>0 and wp_group_w==1 and get_window_assignment_mode(self.manager) == "regular":


            self.shift_direction_to_perm_list = {}
            for shift_direction in ['upper_left', 'lower_right']:

                tmp_list = []


                index = [0, 1, 2, 3]
                perm1 = torch.tensor(index, device='cuda')
                tmp_list.append(perm1)


                index = []
                num_windows_w = self.manager.num_windows_w
                num_windows_h = self.manager.num_windows_h
                total = (num_windows_w * num_windows_h)//self.manager.get_wp_group_size()

                if shift_direction=='upper_left':
                    for start in range(0, total, num_windows_w):
                        index.extend(range(start + 1, start + num_windows_w))
                        index.append(start)
                elif shift_direction=='lower_right':
                    for start in range(0, total, num_windows_w):
                        index.append(start + num_windows_w - 1)
                        index.extend(range(start, start + num_windows_w - 1))

                perm2 = torch.tensor(index, device='cuda')
                tmp_list.append(perm2)


                index = []
                if shift_direction=='upper_left':
                    for start in range(0, total, num_windows_w):
                        index.extend(range(start + 1, start + num_windows_w))
                        index.append(start)
                elif shift_direction=='lower_right':
                    for start in range(0, total, num_windows_w):
                        index.append(start + num_windows_w - 1)
                        index.extend(range(start, start + num_windows_w - 1))
                perm3 = torch.tensor(index, device='cuda')
                tmp_list.append(perm3)

                self.shift_direction_to_perm_list[shift_direction] = tmp_list

        else:
            self.wp_group_size = manager.get_wp_group_size()

            if True:
                self.shift_direction_to_perm_list = {}

                wp_group_size = self.wp_group_size
                wp_rank = manager.get_wp_rank()

                for shift_direction in ['upper_left', 'lower_right']:
                    tmp_list = []


                    local_window_count = num_windows_h * num_windows_w // wp_group_size
                    identity_perm = torch.arange(local_window_count, device='cuda')
                    tmp_list = [identity_perm, identity_perm, identity_perm, identity_perm]


                    self.shift_direction_to_perm_list[shift_direction] = tmp_list


    def wp_ulysses_forward(self, x): # [1, 180, 225, 1536]

        B, num_windows, shard_tokens, C = x.shape
        shortcut = x

        if use_layernorm:

            x = self.norm1(x)


        data_wp_topo = getattr(self.manager, "data_wp_topo", (self.manager.get_wp_group_size(), 1))
        local_h = self.window_h // data_wp_topo[0]
        local_w = self.window_w

        '''
        if self.shift_size > 0:
            x_grid = ulysses_windows_to_stripe_grid(
                x,
                self.manager,
                local_h,
                local_w,
                self.window_size,
                0,
            )
            x = stripe_grid_to_ulysses_windows(
                x_grid,
                self.manager,
                self.window_size,
                self.shift_size,
            )

        x = self.attn(x, mask=self.attn_mask)

        if self.shift_size > 0:
            x_grid = ulysses_windows_to_stripe_grid(
                x,
                self.manager,
                local_h,
                local_w,
                self.window_size,
                self.shift_size,
            )
            x = stripe_grid_to_ulysses_windows(
                x_grid,
                self.manager,
                self.window_size,
                0,
            )
        '''

        if self.shift_size > 0:
            x = ulysses_windows_to_ulysses_windows(
                x,
                self.manager,
                local_h,
                local_w,
                self.window_size,
                0,
                self.shift_size,
            )

        x = self.attn(x, mask=self.attn_mask)

        if self.shift_size > 0:
            x = ulysses_windows_to_ulysses_windows(
                x,
                self.manager,
                local_h,
                local_w,
                self.window_size,
                self.shift_size,
                0,
            )


        if x.shape != shortcut.shape:
            raise RuntimeError(f"WP-Ulysses residual shape mismatch: x={x.shape}, shortcut={shortcut.shape}")
        x = shortcut + x
        return x

    def naive_forward(self, x): # [2, 200, 6*6, 768]
        B, H_W, C = x.shape[0], x.shape[-2], x.shape[-1]

        short_cut = x

        if use_layernorm:
            #all_reduce_and_print_mp_rank0(x, self.manager, description='parallel x sum before layernorm +++++++++++++++++++++++++++++++++++++++++++')
            x = self.norm1(x) # [2, 200, 6, 6, 768]
            #all_reduce_and_print_mp_rank0(x, self.manager, description='parallel x sum after layernorm +++++++++++++++++++++++++++++++++++++++++++')

        if self.shift_size > 0:


            x = rearrange(x, 'b n (h w) d -> b n h w d', h=self.window_size, w=self.window_size) # [2, 200, 6, 6, 768]


            x = rearrange(x, 'b n (gh ph) (gw pw) d -> b n (gh gw) (ph pw) d',
                        gh=2, gw=2, ph=self.shift_size, pw=self.shift_size)


            #all_reduce_and_print_mp_rank0(x[:, :, 3], self.manager, description='parallel x_4 sum  +++++++++++++++++++++++++++++++++++++++++++')


            x_windows = self.dist_window_shift(x,
                                   self.manager,

                                   self.shift_direction_to_perm_list,
                                   'upper_left',
                                   ) # [2, 200, 4, 9, 768], [1, 990, 4, 9, 768]


            x_windows = rearrange(x_windows, 'b n (gh gw) (ph pw) d -> b n (gh ph) (gw pw) d',
                        gh=2, gw=2, ph=self.shift_size, pw=self.shift_size) # [B, 200, 6, 6, 768]
        else:
            x_windows = x # [2, 200, 6, 6, 768]

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) # [400, 36, 768]

        attn_windows = self.attn(x_windows, mask=self.attn_mask) # [400, 36, 768]


        if self.shift_size > 0:


            # step1
            x = attn_windows.view(B, -1, self.window_size, self.window_size, C) # [1, 200, 6, 6, 768]


            x = rearrange(x, 'b n (gh ph) (gw pw) d -> b n (gh gw) (ph pw) d',
                        gh=2, gw=2, ph=self.shift_size, pw=self.shift_size) # [1, 200, 4, 9, 768]

            x_windows = self.dist_window_shift(x,
                                   self.manager,

                                   self.shift_direction_to_perm_list,
                                   'lower_right',
                                   ) # [1, 200, 4, 9, 768]


            x = rearrange(x_windows, 'b n (gh gw) (ph pw) d -> b n (gh ph) (gw pw) d',
                        gh=2, gw=2, ph=self.shift_size, pw=self.shift_size) # [B, 200, 6, 6, 768]


        else:
            x = attn_windows

        x = x.view(B, -1, H_W, C) # [2, 200, 36, 768]

        x = short_cut + x


        #    all_reduce_and_print_mp_rank0(x, self.manager, description='parallel x sum  +++++++++++++++++++++++++++++++++++++++++++')


        return x
    def forward(self, x):
        if self.use_wp_ulysses_attention:
            return self.wp_ulysses_forward(x)
        else:
            return self.naive_forward(x)
