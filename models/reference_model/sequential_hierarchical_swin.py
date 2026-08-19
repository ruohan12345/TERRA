"""Sequential public reference models for exercising the TERRA runtime.

The hierarchical variant combines public Swin/patch operators with CREDIT-derived
FuXi-style sampling blocks. It is not the production model implementation.
"""

import torch

from models.patch_embedding import SeqPatchEmbedding, SeqPatchRecovery
from models.utils import CreditDownBlock, CreditUpBlock

from models.reference_model.parallel_layers import BasicLayer


from utils import get_padded_shape
from core.checkpoint.activation import activation_checkpoint, activation_config_for_sampling
from profiler.memory_timeline import mark_memory_timeline


from core.global_env_config import ON_H200
from models.reference_model.init_utils import REFERENCE_INIT_SEEDS, make_with_seed


# Public hierarchical Swin reference model with CREDIT-derived sampling blocks.
class SequentialHierarchicalSwin(torch.nn.Module):
    def __init__(self,
                padding_scale,
                num_channels = 93,
                kaiming_init = True,
                patch_size = 6,
                window_size = 6,

                height = 720,
                width = 1440,
                embedding_dim = 768,#4320,
                num_heads = -1,
                num_layers = 2, # the number of sequential basic layers
                recompute_config = None,
                manager = None,
                use_attn_mask=True,
                use_relative_position_bias=True,
                use_flash_attention=False,
                padding_spec = None,
                ):
        super().__init__()

        self.manager = manager

        self.patch_size = patch_size

        resolved_padded_shape = None
        if padding_spec is not None:
            resolved_padded_shape = padding_spec.get("padded_shape", None)
        need_padding, initial_padding, padded_shape = get_padded_shape(
            height,
            width,
            patch_size,
            window_size,
            padding_scale=padding_scale,
            padded_shape=resolved_padded_shape,
        ) # (2112, 4320)

        self.need_padding = need_padding
        self.initial_padding = initial_padding
        self.initial_padding_func = torch.nn.ZeroPad2d(initial_padding)
        self.padded_shape = padded_shape # (2112, 4320)


        patches_resolution = [
            padded_shape[0] // patch_size,
            padded_shape[1] // patch_size,
        ]


        self.layers = make_with_seed(
            REFERENCE_INIT_SEEDS["layers"],
            lambda: BasicLayer(
                dim=embedding_dim,
                input_resolution=(patches_resolution[0] // 2, patches_resolution[1] // 2),
                depth=num_layers,
                num_heads=num_heads,
                window_size=window_size,
                recompute_config=recompute_config,
                manager = manager,
                use_attn_mask=use_attn_mask,
                use_relative_position_bias=use_relative_position_bias,
                use_flash_attention=use_flash_attention,
                attention_init_seed_base=REFERENCE_INIT_SEEDS["attention"],
                mlp_init_seed_base=REFERENCE_INIT_SEEDS["mlp"],
            ),
        )


        self.patch_embed = make_with_seed(
            REFERENCE_INIT_SEEDS["patch_embed"],
            lambda: SeqPatchEmbedding(kaiming_init=kaiming_init, patch_size=patch_size, num_channel=num_channels, embedding_dim=embedding_dim),
        )


        self.patch_recovery = make_with_seed(
            REFERENCE_INIT_SEEDS["patch_recovery"],
            lambda: SeqPatchRecovery(kaiming_init=kaiming_init, height=padded_shape[0], width=padded_shape[1],
                                                     patch_size=patch_size, num_channel=num_channels, embedding_dim=embedding_dim),
        )


        self.down_blk = make_with_seed(
            REFERENCE_INIT_SEEDS["down_blk"],
            lambda: CreditDownBlock(embedding_dim, embedding_dim, num_groups=32),
        )
        self.up_blk = make_with_seed(
            REFERENCE_INIT_SEEDS["up_blk"],
            lambda: CreditUpBlock(embedding_dim, embedding_dim, num_groups=32),
        )


        self.recompute_config = recompute_config or {}
        self.recompute_scheduler = self.layers.recompute_scheduler
        self.down_recompute_config = activation_config_for_sampling(self.recompute_config, "down")
        self.up_recompute_config = activation_config_for_sampling(self.recompute_config, "up")
        if bool(self.recompute_config.get("enabled", False)):
            self.checkpoint_down_mode = str(self.recompute_config.get("checkpoint_down_mode", "none"))
            self.checkpoint_up_mode = str(self.recompute_config.get("checkpoint_up_mode", "none"))
        else:
            self.checkpoint_down_mode = "none"
            self.checkpoint_up_mode = "none"
        if self.checkpoint_down_mode.lower() != "none":
            self.checkpoint_down_mode = self.checkpoint_down_mode.upper()
        if self.checkpoint_up_mode.lower() != "none":
            self.checkpoint_up_mode = self.checkpoint_up_mode.upper()
        self._disable_inner_sampling_checkpoint(self.down_blk)
        self._disable_inner_sampling_checkpoint(self.up_blk)

        if self.checkpoint_down_mode != "none" or self.checkpoint_up_mode != "none":
            if self.manager is None or self.manager.get_rank() == 0:
                print(
                    f"[Activation] sampling_checkpoint.down={self.checkpoint_down_mode} "
                    f"sampling_checkpoint.up={self.checkpoint_up_mode}"
                )


        self.div_val = patch_size * 2

        self.input_nchw = ON_H200


    def _disable_inner_sampling_checkpoint(self, module):
        if hasattr(module, "use_checkpointing"):
            module.use_checkpointing = "no"
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = False

    def _checkpoint_module(self, module, x, activation_config=None):
        activation_config = activation_config or self.recompute_config
        label = getattr(module, "__name__", module.__class__.__name__)

        def _checkpoint_body(inp, module=module, label=label):
            mark_memory_timeline("checkpoint_body_pre", label)
            out = module(inp)
            mark_memory_timeline("checkpoint_body_post", label)
            return out

        mark_memory_timeline("checkpoint_edge_pre", label)
        out = activation_checkpoint(_checkpoint_body, x, activation_config=activation_config)
        mark_memory_timeline("checkpoint_edge_post", label)
        return out

    def _patch_embed_to_grid(self, x):
        x = self.patch_embed(x)
        return x.view(
            x.shape[0],
            self.padded_shape[0] // self.patch_size,
            self.padded_shape[1] // self.patch_size,
            x.shape[-1],
        ).permute(0, 3, 1, 2)

    def _patch_embed_down(self, x):
        x = self._patch_embed_to_grid(x)
        return self.down_blk(x)

    def _patch_embed_down_conv(self, x):
        x = self._patch_embed_to_grid(x)
        return self.down_blk.conv(x)

    def _down_residual_first(self, x):
        for idx in range(3):
            x = self.down_blk.b[idx](x)
        return x

    def _down_residual_second(self, x):
        for idx in range(3, len(self.down_blk.b)):
            x = self.down_blk.b[idx](x)
        return x

    def _run_down_d1(self, x):
        mark_memory_timeline("sampling_down_enter", "D1")
        x = self._checkpoint_module(self._patch_embed_down_conv, x, activation_config=self.down_recompute_config)
        mark_memory_timeline("sampling_down_d1_after_conv", "D1")
        shortcut = x.clone()
        mark_memory_timeline("sampling_down_d1_after_shortcut", "D1")
        x = self._checkpoint_module(self._down_residual_first, x, activation_config=self.down_recompute_config)
        mark_memory_timeline("sampling_down_d1_after_residual_first", "D1")
        x = self._checkpoint_module(self._down_residual_second, x, activation_config=self.down_recompute_config)
        mark_memory_timeline("sampling_down_d1_after_residual_second", "D1")
        out = x + shortcut
        mark_memory_timeline("sampling_down_exit", "D1")
        return out

    def _run_patch_embed_down(self, x):
        if self.checkpoint_down_mode == "D0":
            mark_memory_timeline("sampling_down_enter", "D0")
            out = self._checkpoint_module(self._patch_embed_down, x, activation_config=self.down_recompute_config)
            mark_memory_timeline("sampling_down_exit", "D0")
            return out
        if self.checkpoint_down_mode == "D1":
            return self._run_down_d1(x)
        if self.checkpoint_down_mode != "none":
            raise ValueError(f"Unsupported down sampling checkpoint mode: {self.checkpoint_down_mode}")
        mark_memory_timeline("sampling_down_enter", "none")
        x = self._patch_embed_to_grid(x)
        out = self.down_blk(x)
        mark_memory_timeline("sampling_down_exit", "none")
        return out

    def _up_patch_recovery(self, x):
        x = self.up_blk(x)
        x = x.flatten(2).permute(0, 2, 1)
        return self.patch_recovery(x)

    def _up_residual_first(self, x):
        for idx in range(3):
            x = self.up_blk.b[idx](x)
        return x

    def _up_residual_second(self, x):
        for idx in range(3, len(self.up_blk.b)):
            x = self.up_blk.b[idx](x)
        return x

    def _run_up_u1(self, x):
        mark_memory_timeline("sampling_up_enter", "U1")
        x = self._checkpoint_module(self.up_blk.conv, x, activation_config=self.up_recompute_config)
        mark_memory_timeline("sampling_up_u1_after_conv", "U1")
        shortcut = x.clone()
        mark_memory_timeline("sampling_up_u1_after_shortcut", "U1")
        x = self._checkpoint_module(self._up_residual_first, x, activation_config=self.up_recompute_config)
        mark_memory_timeline("sampling_up_u1_after_residual_first", "U1")
        x = self._checkpoint_module(self._up_residual_second, x, activation_config=self.up_recompute_config)
        x = x + shortcut
        mark_memory_timeline("sampling_up_u1_after_residual", "U1")
        x = x.flatten(2).permute(0, 2, 1)
        out = self._checkpoint_module(self.patch_recovery, x, activation_config=self.up_recompute_config)
        mark_memory_timeline("sampling_up_exit", "U1")
        return out

    def _run_up_patch_recovery(self, x):
        if self.checkpoint_up_mode == "U0":
            mark_memory_timeline("sampling_up_enter", "U0")
            out = self._checkpoint_module(self._up_patch_recovery, x, activation_config=self.up_recompute_config)
            mark_memory_timeline("sampling_up_exit", "U0")
            return out
        if self.checkpoint_up_mode == "U1":
            return self._run_up_u1(x)
        if self.checkpoint_up_mode != "none":
            raise ValueError(f"Unsupported up sampling checkpoint mode: {self.checkpoint_up_mode}")
        x = self.up_blk(x)
        x = x.flatten(2).permute(0, 2, 1)
        return self.patch_recovery(x)


    def pre_process(self, x):

        if not self.input_nchw:
            x = x.permute(0, 3, 1, 2).contiguous()

        _, _, raw_lat, raw_lon = x.shape
        if self.need_padding:
            x = self.initial_padding_func(x)

        pad_lat, pad_lon = x.shape[-2], x.shape[-1]
        x = x.permute(0, 2, 3, 1).contiguous()

        x = self._run_patch_embed_down(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x, pad_lat, pad_lon, raw_lat, raw_lon


    def forward(self, input):


        x, pad_lat, pad_lon, raw_lat, raw_lon = self.pre_process(input)

        x = self.layers(x) # [1, 145728, 768]


        x = x.view(x.shape[0], self.padded_shape[0] // self.div_val, self.padded_shape[1] // self.div_val, -1)

        x = x.permute(0, 3, 1, 2)

        x = self._run_up_patch_recovery(x) # [1, 2112, 4416, 93]


        if self.need_padding:
            padding_left, padding_right, padding_top, padding_bottom = self.initial_padding
            x = x[:, padding_top: pad_lat - padding_bottom, padding_left: pad_lon - padding_right, :]
            x = x[:, :raw_lat, :raw_lon, :] # [1, 2041, 4320, 93]


        if self.input_nchw:
            x = x.permute(0, 3, 1, 2).contiguous()

        x = x + input

        return x


class SequentialSwinReference(torch.nn.Module):
    def __init__(self,
                padding_scale,
                num_channels = 93,
                kaiming_init = True,
                patch_size = 6,
                window_size = 6,

                height = 720,
                width = 1440,
                embedding_dim = 768,#4320,
                num_heads = -1,
                num_layers = 2, # the number of sequential basic layers
                recompute_config = None,
                use_attn_mask=True,
                use_relative_position_bias=True,
                use_flash_attention=False,
                manager = None,
                padding_spec = None,
                ):
        super().__init__()

        self.manager = manager


        self.patch_size = patch_size

        resolved_padded_shape = None
        if padding_spec is not None:
            resolved_padded_shape = padding_spec.get("padded_shape", None)
        need_padding, initial_padding, padded_shape = get_padded_shape(
            height,
            width,
            patch_size,
            window_size,
            padding_scale=padding_scale,
            padded_shape=resolved_padded_shape,
        ) # (2112, 4320)

        self.need_padding = need_padding
        self.initial_padding = initial_padding
        self.initial_padding_func = torch.nn.ZeroPad2d(initial_padding)
        self.padded_shape = padded_shape # (2112, 4320)


        patches_resolution = [
            padded_shape[0] // patch_size,
            padded_shape[1] // patch_size,
        ]
        self.layers = BasicLayer(
            dim=embedding_dim,
            input_resolution=(patches_resolution[0] , patches_resolution[1]),
            depth=num_layers,
            num_heads=num_heads,
            window_size=window_size,
            recompute_config=recompute_config,
            use_attn_mask=use_attn_mask,
            use_relative_position_bias=use_relative_position_bias,
            use_flash_attention=use_flash_attention,
            attention_init_seed_base=REFERENCE_INIT_SEEDS["attention"],
            mlp_init_seed_base=REFERENCE_INIT_SEEDS["mlp"],
        )


        self.patch_embed = SeqPatchEmbedding(kaiming_init=kaiming_init, patch_size=patch_size, num_channel=num_channels, embedding_dim=embedding_dim)


        self.patch_recovery = SeqPatchRecovery(kaiming_init=kaiming_init, height=padded_shape[0], width=padded_shape[1],
                                                 patch_size=patch_size, num_channel=num_channels, embedding_dim=embedding_dim)


        self.recompute_config = recompute_config or {}
        self.checkpoint_patch_embed = bool(self.recompute_config.get("enabled", False)) and bool(self.recompute_config.get("checkpoint_patch_embed", False))
        self.checkpoint_patch_recovery = bool(self.recompute_config.get("enabled", False)) and bool(self.recompute_config.get("checkpoint_patch_recovery", False))

        self.div_val = patch_size * 2

    def _checkpoint_module(self, module, x):
        label = getattr(module, "__name__", module.__class__.__name__)

        def _checkpoint_body(inp, module=module, label=label):
            mark_memory_timeline("checkpoint_body_pre", label)
            out = module(inp)
            mark_memory_timeline("checkpoint_body_post", label)
            return out

        mark_memory_timeline("checkpoint_edge_pre", label)
        out = activation_checkpoint(_checkpoint_body, x, activation_config=self.recompute_config)
        mark_memory_timeline("checkpoint_edge_post", label)
        return out

    def _run_patch_embed(self, x):
        if self.checkpoint_patch_embed:
            return self._checkpoint_module(self.patch_embed, x)
        return self.patch_embed(x)

    def _run_patch_recovery(self, x):
        if self.checkpoint_patch_recovery:
            return self._checkpoint_module(self.patch_recovery, x)
        return self.patch_recovery(x)

    def pre_process(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()

        _, _, raw_lat, raw_lon = x.shape
        if self.need_padding:
            x = self.initial_padding_func(x)

        pad_lat, pad_lon = x.shape[-2], x.shape[-1]
        x = x.permute(0, 2, 3, 1).contiguous()

        x = self._run_patch_embed(x) # [1, 145728, 768]

        return x, pad_lat, pad_lon, raw_lat, raw_lon

    def forward(self, input): # [1, 2041, 4320, 93]

        x, pad_lat, pad_lon, raw_lat, raw_lon = self.pre_process(input)

        x = self.layers(x) # [1, 145728, 768]

        x = self._run_patch_recovery(x) # [1, 2112, 4416, 93]


        if self.need_padding:
            padding_left, padding_right, padding_top, padding_bottom = self.initial_padding
            x = x[:, padding_top: pad_lat - padding_bottom, padding_left: pad_lon - padding_right, :]
            x = x[:, :raw_lat, :raw_lon, :] # [1, 2041, 4320, 93]

        x = x + input

        return x
