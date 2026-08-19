"""Distributed public reference models for exercising TERRA.

The hierarchical variant uses CREDIT-derived FuXi sampling blocks wrapped by
TERRA domain parallelism. It is not the production model implementation.
"""

import numpy as np
import random

import torch
from torch import nn


from core.checkpoint.activation import activation_checkpoint, activation_config_for_sampling
from profiler.memory_timeline import (
    get_memory_timeline_context,
    mark_memory_timeline,
    new_memory_timeline_occurrence,
    profile_memory_backward_boundary,
)


def _tensor_mib(tensor):
    if not torch.is_tensor(tensor):
        return ""
    return tensor.numel() * tensor.element_size() / 1024**2


from models.patch_embedding import WrappedLinear
from models.utils import CreditDownBlock, CreditUpBlock
from models.reference_model.parallel_layers import (
    BasicLayer,
    WindowParallelBasicLayer,
    DomainParallelCreditDownBlock,
    DomainParallelCreditUpBlock,
)

from models.transformer import WindowParallelSwinTransformer
from models.mlp import WindowParallelMLP, WPUlyssesTensorParallelMLP


from utils import get_padded_shape
from models.reference_model.init_utils import REFERENCE_INIT_SEEDS, make_with_seed


from core.parallel.layout_transform import (
    round_robin_windows_to_stripe_grid,
    stripe_grid_to_round_robin_windows,

    stripe_grid_to_ulysses_windows,
    ulysses_windows_to_stripe_grid,
)


class ParallelHierarchicalSwin(nn.Module):
    def __init__(self,
                 height,
                 width,
                 num_channels,

                 patch_size,
                 embedding_dim,
                 num_layers,

                 num_heads,
                 window_size,
                 kaiming_init = True,
                 manager = None,
                 device = None,
                 padding_scale = 4,

                 embedding_parallel_type = None,
                 recompute_config = None,
                 use_attn_mask=True,
                 use_relative_position_bias=True,
                 use_flash_attention=False,
                 padding_spec = None,
                ):
        super().__init__()

        self.manager = manager
        self.rank = manager.rank
        self.wp_group = manager.window_parallel_group
        self.wp_rank = manager.get_wp_rank()
        self.wp_group_size = manager.get_wp_group_size()


        self.num_layers = num_layers
        self.patch_size = patch_size
        self.window_size = window_size
        self.embedding_dim = embedding_dim

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
        )

        patches_resolution = [
            (padded_shape[0]) // patch_size,
            (padded_shape[1]) // patch_size
        ]
        self.patches_resolution = patches_resolution


        self.layers = make_with_seed(
            REFERENCE_INIT_SEEDS["layers"],
            lambda: WindowParallelBasicLayer(dim=embedding_dim,
                                   input_resolution=( patches_resolution[0]// 2,
                                                      patches_resolution[1]// 2),
                                   depth=num_layers,
                                   num_heads=num_heads,
                                   window_size=window_size,
                                   manager = manager,
                                   recompute_config=recompute_config,
                                   use_attn_mask=use_attn_mask,
                                   use_relative_position_bias=use_relative_position_bias,
                                   use_flash_attention=use_flash_attention,
                                   attention_init_seed_base=REFERENCE_INIT_SEEDS["attention"],
                                   mlp_init_seed_base=REFERENCE_INIT_SEEDS["mlp"],
                                   ),
        )


        self.patch_embed = make_with_seed(
            REFERENCE_INIT_SEEDS["patch_embed"],
            lambda: WrappedLinear(num_channels*patch_size*patch_size, embedding_dim, bias=True),
        )
        self.patch_recovery = make_with_seed(
            REFERENCE_INIT_SEEDS["patch_recovery"],
            lambda: WrappedLinear(embedding_dim, num_channels*patch_size*patch_size, bias=True),
        )


        self._saved_rng_state = self._get_rng_state()
        down_blk = make_with_seed(
            REFERENCE_INIT_SEEDS["down_blk"],
            lambda: CreditDownBlock(embedding_dim, embedding_dim, num_groups=32),
        )

        self._restore_rng_state(self._saved_rng_state)
        self.down_blk = DomainParallelCreditDownBlock(down_blk, self.manager)


        self._saved_rng_state = self._get_rng_state()
        up_blk = make_with_seed(
            REFERENCE_INIT_SEEDS["up_blk"],
            lambda: CreditUpBlock(embedding_dim, embedding_dim, num_groups=32),
        )
        self._restore_rng_state(self._saved_rng_state)
        self.up_blk = DomainParallelCreditUpBlock(up_blk, self.manager)


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
            if self.manager is not None and self.manager.get_rank() == 0:
                print(
                    f"[Activation] sampling_checkpoint.down={self.checkpoint_down_mode} "
                    f"sampling_checkpoint.up={self.checkpoint_up_mode}"
                )


        self.mp_all_reduce_list = []
        for full_module_name, module in self.named_modules():
            if (
                isinstance(
                    module,
                    (
                        WindowParallelSwinTransformer,
                        WindowParallelMLP,
                        WPUlyssesTensorParallelMLP,
                        #Window_ParaPatchEmbedding,
                        #Window_ParaPatchRecovery,
                        #WindowSequenceParallelSwinTransformer,
                        WrappedLinear,
                        #DownBlock,
                        #UpBlock,
                        DomainParallelCreditDownBlock,
                        DomainParallelCreditUpBlock,

                    )
                )
            ):

                for local_name, param in module.named_parameters(recurse=True):
                    full_name = f"{full_module_name}.{local_name}" if full_module_name else local_name


                    param.allreduce_wp_group = True
                    param.full_name = full_name


                    self.mp_all_reduce_list.append(full_name)


    def _get_rng_state(self):
        """Return current RNG states."""
        return {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'cpu': torch.get_rng_state()
        }

    def _restore_rng_state(self, saved_state):
        """Restore RNG states."""
        random.setstate(saved_state['python'])
        np.random.set_state(saved_state['numpy'])
        torch.set_rng_state(saved_state['cpu'])
        if torch.cuda.is_available() and saved_state['cuda'] is not None:
            torch.cuda.set_rng_state_all(saved_state['cuda'])


    def _disable_inner_sampling_checkpoint(self, module):
        if hasattr(module, "use_checkpointing"):
            module.use_checkpointing = "no"
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = False

    def _checkpoint_module(
        self,
        module,
        x,
        activation_config=None,
        profile_meta=None,
    ):
        activation_config = activation_config or self.recompute_config
        label = getattr(module, "__name__", module.__class__.__name__)
        context = get_memory_timeline_context()
        metadata = dict(profile_meta or {})
        metadata.setdefault("primitive_kind", "sampling_generic")
        metadata.setdefault("primitive_mode", "checkpoint")
        metadata.setdefault("primitive_part", label)
        metadata.setdefault("primitive_id", f"sampling_generic:{label}")
        metadata["lead_idx"] = context.get("lead_idx", "")
        metadata["occurrence_id"] = new_memory_timeline_occurrence(
            metadata["primitive_id"]
        )

        def _checkpoint_body(inp, module=module, label=label, metadata=metadata):
            mark_memory_timeline(
                "primitive_body_pre", label, tensor_mib=_tensor_mib(inp), **metadata
            )
            mark_memory_timeline("checkpoint_body_pre", label)
            try:
                return module(inp)
            finally:
                mark_memory_timeline("checkpoint_body_post", label)
                mark_memory_timeline("primitive_body_post", label, **metadata)

        input_metadata = dict(metadata, tensor_mib=_tensor_mib(x))
        mark_memory_timeline("primitive_forward_pre", label, **input_metadata)
        mark_memory_timeline("checkpoint_edge_pre", label)
        checkpoint_input = profile_memory_backward_boundary(
            x, "primitive_backward_post", label, **input_metadata
        )
        out = activation_checkpoint(
            _checkpoint_body,
            checkpoint_input,
            activation_config=activation_config,
        )
        mark_memory_timeline("checkpoint_edge_post", label)
        output_metadata = dict(metadata, tensor_mib=_tensor_mib(out))
        mark_memory_timeline("primitive_forward_post", label, **output_metadata)
        return profile_memory_backward_boundary(
            out, "primitive_backward_pre", label, **output_metadata
        )

    @staticmethod
    def _sampling_profile(direction, mode, part):
        return {
            "primitive_kind": f"sampling_{direction}",
            "primitive_mode": mode,
            "primitive_part": part,
            "primitive_id": f"sampling_{direction}:{mode}:{part}",
        }

    def _patch_embed_to_grid(self, x):
        x = self.patch_embed(x)
        return x.permute(0, 3, 1, 2).contiguous()

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
        x = self._checkpoint_module(self._patch_embed_down_conv, x, activation_config=self.down_recompute_config, profile_meta=self._sampling_profile("down", "D1", "patch_embed_down_conv"))
        mark_memory_timeline("sampling_down_d1_after_conv", "D1")
        shortcut = x.clone()
        mark_memory_timeline("sampling_down_d1_after_shortcut", "D1")
        x = self._checkpoint_module(self._down_residual_first, x, activation_config=self.down_recompute_config, profile_meta=self._sampling_profile("down", "D1", "residual_first"))
        mark_memory_timeline("sampling_down_d1_after_residual_first", "D1")
        x = self._checkpoint_module(self._down_residual_second, x, activation_config=self.down_recompute_config, profile_meta=self._sampling_profile("down", "D1", "residual_second"))
        mark_memory_timeline("sampling_down_d1_after_residual_second", "D1")
        out = x + shortcut
        mark_memory_timeline("sampling_down_exit", "D1")
        return out

    def _run_patch_embed_down(self, x):
        if self.checkpoint_down_mode == "D0":
            mark_memory_timeline("sampling_down_enter", "D0")
            out = self._checkpoint_module(self._patch_embed_down, x, activation_config=self.down_recompute_config, profile_meta=self._sampling_profile("down", "D0", "fused"))
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
        x = x.permute(0, 2, 3, 1)
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
        x = self._checkpoint_module(self.up_blk.conv, x, activation_config=self.up_recompute_config, profile_meta=self._sampling_profile("up", "U1", "conv"))
        mark_memory_timeline("sampling_up_u1_after_conv", "U1")
        shortcut = x.clone()
        mark_memory_timeline("sampling_up_u1_after_shortcut", "U1")
        x = self._checkpoint_module(self._up_residual_first, x, activation_config=self.up_recompute_config, profile_meta=self._sampling_profile("up", "U1", "residual_first"))
        mark_memory_timeline("sampling_up_u1_after_residual_first", "U1")
        x = self._checkpoint_module(self._up_residual_second, x, activation_config=self.up_recompute_config, profile_meta=self._sampling_profile("up", "U1", "residual_second"))
        x = x + shortcut
        mark_memory_timeline("sampling_up_u1_after_residual", "U1")
        x = x.permute(0, 2, 3, 1)
        out = self._checkpoint_module(self.patch_recovery, x, activation_config=self.up_recompute_config, profile_meta=self._sampling_profile("up", "U1", "patch_recovery"))
        mark_memory_timeline("sampling_up_exit", "U1")
        return out

    def _run_up_patch_recovery(self, x):
        if self.checkpoint_up_mode == "U0":
            mark_memory_timeline("sampling_up_enter", "U0")
            out = self._checkpoint_module(self._up_patch_recovery, x, activation_config=self.up_recompute_config, profile_meta=self._sampling_profile("up", "U0", "fused"))
            mark_memory_timeline("sampling_up_exit", "U0")
            return out
        if self.checkpoint_up_mode == "U1":
            return self._run_up_u1(x)
        if self.checkpoint_up_mode != "none":
            raise ValueError(f"Unsupported up sampling checkpoint mode: {self.checkpoint_up_mode}")
        x = self.up_blk(x)
        x = x.permute(0, 2, 3, 1)
        return self.patch_recovery(x)


    def forward(self, input): # [1, 132, 1104, 1488]


        x = self._run_patch_embed_down(input)


        x = x.permute(0, 2, 3, 1).contiguous()
        B = x.shape[0]
        window_h = x.shape[1]
        window_w = x.shape[2]


        use_factorized_ulysses = (
            self.manager is not None
            and (
                int(getattr(self.manager, "xfmr_sp_size", 1)) > 1
                or int(getattr(self.manager, "xfmr_tp_size", 1)) > 1
            )
        )

        if use_factorized_ulysses:
            x = stripe_grid_to_ulysses_windows(
                x,
                self.manager,
                self.window_size,
                0,
            ) # [1, 180, 225, 768]  [1, 180, 225, 1536]


        else:
            x = stripe_grid_to_round_robin_windows(
                x,
                self.manager,
                self.window_size,
            )

        x = self.layers(x)


        if use_factorized_ulysses:
            x = ulysses_windows_to_stripe_grid(
                x,
                self.manager,
                window_h,
                window_w,
                self.window_size,
                0,
            )
        else:
            x = round_robin_windows_to_stripe_grid(
                x,
                self.manager,
                window_h,
                window_w,
                self.window_size,
            )
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self._run_up_patch_recovery(x) # [1, 132, 1104, 1488]


        x = x + input

        return x


class ParallelSwinReference(nn.Module):
    def __init__(self,
                 height,
                 width,
                 num_channels,

                 patch_size,
                 embedding_dim,
                 num_layers,

                 num_heads,
                 window_size,
                 kaiming_init = True,
                 manager = None,
                 device = None,
                 padding_scale = 4,

                 embedding_parallel_type = None,
                 recompute_config = None,
                 use_attn_mask=True,
                 use_relative_position_bias=True,
                 use_flash_attention=False,
                 padding_spec = None,
                ):
        super().__init__()

        self.manager = manager
        self.rank = manager.rank
        self.wp_group = manager.window_parallel_group
        self.wp_rank = manager.get_wp_rank()
        self.wp_group_size = manager.get_wp_group_size()


        self.num_layers = num_layers
        self.patch_size = patch_size
        self.window_size = window_size
        self.embedding_dim = embedding_dim

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
        )

        patches_resolution = [
            (padded_shape[0]) // patch_size,
            (padded_shape[1]) // patch_size
        ]
        self.patches_resolution = patches_resolution


        self.layers = WindowParallelBasicLayer(dim=embedding_dim,
                               input_resolution=( patches_resolution[0],
                                                  patches_resolution[1]),
                               depth=num_layers,
                               num_heads=num_heads,
                               window_size=window_size,
                               manager = manager,
                               recompute_config=recompute_config,
                               use_attn_mask=use_attn_mask,
                               use_relative_position_bias=use_relative_position_bias,
                               use_flash_attention=use_flash_attention,
                               attention_init_seed_base=REFERENCE_INIT_SEEDS["attention"],
                               mlp_init_seed_base=REFERENCE_INIT_SEEDS["mlp"],
                               )

        self.embedding_parallel_type = embedding_parallel_type


        if embedding_parallel_type == 'window_linear':
            self.patch_embed = WrappedLinear(num_channels*patch_size*patch_size, embedding_dim, bias=True)
            self.patch_recovery = WrappedLinear(embedding_dim, num_channels*patch_size*patch_size, bias=True)
        elif embedding_parallel_type == 'window_embedding':
            self.patch_embed = WrappedLinear(num_channels*patch_size*patch_size, embedding_dim, bias=True)
            self.patch_recovery = WrappedLinear(embedding_dim, num_channels*patch_size*patch_size, bias=True)

        self.recompute_config = recompute_config or {}
        self.checkpoint_patch_embed = bool(self.recompute_config.get("enabled", False)) and bool(self.recompute_config.get("checkpoint_patch_embed", False))
        self.checkpoint_patch_recovery = bool(self.recompute_config.get("enabled", False)) and bool(self.recompute_config.get("checkpoint_patch_recovery", False))


        self.mp_all_reduce_list = []
        for full_module_name, module in self.named_modules():
            if (
                isinstance(
                    module,
                    (
                        WindowParallelSwinTransformer,
                        WindowParallelMLP,
                        WPUlyssesTensorParallelMLP,
                        #Window_ParaPatchEmbedding,
                        #Window_ParaPatchRecovery,
                        #WindowSequenceParallelSwinTransformer,
                        WrappedLinear,
                        #DownBlock,
                        #UpBlock,
                        DomainParallelCreditDownBlock,
                        DomainParallelCreditUpBlock,

                    )
                )
            ):

                for local_name, param in module.named_parameters(recurse=True):
                    full_name = f"{full_module_name}.{local_name}" if full_module_name else local_name


                    param.allreduce_wp_group = True
                    param.full_name = full_name


                    self.mp_all_reduce_list.append(full_name)


    def _get_rng_state(self):
        """Return current RNG states."""
        return {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'cpu': torch.get_rng_state()
        }

    def _restore_rng_state(self, saved_state):
        """Restore RNG states."""
        random.setstate(saved_state['python'])
        np.random.set_state(saved_state['numpy'])
        torch.set_rng_state(saved_state['cpu'])
        if torch.cuda.is_available() and saved_state['cuda'] is not None:
            torch.cuda.set_rng_state_all(saved_state['cuda'])


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

    def forward(self, input): # [1, 66, 552, 5952] for window_linear  [1, 1012, 36, 5952] for window_embedding
        x = self._run_patch_embed(input)

        if self.embedding_parallel_type == 'window_linear':
            x = x.permute(0, 3, 1, 2).contiguous()
            x = x.permute(0, 2, 3, 1).contiguous()

            B = x.shape[0]
            window_h = x.shape[1]
            window_w = x.shape[2]

            use_factorized_ulysses = (
                self.manager is not None
                and (
                    int(getattr(self.manager, "xfmr_sp_size", 1)) > 1
                    or int(getattr(self.manager, "xfmr_tp_size", 1)) > 1
                )
            )

            if use_factorized_ulysses:
                x = stripe_grid_to_ulysses_windows(
                    x,
                    self.manager,
                    self.window_size,
                    0,
                )
            else:
                x = stripe_grid_to_round_robin_windows(
                    x,
                    self.manager,
                    self.window_size,
                )

        x = self.layers(x)

        if self.embedding_parallel_type == 'window_linear':
            if use_factorized_ulysses:
                x = ulysses_windows_to_stripe_grid(
                    x,
                    self.manager,
                    window_h,
                    window_w,
                    self.window_size,
                    0,
                )
            else:
                x = round_robin_windows_to_stripe_grid(
                    x,
                    self.manager,
                    window_h,
                    window_w,
                    self.window_size,
                )
            x = x.contiguous()

        x = self._run_patch_recovery(x)
        x = x + input
        return x
