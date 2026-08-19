import torch
from torch import nn

from models.transformer import SequentialSwinTransformer, WindowParallelSwinTransformer
from models.mlp import SequentialMlp, WindowParallelMLP, WPUlyssesTensorParallelMLP
from models.utils import CreditDownBlock, CreditUpBlock

from credit.domain_parallel.layers import DomainParallelConv2d, DomainParallelGroupNorm, DomainParallelConvTranspose2d

from core.global_env_config import use_MLP
from core.checkpoint.activation import activation_checkpoint
from core.checkpoint.selective_recompute import SelectiveRecomputeScheduler
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


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        manager=None,
        recompute_config=None,
        use_attn_mask=True,
        use_relative_position_bias=True,
        use_flash_attention=False,
        attention_init_seed_base=None,
        mlp_init_seed_base=None,
    ):
        super().__init__()

        blocks = []
        for i in range(depth):
            modules = [
                SequentialSwinTransformer(
                    layer_idx=i,
                    height=input_resolution[0],
                    width=input_resolution[1],
                    embedding_dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    use_attn_mask=use_attn_mask,
                    use_relative_position_bias=use_relative_position_bias,
                    use_flash_attention=use_flash_attention,
                    init_seed_base=attention_init_seed_base,
                )
            ]
            if use_MLP:
                modules.append(
                    SequentialMlp(
                        in_features=dim,
                        hidden_features=dim * 4,
                        out_features=dim,
                        init_seed_base=None if mlp_init_seed_base is None else int(mlp_init_seed_base) + i * 100,
                    )
                )
            blocks.append(nn.Sequential(*modules))
        self.blocks = nn.ModuleList(blocks)


        self.use_checkpointing = "no"
        if not use_MLP:
            self.blocks = nn.ModuleList([nn.Sequential(block[0]) for block in self.blocks])

        self.recompute_scheduler = SelectiveRecomputeScheduler(
            recompute_config,
            depth=depth,
            input_resolution=input_resolution,
            hidden_dim=dim,
            module_name="sequential_basic_layer",
        )
        if self.recompute_scheduler.enabled and manager is not None and manager.get_rank() == 0:
            print("[SelectiveRecompute]", self.recompute_scheduler.summary())

    def run_blocks(self, start, end, x):
        for idx in range(start, end):
            x = self.blocks[idx](x)
        return x

    def _checkpoint_blocks(self, start, end, x):
        label = f"{self.recompute_scheduler.module_name}:{start}-{end}:checkpoint"
        context = get_memory_timeline_context()
        primitive_id = f"{self.recompute_scheduler.module_name}:{start}-{end}"
        metadata = {
            "lead_idx": context.get("lead_idx", ""),
            "primitive_kind": "transformer",
            "primitive_mode": "checkpoint",
            "primitive_part": "segment",
            "primitive_id": primitive_id,
            "occurrence_id": new_memory_timeline_occurrence(primitive_id),
            "segment_start": start,
            "segment_end": end - 1,
        }

        def custom_forward(inp, start=start, end=end, label=label, metadata=metadata):
            mark_memory_timeline(
                "primitive_body_pre", label, tensor_mib=_tensor_mib(inp), **metadata
            )
            mark_memory_timeline("transformer_segment_body_pre", label)
            try:
                return self.run_blocks(start, end, inp)
            finally:
                mark_memory_timeline("transformer_segment_body_post", label)
                mark_memory_timeline("primitive_body_post", label, **metadata)

        input_metadata = dict(metadata, tensor_mib=_tensor_mib(x))
        mark_memory_timeline("primitive_forward_pre", label, **input_metadata)
        mark_memory_timeline("transformer_segment_checkpoint_pre", label)
        checkpoint_input = profile_memory_backward_boundary(
            x, "primitive_backward_post", label, **input_metadata
        )
        out = activation_checkpoint(
            custom_forward,
            checkpoint_input,
            activation_config=self.recompute_scheduler.config_for_block(start),
        )
        mark_memory_timeline("transformer_segment_checkpoint_post", label)
        output_metadata = dict(metadata, tensor_mib=_tensor_mib(out))
        mark_memory_timeline("primitive_forward_post", label, **output_metadata)
        return profile_memory_backward_boundary(
            out, "primitive_backward_pre", label, **output_metadata
        )


    def forward(self, x):

        if self.recompute_scheduler.enabled:
            for start, end, should_checkpoint in self.recompute_scheduler.execution_segments():
                label = f"{self.recompute_scheduler.module_name}:{start}-{end}:{'checkpoint' if should_checkpoint else 'plain'}"
                mark_memory_timeline("transformer_segment_pre", label)
                if should_checkpoint:
                    x = self._checkpoint_blocks(start, end, x)
                else:
                    x = self.run_blocks(start, end, x)
                mark_memory_timeline("transformer_segment_post", label)
            return x

        if self.use_checkpointing=='no':
            for block in self.blocks:
                x = block(x)
        elif self.use_checkpointing=='torch':
            for block in self.blocks:
                x = activation_checkpoint(block, x, mode="torch_recompute")
        else:
            raise ValueError(f"Unsupported checkpointing mode: {self.use_checkpointing}")

        return x


class WindowParallelBasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        kaiming_init=True,
        manager=None,
        recompute_config=None,
        use_attn_mask=True,
        use_relative_position_bias=True,
        use_flash_attention=False,
        attention_init_seed_base=None,
        mlp_init_seed_base=None,
    ):
        super().__init__()

        parallel_mlp_cls = WPUlyssesTensorParallelMLP if (
            manager is not None
            and int(getattr(manager, "xfmr_tp_size", 1)) > 1
        ) else WindowParallelMLP

        self.blocks = nn.ModuleList([
            nn.Sequential(
                WindowParallelSwinTransformer(
                    kaiming_init = kaiming_init,
                    manager = manager,

                    layer_idx = i,
                    height=input_resolution[0],
                    width=input_resolution[1],
                    window_size = window_size,

                    embedding_dim = dim,
                    num_heads = num_heads,
                    use_attn_mask=use_attn_mask,
                    use_relative_position_bias=use_relative_position_bias,
                    use_flash_attention=use_flash_attention,
                    init_seed_base=attention_init_seed_base,
                ),
                parallel_mlp_cls(
                    manager = manager,
                    in_features=dim,
                    hidden_features = dim*4,
                    out_features = dim,
                    use_bias =True,
                    init_seed_base=None if mlp_init_seed_base is None else int(mlp_init_seed_base) + i * 100,
                )
            )
            for i in range(depth)
        ])

        self.use_checkpointing = "no"

        self.checkpoint_every_n_layers = 4

        if not use_MLP:
            self.blocks = nn.ModuleList([nn.Sequential(block[0]) for block in self.blocks])

        self.recompute_scheduler = SelectiveRecomputeScheduler(
            recompute_config,
            depth=depth,
            input_resolution=input_resolution,
            hidden_dim=dim,
            module_name="parallel_basic_layer",
        )
        if self.recompute_scheduler.enabled and manager is not None and manager.get_rank() == 0:
            print("[SelectiveRecompute]", self.recompute_scheduler.summary())

    def run_blocks(self, start, end, x):
        for idx in range(start, end):
            x = self.blocks[idx](x)
        return x


    def _checkpoint_blocks(self, start, end, x):
        label = f"{self.recompute_scheduler.module_name}:{start}-{end}:checkpoint"
        context = get_memory_timeline_context()
        primitive_id = f"{self.recompute_scheduler.module_name}:{start}-{end}"
        metadata = {
            "lead_idx": context.get("lead_idx", ""),
            "primitive_kind": "transformer",
            "primitive_mode": "checkpoint",
            "primitive_part": "segment",
            "primitive_id": primitive_id,
            "occurrence_id": new_memory_timeline_occurrence(primitive_id),
            "segment_start": start,
            "segment_end": end - 1,
        }

        def custom_forward(inp, start=start, end=end, label=label, metadata=metadata):
            mark_memory_timeline(
                "primitive_body_pre", label, tensor_mib=_tensor_mib(inp), **metadata
            )
            mark_memory_timeline("transformer_segment_body_pre", label)
            try:
                return self.run_blocks(start, end, inp)
            finally:
                mark_memory_timeline("transformer_segment_body_post", label)
                mark_memory_timeline("primitive_body_post", label, **metadata)

        input_metadata = dict(metadata, tensor_mib=_tensor_mib(x))
        mark_memory_timeline("primitive_forward_pre", label, **input_metadata)
        mark_memory_timeline("transformer_segment_checkpoint_pre", label)
        checkpoint_input = profile_memory_backward_boundary(
            x, "primitive_backward_post", label, **input_metadata
        )
        out = activation_checkpoint(
            custom_forward,
            checkpoint_input,
            activation_config=self.recompute_scheduler.config_for_block(start),
        )
        mark_memory_timeline("transformer_segment_checkpoint_post", label)
        output_metadata = dict(metadata, tensor_mib=_tensor_mib(out))
        mark_memory_timeline("primitive_forward_post", label, **output_metadata)
        return profile_memory_backward_boundary(
            out, "primitive_backward_pre", label, **output_metadata
        )


    def forward(self, x):
        if self.recompute_scheduler.enabled:
            for start, end, should_checkpoint in self.recompute_scheduler.execution_segments():
                label = f"{self.recompute_scheduler.module_name}:{start}-{end}:{'checkpoint' if should_checkpoint else 'plain'}"
                mark_memory_timeline("transformer_segment_pre", label)
                if should_checkpoint:
                    x = self._checkpoint_blocks(start, end, x)
                else:
                    x = self.run_blocks(start, end, x)
                mark_memory_timeline("transformer_segment_post", label)
            return x

        if self.use_checkpointing=='no':
            for block in self.blocks:
                x = block(x)
        elif self.use_checkpointing=='torch':
            n = self.checkpoint_every_n_layers
            for start in range(0, len(self.blocks), n):

                end = min(start + n, len(self.blocks))

                def custom_forward(inp, start=start, end=end):
                    return self.run_blocks(start, end, inp)

                x = activation_checkpoint(custom_forward, x, mode="torch_recompute")
        else:
            raise ValueError(f"Unsupported checkpointing mode: {self.use_checkpointing}")
        return x

class DomainParallelCreditDownBlock(nn.Module):
    def __init__(self, ref: CreditDownBlock, manager=None):
        super().__init__()

        self.manager = manager

        self.use_checkpointing = "no"
        self.conv = self._wrap_conv(ref.conv, manager)

        blk = []
        for layer in ref.b:
            if isinstance(layer, nn.Conv2d):
                blk.append(self._wrap_conv(layer, manager))
            elif isinstance(layer, nn.GroupNorm):
                #blk.append(DomainParallelGroupNorm(layer))
                blk.append(self._wrap_groupnorm(layer, manager))
            elif isinstance(layer, nn.SiLU):
                blk.append(nn.SiLU())
            else:
                raise TypeError(f"Unsupported layer: {type(layer)}")
        self.b = nn.Sequential(*blk)

    @staticmethod
    def _wrap_conv(layer, manager):

        wrapped = DomainParallelConv2d(
            nn.Conv2d(
                layer.in_channels,
                layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=layer.bias is not None,
            ),
            shard_dim=-2,
            manager=manager,
        )
        wrapped.conv.load_state_dict(layer.state_dict())
        return wrapped


    @staticmethod
    def _wrap_groupnorm(layer, manager):
        """Wrap GroupNorm as a domain-parallel GroupNorm layer."""

        wrapped = DomainParallelGroupNorm(
            torch.nn.GroupNorm(
                num_groups=layer.num_groups,
                num_channels=layer.num_channels,
                eps=layer.eps,
                affine=layer.affine
            ),
            manager=manager,
        )


        if layer.affine:

            if hasattr(layer, 'weight') and layer.weight is not None:
                wrapped.weight.data.copy_(layer.weight.data)
            if hasattr(layer, 'bias') and layer.bias is not None:
                wrapped.bias.data.copy_(layer.bias.data)


        return wrapped


    def forward(self, x):


        x = self.conv(x)


        shortcut = x.clone()

        if self.use_checkpointing=='no':
            x = self.b(x)
        elif self.use_checkpointing=='torch':
            x = activation_checkpoint(self.b, x, mode="torch_recompute")
        else:
            raise ValueError(f"Unsupported checkpointing mode: {self.use_checkpointing}")


        return x + shortcut


class DomainParallelCreditUpBlock(nn.Module):
    def __init__(self, ref: CreditUpBlock, manager=None):
        super().__init__()
        self.manager = manager
        self.use_checkpointing = "no"
        self.conv = self._wrap_conv_transpose(ref.conv, manager)
        self.b = self._wrap_block(ref.b, manager)

    @staticmethod
    def _wrap_conv(layer, manager):
        conv = nn.Conv2d(
            layer.in_channels,
            layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
        )
        wrapped = DomainParallelConv2d(
            conv,
            shard_dim=-2,
            manager = manager,
        )
        wrapped.conv.load_state_dict(layer.state_dict())
        return wrapped

    @staticmethod
    def _wrap_conv_transpose(layer, manager):
        conv = nn.ConvTranspose2d(
            layer.in_channels,
            layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            output_padding=layer.output_padding,
            groups=layer.groups,
            bias=layer.bias is not None,
            dilation=layer.dilation,
        )
        wrapped = DomainParallelConvTranspose2d(
            conv,
            shard_dim=-2,
            manager = manager,
        )
        wrapped.conv.load_state_dict(layer.state_dict())
        return wrapped

    @staticmethod
    def _wrap_groupnorm(layer, manager):
        norm = nn.GroupNorm(
            num_groups=layer.num_groups,
            num_channels=layer.num_channels,
            eps=layer.eps,
            affine=layer.affine,
        )
        if layer.affine:
            norm.weight.data.copy_(layer.weight.data)
            norm.bias.data.copy_(layer.bias.data)
        return DomainParallelGroupNorm(norm, manager=manager)

    @classmethod
    def _wrap_block(cls, block, manager):
        wrapped = []
        for layer in block:
            if isinstance(layer, nn.Conv2d):
                wrapped.append(cls._wrap_conv(layer, manager))
            elif isinstance(layer, nn.GroupNorm):
                wrapped.append(cls._wrap_groupnorm(layer, manager))
            elif isinstance(layer, nn.SiLU):
                wrapped.append(nn.SiLU())
            else:
                raise TypeError(f"Unsupported layer: {type(layer)}")
        return nn.Sequential(*wrapped)

    def forward(self, x):
        x = self.conv(x)


        shortcut = x.clone()


        if self.use_checkpointing=='no':
            x = self.b(x)
        elif self.use_checkpointing=='torch':
            x = activation_checkpoint(self.b, x, mode="torch_recompute")
        else:
            raise ValueError(f"Unsupported checkpointing mode: {self.use_checkpointing}")


        return x + shortcut
