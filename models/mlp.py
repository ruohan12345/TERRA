# we define mlp here, and use tensor parallel to optimize


import torch
from torch import nn
from core.tensor_parallel import _reduce, _CopyToModelParallelRegion, _ReduceFromModelParallelRegion

from core.tensor_parallel import AllGatherReduceScatter_forMLP, ReduceScatterAllGatherDim_forMLP


from utils import all_reduce_and_print_rank0

from models.model_utils.init_weight import init_like_linear, make_linear_with_seed, safe_linear_with_weight

from core.global_env_config import use_layernorm, use_MLP


def _disable_parameters_if_mlp_off(module):
    if not use_MLP:
        for param in module.parameters():
            param.requires_grad_(False)


class _CopyToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group, group_size):
        ctx.group = group
        ctx.group_size = group_size
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.group_size > 1:
            grad_output = grad_output.contiguous()
            torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM, group=ctx.group)
        return grad_output, None, None


def _copy_to_tensor_parallel_region(input_, group, group_size):
    return _CopyToTensorParallelRegion.apply(input_, group, group_size)


def _reduce_scatter_dim1_sum(input_, group, group_size):
        if group_size <= 1:
                return input_
        if input_.shape[1] % group_size != 0:
                raise RuntimeError(
                        f"reduce-scatter along dim=1 requires size {input_.shape[1]} "
                        f"divisible by group_size={group_size}"
                )

        input_ = input_.contiguous()
        perm = [1, 0] + list(range(2, input_.dim()))
        input_dim0 = input_.permute(perm).contiguous()
        chunk0 = input_dim0.shape[0] // group_size
        output_dim0 = torch.empty(
                (chunk0, *input_dim0.shape[1:]),
                dtype=input_.dtype,
                device=input_.device,
        )

        if hasattr(torch.distributed, "reduce_scatter_tensor"):
                torch.distributed.reduce_scatter_tensor(
                        output_dim0,
                        input_dim0,
                        op=torch.distributed.ReduceOp.SUM,
                        group=group,
                )
        else:
                chunks = [chunk.contiguous() for chunk in torch.chunk(input_dim0, group_size, dim=0)]
                torch.distributed.reduce_scatter(
                        output_dim0,
                        chunks,
                        op=torch.distributed.ReduceOp.SUM,
                        group=group,
                )

        return output_dim0.permute(perm).contiguous()


class _GatherWindowBatchToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group, group_size, group_rank):
        ctx.group = group
        ctx.group_size = group_size
        ctx.group_rank = group_rank
        if group_size <= 1:
            return input_

        input_ = input_.contiguous()
        gathered = [torch.empty_like(input_) for _ in range(group_size)]
        torch.distributed.all_gather(gathered, input_, group=group)
        return torch.cat(gathered, dim=1).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.group_size <= 1:
            return grad_output, None, None, None

        grad_input = _reduce_scatter_dim1_sum(
                grad_output,
                ctx.group,
                ctx.group_size,
        )
        return grad_input, None, None, None


class _ReduceScatterWindowBatchFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group, group_size, group_rank):
        ctx.group = group
        ctx.group_size = group_size
        ctx.group_rank = group_rank
        if group_size <= 1:
            return input_

        return _reduce_scatter_dim1_sum(input_, group, group_size)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.group_size <= 1:
            return grad_output, None, None, None

        grad_output = grad_output.contiguous()
        gathered = [torch.empty_like(grad_output) for _ in range(ctx.group_size)]
        torch.distributed.all_gather(gathered, grad_output, group=ctx.group)
        return torch.cat(gathered, dim=1).contiguous(), None, None, None


def _gather_window_batch_to_tensor_parallel_region(input_, group, group_size, group_rank):
    return _GatherWindowBatchToTensorParallelRegion.apply(input_, group, group_size, group_rank)


def _reduce_scatter_window_batch_from_tensor_parallel_region(input_, group, group_size, group_rank):
    return _ReduceScatterWindowBatchFromTensorParallelRegion.apply(input_, group, group_size, group_rank)


class SequentialMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., init_seed_base=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if use_layernorm:
                self.norm2 = torch.nn.LayerNorm(in_features)
        self.fc1 = make_linear_with_seed(in_features, hidden_features, init_seed=None if init_seed_base is None else init_seed_base + 1) # 4320 17280
        self.act = act_layer()
        self.fc2 = make_linear_with_seed(hidden_features, out_features, init_seed=None if init_seed_base is None else init_seed_base + 2)

        _disable_parameters_if_mlp_off(self)

    def forward(self, x):
        if not use_MLP:
            return x
        shortcut = x


        if use_layernorm:
                x = self.norm2(x)


        x = self.fc1(x)


        x = self.act(x)


        x = self.fc2(x)


        x = x + shortcut


        return x


class TensorParallelMLP(nn.Module):
    def __init__(self,
                manager = None,
                in_features=None,
                hidden_features=None,
                out_features=None,
                use_bias =True,
                act_layer=nn.GELU, drop=0.):
        super().__init__()

        self.manager = manager

        if use_layernorm:
                self.norm2 = torch.nn.LayerNorm(in_features)

        self.weight1, self.bias1 = init_like_linear(
                                    model_type = 'parallel',
                                    linear_in_dim = in_features,
                                    linear_out_dim = hidden_features,
                                    mp_rank = manager.get_mp_rank(), # do we need mp_group_size?
                                    mp_group_size = manager.mp_group_size,
                                    split_dim = 0,
                                    split_bias = True,
                                )

        self.weight2, self.bias2 = init_like_linear(
                                    model_type = 'parallel',
                                    linear_in_dim = hidden_features,
                                    linear_out_dim = out_features,
                                    mp_rank = manager.get_mp_rank(), # do we need mp_group_size?
                                    mp_group_size = manager.mp_group_size,
                                    split_dim = -1,
                                    split_bias = False,
                                )

        self.linear1 = safe_linear_with_weight(weight=self.weight1, bias = self.bias1)
        self.linear2 = safe_linear_with_weight(weight=self.weight2, bias = None)
        _disable_parameters_if_mlp_off(self)

        self.act = act_layer()

        self.copy_to_tensor_model_parallel_region = _CopyToModelParallelRegion().apply
        self.reduce_from_tensor_model_parallel_region = _ReduceFromModelParallelRegion().apply


    def forward(self, x):
        if not use_MLP:
            return x


        short_cut = x

        x = self.norm2(x)


        input_parallel = self.copy_to_tensor_model_parallel_region(x, self.manager)


        intermediate_parallel = self.linear1(input_parallel)


        intermediate_parallel = self.act(intermediate_parallel)

        if True:
                intermediate_parallel = self.linear2(intermediate_parallel)
        else:
                intermediate_parallel = self.parallel_linear(intermediate_parallel, self.weight2, None)
                # first allreduce data, then add a common bias
        output = self.reduce_from_tensor_model_parallel_region(intermediate_parallel, self.manager)


        output = output + self.bias2.type(output.dtype)

        output = output + short_cut

        return output


class MegatronSequenceParallelMLP(nn.Module):
    def __init__(self,
                manager = None,
                in_features=None,
                hidden_features=None,
                out_features=None,
                use_bias =True,
                act_layer=nn.GELU, drop=0.):
        super().__init__()

        self.manager = manager

        self.tp_size = manager.get_mp_group_size()
        self.mp_rank = manager.get_mp_rank()

        if use_layernorm:
                self.norm2 = torch.nn.LayerNorm(in_features)

        self.weight1, self.bias1 = init_like_linear(
                                    model_type = 'parallel',
                                    linear_in_dim = in_features,
                                    linear_out_dim = hidden_features,
                                    mp_rank = self.mp_rank, # do we need mp_group_size?
                                    mp_group_size = self.tp_size,
                                    split_dim = 0,
                                    split_bias = True,
                                )

        self.weight2, self.bias2 = init_like_linear(
                                    model_type = 'parallel',
                                    linear_in_dim = hidden_features,
                                    linear_out_dim = out_features,
                                    mp_rank = self.mp_rank, # do we need mp_group_size?
                                    mp_group_size = self.tp_size,
                                    split_dim = -1,
                                    split_bias = False,
                                )

        self.linear1 = safe_linear_with_weight(weight=self.weight1, bias = self.bias1)
        self.linear2 = safe_linear_with_weight(weight=self.weight2, bias = None)
        _disable_parameters_if_mlp_off(self)


        self.weight1 = None
        self.bias1 = None
        self.weight2 = None


        self.act = act_layer()

        # AllGatherReduceScatter, ReduceScatterAllGatherDim

        self.all_gather_dim = AllGatherReduceScatter_forMLP().apply
        self.reduce_scatter_dim = ReduceScatterAllGatherDim_forMLP().apply


    def forward(self, x): # [1, 14400, 768]
        if not use_MLP:
            return x
        short_cut = x


        if use_layernorm:
                x = self.norm2(x) # [1, 14400, 768]


        x = self.all_gather_dim(x, self.manager.model_parallel_group, 1) # [1, 28800, 768]


        x = self.linear1(x) # [1, 28800, 1536]


        x = self.act(x)


        x = self.linear2(x) # [1, 28800, 768]


        output = self.reduce_scatter_dim(x, self.manager.model_parallel_group, 1) # [1, 14400, 768]


        output = output + self.bias2.type(output.dtype)

        output = output + short_cut

        return output


class UlyseesSequenceParallelMLP(nn.Module):
        def __init__(self,
                manager = None,
                in_features=None,
                hidden_features=None,
                out_features=None,
                use_bias =True,
                act_layer=nn.GELU, drop=0.):

                super().__init__()

                self.manager = manager

                if use_layernorm:
                        self.norm2 = torch.nn.LayerNorm(in_features)
                self.fc1 = nn.Linear(in_features, hidden_features) # 4320 17280
                self.act = act_layer()
                self.fc2 = nn.Linear(hidden_features, out_features)
                _disable_parameters_if_mlp_off(self)

        def forward(self, x):
                if not use_MLP:
                        return x
                shortcut = x
                if use_layernorm:
                        x = self.norm2(x)
                x = self.fc1(x)
                x = self.act(x)
                x = self.fc2(x)
                x = x + shortcut # [2, 200, 36, 768]
                return x

class WindowParallelMLP(nn.Module):
        def __init__(self,
                        manager = None,
                        in_features=None,
                        hidden_features=None,
                        out_features=None,
                        use_bias =True,
                        act_layer=nn.GELU, drop=0.,
                        init_seed_base=None):
                super().__init__()

                self.manager = manager

                if use_layernorm:
                        self.norm2 = torch.nn.LayerNorm(in_features)
                self.fc1 = make_linear_with_seed(in_features, hidden_features, init_seed=None if init_seed_base is None else init_seed_base + 1) # 4320 17280
                self.act = act_layer()
                self.fc2 = make_linear_with_seed(hidden_features, out_features, init_seed=None if init_seed_base is None else init_seed_base + 2)
                _disable_parameters_if_mlp_off(self)

        def forward(self, x): # [2, 200, 36, 768]
                if not use_MLP:
                        return x


                #exit(0)


                shortcut = x
                if use_layernorm:
                        x = self.norm2(x)
                x = self.fc1(x)
                x = self.act(x)
                x = self.fc2(x)
                x = x + shortcut # [2, 200, 36, 768]


                return x


class WPUlyssesTensorParallelMLP(nn.Module):
        def __init__(self,
                        manager = None,
                        in_features=None,
                        hidden_features=None,
                        out_features=None,
                        use_bias =True,
                        act_layer=nn.GELU, drop=0.,
                        init_seed_base=None):
                super().__init__()

                self.manager = manager
                self.tp_size = int(getattr(manager, "xfmr_tp_size", 1))
                self.tp_rank = int(getattr(manager, "xfmr_tp_rank", 0))
                self.tp_group = getattr(manager, "xfmr_tp_group", None)

                out_features = out_features or in_features
                hidden_features = hidden_features or in_features
                if hidden_features % self.tp_size != 0:
                        raise ValueError(f"hidden_features={hidden_features} must be divisible by tensor_parallel_size={self.tp_size}")
                if in_features != out_features:
                        raise ValueError("WPUlyssesTensorParallelMLP currently expects in_features == out_features")

                if use_layernorm:
                        self.norm2 = torch.nn.LayerNorm(in_features)

                self.weight1, self.bias1 = init_like_linear(
                                    model_type = 'parallel',
                                    linear_in_dim = in_features,
                                    linear_out_dim = hidden_features,
                                    mp_rank = self.tp_rank,
                                    mp_group_size = self.tp_size,
                                    split_dim = 0,
                                    split_bias = True,
                                    init_seed = None if init_seed_base is None else init_seed_base + 1,
                                )

                self.weight2, self.bias2 = init_like_linear(
                                    model_type = 'parallel',
                                    linear_in_dim = hidden_features,
                                    linear_out_dim = out_features,
                                    mp_rank = self.tp_rank,
                                    mp_group_size = self.tp_size,
                                    split_dim = -1,
                                    split_bias = False,
                                    init_seed = None if init_seed_base is None else init_seed_base + 2,
                                )

                self.linear1 = safe_linear_with_weight(weight=self.weight1, bias=self.bias1)
                self.linear2 = safe_linear_with_weight(weight=self.weight2, bias=None)

                for param in self.linear1.parameters():
                        param.terra_grad_reduce_group = "xfmr_tp_param_group"
                for param in self.linear2.parameters():
                        param.terra_grad_reduce_group = "xfmr_tp_param_group"

                self.weight1 = None
                self.bias1 = None
                self.weight2 = None

                self.act = act_layer()
                _disable_parameters_if_mlp_off(self)

        def forward(self, x):
                if not use_MLP:
                        return x

                shortcut = x
                input_is_4d = x.dim() == 4
                if not input_is_4d and self.tp_size > 1:
                        raise RuntimeError(
                                "WPUlysses TP requires 4D sharded activation: "
                                "[B, local_windows, shard_tokens, C]"
                        )

                if use_layernorm:
                        x = self.norm2(x)

                if input_is_4d:
                        B, local_num_windows, shard_tokens, C = x.shape
                        x = _gather_window_batch_to_tensor_parallel_region(
                                x,
                                self.tp_group,
                                self.tp_size,
                                self.tp_rank,
                        )
                        full_num_windows = x.shape[1]
                        x = x.contiguous().view(B * full_num_windows, shard_tokens, C)
                else:
                        B = None
                        full_num_windows = None
                        shard_tokens = None
                        C = x.shape[-1]
                        x = _copy_to_tensor_parallel_region(x, self.tp_group, self.tp_size)

                x = self.linear1(x)
                x = self.act(x)
                x = self.linear2(x)

                if input_is_4d:
                        x = x.contiguous().view(B, full_num_windows, shard_tokens, C)
                        x = _reduce_scatter_window_batch_from_tensor_parallel_region(
                                x,
                                self.tp_group,
                                self.tp_size,
                                self.tp_rank,
                        )
                elif self.tp_size > 1:
                        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM, group=self.tp_group)

                x = x + self.bias2.type(x.dtype)
                if x.shape != shortcut.shape:
                        raise RuntimeError(f"S-WSTP MLP residual shape mismatch: x={x.shape}, shortcut={shortcut.shape}")
                x = x + shortcut

                return x
