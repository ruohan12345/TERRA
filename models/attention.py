import torch
from timm.models.layers import trunc_normal_

from models.model_utils.init_weight import init_like_linear, make_linear_with_seed, run_with_seed, safe_linear_with_weight

from core.tensor_parallel import _reduce, _CopyToModelParallelRegion, _ReduceFromModelParallelRegion

from core.tensor_parallel import ReduceScatterAllGatherDim

from core.tensor_parallel import generic_all_to_all

from core.tensor_parallel import generic_all_to_all_group


try:
    from flash_attn import flash_attn_func
    _flash_attn_import_error = None
except Exception as exc:
    flash_attn_func = None
    _flash_attn_import_error = exc

use_qkv = False # set to false for default


def require_flash_attention_available():
    if flash_attn_func is None:
        detail = "" if _flash_attn_import_error is None else f" Import error: {_flash_attn_import_error!r}"
        raise RuntimeError(
            "USE_FLASH_ATTENTION=True requires flash-attn installed on every rank/server. "
            "Please install flash-attn or set USE_FLASH_ATTENTION=False."
            + detail
        )


def _can_use_flash_attention(q, mask, use_flash_attention, use_rel_pos_bias):
    return (
        use_flash_attention
        and flash_attn_func is not None
        and mask is None
        and (not use_rel_pos_bias)
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
    )


def _flash_attention_from_bhnd(q, k, v, softmax_scale):
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    x = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
    )
    return x


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


class WindowAttention(torch.nn.Module):
    def __init__(self,
        embedding_dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        use_attn_mask=True,
        use_relative_position_bias=True,
        use_flash_attention=False,
        init_seed_base=None,
        ):
        super().__init__()

        self.use_attn_mask = use_attn_mask
        self.use_relative_position_bias = use_relative_position_bias
        self.use_flash_attention = use_flash_attention
        self.dim = embedding_dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, f"dim {self.dim} must be divisible by num_heads {num_heads}"
        head_dim = self.dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if self.use_relative_position_bias:

            self.relative_position_bias_table = torch.nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            run_with_seed(init_seed_base, lambda: trunc_normal_(self.relative_position_bias_table, std=.02))


        if use_qkv:
            self.qkv = make_linear_with_seed(self.dim, self.dim * 3, bias=qkv_bias, init_seed=None if init_seed_base is None else init_seed_base + 1)
        else:
            self.q_linear = make_linear_with_seed(self.dim, self.dim, bias=qkv_bias, init_seed=None if init_seed_base is None else init_seed_base + 1)
            self.k_linear = make_linear_with_seed(self.dim, self.dim, bias=qkv_bias, init_seed=None if init_seed_base is None else init_seed_base + 2)
            self.v_linear = make_linear_with_seed(self.dim, self.dim, bias=qkv_bias, init_seed=None if init_seed_base is None else init_seed_base + 3)


        self.proj = make_linear_with_seed(self.dim, self.dim, bias=True, init_seed=None if init_seed_base is None else init_seed_base + 4)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape # [800, 36, 4320]
        #  qkv can be transformed into ColumnParallelLinear

        # b, s, H -> b, s, 3H -> b, s, 3, n, h -> 3,b,n,s,h
        # b, s, H -> b, s, 3H/t ->b, s, 3,


        if use_qkv:
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # [3, 800, 60, 36, 72]
            print('seq qkv sum', qkv.sum())
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = self.q_linear(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # [800, 36, 4320] b, s, H-> [800, 60, 36, 72]
            k = self.k_linear(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v_linear(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)


        if _can_use_flash_attention(q, mask, self.use_flash_attention, self.use_relative_position_bias):


            x = _flash_attention_from_bhnd(q, k, v, self.scale).reshape(B_, N, C)
            x = self.proj(x)
            return x

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))

        if self.use_relative_position_bias:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


            attn = attn + relative_position_bias.unsqueeze(0) # [800, 60, 36, 36]


        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn) # [800, 60, 36, 36]


        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)


        x = self.proj(x)

        return x


class ParallelWindowAttention(torch.nn.Module):
    def __init__(self,
        kaiming_init,
        manager,
        tp_size,
        mp_rank,
        embedding_dim,
        window_size,
        num_heads,
        parallel_model_type='tensor_parallel', # 'sequence_parallel',
        qkv_bias=True,
        qk_scale=None,
        ):
        super().__init__()

        self.dim = embedding_dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads

        self.parallel_model_type = parallel_model_type

        self.manager = manager
        self.tp_size = tp_size

        assert self.dim % num_heads == 0, f"dim {self.dim} must be divisible by num_heads {num_heads}"
        assert num_heads % self.tp_size == 0, f"num_heads {num_heads} must be divisible by tp_size {tp_size}"


        head_dim = self.dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5


        if use_relative_position_bias:
            if kaiming_init:


                #    torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads//self.tp_size))  # 2*Wh-1 * 2*Ww-1, nH/tp_size


                tmp_file = torch.nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
                trunc_normal_(tmp_file, std=.02) # [121, 12] -1.2661

                tmp_file = torch.split(tmp_file, tmp_file.shape[-1]//manager.get_mp_group_size(), dim = -1)


                local_bias_table = tmp_file[manager.get_mp_rank()].contiguous()
                self.relative_position_bias_table = torch.nn.Parameter(local_bias_table)


                # get pair-wise relative position index for each token inside the window
                coords_h = torch.arange(self.window_size[0])
                coords_w = torch.arange(self.window_size[1])
                coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
                coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
                relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
                relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
                relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
                relative_coords[:, :, 1] += self.window_size[1] - 1
                relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
                relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
                self.register_buffer("relative_position_index", relative_position_index)


            else:

                self.relative_position_bias_table = torch.nn.Parameter(
                    torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads//self.tp_size))  # 2*Wh-1 * 2*Ww-1, nH/tp_size

                # get pair-wise relative position index for each token inside the window
                coords_h = torch.arange(self.window_size[0])
                coords_w = torch.arange(self.window_size[1])
                coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
                coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
                relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
                relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
                relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
                relative_coords[:, :, 1] += self.window_size[1] - 1
                relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
                relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
                self.register_buffer("relative_position_index", relative_position_index)
                trunc_normal_(self.relative_position_bias_table, std=.02)


        if use_qkv:

            self.weight_qkv, self.bias_qkv = init_like_linear(
                                        model_type = 'parallel',
                                        linear_in_dim = self.dim,
                                        linear_out_dim = self.dim * 3,
                                        mp_rank = mp_rank, # do we need mp_group_size?
                                        mp_group_size = tp_size,
                                        split_dim = 0,
                                        split_bias = True,
                                    )
        else:
            self.weight_q, self.bias_q = init_like_linear(
                                        model_type = 'parallel',
                                        linear_in_dim = self.dim,
                                        linear_out_dim = self.dim,
                                        mp_rank = mp_rank, # do we need mp_group_size?
                                        mp_group_size = tp_size,
                                        split_dim = 0,
                                        split_bias = True,
                                    )
            self.weight_k, self.bias_k = init_like_linear(
                                        model_type = 'parallel',
                                        linear_in_dim = self.dim,
                                        linear_out_dim = self.dim,
                                        mp_rank = mp_rank, # do we need mp_group_size?
                                        mp_group_size = tp_size,
                                        split_dim = 0,
                                        split_bias = True,
                                    )
            self.weight_v, self.bias_v = init_like_linear(
                                        model_type = 'parallel',
                                        linear_in_dim = self.dim,
                                        linear_out_dim = self.dim,
                                        mp_rank = mp_rank, # do we need mp_group_size?
                                        mp_group_size = tp_size,
                                        split_dim = 0,
                                        split_bias = True,
                                    )


            self.q_linear = safe_linear_with_weight(weight=self.weight_q, bias = self.bias_q)
            self.k_linear = safe_linear_with_weight(weight=self.weight_k, bias = self.bias_k)
            self.v_linear = safe_linear_with_weight(weight=self.weight_v, bias = self.bias_v)


            self.weight_q = None
            self.bias_q = None

            self.weight_k = None
            self.bias_k = None

            self.weight_v = None
            self.bias_v = None


        self.copy_to_tensor_model_parallel_region = _CopyToModelParallelRegion().apply
        self.reduce_from_tensor_model_parallel_region = _ReduceFromModelParallelRegion().apply
        self.reduce_scatter_dim = ReduceScatterAllGatherDim().apply


        if self.parallel_model_type=='tensor_parallel':
            my_split_bias = False
        elif self.parallel_model_type=='sequence_parallel':
            my_split_bias = True

        self.weight_proj, self.bias_proj = init_like_linear(
                                    model_type = 'parallel',
                                    linear_in_dim = self.dim,
                                    linear_out_dim = self.dim,
                                    mp_rank = mp_rank, # do we need mp_group_size?
                                    mp_group_size = tp_size,
                                    split_dim = -1,
                                    split_bias = my_split_bias,
                                )

        self.proj = safe_linear_with_weight(weight=self.weight_proj, bias = None)
        self.weight_proj = None


        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape # fp32
        #  qkv can be transformed into ColumnParallelLinear


        if use_qkv:
            if False:
                qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # [3, 800, 60, 36, 72]
            else:
                input_parallel = self.copy_to_tensor_model_parallel_region(x)
                qkv = self.parallel_linear(input_parallel, self.weight_qkv, self.bias_qkv) # [800, 36, 6480]

                qkv = qkv.reshape(B_, N, 3, self.num_heads//self.tp_size, C // self.num_heads).permute(2, 0, 3, 1, 4) # [3, 800, 30, 36, 72]

                print('parallel qkv sum', qkv.sum(), qkv.shape)

            q, k, v = qkv[0], qkv[1], qkv[2]
        else:

            if self.parallel_model_type=='tensor_parallel':

                input_parallel = self.copy_to_tensor_model_parallel_region(x, self.manager)
                q = self.q_linear(input_parallel).reshape(B_, N, self.num_heads//self.tp_size, C // self.num_heads).permute(0, 2, 1, 3)
                k = self.k_linear(input_parallel).reshape(B_, N, self.num_heads//self.tp_size, C // self.num_heads).permute(0, 2, 1, 3)
                v = self.v_linear(input_parallel).reshape(B_, N, self.num_heads//self.tp_size, C // self.num_heads).permute(0, 2, 1, 3)
            elif self.parallel_model_type=='sequence_parallel':
                q = self.q_linear(x).reshape(B_, N, self.num_heads//self.tp_size, C // self.num_heads).permute(0, 2, 1, 3)
                k = self.k_linear(x).reshape(B_, N, self.num_heads//self.tp_size, C // self.num_heads).permute(0, 2, 1, 3)
                v = self.v_linear(x).reshape(B_, N, self.num_heads//self.tp_size, C // self.num_heads).permute(0, 2, 1, 3)


        q = q * self.scale

        attn = (q @ k.transpose(-2, -1)) # [800, 30, 36, 36]

        if use_relative_position_bias:


            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww   [60, 36, 36]


            attn = attn + relative_position_bias.unsqueeze(0) # [800, 60, 36, 36]

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads//self.tp_size, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads//self.tp_size, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn) # [800, 60, 36, 36]


        x = (attn @ v).transpose(1, 2).reshape(B_, N, C//self.tp_size) # [800, 36, 2160]


        if True:
            x = self.proj(x)
            if self.parallel_model_type=='tensor_parallel':
                output = self.reduce_from_tensor_model_parallel_region(x, self.manager)
                output = output + self.bias_proj.type(output.dtype)
            elif self.parallel_model_type=='sequence_parallel':


                output = self.reduce_scatter_dim(x, self.manager.model_parallel_group, -1)

                output = output + self.bias_proj.type(output.dtype)
        else:
            x = self.parallel_linear(x, self.weight_proj, None)

            output = self.reduce_from_tensor_model_parallel_region(x, self.manager)

            output = output + self.bias_proj.type(output.dtype)

        return output


class UlyssesWindowAttention(torch.nn.Module):
    def __init__(self,
        manager,
        embedding_dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        ):
        super().__init__()
        self.manager = manager
        self.dim = embedding_dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, f"dim {self.dim} must be divisible by num_heads {num_heads}"
        head_dim = self.dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.mp_size = manager.get_mp_group_size()

        if use_relative_position_bias:
            print('we do not consider use_relative_position_bias yet')
            exit(0)

        if use_qkv:

            print('We should probably avoid using use_qkv')
            exit(0)
        else:
            self.q_linear = torch.nn.Linear(self.dim, self.dim, bias=qkv_bias)
            self.k_linear = torch.nn.Linear(self.dim, self.dim, bias=qkv_bias)
            self.v_linear = torch.nn.Linear(self.dim, self.dim, bias=qkv_bias)

        self.proj = torch.nn.Linear(self.dim, self.dim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, mask=None): # [800, 18, 768]
        B_, N, C = x.shape
        N = N * self.mp_size # 36

        q = self.q_linear(x) # [800, 18, 768]
        k = self.k_linear(x)
        v = self.v_linear(x)

        q = generic_all_to_all(q, self.manager, split_dim=2, concat_dim=1) # [800, 18, 768] -> [800, 36, 384]
        k = generic_all_to_all(k, self.manager, split_dim=2, concat_dim=1)
        v = generic_all_to_all(v, self.manager, split_dim=2, concat_dim=1)

        q = q.reshape(B_, N, self.num_heads // self.mp_size, C // self.num_heads).permute(0, 2, 1, 3) # [800, 6, 36, 64]
        k = k.reshape(B_, N, self.num_heads // self.mp_size, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, self.num_heads // self.mp_size, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # [800, 6, 36, 36]

        if use_relative_position_bias:
            print('we do not consider use_relative_position_bias yet')
            exit(0)

        if mask is not None:
            print('we do not consider mask yet')
            exit(0)

            '''
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
            '''
        else:
            attn = self.softmax(attn) # [800, 6, 36, 36]

        x = attn @ v # [800, 6, 36, 64]
        x = x.transpose(1, 2).contiguous().reshape(B_, N, C//self.mp_size) # [800, 36, 384]

        x = generic_all_to_all(x, self.manager, split_dim=1, concat_dim=2) # [800, 18, 768]

        x = self.proj(x)


        return x


class WPUlyssesWindowAttention(torch.nn.Module):
    def __init__(self,
        manager,
        embedding_dim,
        window_size,
        num_heads,
        kaiming_init=True,
        qkv_bias=True,
        qk_scale=None,
        use_attn_mask=True,
        use_relative_position_bias=True,
        use_flash_attention=False,
        init_seed_base=None,
        ):
        super().__init__()
        self.manager = manager
        self.use_attn_mask = use_attn_mask
        self.use_relative_position_bias = use_relative_position_bias
        self.use_flash_attention = use_flash_attention
        self.dim = embedding_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.wp_size = manager.get_wp_group_size()
        self.sp_size = int(getattr(manager, "xfmr_sp_size", self.wp_size))
        self.tp_size = int(getattr(manager, "xfmr_tp_size", 1))
        self.sp_group = getattr(manager, "xfmr_sp_group", manager.window_parallel_group)
        self.sp_rank = int(getattr(manager, "xfmr_sp_rank", manager.get_wp_rank()))
        self.tp_group = getattr(manager, "xfmr_tp_group", None)
        self.tp_rank = int(getattr(manager, "xfmr_tp_rank", 0))
        assert self.dim % num_heads == 0, f"dim {self.dim} must be divisible by num_heads {num_heads}"
        assert self.dim % self.sp_size == 0, f"dim {self.dim} must be divisible by xfmr_sp_size {self.sp_size}"
        assert self.dim % self.tp_size == 0, f"dim {self.dim} must be divisible by tensor_parallel_size {self.tp_size}"
        assert num_heads % (self.sp_size * self.tp_size) == 0, f"num_heads {num_heads} must be divisible by xfmr_sp_size*tensor_parallel_size {self.sp_size * self.tp_size}"


        head_dim = self.dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if self.use_relative_position_bias:
            '''
            self.relative_position_bias_table = torch.nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads // self.wp_size)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            '''


            if kaiming_init:
                full_bias_table = torch.nn.Parameter(
                    torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
                )
                run_with_seed(init_seed_base, lambda: trunc_normal_(full_bias_table, std=.02))
                bias_chunks = torch.split(full_bias_table, num_heads // self.tp_size, dim=-1)
                bias_idx = self.tp_rank
                self.relative_position_bias_table = torch.nn.Parameter(
                    bias_chunks[bias_idx].contiguous()
                )
            else:
                self.relative_position_bias_table = torch.nn.Parameter(
                    torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads // self.tp_size)
                )
                trunc_normal_(self.relative_position_bias_table, std=.02)
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))


            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            self.relative_position_bias_table.terra_grad_reduce_group = "xfmr_tp_param_group"
            #trunc_normal_(self.relative_position_bias_table, std=.02)

        self.local_dim = self.dim // self.tp_size
        self.local_heads = self.num_heads // (self.sp_size * self.tp_size)
        self.weight_q, self.bias_q = init_like_linear(
            model_type='parallel',
            linear_in_dim=self.dim,
            linear_out_dim=self.dim,
            mp_rank=self.tp_rank,
            mp_group_size=self.tp_size,
            split_dim=0,
            split_bias=True,
            use_bias=qkv_bias,
            init_seed=None if init_seed_base is None else init_seed_base + 1,
        )
        self.weight_k, self.bias_k = init_like_linear(
            model_type='parallel',
            linear_in_dim=self.dim,
            linear_out_dim=self.dim,
            mp_rank=self.tp_rank,
            mp_group_size=self.tp_size,
            split_dim=0,
            split_bias=True,
            use_bias=qkv_bias,
            init_seed=None if init_seed_base is None else init_seed_base + 2,
        )
        self.weight_v, self.bias_v = init_like_linear(
            model_type='parallel',
            linear_in_dim=self.dim,
            linear_out_dim=self.dim,
            mp_rank=self.tp_rank,
            mp_group_size=self.tp_size,
            split_dim=0,
            split_bias=True,
            use_bias=qkv_bias,
            init_seed=None if init_seed_base is None else init_seed_base + 3,
        )
        self.q_linear = safe_linear_with_weight(weight=self.weight_q, bias=self.bias_q)
        self.k_linear = safe_linear_with_weight(weight=self.weight_k, bias=self.bias_k)
        self.v_linear = safe_linear_with_weight(weight=self.weight_v, bias=self.bias_v)
        for param in self.q_linear.parameters():
            param.terra_grad_reduce_group = "xfmr_tp_param_group"
        for param in self.k_linear.parameters():
            param.terra_grad_reduce_group = "xfmr_tp_param_group"
        for param in self.v_linear.parameters():
            param.terra_grad_reduce_group = "xfmr_tp_param_group"
        self.weight_q = None
        self.bias_q = None
        self.weight_k = None
        self.bias_k = None
        self.weight_v = None
        self.bias_v = None
        self.weight_proj, self.proj_bias = init_like_linear(
            model_type='parallel',
            linear_in_dim=self.dim,
            linear_out_dim=self.dim,
            mp_rank=self.tp_rank,
            mp_group_size=self.tp_size,
            split_dim=-1,
            split_bias=False,
            use_bias=True,
            init_seed=None if init_seed_base is None else init_seed_base + 4,
        )
        self.proj = safe_linear_with_weight(weight=self.weight_proj, bias=None)
        for param in self.proj.parameters():
            param.terra_grad_reduce_group = "xfmr_tp_param_group"
        self.weight_proj = None
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        input_is_4d = x.dim() == 4
        if not input_is_4d and self.tp_size > 1:
            raise RuntimeError(
                "WPUlysses TP requires 4D sharded activation: "
                "[B, local_windows, shard_tokens, C]"
            )
        if input_is_4d:
            B, local_num_windows, shard_N, C = x.shape
            x = _gather_window_batch_to_tensor_parallel_region(
                x,
                self.tp_group,
                self.tp_size,
                self.tp_rank,
            )
            full_num_windows = x.shape[1]
            B_ = B * full_num_windows
            x = x.contiguous().view(B_, shard_N, C)
        else:
            B_, shard_N, C = x.shape
            B = None
            full_num_windows = None
        full_N = shard_N * self.sp_size
        real_N = self.window_size[0] * self.window_size[1]


        #    exit(0)

        if input_is_4d:
            input_parallel = x
        else:
            input_parallel = _copy_to_tensor_parallel_region(x, self.tp_group, self.tp_size)
        q = self.q_linear(input_parallel)
        k = self.k_linear(input_parallel)
        v = self.v_linear(input_parallel)

        q = generic_all_to_all_group(
            q,
            self.sp_group,
            self.sp_size,
            split_dim=2,
            concat_dim=1,
        )
        k = generic_all_to_all_group(
            k,
            self.sp_group,
            self.sp_size,
            split_dim=2,
            concat_dim=1,
        )
        v = generic_all_to_all_group(
            v,
            self.sp_group,
            self.sp_size,
            split_dim=2,
            concat_dim=1,
        )

        q = q.reshape(B_, full_N, self.local_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B_, full_N, self.local_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B_, full_N, self.local_heads, C // self.num_heads).permute(0, 2, 1, 3)


        '''
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.use_relative_position_bias:
            relative_position_bias = x.new_zeros(full_N, full_N, self.num_heads // self.wp_size)
            real_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                real_N, real_N, -1
            )
            relative_position_bias[:real_N, :real_N] = real_bias
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        if full_N != real_N:
            valid_key = torch.arange(full_N, device=x.device) < real_N
            attn = attn.masked_fill(~valid_key.view(1, 1, 1, full_N), -10000.0)
        attn = self.softmax(attn)
        x = attn @ v
        x = x.transpose(1, 2).contiguous().reshape(B_, full_N, C // self.wp_size)

        x = generic_all_to_all_group(
            x,
            self.manager.window_parallel_group,
            self.wp_size,
            split_dim=1,
            concat_dim=2,
        )
        x = self.proj(x)
        return x
        '''

        if full_N == real_N and _can_use_flash_attention(q, mask, self.use_flash_attention, self.use_relative_position_bias):
            x = _flash_attention_from_bhnd(q, k, v, self.scale).reshape(B_, full_N, self.local_dim // self.sp_size)
            x = generic_all_to_all_group(
                x,
                self.sp_group,
                self.sp_size,
                split_dim=1,
                concat_dim=2,
            )
            x = self.proj(x)
            if input_is_4d:
                x = x.contiguous().view(B, full_num_windows, shard_N, C)
                x = _reduce_scatter_window_batch_from_tensor_parallel_region(
                    x,
                    self.tp_group,
                    self.tp_size,
                    self.tp_rank,
                )
            elif self.tp_size > 1:
                torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM, group=self.tp_group)
            x = x + self.proj_bias
            return x

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.use_relative_position_bias:
            relative_position_bias = x.new_zeros(full_N, full_N, self.local_heads)
            heads_per_sp = self.num_heads // (self.sp_size * self.tp_size)
            sp_head_start = self.sp_rank * heads_per_sp
            sp_head_end = sp_head_start + heads_per_sp
            local_bias_table = self.relative_position_bias_table[:, sp_head_start:sp_head_end]
            real_bias = local_bias_table[self.relative_position_index.view(-1)].view(
                real_N, real_N, -1
            )
            relative_position_bias[:real_N, :real_N] = real_bias
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            if input_is_4d and full_num_windows is not None and full_num_windows > nW:
                pad_windows = full_num_windows - nW
                pad_mask = mask.new_zeros(pad_windows, mask.shape[-2], mask.shape[-1])
                mask = torch.cat([mask, pad_mask], dim=0)
                nW = mask.shape[0]
            if B_ % nW != 0:
                raise ValueError(f"attention batch {B_} must be divisible by mask windows {nW}")
            mask = mask.to(device=attn.device, dtype=attn.dtype)
            if full_N != real_N:
                full_mask = mask.new_zeros(nW, full_N, full_N)
                full_mask[:, :real_N, :real_N] = mask
                mask = full_mask
            attn = attn.view(B_ // nW, nW, self.local_heads, full_N, full_N)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.local_heads, full_N, full_N)
        if full_N != real_N:
            valid_key = torch.arange(full_N, device=x.device) < real_N
            attn = attn.masked_fill(~valid_key.view(1, 1, 1, full_N), -10000.0)
        attn = self.softmax(attn)
        x = attn @ v
        x = x.transpose(1, 2).contiguous().reshape(B_, full_N, self.local_dim // self.sp_size)

        x = generic_all_to_all_group(
            x,
            self.sp_group,
            self.sp_size,
            split_dim=1,
            concat_dim=2,
        )
        x = self.proj(x)
        if input_is_4d:
            x = x.contiguous().view(B, full_num_windows, shard_N, C)
            x = _reduce_scatter_window_batch_from_tensor_parallel_region(
                x,
                self.tp_group,
                self.tp_size,
                self.tp_rank,
            )
        elif self.tp_size > 1:
            torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM, group=self.tp_group)
        x = x + self.proj_bias
        return x
