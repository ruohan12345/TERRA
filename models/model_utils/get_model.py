#
import ast
import inspect

import torch
import torch.cuda.amp as amp


from models.reference_model.sequential_hierarchical_swin import SequentialHierarchicalSwin, SequentialSwinReference
from models.reference_model.parallel_hierarchical_swin import ParallelHierarchicalSwin, ParallelSwinReference
from models.attention import require_flash_attention_available


from optimizer.utils import get_optimizer
from core.checkpoint.activation import normalize_activation_config
from core import global_env_config

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from utils import check_parallel_config, get_padded_shape


def get_ranks_per_dp(data_parallel_group_size, world_size):
    return world_size//data_parallel_group_size


def _get_wp_topo_override(model_config, ranks_per_dp):
    wp_topo = model_config.get('wp_topo', None)
    if wp_topo is None:
        return None

    if isinstance(wp_topo, str):
        wp_topo = ast.literal_eval(wp_topo)

    wp_topo = tuple(wp_topo)
    if len(wp_topo) != 2:
        raise ValueError(f"wp_topo must have 2 dims, got {wp_topo}")
    if wp_topo[0] * wp_topo[1] != ranks_per_dp:
        raise ValueError(f"wp_topo product {wp_topo} must equal ranks_per_dp={ranks_per_dp}")
    return wp_topo


def _get_xfmr_wp_topo_override(model_config, wp_topo, xfmr_sp_size=1, tensor_parallel_size=1):


    xfmr_wp_topo = model_config.get('xfmr_wp_topo', None)
    if xfmr_wp_topo is None:
        return wp_topo

    if isinstance(xfmr_wp_topo, str):
        xfmr_wp_topo = ast.literal_eval(xfmr_wp_topo)

    xfmr_wp_topo = tuple(xfmr_wp_topo)
    if len(xfmr_wp_topo) != 2:
        raise ValueError(f"xfmr_wp_topo must have 2 dims, got {xfmr_wp_topo}")
    if xfmr_wp_topo[0] * xfmr_wp_topo[1] * int(xfmr_sp_size) * int(tensor_parallel_size) != wp_topo[0] * wp_topo[1]:
        raise ValueError(
            f"xfmr_wp_topo product {xfmr_wp_topo} times xfmr_sp_size={xfmr_sp_size} times tensor_parallel_size={tensor_parallel_size} must equal wp_topo product {wp_topo}"
        )
    return xfmr_wp_topo


def _ceil_div(a, b):
    return (a + b - 1) // b


def _get_fsdp_prefetch_policy_from_config(model_config):
    policy = model_config.get(
        'FSDP_CONFIG7_PREFETCH_POLICY',
        model_config.get('fsdp_prefetch_policy', 'none'),
    )
    policy = str(policy or 'none').strip().lower()
    if policy not in ("overlap", "conservative", "none"):
        raise ValueError(
            "FSDP_CONFIG7_PREFETCH_POLICY/fsdp_prefetch_policy must be one of "
            f"'overlap', 'conservative', or 'none', got {policy!r}"
        )
    return policy


def _get_fsdp_sampling_wrapper_cfg_from_config(model_config):
    cfg = int(model_config.get(
        'FSDP_SAMPLING_WRAPPER_CFG',
        model_config.get('fsdp_sampling_wrapper_cfg', 0),
    ))
    if cfg not in (0, 1, 2):
        raise ValueError(
            "FSDP_SAMPLING_WRAPPER_CFG/fsdp_sampling_wrapper_cfg must be one of 0, 1, or 2"
        )
    return cfg


def _get_fsdp_dtype(precision, half_model):
    if half_model:
        if precision == 'fp16':
            return torch.float16
        if precision == 'bf16':
            return torch.bfloat16
        if precision == 'fp32':
            print('fp32 should not set half_model to true')
            exit(0)
    return torch.float32


def _make_fsdp_mixed_precision(precision, half_model):
    fsdp_dtype = _get_fsdp_dtype(precision, half_model)
    return MixedPrecision(
        # Do not let FSDP cast parameters here; this keeps FSDP aligned
        # with the DDP precision path while still controlling comm/buffer dtype.
        param_dtype=None,
        reduce_dtype=fsdp_dtype,
        buffer_dtype=fsdp_dtype,
    )


def _make_fsdp_kwargs(process_group, mp_policy, prefetch_policy="overlap"):
    prefetch_policy = (prefetch_policy or "overlap").lower()
    if prefetch_policy not in ("overlap", "conservative", "none"):
        raise ValueError(f"Unsupported FSDP prefetch policy: {prefetch_policy}")

    kwargs = {
        "process_group": process_group,
        "mixed_precision": mp_policy,
    }
    fsdp_params = inspect.signature(FSDP.__init__).parameters

    if prefetch_policy == "overlap" and "backward_prefetch" in fsdp_params:
        kwargs["backward_prefetch"] = BackwardPrefetch.BACKWARD_PRE
    elif prefetch_policy == "conservative" and "backward_prefetch" in fsdp_params:
        kwargs["backward_prefetch"] = BackwardPrefetch.BACKWARD_POST
    elif prefetch_policy == "none" and "backward_prefetch" in fsdp_params:
        kwargs["backward_prefetch"] = None

    if "forward_prefetch" in fsdp_params:
        kwargs["forward_prefetch"] = False

    if "limit_all_gathers" in fsdp_params:
        kwargs["limit_all_gathers"] = True
    return kwargs


def _make_block_group(block_group):
    if len(block_group) == 1:
        return block_group[0]
    return torch.nn.Sequential(*block_group)


def _wrap_transformer_blocks_with_fsdp(model, process_group, mp_policy, prefetch_policy="overlap", wrapper_cfg=1):
    """Wrap transformer blocks before the root FSDP wrapper."""
    fsdp_kwargs = _make_fsdp_kwargs(process_group, mp_policy, prefetch_policy=prefetch_policy)
    target_layer_names = {"BasicLayer", "WindowParallelBasicLayer"}
    wrapped = 0
    original_blocks = 0

    for _, module in list(model.named_modules()):
        if module.__class__.__name__ not in target_layer_names:
            continue
        blocks = getattr(module, "blocks", None)
        if not isinstance(blocks, torch.nn.ModuleList):
            continue
        block_list = list(blocks)
        new_blocks = []
        for start in range(0, len(block_list), wrapper_cfg):
            group = block_list[start:start + wrapper_cfg]
            if len(group) == 1 and isinstance(group[0], FSDP):
                new_blocks.append(group[0])
                continue
            new_blocks.append(FSDP(_make_block_group(group), **fsdp_kwargs))
            wrapped += 1
            original_blocks += len(group)
        module.blocks = torch.nn.ModuleList(new_blocks)

    return wrapped, original_blocks


def _module_has_trainable_params(module):
    return any(p.requires_grad for p in module.parameters(recurse=True))


def _is_heavy_sampling_module(module):
    name = module.__class__.__name__
    return (
        "Conv" in name
        or isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))
        or name == "WrappedLinear"
    )


def _fsdp_wrap_child(parent, child_name, fsdp_kwargs):
    child = getattr(parent, child_name, None)
    if child is None or isinstance(child, FSDP):
        return 0
    if not _module_has_trainable_params(child):
        return 0
    setattr(parent, child_name, FSDP(child, **fsdp_kwargs))
    return 1


def _fsdp_wrap_sequential_children(sequence, fsdp_kwargs, include_small_affine=False):
    if not isinstance(sequence, torch.nn.Sequential):
        return 0
    wrapped = 0
    for idx, child in enumerate(sequence):
        if isinstance(child, FSDP):
            continue
        if not _module_has_trainable_params(child):
            continue
        if not include_small_affine and not _is_heavy_sampling_module(child):
            continue
        sequence[idx] = FSDP(child, **fsdp_kwargs)
        wrapped += 1
    return wrapped


def _wrap_reference_sampling_with_fsdp(model, process_group, mp_policy, prefetch_policy="overlap", sampling_cfg=0):
    """Wrap reference-model patch/sampling modules before the root FSDP wrapper.

    The sampling path is full-replica with respect to TP, so it must use the
    same process group as the root wrapper instead of the fixed-TP transformer
    block group used by optimizer_config=8/9.
    """
    if int(sampling_cfg) <= 0:
        return 0
    if not (hasattr(model, "down_blk") and hasattr(model, "up_blk")):
        return 0

    fsdp_kwargs = _make_fsdp_kwargs(process_group, mp_policy, prefetch_policy=prefetch_policy)
    include_small_affine = int(sampling_cfg) >= 2
    wrapped = 0

    wrapped += _fsdp_wrap_child(model, "patch_embed", fsdp_kwargs)
    wrapped += _fsdp_wrap_child(model, "patch_recovery", fsdp_kwargs)

    down_blk = getattr(model, "down_blk", None)
    if down_blk is not None:
        wrapped += _fsdp_wrap_child(down_blk, "conv", fsdp_kwargs)
        wrapped += _fsdp_wrap_sequential_children(
            getattr(down_blk, "b", None),
            fsdp_kwargs,
            include_small_affine=include_small_affine,
        )

    up_blk = getattr(model, "up_blk", None)
    if up_blk is not None:
        wrapped += _fsdp_wrap_child(up_blk, "conv", fsdp_kwargs)
        wrapped += _fsdp_wrap_sequential_children(
            getattr(up_blk, "b1", None),
            fsdp_kwargs,
            include_small_affine=include_small_affine,
        )
        wrapped += _fsdp_wrap_sequential_children(
            getattr(up_blk, "b2", None),
            fsdp_kwargs,
            include_small_affine=include_small_affine,
        )

    return wrapped


def _is_xfmr_tp_sharded_param(param):
    return getattr(param, "terra_grad_reduce_group", None) == "xfmr_tp_param_group"


def _collect_non_tp_trainable_params(module):
    return [
        p for p in module.parameters()
        if p.requires_grad and not _is_xfmr_tp_sharded_param(p)
    ]


def _collect_layernorm_modules(module):
    return [
        child for child in module.modules()
        if isinstance(child, torch.nn.LayerNorm)
    ]


def _wrap_transformer_blocks_with_tp_aware_fsdp(model, process_group, mp_policy, prefetch_policy="overlap", wrapper_cfg=1):
    """Wrap TP-sharded transformer blocks while leaving full-replica params to root FSDP.

    Parameters tagged with terra_grad_reduce_group="xfmr_tp_param_group" are
    replicas only across DP/WG/SP ranks at a fixed TP rank, so they use the
    fixed-TP FSDP group.  LayerNorm and full output biases are replicated across
    TP ranks, so they are ignored here and handled by the root FSDP wrapper.
    """
    base_fsdp_kwargs = _make_fsdp_kwargs(process_group, mp_policy, prefetch_policy=prefetch_policy)
    fsdp_params = inspect.signature(FSDP.__init__).parameters
    supports_ignored_states = "ignored_states" in fsdp_params
    supports_ignored_modules = "ignored_modules" in fsdp_params
    target_layer_names = {"BasicLayer", "WindowParallelBasicLayer"}
    wrapped = 0
    original_blocks = 0
    ignored_param_count = 0

    for _, module in list(model.named_modules()):
        if module.__class__.__name__ not in target_layer_names:
            continue
        blocks = getattr(module, "blocks", None)
        if not isinstance(blocks, torch.nn.ModuleList):
            continue
        block_list = list(blocks)
        new_blocks = []
        for start in range(0, len(block_list), wrapper_cfg):
            group = block_list[start:start + wrapper_cfg]
            if len(group) == 1 and isinstance(group[0], FSDP):
                new_blocks.append(group[0])
                continue

            block_group = _make_block_group(group)
            ignored_params = _collect_non_tp_trainable_params(block_group)
            fsdp_kwargs = dict(base_fsdp_kwargs)
            if ignored_params:
                if supports_ignored_states:
                    fsdp_kwargs["ignored_states"] = ignored_params
                elif supports_ignored_modules:
                    ignored_modules = _collect_layernorm_modules(block)
                    covered = {
                        id(p)
                        for ignored_module in ignored_modules
                        for p in ignored_module.parameters()
                    }
                    uncovered = [p for p in ignored_params if id(p) not in covered]
                    if uncovered:
                        raise RuntimeError(
                            "optimizer_config=9 requires FSDP ignored_states support "
                            "because transformer TP blocks contain standalone non-TP "
                            "parameters such as output biases."
                        )
                    fsdp_kwargs["ignored_modules"] = ignored_modules
                else:
                    raise RuntimeError(
                        "optimizer_config=9 requires FSDP ignored_states or "
                        "ignored_modules support."
                    )

            new_blocks.append(FSDP(block_group, **fsdp_kwargs))
            wrapped += 1
            original_blocks += len(group)
            ignored_param_count += len(ignored_params)
        module.blocks = torch.nn.ModuleList(new_blocks)

    return wrapped, ignored_param_count, original_blocks


def _get_padded_token_resolution(height, width, patch_size, window_size, padding_scale):
    align = patch_size * window_size * padding_scale
    padded_h = _ceil_div(height, align) * align
    padded_w = _ceil_div(width, align) * align
    return padded_h // patch_size, padded_w // patch_size


def _as_tuple2(value, name):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    value = tuple(value)
    if len(value) != 2:
        raise ValueError(f"{name} must have 2 dims, got {value}")
    return int(value[0]), int(value[1])


def _get_transformer_downsample_scale(model_architecture):
    if model_architecture == "credit_hierarchical_swin":
        return 2
    return 1


def _build_padding_spec(
    policy,
    height,
    width,
    patch_size,
    window_size,
    model_architecture,
    padded_shape,
):
    padded_shape = _as_tuple2(padded_shape, "padded_shape")
    need_padding, initial_padding, padded_shape = get_padded_shape(
        height,
        width,
        patch_size,
        window_size,
        padded_shape=padded_shape,
    )
    downsample_scale = _get_transformer_downsample_scale(model_architecture)
    if padded_shape[0] % patch_size != 0 or padded_shape[1] % patch_size != 0:
        raise ValueError(f"padded_shape={padded_shape} must be divisible by patch_size={patch_size}")
    patch_token_h = padded_shape[0] // patch_size
    patch_token_w = padded_shape[1] // patch_size
    if patch_token_h % downsample_scale != 0 or patch_token_w % downsample_scale != 0:
        raise ValueError(
            f"patch token resolution {(patch_token_h, patch_token_w)} must be divisible by "
            f"transformer_downsample_scale={downsample_scale}"
        )
    transformer_token_h = patch_token_h // downsample_scale
    transformer_token_w = patch_token_w // downsample_scale
    num_windows = None
    if transformer_token_h % window_size == 0 and transformer_token_w % window_size == 0:
        num_windows = (
            transformer_token_h // window_size,
            transformer_token_w // window_size,
        )
    return {
        "policy": policy,
        "padded_shape": padded_shape,
        "initial_padding": tuple(initial_padding),
        "need_padding": bool(need_padding),
        "patch_token_resolution": (patch_token_h, patch_token_w),
        "transformer_token_resolution": (transformer_token_h, transformer_token_w),
        "num_windows": num_windows,
        "transformer_downsample_scale": downsample_scale,
    }


def _ceil_to_multiple(value, multiple):
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return _ceil_div(value, multiple) * multiple


def _lcm(a, b):
    import math
    return abs(a * b) // math.gcd(a, b)


def _window_multiple_for_token_divisibility(window_size, parallel_dim):

    import math
    return parallel_dim // math.gcd(window_size, parallel_dim)


TERRA_M1_ASSIGNMENT_MODES = {
    "terra_m1_regular",
    "terra_m1_ragged_row",
    "terra_m1_ragged_row_major",
    "terra_m1_ragged_auto",
}

WINDOW_TOPOLOGY_ASSIGNMENT_MODES = {
    "mn": ("regular", "ragged_round_robin"),
    "m1": tuple(sorted(TERRA_M1_ASSIGNMENT_MODES)),
}


def _is_terra_m1_assignment_mode(mode):
    return mode in TERRA_M1_ASSIGNMENT_MODES


def _validate_window_topology_assignment_pair(
    window_topology,
    window_assignment_mode,
    xfmr_wp_topo,
):
    if window_topology not in WINDOW_TOPOLOGY_ASSIGNMENT_MODES:
        supported = ", ".join(WINDOW_TOPOLOGY_ASSIGNMENT_MODES)
        raise ValueError(
            f"Unsupported window_topology={window_topology!r}; supported values are: {supported}. "
            f"Got window_assignment_mode={window_assignment_mode!r}, xfmr_wp_topo={xfmr_wp_topo}."
        )

    supported_modes = {
        mode
        for modes in WINDOW_TOPOLOGY_ASSIGNMENT_MODES.values()
        for mode in modes
    }
    if window_assignment_mode not in supported_modes:
        supported = ", ".join(sorted(supported_modes))
        raise ValueError(
            f"Unsupported window_assignment_mode={window_assignment_mode!r}; supported values are: {supported}. "
            f"Got window_topology={window_topology!r}, xfmr_wp_topo={xfmr_wp_topo}."
        )

    allowed_modes = WINDOW_TOPOLOGY_ASSIGNMENT_MODES[window_topology]
    if window_assignment_mode not in allowed_modes:
        allowed = ", ".join(allowed_modes)
        raise ValueError(
            "Incompatible window topology configuration: "
            f"window_topology={window_topology!r} cannot be used with "
            f"window_assignment_mode={window_assignment_mode!r}. "
            f"Allowed modes for {window_topology!r} are: {allowed}. "
            f"Got xfmr_wp_topo={xfmr_wp_topo}."
        )

    if window_topology == "m1" and xfmr_wp_topo[1] != 1:
        raise ValueError(
            f"window_topology='m1' requires xfmr_wp_topo=(m, 1), got "
            f"xfmr_wp_topo={xfmr_wp_topo} with "
            f"window_assignment_mode={window_assignment_mode!r}."
        )


def _bump_window_count_for_total(num_windows_h, num_windows_w, data_h_multiple, data_w_multiple, required_total):
    while num_windows_h * num_windows_w < required_total:
        next_h = _ceil_to_multiple(num_windows_h + 1, data_h_multiple)
        next_w = _ceil_to_multiple(num_windows_w + 1, data_w_multiple)
        area_if_h = next_h * num_windows_w
        area_if_w = num_windows_h * next_w
        if area_if_h <= area_if_w:
            num_windows_h = next_h
        else:
            num_windows_w = next_w
    return num_windows_h, num_windows_w


def _compute_auto_padded_shape(
    height,
    width,
    patch_size,
    window_size,
    model_architecture,
    wp_topo,
    xfmr_wp_topo,
    window_assignment_mode,
):
    downsample_scale = _get_transformer_downsample_scale(model_architecture)
    padded_unit = patch_size * downsample_scale * window_size
    num_windows_h = _ceil_div(height, padded_unit)
    num_windows_w = _ceil_div(width, padded_unit)

    data_h_multiple = _window_multiple_for_token_divisibility(window_size, wp_topo[0])
    data_w_multiple = _window_multiple_for_token_divisibility(window_size, wp_topo[1])

    if _is_terra_m1_assignment_mode(window_assignment_mode):
        if xfmr_wp_topo[1] != 1:
            raise ValueError(
                f"{window_assignment_mode} requires xfmr_wp_topo=(m, 1), got xfmr_wp_topo={xfmr_wp_topo}"
            )
        num_windows_h = _ceil_to_multiple(num_windows_h, data_h_multiple)
        num_windows_w = _ceil_to_multiple(num_windows_w, data_w_multiple)
        m = xfmr_wp_topo[0]
        if window_assignment_mode == "terra_m1_regular":
            num_windows_h = _ceil_to_multiple(num_windows_h, _lcm(data_h_multiple, m))
        elif window_assignment_mode == "terra_m1_ragged_row":
            num_windows_h = max(num_windows_h, _ceil_to_multiple(m, data_h_multiple))
        elif window_assignment_mode == "terra_m1_ragged_row_major":
            num_windows_h, num_windows_w = _bump_window_count_for_total(
                num_windows_h,
                num_windows_w,
                data_h_multiple,
                data_w_multiple,
                m,
            )
        elif window_assignment_mode == "terra_m1_ragged_auto":
            if num_windows_h < m:
                num_windows_h, num_windows_w = _bump_window_count_for_total(
                    num_windows_h,
                    num_windows_w,
                    data_h_multiple,
                    data_w_multiple,
                    m,
                )
        else:
            raise ValueError(f"Unsupported window_assignment_mode={window_assignment_mode}")
    elif window_assignment_mode == "regular":
        num_windows_h_multiple = _lcm(data_h_multiple, xfmr_wp_topo[0])
        num_windows_w_multiple = _lcm(data_w_multiple, xfmr_wp_topo[1])
        num_windows_h = _ceil_to_multiple(num_windows_h, num_windows_h_multiple)
        num_windows_w = _ceil_to_multiple(num_windows_w, num_windows_w_multiple)
    elif window_assignment_mode == "ragged_round_robin":
        if wp_topo == xfmr_wp_topo and wp_topo[1] > 1:
            raise ValueError(
                "auto padding for ragged_round_robin requires data wp_topo=(m, 1) when transformer topology has n > 1, "
                f"got wp_topo={wp_topo}, xfmr_wp_topo={xfmr_wp_topo}"
            )
        num_windows_h = _ceil_to_multiple(num_windows_h, data_h_multiple)
        num_windows_w = _ceil_to_multiple(num_windows_w, data_w_multiple)
        if xfmr_wp_topo[1] == 1:
            num_windows_h = max(num_windows_h, xfmr_wp_topo[0])
        else:
            while num_windows_h * num_windows_w < xfmr_wp_topo[0] * xfmr_wp_topo[1]:
                next_h = _ceil_to_multiple(num_windows_h + 1, data_h_multiple)
                next_w = _ceil_to_multiple(num_windows_w + 1, data_w_multiple)
                padded_h_if_h = next_h * padded_unit
                padded_w_if_w = next_w * padded_unit
                area_if_h = padded_h_if_h * (num_windows_w * padded_unit)
                area_if_w = (num_windows_h * padded_unit) * padded_w_if_w
                if area_if_h <= area_if_w:
                    num_windows_h = next_h
                else:
                    num_windows_w = next_w
    else:
        raise ValueError(f"Unsupported window_assignment_mode={window_assignment_mode}")

    return num_windows_h * padded_unit, num_windows_w * padded_unit


def _resolve_padding_spec(
    model_config,
    model_architecture,
    height,
    width,
    patch_size,
    window_size,
    wp_topo,
    xfmr_wp_topo,
    window_assignment_mode,
):
    policy = model_config.get("padding_policy", "scale")
    explicit_padded_shape = model_config.get("padded_shape", model_config.get("resolved_padded_shape", None))

    if explicit_padded_shape is not None:
        padded_shape = _as_tuple2(explicit_padded_shape, "padded_shape")
        policy = "explicit"
    elif policy == "auto":
        padded_shape = _compute_auto_padded_shape(
            height,
            width,
            patch_size,
            window_size,
            model_architecture,
            wp_topo,
            xfmr_wp_topo,
            window_assignment_mode,
        )
    elif policy in ("scale", "legacy", "padding_scale"):
        padding_scale = model_config.get("padding_scale", 1)
        _, _, padded_shape = get_padded_shape(
            height,
            width,
            patch_size,
            window_size,
            padding_scale=padding_scale,
        )
        policy = "scale"
    else:
        raise ValueError(f"Unsupported padding_policy={policy}")

    return _build_padding_spec(
        policy,
        height,
        width,
        patch_size,
        window_size,
        model_architecture,
        padded_shape,
    )


def _validate_parallel_layout_config(
    task_type,
    model_architecture,
    embedding_parallel_type,
    wp_topo,
    xfmr_wp_topo,
    height,
    width,
    patch_size,
    window_size,
    padding_scale,
    embedding_dim,
    num_heads,
    window_assignment_mode="regular",
    window_topology=None,
    xfmr_sp_size=1,
    tensor_parallel_size=1,
    padded_shape=None,
):

    if window_topology is None:
        window_topology = "m1" if _is_terra_m1_assignment_mode(window_assignment_mode) else "mn"
    _validate_window_topology_assignment_pair(
        window_topology=window_topology,
        window_assignment_mode=window_assignment_mode,
        xfmr_wp_topo=xfmr_wp_topo,
    )

    wp_size = wp_topo[0] * wp_topo[1]
    xfmr_wp_size = xfmr_wp_topo[0] * xfmr_wp_topo[1]
    xfmr_sp_size = int(xfmr_sp_size)
    tensor_parallel_size = int(tensor_parallel_size)
    if wp_size != xfmr_wp_size * xfmr_sp_size * tensor_parallel_size:
        raise ValueError(
            f"wp_topo={wp_topo} product must equal xfmr_wp_topo={xfmr_wp_topo} product times xfmr_sp_size={xfmr_sp_size} times tensor_parallel_size={tensor_parallel_size}"
        )

    sampling_aware_archs = {
        "credit_hierarchical_swin",
    }
    if model_architecture in sampling_aware_archs:


        if embedding_parallel_type != "window_linear":
            raise ValueError(
                f"{model_architecture} uses halo-based sampling blocks, so embedding_parallel_type must be window_linear, "
                f"got {embedding_parallel_type}"
            )
        if wp_topo[1] != 1:
            raise ValueError(
                f"{model_architecture} uses halo-based sampling blocks, so data wp_topo must be (m, 1), got {wp_topo}"
            )

    if wp_topo != xfmr_wp_topo:

        if wp_topo[1] != 1:
            raise ValueError(f"layout transform only supports data wp_topo=(m, 1) now, got wp_topo={wp_topo}")
        if embedding_parallel_type != "window_linear":
            raise ValueError(
                f"layout transform from wp_topo={wp_topo} to xfmr_wp_topo={xfmr_wp_topo} requires window_linear input layout"
            )


    transformer_downsample_scale = _get_transformer_downsample_scale(model_architecture)

    if padded_shape is None:
        token_h, token_w = _get_padded_token_resolution(height, width, patch_size, window_size, padding_scale)
    else:
        padded_shape = _as_tuple2(padded_shape, "padded_shape")
        if padded_shape[0] % patch_size != 0 or padded_shape[1] % patch_size != 0:
            raise ValueError(f"padded_shape={padded_shape} must be divisible by patch_size={patch_size}")
        token_h = padded_shape[0] // patch_size
        token_w = padded_shape[1] // patch_size


    if token_h % transformer_downsample_scale != 0 or token_w % transformer_downsample_scale != 0:
        raise ValueError(
            f"patch token resolution {(token_h, token_w)} must be divisible by transformer_downsample_scale={transformer_downsample_scale}"
        )
    token_h = token_h // transformer_downsample_scale
    token_w = token_w // transformer_downsample_scale
    if token_h % wp_topo[0] != 0 or token_w % wp_topo[1] != 0:
        raise ValueError(
            f"transformer token resolution {(token_h, token_w)} must be divisible by data wp_topo={wp_topo}"
        )


    # (360 600)

    if token_h % window_size != 0 or token_w % window_size != 0:
        raise ValueError(
            f"padded token resolution {(token_h, token_w)} must be divisible by window_size={window_size}"
        )
    num_windows_h = token_h // window_size
    num_windows_w = token_w // window_size


    if num_windows_h * num_windows_w < xfmr_wp_size:
        raise ValueError(
            f"too few windows for xfmr_wp_topo={xfmr_wp_topo}: num_windows=({num_windows_h}, {num_windows_w})"
        )


    if _is_terra_m1_assignment_mode(window_assignment_mode):
        if wp_topo[1] != 1:
            raise ValueError(
                f"{window_assignment_mode} requires data wp_topo=(m, 1), got wp_topo={wp_topo}"
            )

    if window_assignment_mode == "ragged_round_robin" and wp_topo == xfmr_wp_topo and wp_topo[1] > 1:
        raise ValueError(
            "ragged_round_robin with data wp_topo == xfmr_wp_topo and n > 1 is not supported by the current "
            f"stripe-to-window relayout path. Use data wp_topo=(m, 1), got wp_topo={wp_topo}."
        )


    if window_assignment_mode == "regular":

        if num_windows_h % xfmr_wp_topo[0] != 0 or num_windows_w % xfmr_wp_topo[1] != 0:
            raise ValueError(
                f"regular xfmr_wp_topo={xfmr_wp_topo} requires num_windows=({num_windows_h}, {num_windows_w}) "
                "to be divisible by the transformer topology"
            )

        # ！！！！！！！！！！！！！！！！！！！！！！


    if window_assignment_mode == "terra_m1_regular" and num_windows_h % xfmr_wp_topo[0] != 0:
        raise ValueError(
            f"terra_m1_regular xfmr_wp_topo={xfmr_wp_topo} requires num_windows_h={num_windows_h} "
            "to be divisible by m"
        )
    if window_assignment_mode == "terra_m1_ragged_row" and num_windows_h < xfmr_wp_topo[0]:
        raise ValueError(
            f"terra_m1_ragged_row xfmr_wp_topo={xfmr_wp_topo} requires num_windows_h={num_windows_h} >= m"
        )

    if xfmr_sp_size > 1 or tensor_parallel_size > 1:
        # Factorized WPxSP only shards heads/channels inside each SP subgroup.
        if embedding_dim % (xfmr_sp_size * tensor_parallel_size) != 0:
            raise ValueError(f"Ulysses/TP attention requires embedding_dim={embedding_dim} divisible by xfmr_sp_size*tensor_parallel_size={xfmr_sp_size * tensor_parallel_size}")
        if num_heads % (xfmr_sp_size * tensor_parallel_size) != 0:
            raise ValueError(f"Ulysses/TP attention requires num_heads={num_heads} divisible by xfmr_sp_size*tensor_parallel_size={xfmr_sp_size * tensor_parallel_size}")


def get_model_archi_params_and_other_params(task_type, model_config, data_parallel_group_size=-1, world_size=-1):
    if task_type != 'glorys':
        raise ValueError(f"Unsupported task_type: {task_type}. TERRA supports GLORYS only.")


    model_architecture = model_config.get('model_architecture', None)
    patch_size = model_config['patch_size']
    embedding_dim = model_config['embedding_dim']
    num_layers = model_config['num_layers']
    num_heads = model_config['num_heads']
    num_channels = int(model_config.get('num_channels', 93))
    window_size = model_config['window_size']
    padding_scale = model_config.get('padding_scale', 1)
    if task_type=='glorys':
        height = 2041
        width = 4320
        model_archi_params = {
            'height': height,
            'width': width,
            'num_layers': num_layers,
            'embedding_dim': embedding_dim,
            'num_heads': num_heads,
            'num_channels': num_channels,
            'patch_size': patch_size,
            'window_size': window_size,
        }

        model_archi_params['padding_scale'] = padding_scale
        model_archi_params['activation'] = normalize_activation_config(model_config)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}. TERRA supports GLORYS only.")


    model_archi_params['model_architecture'] = model_architecture
    model_archi_params.setdefault('activation', normalize_activation_config(model_config))
    model_archi_params['use_attn_mask'] = bool(model_config.get('use_attn_mask', True))
    model_archi_params['use_relative_position_bias'] = bool(model_config.get('use_relative_position_bias', True))
    model_archi_params['USE_FLASH_ATTENTION'] = bool(model_config.get('USE_FLASH_ATTENTION', model_config.get('use_flash_attention', False)))
    if model_archi_params['USE_FLASH_ATTENTION']:
        require_flash_attention_available()
    model_archi_params['FSDP_CONFIG7_PREFETCH_POLICY'] = _get_fsdp_prefetch_policy_from_config(model_config)
    model_archi_params['FSDP_SAMPLING_WRAPPER_CFG'] = _get_fsdp_sampling_wrapper_cfg_from_config(model_config)

    other_params = {}
    model_type = model_config['model_type']

    if model_type == 'sequential':
        other_params['mp_size'] = 1
        other_params['wp_topo'] = (1, 1)
        other_params['xfmr_wp_topo'] = (1, 1)
        other_params['xfmr_sp_size'] = 1
        other_params['tensor_parallel_size'] = 1
        other_params['domain_topo'] = (1, 1)
        other_params['sp_tp_placement'] = 'tp_first'

        other_params['embedding_parallel_type'] = None
        other_params['attn_parallel_type'] = None
        other_params['mlp_parallel_type'] = None
        other_params['window_assignment_mode'] = 'regular'
        other_params['window_topology'] = 'mn'

    elif model_type == 'parallel':
        ranks_per_dp = get_ranks_per_dp(data_parallel_group_size, world_size)

        parallel_cfg = 'wp'

        xfmr_wp_topo = None
        xfmr_sp_size = int(model_config.get('xfmr_sp_size', 1))
        tensor_parallel_size = int(model_config.get('tensor_parallel_size', 1))
        window_assignment_mode = model_config.get('window_assignment_mode', 'regular')
        window_topology = model_config.get(
            'window_topology',
            'm1' if _is_terra_m1_assignment_mode(window_assignment_mode) else 'mn',
        )
        sp_tp_placement = model_config.get('sp_tp_placement', model_config.get('topology_placement', 'tp_first'))


        if parallel_cfg == 'wp':
            attn_parallel_type = 'wp'
            mlp_parallel_type = 'wp'

            embedding_parallel_type = model_config.get('embedding_parallel_type', 'window_linear')
            wp_topo_override = _get_wp_topo_override(model_config, ranks_per_dp)

            if ranks_per_dp <= 0:
                raise ValueError(f"ranks_per_dp must be positive for window parallel, got {ranks_per_dp}")
            mp_size = 1
            wp_topo = wp_topo_override if wp_topo_override is not None else (ranks_per_dp, 1)
            domain_topo = (1, 1)


            xfmr_wp_topo = _get_xfmr_wp_topo_override(model_config, wp_topo, xfmr_sp_size=xfmr_sp_size, tensor_parallel_size=tensor_parallel_size)

        else:
            print('unrecognized parallel_cfg', parallel_cfg)
            exit(0)

        assert attn_parallel_type in [
            'wp',
        ]

        assert mlp_parallel_type in [
            'wp',
        ]
        assert embedding_parallel_type in [
            'window_embedding',
            'domain_parallel',
            'window_domain',
            'window_linear',
        ]

        wp_group_h, wp_group_w = wp_topo


        padding_spec = _resolve_padding_spec(
            model_config=model_config,
            model_architecture=model_config['model_architecture'],
            height=height,
            width=width,
            patch_size=patch_size,
            window_size=window_size,
            wp_topo=wp_topo,
            xfmr_wp_topo=xfmr_wp_topo,
            window_assignment_mode=window_assignment_mode,
        )
        model_archi_params['padding_spec'] = padding_spec
        model_archi_params['padded_shape'] = padding_spec['padded_shape']
        model_archi_params['initial_padding'] = padding_spec['initial_padding']
        model_archi_params['padding_policy'] = padding_spec['policy']

        _validate_parallel_layout_config(
            task_type=task_type,
            model_architecture=model_config['model_architecture'],
            embedding_parallel_type=embedding_parallel_type,
            wp_topo=wp_topo,
            xfmr_wp_topo=xfmr_wp_topo,
            height=height,
            width=width,
            patch_size=patch_size,
            window_size=window_size,
            padding_scale=padding_scale,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            window_assignment_mode=window_assignment_mode,
            window_topology=window_topology,
            xfmr_sp_size=xfmr_sp_size,
            tensor_parallel_size=tensor_parallel_size,
            padded_shape=padding_spec['padded_shape'],
        )


        other_params['mp_size'] = mp_size
        other_params['wp_topo'] = wp_topo
        other_params['xfmr_wp_topo'] = xfmr_wp_topo
        other_params['xfmr_sp_size'] = xfmr_sp_size
        other_params['tensor_parallel_size'] = tensor_parallel_size
        other_params['sp_tp_placement'] = sp_tp_placement

        other_params['domain_topo'] = domain_topo

        other_params['embedding_parallel_type'] = embedding_parallel_type
        other_params['attn_parallel_type'] = attn_parallel_type
        other_params['mlp_parallel_type'] = mlp_parallel_type
        other_params['window_assignment_mode'] = window_assignment_mode
        other_params['window_topology'] = window_topology
    else:
        print('unrecognized model_type in get_model_archi_params_and_other_params', model_type)
        exit(0)

    if 'padding_spec' not in model_archi_params:
        padding_spec = _resolve_padding_spec(
            model_config=model_config,
            model_architecture=model_architecture,
            height=height,
            width=width,
            patch_size=patch_size,
            window_size=window_size,
            wp_topo=other_params['wp_topo'],
            xfmr_wp_topo=other_params['xfmr_wp_topo'],
            window_assignment_mode=other_params['window_assignment_mode'],
        )
        model_archi_params['padding_spec'] = padding_spec
        model_archi_params['padded_shape'] = padding_spec['padded_shape']
        model_archi_params['initial_padding'] = padding_spec['initial_padding']
        model_archi_params['padding_policy'] = padding_spec['policy']

    other_params['padding_spec'] = model_archi_params['padding_spec']
    other_params['use_splited_data'] = True
    other_params['norm_type'] = model_config.get('norm_type', 'zs')
    other_params['data_precision'] = model_config.get('data_precision', 'fp16')
    glorys_data_root = (
        model_config.get('glorys_data_root')
        or model_config.get('sequential_data_root')
        or model_config.get('data_root')
    )
    if glorys_data_root is not None:
        other_params['glorys_data_root'] = glorys_data_root


    return model_archi_params, other_params


def get_model(
        model_archi_params,
        task_type = 'glorys',

        model_architecture = None,

        kaiming_init = True,
        model_type = 'sequential',
        embedding_parallel_type = 'x_parallel',
        attn_parallel_type = None,
        mlp_parallel_type = None,

        manager = None,
        device = None,
        ):


    if task_type == 'glorys':
        if model_architecture=='credit_hierarchical_swin':
            if model_type == 'sequential':
                model = SequentialHierarchicalSwin(
                    padding_scale = model_archi_params['padding_scale'],
                    num_channels = model_archi_params['num_channels'],
                    patch_size = model_archi_params['patch_size'],
                    window_size = model_archi_params['window_size'],

                    height = model_archi_params['height'],
                    width = model_archi_params['width'],
                    embedding_dim = model_archi_params['embedding_dim'],
                    num_heads = model_archi_params['num_heads'],
                    manager = manager,
                    num_layers = model_archi_params['num_layers'],
                    recompute_config = model_archi_params.get('activation', None),
                    use_attn_mask = model_archi_params['use_attn_mask'],
                    use_relative_position_bias = model_archi_params['use_relative_position_bias'],
                    use_flash_attention = model_archi_params['USE_FLASH_ATTENTION'],
                    padding_spec = model_archi_params.get('padding_spec', None),
                )
            elif model_type == 'parallel':
                model = ParallelHierarchicalSwin(
                    height = model_archi_params['height'],
                    width = model_archi_params['width'],

                    patch_size = model_archi_params['patch_size'],
                    embedding_dim = model_archi_params['embedding_dim'], #768*2
                    num_layers = model_archi_params['num_layers'],

                    num_heads=model_archi_params['num_heads'],
                    window_size=model_archi_params['window_size'],

                    manager = manager,
                    device = device,

                    padding_scale = model_archi_params['padding_scale'],
                    num_channels = model_archi_params['num_channels'],

                    embedding_parallel_type = embedding_parallel_type,
                    recompute_config = model_archi_params.get('activation', None),
                    use_attn_mask = model_archi_params['use_attn_mask'],
                    use_relative_position_bias = model_archi_params['use_relative_position_bias'],
                    use_flash_attention = model_archi_params['USE_FLASH_ATTENTION'],
                    padding_spec = model_archi_params.get('padding_spec', None),
                )
        elif model_architecture=='swin_reference':
            if model_type == 'sequential':
                model = SequentialSwinReference(
                    padding_scale = model_archi_params['padding_scale'],
                    num_channels = model_archi_params['num_channels'],
                    patch_size = model_archi_params['patch_size'],
                    window_size = model_archi_params['window_size'],
                    height = model_archi_params['height'],
                    width = model_archi_params['width'],
                    embedding_dim = model_archi_params['embedding_dim'],
                    num_heads = model_archi_params['num_heads'],
                    manager = manager,
                    num_layers = model_archi_params['num_layers'],
                    recompute_config = model_archi_params.get('activation', None),
                    use_attn_mask = model_archi_params['use_attn_mask'],
                    use_relative_position_bias = model_archi_params['use_relative_position_bias'],
                    use_flash_attention = model_archi_params['USE_FLASH_ATTENTION'],
                    padding_spec = model_archi_params.get('padding_spec', None),
                )
            elif model_type == 'parallel':
                model = ParallelSwinReference(
                    height = model_archi_params['height'],
                    width = model_archi_params['width'],

                    patch_size = model_archi_params['patch_size'],
                    embedding_dim = model_archi_params['embedding_dim'], #768*2
                    num_layers = model_archi_params['num_layers'],

                    num_heads=model_archi_params['num_heads'],
                    window_size=model_archi_params['window_size'],

                    manager = manager,
                    device = device,

                    padding_scale = model_archi_params['padding_scale'],
                    num_channels = model_archi_params['num_channels'],

                    embedding_parallel_type = embedding_parallel_type,
                    recompute_config = model_archi_params.get('activation', None),
                    use_attn_mask = model_archi_params['use_attn_mask'],
                    use_relative_position_bias = model_archi_params['use_relative_position_bias'],
                    use_flash_attention = model_archi_params['USE_FLASH_ATTENTION'],
                    padding_spec = model_archi_params.get('padding_spec', None),
                )


        else:
            raise ValueError(f"Unsupported GLORYS model_architecture: {model_architecture}")
    else:
        raise ValueError(f"Unsupported task_type: {task_type}; only GLORYS reference workload is available")

    return model.to(device)


def get_model_for_train(
    model_archi_params,

    precision,
    half_model,
    optimizer_config,

    task_type = 'glorys',

    kaiming_init = True,
    model_type = 'sequential',

    model_architecture = None,

    embedding_parallel_type = 'x_parallel',
    attn_parallel_type = None,
    mlp_parallel_type = None,

    manager = None,

    device = None,
    local_rank = None,

    learning_rate = None,
):


    model = get_model( #
        model_archi_params,
        task_type = task_type,

        model_architecture = model_architecture,

        kaiming_init = kaiming_init,
        model_type = model_type,
        embedding_parallel_type = embedding_parallel_type,
        attn_parallel_type = attn_parallel_type,
        mlp_parallel_type = mlp_parallel_type,

        manager = manager,
        device = device,
    )

    pre_wrap_parameter_count = sum(p.numel() for p in model.parameters())
    model.pre_wrap_parameter_count = pre_wrap_parameter_count
    model.pre_wrap_parameter_count_method = "before_ddp_fsdp_optimizer_wrapper"


    USE_DIST_OPT = False

    tensor_parallel_size = int(getattr(manager, "xfmr_tp_size", 1)) if manager is not None else 1
    if optimizer_config in (5, 7) and tensor_parallel_size > 1:
        raise ValueError(
            f"optimizer_config={optimizer_config} (FSDP) is currently only allowed when tensor_parallel_size == 1. "
            f"Got tensor_parallel_size={tensor_parallel_size}. Use optimizer_config=8/9 for TP-aware FSDP "
            "or optimizer_config=6 for the custom distributed optimizer."
        )
    if optimizer_config in (8, 9):
        if tensor_parallel_size <= 1:
            raise ValueError(
                f"optimizer_config={optimizer_config} (TP-aware FSDP) requires tensor_parallel_size > 1. "
                f"Got tensor_parallel_size={tensor_parallel_size}. Use optimizer_config=7 for block-level FSDP."
            )
        if manager is None:
            raise ValueError(f"optimizer_config={optimizer_config} requires ParallelManager to provide TP-aware FSDP groups.")
        if getattr(manager, "full_param_replica_group", None) is None:
            raise ValueError(f"optimizer_config={optimizer_config} requires manager.full_param_replica_group.")
        if getattr(manager, "xfmr_tp_param_replica_group", None) is None:
            raise ValueError(f"optimizer_config={optimizer_config} requires manager.xfmr_tp_param_replica_group.")
    if optimizer_config == 6:
        if tensor_parallel_size <= 1:
            raise ValueError(
                "optimizer_config=6 (TERRA distributed optimizer) is reserved for tensor_parallel_size > 1. "
                f"Got tensor_parallel_size={tensor_parallel_size}. Use optimizer_config=5 for FSDP or optimizer_config=4 for DDP."
            )
        if model_type not in ("parallel", "hybrid"):
            raise ValueError(
                f"optimizer_config=6 requires a parallel/hybrid model with TP, got model_type={model_type}."
            )


    if optimizer_config==4: # DDP
        USE_DDP = True
        USE_FSDP = False
        ZERO_STAGE_NUMBER = None

        if precision=='fp16':
            gscaler = amp.GradScaler()
        else:
            gscaler = None


    elif optimizer_config in (5, 7, 8, 9): # FSDP
        USE_DDP = False
        USE_FSDP = True
        ZERO_STAGE_NUMBER = None


        if precision=='fp16':
            gscaler = ShardedGradScaler()
        else:
            gscaler = None


        mp_policy = _make_fsdp_mixed_precision(precision, half_model)
        fsdp_root_group = torch.distributed.group.WORLD
        fsdp_block_group = fsdp_root_group

        if optimizer_config in (8, 9):
            # Transformer blocks contain TP-sharded parameters, so different TP
            # ranks are not replicas of each other.  Non-TP parameters remain
            # replicated across TP ranks and are handled by the root wrapper.
            fsdp_root_group = manager.full_param_replica_group
            fsdp_block_group = manager.xfmr_tp_param_replica_group

        if optimizer_config in (7, 8, 9):
            fsdp_prefetch_policy = model_archi_params.get('FSDP_CONFIG7_PREFETCH_POLICY', 'none')
            fsdp_wrapper_cfg = global_env_config.FSDP_WRAPPER_CFG
            fsdp_sampling_wrapper_cfg = int(model_archi_params.get('FSDP_SAMPLING_WRAPPER_CFG', 0))
            if optimizer_config == 9:
                wrapped_blocks, ignored_params, original_blocks = _wrap_transformer_blocks_with_tp_aware_fsdp(
                    model,
                    fsdp_block_group,
                    mp_policy,
                    prefetch_policy=fsdp_prefetch_policy,
                    wrapper_cfg=fsdp_wrapper_cfg,
                )
            else:
                wrapped_blocks, original_blocks = _wrap_transformer_blocks_with_fsdp(
                    model,
                    fsdp_block_group,
                    mp_policy,
                    prefetch_policy=fsdp_prefetch_policy,
                    wrapper_cfg=fsdp_wrapper_cfg,
                )
                ignored_params = 0
            wrapped_sampling = _wrap_reference_sampling_with_fsdp(
                model,
                fsdp_root_group,
                mp_policy,
                prefetch_policy=fsdp_prefetch_policy,
                sampling_cfg=fsdp_sampling_wrapper_cfg,
            )
            if manager is None or manager.get_rank() == 0:
                if optimizer_config == 8:
                    print(
                        "[FSDP] optimizer_config=8 TP-aware wrapper, "
                        f"original_blocks={original_blocks}, wrapped_units={wrapped_blocks}, "
                        f"wrapper_cfg={fsdp_wrapper_cfg}, sampling_wrapper_cfg={fsdp_sampling_wrapper_cfg}, "
                        f"wrapped_sampling_units={wrapped_sampling}, prefetch_policy={fsdp_prefetch_policy}, "
                        "block_group=DPxWGxSP fixed TP, root_group=DPxDMP including TP"
                    )
                elif optimizer_config == 9:
                    print(
                        "[FSDP] optimizer_config=9 TP-aware block wrapper, "
                        f"original_blocks={original_blocks}, wrapped_units={wrapped_blocks}, "
                        f"wrapper_cfg={fsdp_wrapper_cfg}, sampling_wrapper_cfg={fsdp_sampling_wrapper_cfg}, "
                        f"wrapped_sampling_units={wrapped_sampling}, ignored_non_tp_params={ignored_params}, "
                        f"prefetch_policy={fsdp_prefetch_policy}, "
                        "TP-sharded block params use fixed-TP group, ignored params use full replica root group"
                    )
                else:
                    print(
                        "[FSDP] optimizer_config=7 block-level wrapper, "
                        f"original_blocks={original_blocks}, wrapped_units={wrapped_blocks}, "
                        f"wrapper_cfg={fsdp_wrapper_cfg}, sampling_wrapper_cfg={fsdp_sampling_wrapper_cfg}, "
                        f"wrapped_sampling_units={wrapped_sampling}, prefetch_policy={fsdp_prefetch_policy}"
                    )

        model = FSDP(
            model,

            process_group = fsdp_root_group,

            mixed_precision=mp_policy,


            )
        model.pre_wrap_parameter_count = pre_wrap_parameter_count
        model.pre_wrap_parameter_count_method = "before_ddp_fsdp_optimizer_wrapper"


    elif(optimizer_config==0 or optimizer_config==1 or optimizer_config==2 or optimizer_config==3):
        USE_DDP = False
        USE_FSDP = False
        gscaler = None
        ZERO_STAGE_NUMBER = optimizer_config
    elif optimizer_config==6:
        USE_DDP = False
        USE_FSDP = False
        USE_DIST_OPT = True
        ZERO_STAGE_NUMBER = None

        if precision=='fp16':
            gscaler = amp.GradScaler()
        else:
            gscaler = None
    else:
        print('invalid optimizer_config', optimizer_config)
        exit(0)


    if USE_DDP:
        if half_model:

            if precision=='fp16':
                print('DDP + fp16 + amp is invalid')

                exit(0)


            elif precision=='bf16':
                model = model.to(torch.bfloat16)


    elif ZERO_STAGE_NUMBER is not None:
        if half_model:
            print('zero do not need manually set parameter dtype')
            exit(0)


    if USE_DDP:


        my_ddp_group = manager.data_parallel_group


        model = torch.nn.parallel.DistributedDataParallel(model,
                            device_ids    = [local_rank],
                            output_device = local_rank,
                            process_group = my_ddp_group,
                            find_unused_parameters = False,


                            )
        model.pre_wrap_parameter_count = pre_wrap_parameter_count
        model.pre_wrap_parameter_count_method = "before_ddp_fsdp_optimizer_wrapper"


    model.train()

    engine, optimizer = get_optimizer(
        #model.parameters(),
        precision,
        model,
        learning_rate,
        manager = manager,
        model_type = model_type,

        zero_stage_number = ZERO_STAGE_NUMBER,
        use_distributed_optimizer = USE_DIST_OPT,


    )

    return model, engine, optimizer, gscaler, (USE_DDP, USE_FSDP, ZERO_STAGE_NUMBER, USE_DIST_OPT)


def load_model_ckpt(
                    task_type,
                    manager,
                    model,
                    ckpt_path,
                    device,
                    local_rank,
                    epoch,
                    ):
    #model.to(device)
    if task_type=='glorys':
        checkpoint = torch.load(ckpt_path+'/checkpoint/'+ "mp_rank"+str(0)+'_'+str(manager.get_mp_rank())  +'/epoch'+str(epoch)+'.pth',
                            map_location=device, weights_only=True)
        model=torch.nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank,
                                                        process_group = manager.data_parallel_group,
                                                        )
        model.load_state_dict({k: v for k, v in checkpoint['model_state_dict'].items() if k in model.state_dict()})
    else:
        raise ValueError(f"Unsupported task_type: {task_type}; only GLORYS reference workload is available")


    model.eval()
    return model
