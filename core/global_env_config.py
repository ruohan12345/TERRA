import ast
import os


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return bool(default)
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean env {name}={value!r}")


def _env_int_list(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return list(default)
    value = value.strip()
    try:
        if value.startswith("["):
            parsed = ast.literal_eval(value)
        else:
            parsed = [x.strip() for x in value.split(",")]
        parsed = [int(x) for x in parsed]
    except Exception as exc:
        raise ValueError(f"Invalid integer-list env {name}={value!r}") from exc
    if len(parsed) != 4:
        raise ValueError(f"{name} must contain 4 integers [wait,warmup,active,repeat], got {parsed}")
    return parsed


def _detect_on_h200():
    override = os.environ.get("TERRA_ON_H200")
    if override is not None and override != "":
        return _env_bool("TERRA_ON_H200", False)
    return False


ON_H200 = _detect_on_h200()

ENABLE_TORCH_PROF = _env_bool("ENABLE_TORCH_PROF", False)
if _env_bool("DISABLE_TORCH_PROF", False):
    ENABLE_TORCH_PROF = False
TORCH_PROF_WRITE_TRACE = _env_bool("TORCH_PROF_WRITE_TRACE", False)
TORCH_PROF_step_list = _env_int_list("TORCH_PROF_STEP_LIST", [1, 1, 3, 1]) # [1, 1, 3, 1]
# wait_iters, warmup_iters, active_iters, repeat_iters


DEBUG_GRAD_HOOK = _env_bool("DEBUG_GRAD_HOOK", False)
DEBUG_GRAD_HOOK_PRINT_LIMIT = int(os.environ.get("DEBUG_GRAD_HOOK_PRINT_LIMIT", "64"))


# FSDP block-wrapper prefetch policy for optimizer_config=7 only.
# overlap: keep the current behavior; better overlap, usually higher reserved memory.
# conservative: reduce aggressive parameter prefetching; useful for memory diagnostics.
# none: disable FSDP block prefetch as much as possible; diagnostic slow path.


#    raise ValueError(
#        "FSDP_CONFIG7_PREFETCH_POLICY must be one of "
#        f"'overlap', 'conservative', or 'none', got {FSDP_CONFIG7_PREFETCH_POLICY!r}"
#    )


# FSDP wrapper granularity for optimizer_config=7/8/9.
# 1: current behavior, one Swin Transformer block per FSDP wrapper.
# N>1: group every N consecutive blocks into one FSDP wrapper.
# 0 is reserved for future attention/MLP-level wrapping and is not enabled yet.
FSDP_WRAPPER_CFG = int(os.environ.get("FSDP_WRAPPER_CFG", "1"))
if FSDP_WRAPPER_CFG < 1:
    raise ValueError("FSDP_WRAPPER_CFG currently supports cfg >= 1 only")


# Optional nested FSDP wrappers for reference-model sampling modules.
# 0: disabled, current behavior.
# 1: wrap heavy sampling operators only: patch embed/recovery and conv/tconv layers.
# 2: additionally wrap small affine sampling modules such as GroupNorm.


#    raise ValueError("FSDP_SAMPLING_WRAPPER_CFG must be one of 0, 1, or 2")


MEM_PROF = 0


USE_FAKE_INPUT = _env_bool("TERRA_USE_FAKE_INPUT", False)


# 'no', 'torch', 'ours', 'ours_reentrant_legacy'

# 'no',
# 'torch',
# 'ours',


USE_FLASH_ATTENTION = 'no use now'

use_Transformer = True
use_MLP = True
use_layernorm = True

use_shift_size_0 = False


use_attn_mask = 'no use now' #True
use_relative_position_bias = 'no use now' #True


emb_cfg = 4

if emb_cfg==0:


    EMB_1_input_split = '(1,n)'
    EMB_1_output_split = '(m,1)'
    EMB_1_SPLIT_PARAM = True


    EMB_2_input_split = '(m,1)'
    EMB_2_output_split = EMB_1_input_split
    EMB_2_SPLIT_PARAM = True
    # patch_embed.weight grad sum: -6.03016853332519531250
    # patch_embed.bias grad sum: -0.00299162417650222778
elif emb_cfg==1:

    EMB_1_input_split = '(m,1)'
    EMB_1_output_split = '(m,1)'
    EMB_1_SPLIT_PARAM = True

    EMB_2_input_split = '(m,1)'
    EMB_2_output_split = EMB_1_input_split
    EMB_2_SPLIT_PARAM = True
    #patch_embed.weight grad sum: -6.03256797790527343750
    #patch_embed.bias grad sum: -0.00299304723739624023
elif emb_cfg==2:
    EMB_1_input_split = '(m,1)'
    EMB_1_output_split = '(m,1)'
    EMB_1_SPLIT_PARAM = False

    EMB_2_input_split = '(m,1)'
    EMB_2_output_split = EMB_1_input_split
    EMB_2_SPLIT_PARAM = False
    #patch_embed.linear.weight grad sum: -6.03154373168945312500
    #patch_embed.linear.bias grad sum: -0.00299250334501266479
elif emb_cfg==3:
    EMB_1_input_split = '(1,n)'
    EMB_1_output_split = '(1,n)'
    EMB_1_SPLIT_PARAM = True

    EMB_2_input_split = '(1,n)'
    EMB_2_output_split = EMB_1_input_split
    EMB_2_SPLIT_PARAM = True

elif emb_cfg==4:
    EMB_1_input_split = '(m,1)'
    EMB_1_output_split = '(m,1)'
    EMB_1_SPLIT_PARAM = True

    EMB_2_input_split = '(m,1)'
    EMB_2_output_split = EMB_1_input_split
    EMB_2_SPLIT_PARAM = True
