import warnings
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release"
)

import os
import sys
import time
import argparse
import csv
import json
import itertools
import yaml

import torch
import torch.distributed as dist


from core.global_env_config import ENABLE_TORCH_PROF, TORCH_PROF_step_list, MEM_PROF, USE_FAKE_INPUT, TORCH_PROF_WRITE_TRACE
from core.logging.logging import set_dp_rank_print_redirect
from core.rank_manager import ParallelManager
from core.runtime.run_model import one_step_finetune


from dataloader.dataloader_utils import (
    get_finetune_dataloader_for_task,
    resolve_required_tensors_online_split_for_parallel,
    resolve_required_tensors_from_dataloader,
    resolve_glorys_finetune_double_buffer,
)
from dataloader.task_specific_data import get_task_specific_data

from optimizer.utils import get_finetune_lr_by_global_step, set_lr

from profiler.memory_timeline import (
    CudaMemoryTimeline,
    get_memory_timeline_peak_mib,
    mark_memory_timeline,
    set_memory_timeline_context,
)
from profiler.flops_profiler import TorchFlopsProfiler
from core.checkpoint.activation import activation_config_for_sampling
from core.checkpoint.boundary_offload import (
    get_boundary_offload_stats,
    reset_boundary_offload_stats,
)

from train_scripts.test_utils import process_loss_and_output_in_test, gather_scalar

from models.model_utils.get_model import get_model_for_train, get_model_archi_params_and_other_params, get_ranks_per_dp

from utils import init_distributed, set_random_seed


test_iter_num = 10 #20 #20
use_fake_input = USE_FAKE_INPUT
use_splited_data = False


show_gradient_flag = True


dataset_config = None


rank, local_rank, device, world_size = init_distributed()
set_random_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--data_parallel_group_size", type=int, required=True)
parser.add_argument("--model_cfg", required=True)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--metrics_csv", default=None)
parser.add_argument("--metrics_json", default=None)
parser.add_argument("--memory_timeline_csv", default=None)
parser.add_argument("--metrics_no_json", action="store_true")
parser.add_argument("--disable_torch_prof", action="store_true")
parser.add_argument("--quiet_metrics", action="store_true")
parser.add_argument("--disable_rank_log_redirect", action="store_true")
parser.add_argument("--mem_prof", type=int, default=None)
args = parser.parse_args()

if args.steps is not None:
    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}")
    test_iter_num = args.steps

if args.disable_torch_prof:
    ENABLE_TORCH_PROF = False
if args.mem_prof is not None:
    MEM_PROF = int(args.mem_prof)
elif os.environ.get("MEM_PROF", "") != "":
    MEM_PROF = int(os.environ["MEM_PROF"])
MEMORY_TIMELINE_OUTPUT_DIR = os.environ.get("MEMORY_TIMELINE_OUTPUT_DIR", "./log/memory_timeline")

torch.distributed.barrier()
reset_boundary_offload_stats()

with open(args.model_cfg, 'r') as f:
    model_config = yaml.safe_load(f)

model_config.setdefault('finetune_lead_time', 1)
model_config.setdefault('finetune_loss_reduction', 'mean')
if model_config['finetune_loss_reduction'] not in ('mean', 'sum'):
    raise ValueError(f"unsupported finetune_loss_reduction: {model_config['finetune_loss_reduction']}")
use_fake_input = bool(model_config.get('use_fake_input', use_fake_input))
use_splited_data = bool(model_config.get('use_splited_data', use_splited_data))
fake_input_random = bool(model_config.get('fake_input_random', True))
fake_input_seed = int(model_config.get('fake_input_seed', 1234))
fake_input_dmp_local = bool(model_config.get('fake_input_dmp_local', False))
release_cpu_batch_after_transfer = bool(
    model_config.get('release_cpu_batch_after_transfer', False)
)
dataloader_pin_memory = bool(model_config.get('dataloader_pin_memory', True))
input_transfer_non_blocking = bool(
    model_config.get('input_transfer_non_blocking', False)
)
include_input_transfer_in_step_time = bool(
    model_config.get('include_input_transfer_in_step_time', False)
)
input_double_buffer = bool(model_config.get('input_double_buffer', False))
if input_double_buffer and not input_transfer_non_blocking:
    raise ValueError("input_double_buffer requires input_transfer_non_blocking=true")
if input_double_buffer and not dataloader_pin_memory:
    raise ValueError("input_double_buffer requires dataloader_pin_memory=true")
if input_transfer_non_blocking and not dataloader_pin_memory and rank == 0:
    print(
        "warning: input_transfer_non_blocking=true without "
        "dataloader_pin_memory=true; H2D overlap is not guaranteed"
    )


task_type = model_config['task_type']
if task_type != 'glorys':
    raise ValueError("Unsupported task_type; only glorys is supported")
ranks_per_dp = get_ranks_per_dp(args.data_parallel_group_size, world_size)

model_archi_params, other_params = get_model_archi_params_and_other_params(task_type, model_config, data_parallel_group_size = args.data_parallel_group_size, world_size = world_size)
other_params['use_splited_data'] = use_splited_data

precision = model_config['precision']
assert precision in ['bf16', 'fp32', 'fp16']
if precision=='fp32':
    my_dtype = torch.float32
elif precision=='fp16':
    my_dtype = torch.float16
elif precision=='bf16':
    my_dtype = torch.bfloat16

model_type = model_config['model_type']
assert model_type in ['sequential', 'parallel', 'hybrid']

if fake_input_dmp_local:
    if not use_fake_input:
        raise ValueError('fake_input_dmp_local requires use_fake_input=true')
    if task_type != 'glorys' or model_type not in ('parallel', 'hybrid'):
        raise ValueError(
            'fake_input_dmp_local only supports parallel/hybrid GLORYS finetune'
        )

loss_fn = torch.nn.L1Loss()

def _get_debug_log_wp_ranks(manager, model_type, other_params):
    if model_type not in ('parallel', 'hybrid'):
        return [0]
    tp_size = int(other_params.get('tensor_parallel_size', 1))
    if tp_size <= 1:
        return [0]
    if manager is not None and hasattr(manager, "xfmr_coord_to_wp_rank"):
        return [manager.xfmr_coord_to_wp_rank(0, 0, tp_rank) for tp_rank in range(tp_size)]
    return list(range(tp_size))


def _optimizer_state_tuple_for_log(optimizer_config):
    if optimizer_config == 4:
        return (True, False, None, False)
    if optimizer_config in (5, 7, 8, 9):
        return (False, True, None, False)
    if optimizer_config == 6:
        return (False, False, None, True)
    if optimizer_config in (0, 1, 2, 3):
        return (False, False, optimizer_config, False)
    return (False, False, None, False)

manager = ParallelManager(
    dp_size = args.data_parallel_group_size,
    mp_size = other_params['mp_size'],
    wp_topo = other_params['wp_topo'],
    xfmr_wp_topo = other_params['xfmr_wp_topo'],
    domain_topo = other_params['domain_topo'],
    rank = rank,
    world_size = world_size,
    device = device,
    window_assignment_mode = other_params.get('window_assignment_mode', 'regular'),
    xfmr_sp_size = other_params.get('xfmr_sp_size', 1),
    tensor_parallel_size = other_params.get('tensor_parallel_size', 1),
    sp_tp_placement = other_params.get('sp_tp_placement', 'tp_first'),
)

log_wp_ranks = _get_debug_log_wp_ranks(manager, model_type, other_params)
early_optimizer_state_tuple = _optimizer_state_tuple_for_log(model_config['optimizer_config'])

if args.quiet_metrics:
    devnull = open(os.devnull, "w", buffering=1, encoding="utf-8")
    sys.stdout = devnull
    sys.stderr = devnull
elif not args.disable_rank_log_redirect:
    set_dp_rank_print_redirect(
                            rank = rank,
                            dp_rank = dist.get_rank(group=manager.data_parallel_group),
                            mode = model_type,
                            embedding_parallel_type = other_params['embedding_parallel_type'],
                            optimizer_state_tuple = early_optimizer_state_tuple,
                            manager = manager,
                            only_wp_ranks = log_wp_ranks)

model, engine, optimizer, gscaler, optimizer_state_tuple = get_model_for_train(
    model_archi_params,

    precision,
    model_config['half_model'],
    model_config['optimizer_config'],

    task_type = model_config['task_type'],

    kaiming_init = model_config['kaiming_init'],
    model_type = model_type,
    model_architecture = model_config['model_architecture'],

    embedding_parallel_type = other_params['embedding_parallel_type'],
    attn_parallel_type = other_params['attn_parallel_type'],
    mlp_parallel_type = other_params['mlp_parallel_type'],

    manager = manager,

    device = device,
    local_rank = local_rank,
    learning_rate = model_config['learning_rate'],
)


USE_DDP, USE_FSDP, ZERO_STAGE_NUMBER, USE_DIST_OPT = optimizer_state_tuple


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
total_params = count_parameters(model)


def _load_pretrained_state_dict_for_test(model, saved_ckpt_path, device, rank):
    checkpoint = torch.load(saved_ckpt_path, map_location=device, weights_only=True)
    state_dict = checkpoint['model_state_dict']
    model_state = model.state_dict()
    new_state_dict = {}
    skipped_keys = []

    for k, v in state_dict.items():
        candidates = [k]
        if k.startswith("module."):
            candidates.append(k[7:])
        else:
            candidates.append("module." + k)

        target_key = None
        for candidate in candidates:
            if candidate in model_state:
                target_key = candidate
                break

        if target_key is None:
            skipped_keys.append(k)
            continue
        if target_key.endswith("attn_mask"):
            skipped_keys.append(target_key)
            continue
        if tuple(model_state[target_key].shape) != tuple(v.shape):
            skipped_keys.append(target_key)
            continue

        new_state_dict[target_key] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if rank == 0 and not args.quiet_metrics:
        print(
            "loaded pretrained checkpoint for finetune test:",
            saved_ckpt_path,
            "loaded_keys=", len(new_state_dict),
            "skipped_keys=", len(skipped_keys),
            "missing_keys=", len(missing_keys),
            "unexpected_keys=", len(unexpected_keys),
        )


if True:

    trained_model_config_path = model_config.get('trained_model_config_path', None)
    trained_model_checkpoint_path = model_config.get('trained_model_checkpoint_path', None)
    if trained_model_checkpoint_path:
        _load_pretrained_state_dict_for_test(model, trained_model_checkpoint_path, device, rank)
    elif trained_model_config_path:
        with open(trained_model_config_path, 'r') as f:
            pretrained_model_config = yaml.safe_load(f)


            pretrained_model_ckpt_path = pretrained_model_config['train_path']

            ckpt_path = pretrained_model_config['train_path'] +'/'+ pretrained_model_config['description'] + '/ckpt/' +  "wp_rank"+'_'+str(manager.get_wp_rank())
            pretrained_epoch = model_config.get('trained_model_epoch', pretrained_model_config.get('ckpt_epoch', 200))
            saved_ckpt_path = ckpt_path+'/epoch'+str(pretrained_epoch)+'.pth'
            _load_pretrained_state_dict_for_test(model, saved_ckpt_path, device, rank)
    else:
        if rank==0 and not args.quiet_metrics:
            print('No trained_model_config_path is provided; finetune starts from current model initialization.')


log_wp_ranks = _get_debug_log_wp_ranks(manager, model_type, other_params)


if args.quiet_metrics:
    devnull = open(os.devnull, "w", buffering=1, encoding="utf-8")
    sys.stdout = devnull
    sys.stderr = devnull
elif not args.disable_rank_log_redirect:
    set_dp_rank_print_redirect(
                            rank = rank,
                            dp_rank = dist.get_rank(group=manager.data_parallel_group),
                            mode = model_type,
                            embedding_parallel_type = other_params['embedding_parallel_type'],
                            optimizer_state_tuple = optimizer_state_tuple,
                            manager = manager,
                            only_wp_ranks = log_wp_ranks,
                            append = True)


if (not args.quiet_metrics) and (not args.disable_rank_log_redirect) and manager.get_wp_rank() in log_wp_ranks:
    print(
        "[TERRA parallel config]",
        "task_type=", task_type,
        "model_architecture=", model_config['model_architecture'],
        "model_type=", model_type,
        "dp_size=", args.data_parallel_group_size,
        "ranks_per_dp=", ranks_per_dp,
        "wp_topo=", other_params['wp_topo'],
        "xfmr_wp_topo=", other_params['xfmr_wp_topo'],
        "xfmr_sp_size=", other_params.get('xfmr_sp_size', 1),
        "tensor_parallel_size=", other_params.get('tensor_parallel_size', 1),
        "sp_tp_placement=", other_params.get('sp_tp_placement', 'tp_first'),
        "embedding_parallel_type=", other_params['embedding_parallel_type'],
        "attn_parallel_type=", other_params['attn_parallel_type'],
        "use_wp_ulysses_attention=", (
            int(other_params.get('xfmr_sp_size', 1)) > 1
            or int(other_params.get('tensor_parallel_size', 1)) > 1
        ),
        "optimizer_config=", model_config['optimizer_config'],
        "optimizer_state_tuple=", optimizer_state_tuple,
    )


task_specific_data_dict = get_task_specific_data(
    task_type,
    device,
    optimizer_state_tuple,
    model_archi_params,
    other_params,
    manager,
    micro_batch_size=model_config['micro_batch_size'],
    model_type=model_type,
    dataset_config=dataset_config,
)


task_specific_data_dict['embedding_parallel_type'] = other_params['embedding_parallel_type']
task_specific_data_dict['test_iter_num'] = test_iter_num
task_specific_data_dict['show_gradient_flag'] = show_gradient_flag


torch.distributed.barrier()

def _normalize_lead_checkpoint_schedule(model_config):
    activation = model_config.get("activation", {}) or {}
    raw_schedule = activation.get("lead_checkpoint_schedule", activation.get("lead_checkpoint_layers", None))
    if not raw_schedule:
        return []

    normalized = []
    if isinstance(raw_schedule, dict):
        iterable = []
        for key, value in raw_schedule.items():
            text = str(key).strip()
            if "-" in text:
                start_text, end_text = [part.strip() for part in text.split("-", 1)]
                lead_start = int(start_text)
                lead_end = int(end_text)
            else:
                lead_start = lead_end = int(text)
            iterable.append({
                "lead_start": lead_start,
                "lead_end": lead_end,
                "checkpoint_layers": value,
            })
    else:
        iterable = list(raw_schedule)

    for item in iterable:
        if not isinstance(item, dict):
            raise ValueError(f"lead_checkpoint_schedule entries must be dicts, got: {item}")
        lead_start = int(item.get("lead_start", item.get("start", item.get("lead", 1))))
        lead_end = int(item.get("lead_end", item.get("end", lead_start)))
        checkpoint_layers = item.get("checkpoint_layers", item.get("layers", None))
        if checkpoint_layers is None:
            raise ValueError(f"lead_checkpoint_schedule entry missing checkpoint_layers: {item}")
        if isinstance(checkpoint_layers, str):
            checkpoint_layers = [part.strip() for part in checkpoint_layers.replace(",", " ").split() if part.strip()]
        activation_mode = item.get("activation_mode", item.get("mode", activation.get("mode", "torch_recompute")))
        segment_activation_modes = item.get(
            "segment_activation_modes", None
        )
        if segment_activation_modes is None:
            segment_activation_modes = [
                str(activation_mode) for _ in checkpoint_layers
            ]
        elif isinstance(segment_activation_modes, str):
            segment_activation_modes = [
                part.strip()
                for part in segment_activation_modes.replace(",", " ").split()
                if part.strip()
            ]
        else:
            segment_activation_modes = [
                str(mode) for mode in segment_activation_modes
            ]
        if len(segment_activation_modes) != len(checkpoint_layers):
            raise ValueError(
                "lead_checkpoint_schedule segment_activation_modes must "
                "match checkpoint_layers: "
                f"{len(segment_activation_modes)} != "
                f"{len(checkpoint_layers)} in {item}"
            )
        offload = item.get("offload", activation.get("offload", {}))
        if offload is None:
            offload = {}
        if not isinstance(offload, dict):
            raise ValueError(
                "lead_checkpoint_schedule offload must be a dict: "
                f"{item}"
            )
        normalized.append({
            "lead_start": lead_start,
            "lead_end": lead_end,
            "checkpoint_layers": list(checkpoint_layers),
            "activation_mode": str(activation_mode),
            "segment_activation_modes": segment_activation_modes,
            "offload": dict(offload),
            "label": item.get("label", f"lead{lead_start}-{lead_end}"),
        })

    normalized.sort(key=lambda item: (item["lead_start"], item["lead_end"]))
    return normalized


def _normalize_lead_sampling_checkpoint_schedule(model_config):
    activation = model_config.get("activation", {}) or {}
    raw_schedule = activation.get("lead_sampling_checkpoint_schedule", None)
    if not raw_schedule:
        return []

    normalized = []
    iterable = []
    if isinstance(raw_schedule, dict):
        for key, value in raw_schedule.items():
            text = str(key).strip()
            if "-" in text:
                start_text, end_text = [part.strip() for part in text.split("-", 1)]
                lead_start = int(start_text)
                lead_end = int(end_text)
            else:
                lead_start = lead_end = int(text)
            item = dict(value or {}) if isinstance(value, dict) else {"sampling_checkpoint": value}
            item["lead_start"] = lead_start
            item["lead_end"] = lead_end
            iterable.append(item)
    else:
        iterable = list(raw_schedule)

    for item in iterable:
        if not isinstance(item, dict):
            raise ValueError(f"lead_sampling_checkpoint_schedule entries must be dicts, got: {item}")
        lead_start = int(item.get("lead_start", item.get("start", item.get("lead", 1))))
        lead_end = int(item.get("lead_end", item.get("end", lead_start)))
        sampling = item.get("sampling_checkpoint", item.get("sampling", item))
        if not isinstance(sampling, dict):
            raise ValueError(f"lead_sampling_checkpoint_schedule entry missing sampling checkpoint dict: {item}")
        down = sampling.get("down", None)
        up = sampling.get("up", None)
        if down is None and up is None:
            raise ValueError(f"lead_sampling_checkpoint_schedule entry must contain down or up: {item}")
        sampling_modes = item.get("sampling_modes", {})
        if sampling_modes is None:
            sampling_modes = {}
        if not isinstance(sampling_modes, dict):
            raise ValueError(
                "lead_sampling_checkpoint_schedule sampling_modes must be "
                f"a down/up dict: {item}"
            )
        unknown_mode_axes = set(sampling_modes) - {"down", "up"}
        if unknown_mode_axes:
            raise ValueError(
                "lead_sampling_checkpoint_schedule sampling_modes contains "
                f"unsupported axes: {sorted(unknown_mode_axes)}"
            )
        offload = item.get("offload", None)
        if offload is not None and not isinstance(offload, dict):
            raise ValueError(
                "lead_sampling_checkpoint_schedule offload must be a dict: "
                f"{item}"
            )
        normalized.append({
            "lead_start": lead_start,
            "lead_end": lead_end,
            "down": str(down).upper() if down is not None else None,
            "up": str(up).upper() if up is not None else None,
            "down_activation_mode": sampling_modes.get("down", None),
            "up_activation_mode": sampling_modes.get("up", None),
            "offload": dict(offload) if offload is not None else None,
            "label": item.get("label", f"lead{lead_start}-{lead_end}:sampling"),
        })

    normalized.sort(key=lambda item: (item["lead_start"], item["lead_end"]))
    return normalized


def _checkpoint_layers_for_lead(
    schedule,
    lead_number,
    base_activation_mode,
    base_offload,
):
    for item in schedule:
        if item["lead_start"] <= lead_number <= item["lead_end"]:
            return (
                item["checkpoint_layers"],
                item["activation_mode"],
                item["segment_activation_modes"],
                item["offload"],
                item["label"],
            )
    return None, base_activation_mode, None, base_offload, ""


def _sampling_checkpoint_for_lead(
    schedule,
    lead_number,
    base_down,
    base_up,
    base_down_activation_mode,
    base_up_activation_mode,
    base_offload,
):
    for item in schedule:
        if item["lead_start"] <= lead_number <= item["lead_end"]:
            return (
                item["down"] or base_down,
                item["up"] or base_up,
                item["down_activation_mode"] or base_down_activation_mode,
                item["up_activation_mode"] or base_up_activation_mode,
                item["offload"] if item["offload"] is not None else base_offload,
                item["label"],
            )
    return (
        base_down,
        base_up,
        base_down_activation_mode,
        base_up_activation_mode,
        base_offload,
        "",
    )


def _apply_checkpoint_layers_to_model(
    model,
    checkpoint_layers,
    activation_mode,
    segment_activation_modes,
    offload_config,
):
    updated = 0
    for module in model.modules():
        scheduler = getattr(module, "recompute_scheduler", None)
        if scheduler is None:
            continue
        if hasattr(scheduler, "set_checkpoint_plan"):
            scheduler.set_checkpoint_plan(
                checkpoint_layers,
                activation_mode=activation_mode,
                offload_config=offload_config,
                segment_activation_modes=segment_activation_modes,
            )
        elif hasattr(scheduler, "set_checkpoint_layers"):
            scheduler.set_checkpoint_layers(checkpoint_layers)
        else:
            continue
        updated += 1
    return updated


def _apply_sampling_checkpoint_to_model(
    model,
    down_mode,
    up_mode,
    down_activation_mode,
    up_activation_mode,
    offload_config,
):
    updated = 0
    for module in model.modules():
        if not (hasattr(module, "checkpoint_down_mode") or hasattr(module, "checkpoint_up_mode")):
            continue
        base_config = dict(getattr(module, "recompute_config", {}) or {})
        sampling = dict(base_config.get("sampling_checkpoint", {}) or {})
        sampling_modes = dict(base_config.get("sampling_modes", {}) or {})
        if down_mode is not None and hasattr(module, "checkpoint_down_mode"):
            module.checkpoint_down_mode = str(down_mode)
            sampling["down"] = str(down_mode)
            base_config["checkpoint_down_mode"] = str(down_mode)
            sampling_modes["down"] = str(down_activation_mode)
        if up_mode is not None and hasattr(module, "checkpoint_up_mode"):
            module.checkpoint_up_mode = str(up_mode)
            sampling["up"] = str(up_mode)
            base_config["checkpoint_up_mode"] = str(up_mode)
            sampling_modes["up"] = str(up_activation_mode)
        base_config["sampling_checkpoint"] = sampling
        base_config["sampling_modes"] = sampling_modes
        base_config["offload"] = dict(offload_config or {})
        module.recompute_config = base_config
        if hasattr(module, "down_recompute_config"):
            module.down_recompute_config = activation_config_for_sampling(base_config, "down")
        if hasattr(module, "up_recompute_config"):
            module.up_recompute_config = activation_config_for_sampling(base_config, "up")
        updated += 1
    return updated


lead_checkpoint_schedule = _normalize_lead_checkpoint_schedule(model_config)
lead_sampling_checkpoint_schedule = _normalize_lead_sampling_checkpoint_schedule(model_config)
base_activation = (model_config.get("activation", {}) or {})
base_transformer_activation_mode = str(base_activation.get("mode", "torch_recompute"))
base_transformer_offload = dict(base_activation.get("offload", {}) or {})
base_sampling = (base_activation.get("sampling_checkpoint", {}) or {})
base_sampling_down = str(base_sampling.get("down", "none")).upper()
base_sampling_up = str(base_sampling.get("up", "none")).upper()
base_sampling_modes = ((model_config.get("activation", {}) or {}).get("sampling_modes", {}) or {})
base_sampling_down_activation_mode = str(
    base_sampling_modes.get("down", "torch_recompute")
)
base_sampling_up_activation_mode = str(
    base_sampling_modes.get("up", "torch_recompute")
)
base_sampling_offload = dict(
    ((model_config.get("activation", {}) or {}).get("offload", {}) or {})
)
memory_timeline_rows = []
current_iter_idx = -1
_last_applied_checkpoint_layers = None
_last_applied_checkpoint_plan = None
_last_applied_checkpoint_label = ""
_last_applied_sampling_checkpoint = None
_last_applied_sampling_checkpoint_label = ""


def _lead_callback(event, lead_idx, cur_loss):
    global _last_applied_checkpoint_layers, _last_applied_checkpoint_plan
    global _last_applied_checkpoint_label
    global _last_applied_sampling_checkpoint, _last_applied_sampling_checkpoint_label
    lead_number = int(lead_idx) + 1
    (
        checkpoint_layers,
        transformer_activation_mode,
        transformer_segment_activation_modes,
        transformer_offload,
        checkpoint_label,
    ) = _checkpoint_layers_for_lead(
        lead_checkpoint_schedule,
        lead_number,
        base_transformer_activation_mode,
        base_transformer_offload,
    )
    (
        down_mode,
        up_mode,
        down_activation_mode,
        up_activation_mode,
        sampling_offload,
        sampling_label,
    ) = _sampling_checkpoint_for_lead(
        lead_sampling_checkpoint_schedule,
        lead_number,
        base_sampling_down,
        base_sampling_up,
        base_sampling_down_activation_mode,
        base_sampling_up_activation_mode,
        base_sampling_offload,
    )
    if event == "before_forward":
        set_memory_timeline_context(lead_idx=lead_number)
        sampling_key = (
            down_mode,
            up_mode,
            down_activation_mode,
            up_activation_mode,
            json.dumps(sampling_offload, sort_keys=True),
        )
        if lead_sampling_checkpoint_schedule and sampling_key != _last_applied_sampling_checkpoint:
            updated_sampling = _apply_sampling_checkpoint_to_model(
                model,
                down_mode,
                up_mode,
                down_activation_mode,
                up_activation_mode,
                sampling_offload,
            )
            if updated_sampling <= 0:
                raise RuntimeError("lead_sampling_checkpoint_schedule is set, but no sampling checkpoint module was found in model")
            _last_applied_sampling_checkpoint = sampling_key
            _last_applied_sampling_checkpoint_label = sampling_label
        elif lead_sampling_checkpoint_schedule and sampling_label:
            _last_applied_sampling_checkpoint_label = sampling_label

        checkpoint_plan = (
            tuple(checkpoint_layers) if checkpoint_layers is not None else None,
            transformer_activation_mode,
            tuple(transformer_segment_activation_modes)
            if transformer_segment_activation_modes is not None
            else None,
            json.dumps(transformer_offload, sort_keys=True),
        )
        if checkpoint_layers is not None and checkpoint_plan != _last_applied_checkpoint_plan:
            updated = _apply_checkpoint_layers_to_model(
                model,
                checkpoint_layers,
                transformer_activation_mode,
                transformer_segment_activation_modes,
                transformer_offload,
            )
            if updated <= 0:
                raise RuntimeError("lead_checkpoint_schedule is set, but no recompute_scheduler was found in model")
            _last_applied_checkpoint_layers = list(checkpoint_layers)
            _last_applied_checkpoint_plan = checkpoint_plan
            _last_applied_checkpoint_label = checkpoint_label
        elif checkpoint_layers is not None:
            _last_applied_checkpoint_label = checkpoint_label
        mark_memory_timeline(
            "rollout_lead_pre",
            f"lead_{lead_number}",
            lead_idx=lead_number,
        )
        return

    if event == "after_loss":
        mark_memory_timeline(
            "rollout_lead_post",
            f"lead_{lead_number}",
            lead_idx=lead_number,
        )
        if args.memory_timeline_csv is None or rank != 0:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            allocated_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
            reserved_mb = torch.cuda.memory_reserved(device) / 1024 / 1024
            peak_mb = get_memory_timeline_peak_mib()
        else:
            allocated_mb = reserved_mb = peak_mb = 0.0
        loss_value = ""
        if cur_loss is not None:
            try:
                loss_value = float(cur_loss.detach().float().item())
            except Exception:
                loss_value = ""
        memory_timeline_rows.append({
            "iter": current_iter_idx,
            "lead": lead_number,
            "checkpoint_label": checkpoint_label or _last_applied_checkpoint_label,
            "checkpoint_layers": json.dumps(checkpoint_layers if checkpoint_layers is not None else _last_applied_checkpoint_layers),
            "transformer_activation_mode": transformer_activation_mode,
            "transformer_segment_activation_modes": json.dumps(
                transformer_segment_activation_modes
                if transformer_segment_activation_modes is not None
                else []
            ),
            "transformer_offload": json.dumps(transformer_offload, sort_keys=True),
            "sampling_checkpoint_label": sampling_label or _last_applied_sampling_checkpoint_label,
            "sampling_down": down_mode,
            "sampling_up": up_mode,
            "sampling_down_activation_mode": down_activation_mode,
            "sampling_up_activation_mode": up_activation_mode,
            "allocated_memory_mb": allocated_mb,
            "reserved_memory_mb": reserved_mb,
            "peak_memory_mb": peak_mb,
            "loss": loss_value,
        })


lead_callback = _lead_callback if (lead_checkpoint_schedule or lead_sampling_checkpoint_schedule or args.memory_timeline_csv is not None) else None

peak_memory_list = []
loss_list =[]
output_sum_list = []
output_tensor_dtype_list = []

one_step_train_time_s_list = []
forward_time_s_list = []
load_data_time_list = []
timer_name_to_total_time_list = []
current_external_step_start_time_s = None
current_input_transfer_start_event = None
current_input_transfer_end_event = None

flops_profiler = TorchFlopsProfiler(
    enabled=ENABLE_TORCH_PROF,
    rank=rank,
    schedule_steps=TORCH_PROF_step_list,
    trace_dir="./log/profiler/rank0",
    write_trace=TORCH_PROF_WRITE_TRACE,
)
flops_profiler.start()


def finetune_one_step():
    return one_step_finetune(
                    task_type,
                    model_type, # 'parallel', 'sequential',
                    model,
                    engine,
                    optimizer,

                    required_tensors,
                    task_specific_data_dict,
                    other_params,

                    precision,
                    #model_config['half_model'],
                    my_dtype,
                    loss_fn,
                    gscaler,
                    manager = manager,
                    optimizer_state_tuple = optimizer_state_tuple,
                    memory_timeline = mem_trace,

                    lead_time = model_config['finetune_lead_time'],
                    loss_reduction = model_config['finetune_loss_reduction'],
                    lead_callback = lead_callback,
                    external_step_start_time_s = current_external_step_start_time_s,
                )


global_step = 0
num_pre_train_epochs = 300


if task_type == 'glorys':
    dataloader_num_workers = int(model_config.get('num_workers', model_config.get('dataloader_num_workers', 8)))
    train_dataloader = get_finetune_dataloader_for_task(task_type, model_config['micro_batch_size'], use_splited_data, status=0, num_workers=dataloader_num_workers,
                                               manager=manager, model_type= model_type, model_archi_params= model_archi_params,
                                               other_params = other_params, use_fake_input = use_fake_input, lead_time = model_config['finetune_lead_time'],
                                               fake_input_random = fake_input_random,
                                               fake_input_dmp_local = fake_input_dmp_local,
                                               fake_input_seed = fake_input_seed,
                                               pin_memory = dataloader_pin_memory)


    for i, x in enumerate(itertools.islice(train_dataloader, test_iter_num)):
        torch.cuda.reset_peak_memory_stats(device)

        current_iter_idx = i

        lr = get_finetune_lr_by_global_step(
                global_step,
                model_config['learning_rate'],
                warmup_steps=10,
            )
        if ZERO_STAGE_NUMBER is not None:
            set_lr(engine.optimizer, lr)
        else:
            set_lr(optimizer, lr)


        # seq
        #x[0] shape torch.Size([1, 4, 70, 969, 1878]) torch.Size([1, 4, 70, 968, 1879])
        #torch.Size([1, 4, 70, 969, 1879]) torch.Size([1, 4, 70, 969, 1879]) torch.Size([1, 4, 1, 969, 1879]) torch.Size([1, 3, 16, 969, 1879])

        # para
        # x[0] shape torch.Size([1, 5, 144, 960, 280]) torch.Size([1, 5, 144, 960, 280]) torch.Size([1, 5, 144, 960, 280]) torch.Size([1, 5, 144, 960, 280]) torch.Size([1, 5, 144, 960, 4]) torch.Size([1, 5, 144, 960, 64])


        current_external_step_start_time_s = None
        current_input_transfer_start_event = None
        current_input_transfer_end_event = None
        if include_input_transfer_in_step_time:
            # Align ranks before H2D. one_step_finetune skips its legacy
            # pre-forward synchronization in this opt-in mode, so the final
            # synchronized wall time includes the queued input transfer.
            torch.cuda.synchronize()
            torch.distributed.barrier()
            current_external_step_start_time_s = time.time()
            if not input_double_buffer:
                current_input_transfer_start_event = torch.cuda.Event(
                    enable_timing=True
                )
                current_input_transfer_end_event = torch.cuda.Event(
                    enable_timing=True
                )
                current_input_transfer_start_event.record()

        if input_double_buffer:
            if task_type != 'glorys' or model_type == 'sequential':
                raise RuntimeError(
                    "input_double_buffer currently supports parallel GLORYS only"
                )
            required_tensors = resolve_glorys_finetune_double_buffer(
                x,
                device,
                task_specific_data_dict,
                my_dtype,
                lead_time=model_config['finetune_lead_time'],
            )
        elif (
            task_type == 'glorys'
            and model_type != 'sequential'
            and (not use_splited_data)
            and (not fake_input_dmp_local)
        ):
            required_tensors = resolve_required_tensors_online_split_for_parallel(
                task_type,
                x,
                device,
                model_archi_params,
                other_params,
                manager,
                my_dtype,
                is_pretrain=False,
                lead_time=model_config['finetune_lead_time'],
            )
        else:
            required_tensors = resolve_required_tensors_from_dataloader(
                task_type,
                x,
                device,
                task_specific_data_dict,
                my_dtype,
                is_pretrain=False,
                lead_time=model_config['finetune_lead_time'],
                input_non_blocking=input_transfer_non_blocking,
            )

        if current_input_transfer_end_event is not None:
            current_input_transfer_end_event.record()

        if release_cpu_batch_after_transfer:
            # required_tensors now owns independent device tensors; release the
            # potentially very large pinned/pageable CPU batch before forward.
            del x

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


        if MEM_PROF==0:
            mem_trace = None
            thetao_out, loss, one_step_train_time_s, forward_time_s = finetune_one_step()

        elif MEM_PROF==1:
            with CudaMemoryTimeline(
                    model=model,
                    device=device,
                    output_dir=MEMORY_TIMELINE_OUTPUT_DIR,

                    model_type = model_type,
                    rank=rank,
                    dp_rank = dist.get_rank(group=manager.data_parallel_group),
                    wp_rank = manager.get_wp_rank(),
                    iter_idx=i,
                    name=task_type,

                    max_depth = None,
                    record_leaf_only=True,
                    module_filter=None,
                    synchronize=True,
                ) as mem_trace:
                thetao_out, loss, one_step_train_time_s, forward_time_s = finetune_one_step()

        elif MEM_PROF==2:
            '''
            module_filter=[
                "DomainParallelDownBlock",
                "DomainParallelUpBlock",
                "Parallel_BasicLayer",
                "WindowParallel",
            ]
            '''
            with CudaMemoryTimeline(
                    model=model,
                    device=device,
                    output_dir=MEMORY_TIMELINE_OUTPUT_DIR,

                    model_type = model_type,
                    rank=rank,
                    dp_rank = dist.get_rank(group=manager.data_parallel_group),
                    wp_rank = manager.get_wp_rank(),
                    iter_idx=i,
                    name=task_type,

                    max_depth = 4,
                    record_leaf_only=False,
                    module_filter= None,
                    synchronize=False,
                ) as mem_trace:
                thetao_out, loss, one_step_train_time_s, forward_time_s = finetune_one_step()

        flops_profiler.step()
        global_step = global_step + 1

        loss, thetao_out = process_loss_and_output_in_test(loss, thetao_out, model_type, other_params, manager)

        peak_memory = get_memory_timeline_peak_mib()
        if mem_trace is not None:
            peak_memory = max(peak_memory, mem_trace.overall_peak_allocated_mib)
        peak_memory_list.append(peak_memory)

        output_tensor_dtype_list.append(thetao_out.dtype)
        output_sum_list.append(thetao_out.type(torch.float32).sum().item())
        loss_list.append(loss.item())

        one_step_train_time_s_list.append(one_step_train_time_s)
        forward_time_s_list.append(forward_time_s)

        if input_double_buffer:
            input_transfer_time_s = required_tensors.transfer_time_s()
            required_tensors.release()
        elif current_input_transfer_start_event is not None:
            input_transfer_time_s = (
                current_input_transfer_start_event.elapsed_time(
                    current_input_transfer_end_event
                )
                / 1000.0
            )
        else:
            input_transfer_time_s = 0.0
        load_data_time_list.append(input_transfer_time_s)


flops_profiler.stop()


if (rank % ranks_per_dp)==0:

    print('optimizer_config', model_config['optimizer_config'], 'precision', model_config['precision'], 'half_model', model_config['half_model'])
    print('rank', rank, 'dp_rank', dist.get_rank(group=manager.data_parallel_group), f"Total parameters: {total_params:,}")


if test_iter_num>2:
    all_mem_peak_0 = gather_scalar(peak_memory_list[0])
    all_mem_peak_1 = gather_scalar(peak_memory_list[1])

all_parameters = gather_scalar(total_params)
all_mem_peak_last = gather_scalar(peak_memory_list[-1])
all_time_peak = gather_scalar(one_step_train_time_s_list[-1])
local_boundary_offload_stats = get_boundary_offload_stats()
boundary_offload_stats_by_rank = {
    key: gather_scalar(value)
    for key, value in sorted(local_boundary_offload_stats.items())
}
flops_summary = flops_profiler.summarize()

def write_test_metrics():
    if rank != 0:
        return

    csv_path = os.path.abspath(args.metrics_csv) if args.metrics_csv is not None else None
    json_path = None
    if not args.metrics_no_json:
        if args.metrics_json is not None:
            json_path = os.path.abspath(args.metrics_json)
        elif csv_path is not None:
            root, _ = os.path.splitext(csv_path)
            json_path = root + ".json"

    if csv_path is None and json_path is None:
        return

    row_count = min(
        len(loss_list),
        len(output_sum_list),
        len(output_tensor_dtype_list),
        len(peak_memory_list),
        len(one_step_train_time_s_list),
    )
    lead_time_value = int(model_config.get('finetune_lead_time', 1))
    loss_reduction_value = model_config.get('finetune_loss_reduction', 'mean')
    local_flops_per_step = flops_summary.local_flops_per_step

    if csv_path is not None:
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "iter",
                    "lead_time",
                    "loss_reduction",
                    "loss",
                    "loss_per_lead",
                    "loss_sum_estimated",
                    "output_sum",
                    "output_dtype",
                    "peak_memory_mb",
                    "step_time_s",
                    "forward_time_s",
                    "rest_time_s",
                    "flops_tflops",
                    "load_data_time_s",
                ],
            )
            writer.writeheader()
            for idx in range(row_count):
                step_time = one_step_train_time_s_list[idx]
                forward_time = forward_time_s_list[idx] if idx < len(forward_time_s_list) else None
                load_time = load_data_time_list[idx] if idx < len(load_data_time_list) else None
                writer.writerow({
                    "iter": idx,
                    "lead_time": lead_time_value,
                    "loss_reduction": loss_reduction_value,
                    "loss": loss_list[idx],
                    "loss_per_lead": loss_list[idx] if loss_reduction_value == "mean" else loss_list[idx] / max(1, lead_time_value),
                    "loss_sum_estimated": loss_list[idx] * max(1, lead_time_value) if loss_reduction_value == "mean" else loss_list[idx],
                    "output_sum": output_sum_list[idx],
                    "output_dtype": str(output_tensor_dtype_list[idx]),
                    "peak_memory_mb": peak_memory_list[idx],
                    "step_time_s": step_time,
                    "forward_time_s": forward_time,
                    "rest_time_s": (step_time - forward_time) if forward_time is not None else None,
                    "flops_tflops": (local_flops_per_step / step_time / 1e12) if step_time and local_flops_per_step else 0.0,
                    "load_data_time_s": load_time,
                })

    if json_path is not None:
        json_dir = os.path.dirname(json_path)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        padding_spec = model_archi_params.get("padding_spec", {}) or {}
        metadata = {
            "script": "test_finetune.py",
            "model_cfg": os.path.abspath(args.model_cfg),
            "task_type": task_type,
            "model_architecture": model_config.get("model_architecture"),
            "model_type": model_type,
            "precision": precision,
            "optimizer_config": model_config.get("optimizer_config"),
            "dp_size": args.data_parallel_group_size,
            "world_size": world_size,
            "ranks_per_dp": ranks_per_dp,
            "wp_topo": other_params.get("wp_topo"),
            "xfmr_wp_topo": other_params.get("xfmr_wp_topo"),
            "xfmr_sp_size": other_params.get("xfmr_sp_size", 1),
            "tensor_parallel_size": other_params.get("tensor_parallel_size", 1),
            "window_assignment_mode": other_params.get("window_assignment_mode", "regular"),
            "lead_time": lead_time_value,
            "finetune_loss_reduction": loss_reduction_value,
            "micro_batch_size": model_config.get("micro_batch_size"),
            "use_fake_input": use_fake_input,
            "fake_input_random": fake_input_random,
            "fake_input_seed": fake_input_seed,
            "fake_input_dmp_local": fake_input_dmp_local,
            "release_cpu_batch_after_transfer": release_cpu_batch_after_transfer,
            "dataloader_pin_memory": dataloader_pin_memory,
            "input_transfer_non_blocking": input_transfer_non_blocking,
            "include_input_transfer_in_step_time": include_input_transfer_in_step_time,
            "input_double_buffer": input_double_buffer,
            "num_workers": int(model_config.get('num_workers', model_config.get('dataloader_num_workers', 8))),
            "test_iter_num": test_iter_num,
            "metrics_csv": csv_path,
            "parameter_count_by_rank": all_parameters,
            "peak_memory_last_by_rank": all_mem_peak_last,
            "step_time_last_by_rank": all_time_peak,
            "boundary_offload_stats_by_rank": boundary_offload_stats_by_rank,
            "padding_policy": model_archi_params.get("padding_policy"),
            "padded_shape": model_archi_params.get("padded_shape"),
            "initial_padding": model_archi_params.get("initial_padding"),
            "patch_token_resolution": padding_spec.get("patch_token_resolution"),
            "transformer_token_resolution": padding_spec.get("transformer_token_resolution"),
            "num_windows": padding_spec.get("num_windows"),
            "flops_profiler_enabled": flops_summary.enabled,
            "flops_profiler_active_steps": flops_summary.active_steps,
            "flops_profiler_event_count": flops_summary.event_count,
            "flops_profiler_local_flops_per_step": flops_summary.local_flops_per_step,
            "flops_profiler_world_flops_per_step": flops_summary.world_flops_per_step,
            "flops_profiler_world_avg_flops_per_step": flops_summary.world_avg_flops_per_step,
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

def write_memory_timeline_metrics():
    if rank != 0 or args.memory_timeline_csv is None:
        return
    timeline_path = os.path.abspath(args.memory_timeline_csv)
    timeline_dir = os.path.dirname(timeline_path)
    if timeline_dir:
        os.makedirs(timeline_dir, exist_ok=True)
    with open(timeline_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iter",
                "lead",
            "checkpoint_label",
            "checkpoint_layers",
            "transformer_activation_mode",
            "transformer_segment_activation_modes",
            "transformer_offload",
            "sampling_checkpoint_label",
            "sampling_down",
            "sampling_up",
            "sampling_down_activation_mode",
            "sampling_up_activation_mode",
            "allocated_memory_mb",
                "reserved_memory_mb",
                "peak_memory_mb",
                "loss",
            ],
        )
        writer.writeheader()
        writer.writerows(memory_timeline_rows)

write_test_metrics()
write_memory_timeline_metrics()
actual_iter_num = len(loss_list)


if (rank % ranks_per_dp)==0:
    print('all_parameters', all_parameters)
    if test_iter_num>2:
        print('all_mem_peak_0', all_mem_peak_0)
        print('all_mem_peak_1', all_mem_peak_1)
    print('all_mem_peak_last', all_mem_peak_last)
    print('all_time_peak', all_time_peak)

    if ENABLE_TORCH_PROF:
        flops_profiler.print_top_ops(row_limit=20)
        print('local_FLOPs_per_step', flops_summary.local_flops_per_step)
        print('world_FLOPs_per_step', flops_summary.world_flops_per_step)
        print('world_avg_FLOPs_per_step', flops_summary.world_avg_flops_per_step)
    total_FLOPs = flops_summary.local_flops_per_step

    print('-----------------------------Accuracy------------------------------')
    for i in range(0, actual_iter_num):
        print('iter', i, output_tensor_dtype_list[i], "sum:", output_sum_list[i], 'loss', loss_list[i])
    print('-----------------------------Per-step memory------------------------------')
    for i in range(0, actual_iter_num):
        print('iter', i, 'peak_memory', peak_memory_list[i])
    print('-----------------------------Per-step time and FLOPS------------------------------')#
    for i in range(0, actual_iter_num):
        print('iter', i, f"Total Time  : {one_step_train_time_s_list[i]*1000:.3f} ms", f"FLOPS : {total_FLOPs/one_step_train_time_s_list[i]/1e12:.3f} TFLOPS", f"forward Time  : {forward_time_s_list[i]*1000:.3f} ms", f"load data time {load_data_time_list[i]*1000:.3f} ms")
    print('-----------------------------Per-step time break down------------------------------')
    for i in range(0, actual_iter_num):
        print('iter', i, f"Total Time  : {one_step_train_time_s_list[i]*1000:.3f} ms", f"forward Time  : {forward_time_s_list[i]*1000:.3f} ms", f"Rest Time  : {(one_step_train_time_s_list[i] - forward_time_s_list[i])*1000:.3f} ms")


torch.distributed.destroy_process_group()
