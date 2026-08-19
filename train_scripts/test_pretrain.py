import warnings
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release"
)

#warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")

import os
import sys
import time
from einops import rearrange
from glob import glob
import argparse
import yaml
import csv
import json

import torch
import torch.distributed as dist
import torch.cuda.amp as amp

from core.rank_manager import ParallelManager
from core.runtime.run_model import one_step_train
from core.logging.logging import set_dp_rank_print_redirect
from core.global_env_config import ENABLE_TORCH_PROF, TORCH_PROF_step_list, MEM_PROF, USE_FAKE_INPUT, TORCH_PROF_WRITE_TRACE

from dataloader.dataloader_utils import (
    get_dataloader_for_task,
    resolve_required_tensors_online_split_for_parallel,
    resolve_required_tensors_from_dataloader,
)
from dataloader.task_specific_data import get_task_specific_data

from models.model_utils.get_model import get_model_for_train, get_model_archi_params_and_other_params, get_ranks_per_dp

from train_scripts.test_utils import process_loss_and_output_in_test, gather_scalar
from profiler.memory_timeline import CudaMemoryTimeline
from profiler.flops_profiler import TorchFlopsProfiler


from utils import init_distributed, set_random_seed, sort_key


test_iter_num = 20 #20 #20 #20 #10 #20
use_fake_input = USE_FAKE_INPUT
use_splited_data = False
show_gradient_flag = True


if use_fake_input:
    if use_splited_data:
        print('we do not support use_fake_input and use_splited_data')

        exit(0)


rank, local_rank, device, world_size = init_distributed()
set_random_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--data_parallel_group_size", type=int, required=True)
parser.add_argument("--model_cfg", required=True)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--metrics_csv", default=None)
parser.add_argument("--metrics_json", default=None)
parser.add_argument("--metrics_no_json", action="store_true")
parser.add_argument("--disable_torch_prof", action="store_true")
parser.add_argument("--quiet_metrics", action="store_true")
parser.add_argument("--disable_rank_log_redirect", action="store_true")
args = parser.parse_args()

if args.steps is not None:
    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}")
    test_iter_num = args.steps

if args.disable_torch_prof:
    ENABLE_TORCH_PROF = False

torch.distributed.barrier()

with open(args.model_cfg, 'r') as f:
    model_config = yaml.safe_load(f)


use_splited_data = bool(model_config.get('use_splited_data', use_splited_data))


task_type = model_config['task_type']
if task_type != 'glorys':
    raise ValueError("Unsupported task_type; only glorys is supported")


ranks_per_dp = get_ranks_per_dp(args.data_parallel_group_size, world_size)


model_archi_params, other_params = get_model_archi_params_and_other_params(task_type, model_config, data_parallel_group_size = args.data_parallel_group_size, world_size = world_size)


#other_params['use_splited_data'] = True
other_params['use_splited_data'] = use_splited_data


precision = model_config['precision']
assert precision in ['bf16', 'fp32', 'fp16']
if precision=='fp32':
    my_dtype = torch.float32
elif precision=='fp16':
    my_dtype = torch.float16
elif precision=='bf16':
    my_dtype = torch.bfloat16

norm_type = model_config['norm_type']
assert norm_type in ['zs', 'mm']

model_type = model_config['model_type']
assert model_type in ['sequential', 'parallel', 'hybrid']

micro_batch_size = model_config['micro_batch_size']

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
post_wrap_local_parameter_count = int(count_parameters(model))
total_params = post_wrap_local_parameter_count

def get_pre_wrap_parameter_count(model):
    if hasattr(model, "pre_wrap_parameter_count"):
        return int(model.pre_wrap_parameter_count)
    if hasattr(model, "module") and hasattr(model.module, "pre_wrap_parameter_count"):
        return int(model.module.pre_wrap_parameter_count)
    return post_wrap_local_parameter_count

pre_wrap_parameter_count = get_pre_wrap_parameter_count(model)


# 0 1 2 3 dp rank0
# 4 5 6 7 dp rank1
# 8 9 10 11 dp rank2
# 12 13 14 15 dp rank3

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


torch.distributed.barrier()

peak_memory_list = []
loss_list =[]
output_sum_list = []
output_tensor_dtype_list = []

one_step_train_time_s_list = []
timer_name_to_total_time_list = []

flops_profiler = TorchFlopsProfiler(
    enabled=ENABLE_TORCH_PROF,
    rank=rank,
    schedule_steps=TORCH_PROF_step_list,
    trace_dir="./log/profiler/rank0",
    write_trace=TORCH_PROF_WRITE_TRACE,
)
flops_profiler.start()


'''
memory_tracer = CudaMemoryTimeline(
    model = model,
    device = device,
    output_dir="./log/memory_timeline",
    rank=rank,
)
'''


def run_one_step():
    return one_step_train(
                    task_type,
                    model_type, # 'parallel', 'sequential',
                    model,
                    engine,
                    optimizer,

                    required_tensors,
                    task_specific_data_dict,
                    other_params,

                    precision,
                    model_config['half_model'],

                    my_dtype,
                    loss_fn,
                    gscaler,
                    manager = manager,
                    optimizer_state_tuple = optimizer_state_tuple,
                    memory_timeline = mem_trace,
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
    dataset_config=None,
)


task_specific_data_dict['embedding_parallel_type'] = other_params['embedding_parallel_type']
task_specific_data_dict['test_iter_num'] = test_iter_num
task_specific_data_dict['show_gradient_flag'] = show_gradient_flag


global_step = 0
num_pre_train_epochs = 300
from optimizer.utils import get_lr_by_global_step, set_lr


set_random_seed(1234)

if task_type == 'glorys':


    train_dataloader = get_dataloader_for_task(task_type, model_config['micro_batch_size'], use_splited_data, status=0, num_workers=8, simplified = False,
                                               manager=manager, model_type= model_type, model_archi_params= model_archi_params,
                                               other_params = other_params, use_fake_input = use_fake_input)
    test_dataloader = get_dataloader_for_task(task_type, model_config['micro_batch_size'], use_splited_data, status=1, num_workers=6, simplified = False,
                                              manager=manager, model_type=model_type, model_archi_params= model_archi_params,
                                              other_params = other_params, use_fake_input = use_fake_input)

    for i, x in enumerate(train_dataloader):
        if i==test_iter_num:
            break


        lr = get_lr_by_global_step(model_config['task_type'],
                                   global_step,
                                   model_config['learning_rate'],
                                   warmup_steps=10,
                                   num_pre_train_epochs = num_pre_train_epochs)


        if ZERO_STAGE_NUMBER is not None:
            set_lr(engine.optimizer, lr)
        else:
            set_lr(optimizer, lr)

        if task_type == 'glorys' and model_type != 'sequential' and (not use_splited_data):
            required_tensors = resolve_required_tensors_online_split_for_parallel(
                task_type,
                x,
                device,
                model_archi_params,
                other_params,
                manager,
                my_dtype,
                is_pretrain=True,
            )
        else:
            required_tensors = resolve_required_tensors_from_dataloader(task_type, x, device, task_specific_data_dict, my_dtype, is_pretrain = True)


        torch.cuda.reset_peak_memory_stats(device) # Count peak model memory after input transfer/splitting is complete.

        if MEM_PROF==0:
            mem_trace = None
            thetao_out, loss, one_step_train_time_s, _forward_time_s = run_one_step()
        elif MEM_PROF==1:
            with CudaMemoryTimeline(
                    model=model,
                    device=device,
                    output_dir="./log/memory_timeline",

                    model_type = model_type,
                    rank=rank,
                    dp_rank = dist.get_rank(group=manager.data_parallel_group),
                    wp_rank = manager.get_wp_rank(),
                    iter_idx=i,
                    name=model_config.get("description", task_type),

                    max_depth = None,
                    record_leaf_only=True,
                    module_filter=None,
                    synchronize=True,
                ) as mem_trace:
                thetao_out, loss, one_step_train_time_s, _forward_time_s = run_one_step()

        elif MEM_PROF==2:
            '''
            module_filter=[
                "DomainParallelCreditDownBlock",
                "DomainParallelCreditUpBlock",
                "Parallel_BasicLayer",
                "WindowParallel",
            ]
            '''
            with CudaMemoryTimeline(
                    model=model,
                    device=device,
                    output_dir="./log/memory_timeline",

                    model_type = model_type,
                    rank=rank,
                    dp_rank = dist.get_rank(group=manager.data_parallel_group),
                    wp_rank = manager.get_wp_rank(),
                    iter_idx=i,
                    name=model_config.get("description", task_type),

                    max_depth = 4,
                    record_leaf_only=False,
                    module_filter= None,
                    synchronize=False,
                ) as mem_trace:
                thetao_out, loss, one_step_train_time_s, _forward_time_s = run_one_step()


        flops_profiler.step()

        global_step = global_step + 1


        #loss, thetao_out = process_loss_and_output_in_test(loss, thetao_out)
        loss, thetao_out = process_loss_and_output_in_test(loss, thetao_out, model_type, other_params, manager)

        peak_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        peak_memory_list.append(peak_memory)

        output_tensor_dtype_list.append(thetao_out.dtype)
        output_sum_list.append(thetao_out.type(torch.float32).sum().item())
        loss_list.append(loss.item())

        one_step_train_time_s_list.append(one_step_train_time_s)


flops_profiler.stop()


if (rank % ranks_per_dp)==0:

    print('optimizer_config', model_config['optimizer_config'], 'precision', model_config['precision'], 'half_model', model_config['half_model'])
    print('rank', rank, 'dp_rank', dist.get_rank(group=manager.data_parallel_group), f"Total parameters: {total_params:,}")


if test_iter_num>2:
    all_mem_peak_0 = gather_scalar(peak_memory_list[0])
    all_mem_peak_1 = gather_scalar(peak_memory_list[1])


def gather_int64_scalar(x):
    t = torch.tensor(int(x), device=torch.cuda.current_device(), dtype=torch.int64)
    out = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(out, t)
    return [int(v.item()) for v in out]


all_parameters = gather_int64_scalar(post_wrap_local_parameter_count)
all_pre_wrap_parameters = gather_int64_scalar(pre_wrap_parameter_count)
all_mem_peak_last = gather_scalar(peak_memory_list[-1])
all_time_peak = gather_scalar(one_step_train_time_s_list[-1])
flops_summary = flops_profiler.summarize()

def infer_parameter_count_summary():
    tp_size = int(other_params.get("tensor_parallel_size", 1))
    fsdp_group_size = world_size if USE_FSDP else 1

    if USE_FSDP:
        global_parameter_count = max(all_pre_wrap_parameters)
        parameter_count_method = "pre_wrap_full_count_before_fsdp_wrap"
    elif tp_size > 1:
        global_parameter_count = None
        parameter_count_method = "tp_sharded_local_count_only"
    else:
        global_parameter_count = max(all_pre_wrap_parameters)
        parameter_count_method = "pre_wrap_replicated_full_count"

    return {
        "local_parameter_count": post_wrap_local_parameter_count,
        "pre_wrap_parameter_count": pre_wrap_parameter_count,
        "post_wrap_local_parameter_counts_by_rank": all_parameters,
        "pre_wrap_parameter_counts_by_rank": all_pre_wrap_parameters,
        "global_parameter_count": global_parameter_count,
        "parameter_count_method": parameter_count_method,
        "fsdp_group_size": fsdp_group_size,
        "tensor_parallel_size": tp_size,
    }

parameter_count_summary = infer_parameter_count_summary()


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
    padding_spec = model_archi_params.get("padding_spec", {}) or {}
    padding_policy = model_archi_params.get("padding_policy", padding_spec.get("policy"))
    padded_shape = model_archi_params.get("padded_shape", padding_spec.get("padded_shape"))
    initial_padding = model_archi_params.get("initial_padding", padding_spec.get("initial_padding"))
    need_padding = padding_spec.get("need_padding")
    patch_token_resolution = padding_spec.get("patch_token_resolution")
    transformer_token_resolution = padding_spec.get("transformer_token_resolution")
    num_windows = padding_spec.get("num_windows")
    transformer_downsample_scale = padding_spec.get("transformer_downsample_scale")
    input_shape = (model_archi_params.get("height"), model_archi_params.get("width"))
    padding_overhead_pct = None
    if padded_shape is not None and input_shape[0] is not None and input_shape[1] is not None:
        padding_overhead_pct = (
            (int(padded_shape[0]) * int(padded_shape[1]))
            / (int(input_shape[0]) * int(input_shape[1]))
            - 1.0
        ) * 100.0
    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "iter",
                    "loss",
                    "output_sum",
                    "output_dtype",
                    "peak_memory_mb",
                    "step_time_s",
                    "padding_policy",
                    "padding_scale",
                    "input_shape",
                    "need_padding",
                    "padded_shape",
                    "initial_padding",
                    "padding_overhead_pct",
                    "patch_token_resolution",
                    "transformer_token_resolution",
                    "num_windows",
                    "transformer_downsample_scale",
                ],
            )
            writer.writeheader()
            for i in range(row_count):
                writer.writerow({
                    "iter": i,
                    "loss": loss_list[i],
                    "output_sum": output_sum_list[i],
                    "output_dtype": str(output_tensor_dtype_list[i]),
                    "peak_memory_mb": peak_memory_list[i],
                    "step_time_s": one_step_train_time_s_list[i],
                    "padding_policy": padding_policy,
                    "padding_scale": model_archi_params.get("padding_scale"),
                    "input_shape": str(input_shape),
                    "need_padding": need_padding,
                    "padded_shape": str(padded_shape),
                    "initial_padding": str(initial_padding),
                    "padding_overhead_pct": padding_overhead_pct,
                    "patch_token_resolution": str(patch_token_resolution),
                    "transformer_token_resolution": str(transformer_token_resolution),
                    "num_windows": str(num_windows),
                    "transformer_downsample_scale": transformer_downsample_scale,
                })

    if json_path is None:
        if csv_path is not None:
            print(f"[test_pretrain] wrote metrics csv to {csv_path}")
        return

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    metadata = {
        "model_cfg": os.path.abspath(args.model_cfg),
        "steps": test_iter_num,
        "world_size": world_size,
        "data_parallel_group_size": args.data_parallel_group_size,
        "ranks_per_dp": ranks_per_dp,
        "task_type": task_type,
        "model_architecture": model_config.get("model_architecture"),
        "model_type": model_type,
        "optimizer_config": model_config.get("optimizer_config"),
        "precision": model_config.get("precision"),
        "micro_batch_size": model_config.get("micro_batch_size"),
        "learning_rate": model_config.get("learning_rate"),
        "wp_topo": other_params.get("wp_topo"),
        "xfmr_wp_topo": other_params.get("xfmr_wp_topo"),
        "xfmr_sp_size": other_params.get("xfmr_sp_size"),
        "tensor_parallel_size": other_params.get("tensor_parallel_size"),
        "sp_tp_placement": other_params.get("sp_tp_placement"),
        "window_assignment_mode": other_params.get("window_assignment_mode"),
        "num_layers": model_config.get("num_layers"),
        "embedding_dim": model_config.get("embedding_dim"),
        "num_heads": model_config.get("num_heads"),
        "patch_size": model_config.get("patch_size"),
        "window_size": model_config.get("window_size"),
        "padding_scale": model_config.get("padding_scale"),
        "padding_policy": model_archi_params.get("padding_policy"),
        "input_shape": input_shape,
        "need_padding": padding_spec.get("need_padding"),
        "padded_shape": model_archi_params.get("padded_shape"),
        "initial_padding": model_archi_params.get("initial_padding"),
        "padding_overhead_pct": padding_overhead_pct,
        "patch_token_resolution": padding_spec.get("patch_token_resolution"),
        "transformer_token_resolution": padding_spec.get("transformer_token_resolution"),
        "num_windows": padding_spec.get("num_windows"),
        "transformer_downsample_scale": padding_spec.get("transformer_downsample_scale"),
        "metrics_csv": csv_path,
        "parameter_count": parameter_count_summary,
        "local_parameter_count": parameter_count_summary["local_parameter_count"],
        "pre_wrap_parameter_count": parameter_count_summary["pre_wrap_parameter_count"],
        "global_parameter_count": parameter_count_summary["global_parameter_count"],
        "parameter_count_method": parameter_count_summary["parameter_count_method"],
        "post_wrap_local_parameter_counts_by_rank": parameter_count_summary["post_wrap_local_parameter_counts_by_rank"],
        "pre_wrap_parameter_counts_by_rank": parameter_count_summary["pre_wrap_parameter_counts_by_rank"],
        "fsdp_group_size": parameter_count_summary["fsdp_group_size"],
        "flops_profiler_enabled": flops_summary.enabled,
        "flops_profiler_active_steps": flops_summary.active_steps,
        "flops_profiler_event_count": flops_summary.event_count,
        "flops_profiler_local_flops_per_step": flops_summary.local_flops_per_step,
        "flops_profiler_world_flops_per_step": flops_summary.world_flops_per_step,
        "flops_profiler_world_avg_flops_per_step": flops_summary.world_avg_flops_per_step,
    }
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if csv_path is not None:
        print(f"[test_pretrain] wrote metrics csv to {csv_path}")
    print(f"[test_pretrain] wrote metrics json to {json_path}")


write_test_metrics()


if (rank % ranks_per_dp)==0:
    print('all_parameters', all_parameters)
    print('all_pre_wrap_parameters', all_pre_wrap_parameters)
    print('parameter_count_summary', parameter_count_summary)
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
    for i in range(0, test_iter_num):
        print('iter', i, output_tensor_dtype_list[i], "sum:", output_sum_list[i], 'loss', loss_list[i])
    print('-----------------------------Per-step memory------------------------------')
    for i in range(0, test_iter_num):
        print('iter', i, 'peak_memory', peak_memory_list[i])
    print('-----------------------------Per-step time and FLOPS------------------------------')#
    for i in range(0, test_iter_num):
        print('iter', i, f"Total Time  : {one_step_train_time_s_list[i]*1000:.3f} ms", f"FLOPS : {total_FLOPs/one_step_train_time_s_list[i]/1e12:.3f} TFLOPS")


torch.distributed.destroy_process_group()
