import argparse
import os
import pickle
import warnings

import torch
import torch.distributed as dist
import yaml

from core.rank_manager import ParallelManager
from core.runtime.run_model import one_step_train_production
from dataloader.dataloader_utils import (
    get_dataloader_for_task,
    manually_split_data_for_parallel_training,
    resolve_required_tensors_online_split_for_parallel,
    resolve_required_tensors_from_dataloader,
)
from dataloader.task_specific_data import get_task_specific_data
from models.model_utils.get_model import (
    get_model_archi_params_and_other_params,
    get_model_for_train,
)
from optimizer.utils import get_lr_by_global_step, set_lr
from utils import get_criterion, init_distributed, set_random_seed


warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")


def _dtype_from_precision(precision):
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise RuntimeError(f"unsupported precision: {precision}")


def _set_default_config(model_config, model_cfg_path):
    model_config.setdefault("description", os.path.splitext(os.path.basename(model_cfg_path))[0])
    model_config.setdefault("train_path", "./runs/pretrain")
    model_config.setdefault("ckpt_start", False)
    model_config.setdefault("ckpt_epoch", 0)
    model_config.setdefault("start_epoch", 0)
    model_config.setdefault("num_pre_train_epochs", 300)
    model_config.setdefault("loss_func", "L1")
    model_config.setdefault("clip_grad", False)
    model_config.setdefault("use_splited_data", True)
    model_config.setdefault("use_fake_input", False)
    model_config.setdefault("save_interval_epochs", 1)
    model_config.setdefault("warmup_steps", 10)
    return model_config


def _checkpoint_root(model_config):
    return os.path.join(model_config["train_path"], model_config["description"], "ckpt")


def _checkpoint_kind(model_type, optimizer_state_tuple):
    _use_ddp, use_fsdp, _zero_stage, use_dist_opt = optimizer_state_tuple
    if use_fsdp:
        return "fsdp"
    if use_dist_opt:
        return "rank_local"
    if model_type in ("parallel", "hybrid"):
        return "wp_rank"
    return "single"


def _checkpoint_dir(model_config, model_type, manager, optimizer_state_tuple):
    root = _checkpoint_root(model_config)
    kind = _checkpoint_kind(model_type, optimizer_state_tuple)
    if kind == "fsdp":
        fsdp_ckpt_type = model_config.get("fsdp_checkpoint_type", "full")
        if fsdp_ckpt_type == "sharded":
            return os.path.join(root, "fsdp_sharded", f"rank_{manager.get_rank()}")
        return root
    if kind == "rank_local":
        return os.path.join(root, f"rank_{manager.get_rank()}")
    if kind == "wp_rank":
        return os.path.join(root, f"wp_rank_{manager.get_wp_rank()}")
    return root


def _checkpoint_path(model_config, model_type, manager, optimizer_state_tuple, epoch):
    return os.path.join(
        _checkpoint_dir(model_config, model_type, manager, optimizer_state_tuple),
        f"epoch{epoch}.pth",
    )


def _should_save_checkpoint(model_type, manager, optimizer_state_tuple):
    _use_ddp, use_fsdp, _zero_stage, use_dist_opt = optimizer_state_tuple
    if use_fsdp:
        return True
    if use_dist_opt:
        return True
    if model_type in ("parallel", "hybrid"):
        return manager.get_dp_rank() == 0
    return manager.get_rank() == 0


def _strip_or_add_module_prefix(model, state_dict):
    model_state = model.state_dict()
    model_has_module_prefix = next(iter(model_state)).startswith("module.")
    out = {}
    for key, value in state_dict.items():
        if key.startswith("module.") and not model_has_module_prefix:
            out[key[7:]] = value
        elif (not key.startswith("module.")) and model_has_module_prefix:
            out["module." + key] = value
        else:
            out[key] = value
    return out


def _optimizer_for_state(optimizer, engine, zero_stage_number):
    return engine.optimizer if zero_stage_number is not None else optimizer


def _fsdp_optim_state_to_load(model, optimizer, optim_state):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if hasattr(FSDP, "optim_state_dict_to_load"):
        return FSDP.optim_state_dict_to_load(model, optimizer, optim_state)
    return FSDP.shard_full_optim_state_dict(optim_state, model, optim=optimizer)


def _save_fsdp_checkpoint(model, optimizer, gscaler, loss, epoch, global_step, path, model_config, manager):
    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        StateDictType,
    )

    fsdp_ckpt_type = model_config.get("fsdp_checkpoint_type", "full")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if fsdp_ckpt_type == "sharded":
        try:
            from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig
        except ImportError as exc:
            raise RuntimeError("fsdp_checkpoint_type='sharded' requires ShardedStateDictConfig support") from exc

        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
            ShardedOptimStateDictConfig(offload_to_cpu=True),
        ):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
        _write_checkpoint(path, model_state, optim_state, gscaler, loss, epoch, global_step, "fsdp_sharded")
        return

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)

    if manager.get_rank() == 0:
        _write_checkpoint(path, model_state, optim_state, gscaler, loss, epoch, global_step, "fsdp_full")


def _load_fsdp_checkpoint(model, optimizer, gscaler, path, model_config, map_location):
    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        StateDictType,
    )

    fsdp_ckpt_type = model_config.get("fsdp_checkpoint_type", "full")
    checkpoint = torch.load(path, map_location=map_location)

    if fsdp_ckpt_type == "sharded":
        try:
            from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig
        except ImportError as exc:
            raise RuntimeError("fsdp_checkpoint_type='sharded' requires ShardedStateDictConfig support") from exc

        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
            ShardedOptimStateDictConfig(offload_to_cpu=True),
        ):
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(
                _fsdp_optim_state_to_load(model, optimizer, checkpoint["optimizer_state_dict"])
            )
    else:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(
                _fsdp_optim_state_to_load(model, optimizer, checkpoint["optimizer_state_dict"])
            )

    if gscaler is not None and "scaler" in checkpoint:
        gscaler.load_state_dict(checkpoint["scaler"])
    optimizer.global_step = checkpoint["global_step"]
    return checkpoint


def _write_checkpoint(path, model_state, optim_state, gscaler, loss, epoch, global_step, ckpt_layout):
    checkpoint = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optim_state,
        "loss": float(loss.detach().cpu()) if torch.is_tensor(loss) else float(loss),
        "epoch": epoch,
        "global_step": global_step,
        "ckpt_layout": ckpt_layout,
    }
    if gscaler is not None:
        checkpoint["scaler"] = gscaler.state_dict()
    torch.save(checkpoint, path)


def _save_checkpoint(model, optimizer, engine, gscaler, loss, epoch, model_config, model_type, manager, optimizer_state_tuple):
    use_ddp, use_fsdp, zero_stage_number, use_dist_opt = optimizer_state_tuple
    path = _checkpoint_path(model_config, model_type, manager, optimizer_state_tuple, epoch)

    if use_fsdp:
        _save_fsdp_checkpoint(model, optimizer, gscaler, loss, epoch, optimizer.global_step, path, model_config, manager)
        dist.barrier()
        if manager.get_rank() == 0:
            print(f"saved FSDP checkpoint: {path}")
        return

    if not _should_save_checkpoint(model_type, manager, optimizer_state_tuple):
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    active_optimizer = _optimizer_for_state(optimizer, engine, zero_stage_number)
    model_state = model.state_dict()
    optim_state = active_optimizer.state_dict()
    layout = "rank_local" if use_dist_opt else ("wp_rank" if model_type in ("parallel", "hybrid") else "single")
    _write_checkpoint(path, model_state, optim_state, gscaler, loss, epoch, active_optimizer.global_step, layout)
    print(f"saved checkpoint: {path}")


def _load_checkpoint_if_requested(model, optimizer, engine, gscaler, model_config, model_type, manager, optimizer_state_tuple, device):
    if not model_config.get("ckpt_start", False):
        return model_config.get("start_epoch", 0)

    use_ddp, use_fsdp, zero_stage_number, _use_dist_opt = optimizer_state_tuple
    path = _checkpoint_path(model_config, model_type, manager, optimizer_state_tuple, model_config["ckpt_epoch"])

    if manager.get_rank() == 0:
        print(f"loading checkpoint: {path}")

    if use_fsdp:
        checkpoint = _load_fsdp_checkpoint(model, optimizer, gscaler, path, model_config, "cpu")
    else:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(_strip_or_add_module_prefix(model, checkpoint["model_state_dict"]))
        active_optimizer = _optimizer_for_state(optimizer, engine, zero_stage_number)
        active_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        active_optimizer.global_step = checkpoint["global_step"]
        if gscaler is not None and "scaler" in checkpoint:
            gscaler.load_state_dict(checkpoint["scaler"])

    start_epoch = checkpoint["epoch"] + 1
    if manager.get_rank() == 0:
        print("resumed epoch:", start_epoch)
        print("resumed global_step:", checkpoint["global_step"])
        print("resumed loss:", checkpoint.get("loss"))
    dist.barrier()
    return start_epoch


def _load_or_make_task_specific_data(task_type, device, optimizer_state_tuple, model_archi_params, other_params, manager, model_config, model_type, rank):
    if task_type == "glorys":
        task_data = get_task_specific_data(
            task_type,
            device,
            optimizer_state_tuple,
            model_archi_params,
            other_params,
            manager,
            micro_batch_size=model_config["micro_batch_size"],
            model_type=model_type,
            dataset_config=None,
        )
    else:
        saved_root = model_config.get(
            "saved_data_path",
            os.path.join(model_config["train_path"], model_config["description"], "saved_data_path"),
        )
        rank_path = os.path.join(saved_root, f"rank_{rank}")
        if model_config.get("load_saved_data_dict", False):
            with open(os.path.join(rank_path, "task_specific_data_dict.pkl"), "rb") as f:
                task_data = pickle.load(f)
            print(f"Rank {rank} loaded data from {rank_path}")
        else:
            task_data = get_task_specific_data(
                task_type,
                device,
                optimizer_state_tuple,
                model_archi_params,
                other_params,
                manager,
                micro_batch_size=model_config["micro_batch_size"],
                model_type=model_type,
                dataset_config=None,
            )
            if model_config.get("save_task_specific_data_dict", True):
                os.makedirs(rank_path, exist_ok=True)
                with open(os.path.join(rank_path, "task_specific_data_dict.pkl"), "wb") as f:
                    pickle.dump(task_data, f)
                print(f"Rank {rank} saved data to {rank_path}")

    task_data["test_iter_num"] = -1
    task_data["show_gradient_flag"] = False
    return task_data


def _prepare_required_tensors(x, task_type, model_type, device, task_data, model_archi_params, other_params, manager, my_dtype, use_splited_data):
    if task_type == "glorys" and model_type != "sequential" and not use_splited_data:
        return resolve_required_tensors_online_split_for_parallel(
            task_type,
            x,
            device,
            model_archi_params,
            other_params,
            manager,
            my_dtype,
            is_pretrain=True,
        )

    required_tensors = resolve_required_tensors_from_dataloader(
        task_type,
        x,
        device,
        task_data,
        my_dtype,
        is_pretrain=True,
    )
    if model_type != "sequential" and not use_splited_data:
        manually_split_data_for_parallel_training(
            required_tensors,
            task_type,
            model_archi_params,
            my_dtype,
            other_params,
            manager,
            data_format="NCHW",
            padding_scale=model_archi_params["padding_scale"],
        )
    return required_tensors


def _log_train_step(log_file, rank, message_parts):
    if rank != 0:
        return
    print(*message_parts, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        print(*message_parts, file=f)


def main():
    rank, local_rank, device, world_size = init_distributed()
    set_random_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_parallel_group_size", type=int, required=True)
    parser.add_argument("--model_cfg", required=True)
    args = parser.parse_args()

    with open(args.model_cfg, "r", encoding="utf-8") as f:
        model_config = _set_default_config(yaml.safe_load(f), args.model_cfg)

    task_type = model_config["task_type"]
    model_type = model_config["model_type"]
    precision = model_config["precision"]
    my_dtype = _dtype_from_precision(precision)

    model_archi_params, other_params = get_model_archi_params_and_other_params(
        task_type,
        model_config,
        data_parallel_group_size=args.data_parallel_group_size,
        world_size=world_size,
    )

    manager = ParallelManager(
        dp_size=args.data_parallel_group_size,
        mp_size=other_params["mp_size"],
        wp_topo=other_params["wp_topo"],
        xfmr_wp_topo=other_params["xfmr_wp_topo"],
        domain_topo=other_params["domain_topo"],
        rank=rank,
        world_size=world_size,
        device=device,
        window_assignment_mode=other_params.get("window_assignment_mode", "regular"),
        xfmr_sp_size=other_params.get("xfmr_sp_size", 1),
        tensor_parallel_size=other_params.get("tensor_parallel_size", 1),
        sp_tp_placement=other_params.get("sp_tp_placement", "tp_first"),
    )

    model, engine, optimizer, gscaler, optimizer_state_tuple = get_model_for_train(
        model_archi_params,
        precision,
        model_config["half_model"],
        model_config["optimizer_config"],
        task_type=task_type,
        kaiming_init=model_config["kaiming_init"],
        model_type=model_type,
        model_architecture=model_config["model_architecture"],
        embedding_parallel_type=other_params["embedding_parallel_type"],
        attn_parallel_type=other_params["attn_parallel_type"],
        mlp_parallel_type=other_params["mlp_parallel_type"],
        manager=manager,
        device=device,
        local_rank=local_rank,
        learning_rate=model_config["learning_rate"],
    )
    use_ddp, use_fsdp, zero_stage_number, use_dist_opt = optimizer_state_tuple

    if rank == 0:
        print("training description:", model_config["description"])
        print("task/model:", task_type, model_config["model_architecture"], model_type)
        print(
            "optimizer_config:",
            model_config["optimizer_config"],
            "DDP/FSDP/Zero/DistOpt:",
            use_ddp,
            use_fsdp,
            zero_stage_number,
            use_dist_opt,
        )
        print(
            "parallel:",
            "wp_topo=", other_params.get("wp_topo"),
            "xfmr_wp_topo=", other_params.get("xfmr_wp_topo"),
            "sp=", other_params.get("xfmr_sp_size"),
            "tp=", other_params.get("tensor_parallel_size"),
        )

    criterion = get_criterion(model_config["loss_func"])
    task_data = _load_or_make_task_specific_data(
        task_type,
        device,
        optimizer_state_tuple,
        model_archi_params,
        other_params,
        manager,
        model_config,
        model_type,
        rank,
    )

    use_splited_data = bool(model_config.get("use_splited_data", True))
    use_fake_input = bool(model_config.get("use_fake_input", False))
    num_workers = int(model_config.get("num_workers", model_config.get("dataloader_num_workers", 8)))
    train_dataloader = get_dataloader_for_task(
        task_type,
        model_config["micro_batch_size"],
        use_splited_data=use_splited_data,
        status=0,
        num_workers=num_workers,
        simplified=False,
        manager=manager,
        model_type=model_type,
        model_archi_params=model_archi_params,
        other_params=other_params,
        use_fake_input=use_fake_input,
    )

    loss_dir = os.path.join(model_config["train_path"], model_config["description"], "loss")
    if rank == 0:
        os.makedirs(loss_dir, exist_ok=True)
        os.makedirs(_checkpoint_root(model_config), exist_ok=True)
    dist.barrier()
    log_file = os.path.join(loss_dir, "rank0.txt")

    start_epoch = _load_checkpoint_if_requested(
        model,
        optimizer,
        engine,
        gscaler,
        model_config,
        model_type,
        manager,
        optimizer_state_tuple,
        device,
    )

    num_epochs = int(model_config["num_pre_train_epochs"])
    save_interval = int(model_config.get("save_interval_epochs", 1))
    last_loss = None

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_dataloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            active_optimizer = _optimizer_for_state(optimizer, engine, zero_stage_number)
            lr = get_lr_by_global_step(
                task_type,
                active_optimizer.global_step,
                model_config["learning_rate"],
                warmup_steps=int(model_config.get("warmup_steps", 10)),
                num_pre_train_epochs=num_epochs,
            )
            set_lr(active_optimizer, lr)

            required_tensors = _prepare_required_tensors(
                batch,
                task_type,
                model_type,
                device,
                task_data,
                model_archi_params,
                other_params,
                manager,
                my_dtype,
                use_splited_data,
            )

            _output, loss, step_time_s, forward_time_s = one_step_train_production(
                task_type,
                model_type,
                model,
                engine,
                optimizer,
                required_tensors,
                task_data,
                other_params,
                precision,
                model_config["half_model"],
                my_dtype,
                criterion,
                gscaler,
                manager=manager,
                optimizer_state_tuple=optimizer_state_tuple,
                memory_timeline=None,
                clip_grad=bool(model_config.get("clip_grad", False)),
            )
            last_loss = loss.detach()

            log_parts = [
                model_config["description"],
                "epoch",
                epoch,
                "train step:",
                step,
                "loss",
                loss.item(),
                "lr:",
                f"{active_optimizer.param_groups[0]['lr']:.8f}",
                "train_gs",
                active_optimizer.global_step,
                "step_time_s",
                f"{step_time_s:.4f}",
                "forward_time_s",
                f"{forward_time_s:.4f}",
            ]
            if gscaler is not None:
                log_parts.extend(["scale", gscaler.get_scale()])
            _log_train_step(log_file, rank, log_parts)

        if last_loss is not None and (save_interval > 0) and ((epoch + 1) % save_interval == 0):
            _save_checkpoint(
                model,
                optimizer,
                engine,
                gscaler,
                last_loss,
                epoch,
                model_config,
                model_type,
                manager,
                optimizer_state_tuple,
            )
            dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
