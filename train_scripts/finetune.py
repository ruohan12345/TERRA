import os
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
import yaml

from core.rank_manager import ParallelManager
from core.runtime.finetune_one_step import (
    run_one_step_for_finetune_train,
)
from core.runtime.run_model import make_global_report_loss
from dataloader.dataloader_utils import (
    get_finetune_dataloader_for_task,
    resolve_required_tensors_online_split_for_parallel,
    resolve_required_tensors_from_dataloader,
)
from dataloader.task_specific_data import get_task_specific_data
from models.model_utils.get_model import (
    get_model_archi_params_and_other_params,
    get_model_for_train,
    get_ranks_per_dp,
)
from optimizer.utils import get_finetune_lr_by_global_step, set_lr
from train_scripts.train import _load_fsdp_checkpoint, _save_fsdp_checkpoint
from utils import get_criterion, init_distributed, set_random_seed


warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release",
)


def _dtype_from_precision(precision):
    if precision == 'fp32':
        return torch.float32
    if precision == 'fp16':
        return torch.float16
    if precision == 'bf16':
        return torch.bfloat16
    raise RuntimeError(f"unsupported precision: {precision}")


def _checkpoint_dir(model_config, model_type, manager):
    base = os.path.join(model_config['train_path'], model_config['description'], 'ckpt')
    if model_type == 'parallel' or model_type == 'hybrid':
        return os.path.join(base, f"wp_rank_{manager.get_wp_rank()}")
    return base


def _checkpoint_dir_with_wp_rank(model_config, model_type, wp_rank):
    base = os.path.join(model_config['train_path'], model_config['description'], 'ckpt')
    if model_type == 'parallel' or model_type == 'hybrid':
        return os.path.join(base, f"wp_rank_{wp_rank}")
    return base


def _finetune_output_dir(model_config):
    return os.path.join(
        model_config['train_path'],
        model_config['description'],
        f"lead_{model_config['finetune_lead_time']}",
    )


def _finetune_checkpoint_dir(model_config, model_type, manager, optimizer_state_tuple=None):
    base = os.path.join(_finetune_output_dir(model_config), 'ckpt')
    if optimizer_state_tuple is not None:
        _use_ddp, use_fsdp, _zero_stage, _use_dist_opt = optimizer_state_tuple
        if use_fsdp:
            if model_config.get("fsdp_checkpoint_type", "full") == "sharded":
                return os.path.join(base, "fsdp_sharded", f"rank_{manager.get_rank()}")
            return base
    if model_type == 'parallel' or model_type == 'hybrid':
        return os.path.join(base, f"wp_rank_{manager.get_wp_rank()}")
    return base


def _load_state_dict_compat(model, state_dict):
    new_state_dict = _strip_or_add_module_prefix(model, state_dict)
    model.load_state_dict(new_state_dict)


def _strip_or_add_module_prefix(model, state_dict):
    model_state = model.state_dict()
    model_has_module_prefix = next(iter(model_state)).startswith("module.")
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module.") and not model_has_module_prefix:
            new_state_dict[key[7:]] = value
        elif (not key.startswith("module.")) and model_has_module_prefix:
            new_state_dict["module." + key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def _filter_pretrained_state_dict(model, state_dict, skipped_suffixes):
    model_state = model.state_dict()
    filtered_state_dict = {}
    skipped_keys = []
    missing_in_model = []
    shape_mismatch = []

    for key, value in state_dict.items():
        if any(key.endswith(suffix) for suffix in skipped_suffixes):
            skipped_keys.append(key)
            continue
        if key not in model_state:
            missing_in_model.append(key)
            continue
        if hasattr(value, "shape") and hasattr(model_state[key], "shape") and value.shape != model_state[key].shape:
            shape_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered_state_dict[key] = value

    return filtered_state_dict, skipped_keys, missing_in_model, shape_mismatch


def _pretrained_checkpoint_path(model_config, manager):
    explicit_path = model_config.get('trained_model_checkpoint_path', None)
    if explicit_path:
        return explicit_path

    trained_model_config_path = model_config.get('trained_model_config_path', None)

    if not trained_model_config_path:
        return None

    with open(trained_model_config_path, 'r') as f:
        pretrained_config = yaml.safe_load(f)

    epoch = model_config.get('trained_model_epoch', pretrained_config.get('ckpt_epoch', 200))
    pretrained_model_type = pretrained_config['model_type']
    pretrained_optimizer_config = int(pretrained_config.get('optimizer_config', -1))
    pretrained_ckpt_root = os.path.join(
        pretrained_config['train_path'],
        pretrained_config['description'],
        'ckpt',
    )

    if pretrained_optimizer_config in (5, 7, 8, 9):
        if pretrained_config.get("fsdp_checkpoint_type", "full") == "sharded":
            pretrained_ckpt_root = os.path.join(
                pretrained_ckpt_root,
                "fsdp_sharded",
                f"rank_{manager.get_rank()}",
            )
        return os.path.join(pretrained_ckpt_root, f"epoch{epoch}.pth")

    if pretrained_optimizer_config == 6:
        return os.path.join(
            pretrained_ckpt_root,
            f"rank_{manager.get_rank()}",
            f"epoch{epoch}.pth",
        )

    if model_config.get('load_pretrained_from_wp_rank0', True):
        ckpt_dir = _checkpoint_dir_with_wp_rank(pretrained_config, pretrained_model_type, 0)
    else:
        ckpt_dir = _checkpoint_dir(pretrained_config, pretrained_model_type, manager)


    return os.path.join(ckpt_dir, f"epoch{epoch}.pth")


def _fsdp_full_state_context(model):
    try:
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )
    except ImportError:
        return nullcontext()

    if not isinstance(model, FSDP):
        return nullcontext()

    return FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )


def _load_pretrained_if_requested(model, model_config, manager, device, rank):
    ckpt_path = _pretrained_checkpoint_path(model_config, manager)
    if ckpt_path is None:
        if rank == 0:
            print('No pretrained checkpoint is provided; finetune starts from current model initialization.')
        return

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    skipped_suffixes = model_config.get('skip_pretrained_state_suffixes', ['attn_mask'])
    with _fsdp_full_state_context(model):
        pretrained_state_dict = _strip_or_add_module_prefix(model, checkpoint['model_state_dict'])
        pretrained_state_dict, skipped_keys, missing_in_model, shape_mismatch = _filter_pretrained_state_dict(
            model,
            pretrained_state_dict,
            skipped_suffixes,
        )
        load_result = model.load_state_dict(pretrained_state_dict, strict=False)
    if rank == 0:
        print('loaded pretrained checkpoint:', ckpt_path)
        print('skipped pretrained state suffixes:', skipped_suffixes)
        print('skipped pretrained keys:', skipped_keys[:20], 'num_skipped:', len(skipped_keys))
        if missing_in_model:
            print('pretrained keys missing in current model:', missing_in_model[:20], 'num_missing:', len(missing_in_model))
        if shape_mismatch:
            print('pretrained shape mismatch keys:', shape_mismatch[:20], 'num_shape_mismatch:', len(shape_mismatch))
        print('load_state missing keys:', load_result.missing_keys[:20], 'num_missing:', len(load_result.missing_keys))
        print('load_state unexpected keys:', load_result.unexpected_keys[:20], 'num_unexpected:', len(load_result.unexpected_keys))
    dist.barrier()


def _load_resume_checkpoint_if_requested(
        model,
        optimizer,
        engine,
        gscaler,
        model_config,
        ckpt_path,
        optimizer_state_tuple,
        device,
        rank,
        steps_per_epoch,
        ):
    if not model_config.get('ckpt_start', False):
        return model_config.get('start_epoch', 0), 0, 0

    USE_DDP, USE_FSDP, ZERO_STAGE_NUMBER, USE_DIST_OPT = optimizer_state_tuple
    resume_by_step = model_config.get('ckpt_step', None) is not None
    if resume_by_step:
        saved_ckpt_path = os.path.join(ckpt_path, f"step{model_config['ckpt_step']}.pth")
    else:
        saved_ckpt_path = os.path.join(ckpt_path, f"epoch{model_config['ckpt_epoch']}.pth")

    if USE_FSDP:
        checkpoint = _load_fsdp_checkpoint(model, optimizer, gscaler, saved_ckpt_path, model_config, "cpu")
    else:
        checkpoint = torch.load(saved_ckpt_path, map_location=device)
        _load_state_dict_compat(model, checkpoint['model_state_dict'])

    if USE_FSDP:
        optimizer.global_step = checkpoint['global_step']
    elif ZERO_STAGE_NUMBER is not None:
        engine.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        engine.optimizer.global_step = checkpoint['global_step']
    else:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.global_step = checkpoint['global_step']

    if gscaler is not None and 'scaler' in checkpoint:
        gscaler.load_state_dict(checkpoint['scaler'])

    global_step = checkpoint['global_step']
    if resume_by_step:
        if steps_per_epoch <= 0:
            raise RuntimeError(f"invalid steps_per_epoch for finetune step resume: {steps_per_epoch}")
        start_epoch = global_step // steps_per_epoch
        resume_step_in_epoch = global_step % steps_per_epoch
    else:
        start_epoch = checkpoint['epoch'] + 1
        resume_step_in_epoch = 0
    if rank == 0:
        print('resumed finetune checkpoint:', saved_ckpt_path)
        print('resume start_epoch:', start_epoch)
        print('resume skip_steps_in_epoch:', resume_step_in_epoch)
        print('resume global_step:', global_step)
        print('ckpt loss:', checkpoint['loss'])
    return start_epoch, global_step, resume_step_in_epoch


def _save_checkpoint(
        model,
        optimizer,
        engine,
        gscaler,
        loss,
        epoch,
        global_step,
        ckpt_path,
        optimizer_state_tuple,
        manager,
        model_config,
        checkpoint_name=None,
        ):
    USE_DDP, USE_FSDP, ZERO_STAGE_NUMBER, USE_DIST_OPT = optimizer_state_tuple
    checkpoint_name = checkpoint_name or f"epoch{epoch}.pth"
    checkpoint_path = os.path.join(ckpt_path, checkpoint_name)
    if USE_FSDP:
        _save_fsdp_checkpoint(model, optimizer, gscaler, loss, epoch, global_step, checkpoint_path, model_config, manager)
        torch.distributed.barrier()
        if manager.get_rank() == 0:
            print(f"saved FSDP finetune checkpoint: {checkpoint_path}")
        return

    if manager.get_dp_rank() != 0:
        return

    if ZERO_STAGE_NUMBER is not None:
        optimizer_state = engine.optimizer.state_dict()
    else:
        optimizer_state = optimizer.state_dict()

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_state,
        'loss': loss,
        'epoch': epoch,
        'global_step': global_step,
    }
    if gscaler is not None:
        checkpoint['scaler'] = gscaler.state_dict()
    torch.save(checkpoint, checkpoint_path)


def main():
    rank, local_rank, device, world_size = init_distributed()
    set_random_seed(1234)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_parallel_group_size", type=int, required=True)
    parser.add_argument("--model_cfg", required=True)
    args = parser.parse_args()

    with open(args.model_cfg, 'r') as f:
        model_config = yaml.safe_load(f)

    model_config.setdefault('description', os.path.splitext(os.path.basename(args.model_cfg))[0])
    model_config.setdefault('train_path', './runs/finetune')
    model_config.setdefault('ckpt_start', False)
    model_config.setdefault('ckpt_epoch', 0)
    model_config.setdefault('start_epoch', 0)
    model_config.setdefault('num_finetune_epochs', model_config.get('num_pre_train_epochs', 300))
    model_config.setdefault('loss_func', 'L1')
    model_config.setdefault('clip_grad', False)


    model_config.setdefault('clip_grad', False)
    model_config.setdefault('use_splited_data', True)
    model_config.setdefault('use_fake_input', False)
    model_config.setdefault('finetune_lead_time', 1)
    model_config.setdefault('finetune_loss_reduction', 'mean')
    model_config.setdefault('save_interval_steps', 50)
    model_config.setdefault('max_train_steps_per_epoch', None)
    if model_config['finetune_loss_reduction'] not in ('mean', 'sum'):
        raise ValueError(f"unsupported finetune_loss_reduction: {model_config['finetune_loss_reduction']}")
    model_config.setdefault('load_pretrained_from_wp_rank0', True)


    task_type = model_config['task_type']
    if task_type != 'glorys':
        raise ValueError("Unsupported task_type; only glorys is supported")
    model_type = model_config['model_type']
    precision = model_config['precision']
    my_dtype = _dtype_from_precision(precision)

    ranks_per_dp = get_ranks_per_dp(args.data_parallel_group_size, world_size)
    model_archi_params, other_params = get_model_archi_params_and_other_params(
        task_type,
        model_config,
        data_parallel_group_size=args.data_parallel_group_size,
        world_size=world_size,
    )
    other_params['use_splited_data'] = model_config['use_splited_data']

    manager = ParallelManager(
        dp_size=args.data_parallel_group_size,
        mp_size=other_params['mp_size'],
        wp_topo=other_params['wp_topo'],
        xfmr_wp_topo = other_params['xfmr_wp_topo'],
        domain_topo=other_params['domain_topo'],
        rank=rank,
        world_size=world_size,
        device=device,
        window_assignment_mode=other_params.get('window_assignment_mode', 'regular'),
        xfmr_sp_size=other_params.get('xfmr_sp_size', 1),
        tensor_parallel_size=other_params.get('tensor_parallel_size', 1),
        sp_tp_placement=other_params.get('sp_tp_placement', 'tp_first'),
    )


    model, engine, optimizer, gscaler, optimizer_state_tuple = get_model_for_train(
        model_archi_params,
        precision,
        model_config['half_model'],
        model_config['optimizer_config'],
        task_type=task_type,
        kaiming_init=model_config['kaiming_init'],
        model_type=model_type,
        model_architecture=model_config['model_architecture'],
        embedding_parallel_type=other_params['embedding_parallel_type'],
        attn_parallel_type=other_params['attn_parallel_type'],
        mlp_parallel_type=other_params['mlp_parallel_type'],
        manager=manager,
        device=device,
        local_rank=local_rank,
        learning_rate=model_config['learning_rate'],
    )


    USE_DDP, USE_FSDP, ZERO_STAGE_NUMBER, USE_DIST_OPT = optimizer_state_tuple
    criterion = get_criterion(model_config['loss_func'])

    if model_config.get('ckpt_start', False):
        if rank == 0:
            print('ckpt_start=True; skip loading pretrained checkpoint and resume finetune checkpoint directly.')
    else:
        _load_pretrained_if_requested(model, model_config, manager, device, rank)

    num_workers = int(model_config.get('num_workers', model_config.get('dataloader_num_workers', 8)))

    train_dataloader = get_finetune_dataloader_for_task(
        task_type,
        model_config['micro_batch_size'],
        model_config['use_splited_data'],
        status=0,
        num_workers=num_workers,
        manager=manager,
        model_type=model_type,
        model_archi_params=model_archi_params,
        other_params=other_params,
        use_fake_input=model_config['use_fake_input'],
        lead_time=model_config['finetune_lead_time'],
    )
    if rank==-1:
        print('model_config[train_path]', model_config['train_path'])


    ckpt_path = _finetune_checkpoint_dir(model_config, model_type, manager, optimizer_state_tuple)
    log_path = os.path.join(_finetune_output_dir(model_config), 'loss')

    if rank==0:
        print('[TERRA finetune config]',
              'description=', model_config['description'],
              'task_type=', task_type,
              'model_type=', model_type,
              'optimizer_config=', model_config['optimizer_config'],
              'precision=', precision,
              'norm_type=', model_config.get('norm_type'),
              'data_precision=', model_config.get('data_precision'),
              'use_splited_data=', model_config['use_splited_data'],
              'lead_time=', model_config['finetune_lead_time'],
              'wp_topo=', model_config.get('wp_topo'),
              'xfmr_wp_topo=', model_config.get('xfmr_wp_topo'),
              'xfmr_sp_size=', model_config.get('xfmr_sp_size', 1),
              'tensor_parallel_size=', model_config.get('tensor_parallel_size', 1))
        print('ckpt_path', ckpt_path)
        print('log_path', log_path)


    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    my_log_dir = os.path.join(log_path, f"rank{rank}.txt")


    task_specific_data_dict = get_task_specific_data(
        task_type,
        device,
        optimizer_state_tuple,
        model_archi_params,
        other_params,
        manager,
        micro_batch_size=model_config['micro_batch_size'],
        model_type=model_type,
    )


    num_finetune_epochs = model_config['num_finetune_epochs']
    warmup_steps = model_config.get('warmup_steps', 10)
    save_interval_steps = int(model_config.get('save_interval_steps', 50))
    max_train_steps_per_epoch = model_config.get('max_train_steps_per_epoch', None)
    steps_per_epoch = len(train_dataloader)
    if max_train_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, int(max_train_steps_per_epoch))

    start_epoch, global_step, resume_step_in_epoch = _load_resume_checkpoint_if_requested(
        model,
        optimizer,
        engine,
        gscaler,
        model_config,
        ckpt_path,
        optimizer_state_tuple,
        device,
        rank,
        steps_per_epoch,
    )

    if ZERO_STAGE_NUMBER is not None:
        engine.optimizer.global_step = global_step
    else:
        optimizer.global_step = global_step

    def process_loss_for_show(loss):
        with torch.no_grad():
            report_loss = loss.detach().float().clone()
            embedding_parallel_type = other_params['embedding_parallel_type']

            # Match pretrain reporting: FSDP keeps window-parallel loss enlarged

            if USE_FSDP and model_type != 'sequential':
                if embedding_parallel_type == 'window_embedding' or embedding_parallel_type == 'window_linear':
                    report_loss.div_(manager.get_wp_group_size())

            return make_global_report_loss(report_loss, manager)

    def prepare_required_tensors(batch):
        if task_type == 'glorys' and model_type != 'sequential' and not model_config['use_splited_data']:
            return resolve_required_tensors_online_split_for_parallel(
                task_type,
                batch,
                device,
                model_archi_params,
                other_params,
                manager,
                my_dtype,
                is_pretrain=False,
                lead_time=model_config['finetune_lead_time'],
            )

        required_tensors = resolve_required_tensors_from_dataloader(
            task_type,
            batch,
            device,
            task_specific_data_dict,
            my_dtype,
            is_pretrain=False,
            lead_time=model_config['finetune_lead_time'],
        )
        return required_tensors

    for epoch in range(start_epoch, num_finetune_epochs):
        model.train()
        train_dataloader.sampler.set_epoch(epoch)
        last_loss_for_ckpt = None

        for step, batch in enumerate(train_dataloader):
            if epoch == start_epoch and step < resume_step_in_epoch:
                continue
            if max_train_steps_per_epoch is not None and step >= int(max_train_steps_per_epoch):
                break


            lr = get_finetune_lr_by_global_step(
                global_step,
                model_config['learning_rate'],
                warmup_steps=warmup_steps,
            )


            if ZERO_STAGE_NUMBER is not None:
                set_lr(engine.optimizer, lr)
            else:
                set_lr(optimizer, lr)

            required_tensors = prepare_required_tensors(batch)
            output_info, output_tensor_list, loss = run_one_step_for_finetune_train(
                task_type,
                model_type,
                model,
                optimizer,
                engine,
                gscaler,
                precision,
                my_dtype,
                optimizer_state_tuple,
                required_tensors,
                task_specific_data_dict,
                other_params,
                manager,
                criterion,
                lead_time=model_config['finetune_lead_time'],
                loss_reduction=model_config['finetune_loss_reduction'],
                clip_grad=model_config['clip_grad'],
            )
            global_step += 1
            last_loss_for_ckpt = loss

            with torch.no_grad():
                shown_loss = process_loss_for_show(loss.detach())

            if rank == 0:
                log_args = (
                    model_config['description'], 'epoch', epoch, 'finetune step:', step,
                    'loss', shown_loss.item(),
                    'lr:', f"{lr:.8f}",
                    'train_gs', global_step - 1,
                )
                if gscaler is not None:
                    log_args = log_args + ('scale', gscaler.get_scale())
                if model_config['clip_grad'] and 'grad_norm' in output_info:
                    log_args = log_args + ('grad_norm', output_info['grad_norm'].item())
                print(*log_args)
                with open(my_log_dir, "a") as file:
                    print(*log_args, file=file)

            if save_interval_steps > 0 and global_step % save_interval_steps == 0:
                _save_checkpoint(
                    model,
                    optimizer,
                    engine,
                    gscaler,
                    last_loss_for_ckpt.detach(),
                    epoch,
                    global_step,
                    ckpt_path,
                    optimizer_state_tuple,
                    manager,
                    model_config,
                    checkpoint_name=f"step{global_step}.pth",
                )

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
