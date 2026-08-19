import torch
import math


from optimizer.distributed_adamw import DistributedAdamW


def get_optimizer(
    precision,
    model,
    learning_rate,
    manager,
    model_type,

    zero_stage_number,
    use_distributed_optimizer=False,


    optimizer_type = 'AdamW',
):


    if use_distributed_optimizer:
        model_engine = None
        optimizer = DistributedAdamW(
            model.named_parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            process_group = None,

            state_dtype=torch.float32,
            manager = manager,
        )
    elif optimizer_type=='AdamW':
        model_engine = None
        optimizer = torch.optim.AdamW(model.parameters(),
                                    lr = learning_rate, #0.0001
                                    betas=(0.9, 0.95),
                                    weight_decay = 0.1) # 0.1
    else:
        print('unrecognized optimizer_type', optimizer_type)
        exit(0)


    if zero_stage_number is not None:
        try:
            import deepspeed
        except ImportError as exc:
            raise ImportError(
                "zero_stage_number is enabled, but deepspeed is not installed. "
                "Set optimizer_config to DDP/FSDP/custom optimizer, or install deepspeed for Zero."
            ) from exc


        ds_config = {
            "train_micro_batch_size_per_gpu": 1,

            "gradient_accumulation_steps": 1,

            "zero_optimization": {
                "stage": zero_stage_number,
                "allgather_partitions": False, #True,
                "allgather_bucket_size": 5e8,
                "reduce_scatter": False, #True,
                "reduce_bucket_size": 5e8,
                "overlap_comm": False, #True,
                "contiguous_gradients": False #True
            },


        }

        if precision == 'fp32':
            pass
        elif precision == 'bf16':
            ds_config["bf16"] = {
                "enabled": True
            }
        elif precision == 'fp16':
            ds_config["fp16"] = {
                "enabled": True,
                "loss_scale": 0
            }


        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            config=ds_config
        )

        model_engine.dp_process_group = manager.data_parallel_group

    optimizer.global_step = 0

    return model_engine, optimizer


def get_lr_by_global_step(task_type, step, base_lr, warmup_steps, num_pre_train_epochs=None, min_lr=1e-6):
    if task_type != 'glorys':
        raise ValueError(f"Unsupported task_type: {task_type}; only GLORYS reference workload is available")
    total_steps = num_pre_train_epochs*(594)


    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_decay


def get_finetune_lr_by_global_step(step, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    return base_lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
