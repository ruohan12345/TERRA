import torch
import time
from utils import all_reduce_and_print_rank0
from core.runtime.run_one_step import backbone_run, cal_loss, loss_post_process, grad_post_process
from core.runtime.finetune_one_step import funetune_run
from core.global_env_config import DEBUG_GRAD_HOOK, DEBUG_GRAD_HOOK_PRINT_LIMIT
from profiler.memory_timeline import (
    get_memory_timeline_context,
    set_memory_timeline_context,
)


def make_global_report_loss(loss, manager):
    report_loss = loss.detach().float().clone()
    if manager is None:
        return report_loss
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return report_loss

    if manager.get_wp_group_size() > 1:
        torch.distributed.all_reduce(
            report_loss,
            op=torch.distributed.ReduceOp.SUM,
            group=manager.window_parallel_group,
        )

    if manager.get_dp_group_size() > 1:
        torch.distributed.all_reduce(
            report_loss,
            op=torch.distributed.ReduceOp.SUM,
            group=manager.data_parallel_group,
        )
        report_loss.div_(manager.get_dp_group_size())

    return report_loss


def should_show_window_debug_rank(manager):
    if manager is None:
        return True
    if int(getattr(manager, "xfmr_tp_size", 1)) <= 1:
        return manager.get_wp_rank() == 0
    return (
        getattr(manager, "xfmr_window_group_rank", 0) == 0
        and getattr(manager, "xfmr_sp_rank", 0) == 0
    )


def _clean_grad_debug_name(name):
    while name.startswith("module."):
        name = name[len("module."):]
    return name.replace("_fsdp_wrapped_module.", "")


def _grad_debug_targets(task_type):
    if task_type == "glorys":
        return {
            "patch_embed.weight",
            "patch_embed.bias",
            "patch_embed.linear.weight",
            "patch_embed.linear.bias",
            "patch_recovery.weight",
            "patch_recovery.bias",
            "patch_recovery.linear.weight",
            "patch_recovery.linear.bias",
            "down_blk.conv.weight",
            "down_blk.conv.bias",
            "down_blk.conv.conv.weight",
            "down_blk.conv.conv.bias",
            "up_blk.conv.weight",
            "up_blk.conv.bias",
            "up_blk.conv.conv.weight",
            "up_blk.conv.conv.bias",
            "norm1.weight",
            "norm1.bias",
            "attn.q_linear.weight",
            "attn.q_linear.bias",
            "attn.k_linear.weight",
            "attn.k_linear.bias",
            "attn.v_linear.weight",
            "attn.v_linear.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "attn.proj_bias",
            "attn.relative_position_bias_table",
            "linear1.weight",
            "linear1.bias",
            "linear2.weight",
            "linear2.bias",
            "bias2",
        }
    return set()


def _grad_debug_name_matches(name, targets):
    name = _clean_grad_debug_name(name)
    return any(name == target or name.endswith("." + target) or target.endswith("." + name) for target in targets)


def _shape_numel(shape):
    numel = 1
    for dim in tuple(shape):
        numel *= int(dim)
    return numel


def _flat_param_fqns_and_numels(param):
    fqns = getattr(param, "_fqns", None)
    numels = getattr(param, "_numels", None)

    if fqns is None:
        infos = getattr(param, "_param_infos", None)
        if infos is not None:
            fqns = []
            for info in infos:
                module_name = getattr(info, "module_name", "")
                param_name = getattr(info, "param_name", "")
                fqns.append(f"{module_name}.{param_name}" if module_name else param_name)

    if numels is None:
        shapes = getattr(param, "_shapes", None)
        if shapes is not None:
            numels = [_shape_numel(shape) for shape in shapes]

    if fqns is None or numels is None:
        return None, None
    return list(fqns), [int(x) for x in numels]


def _flat_param_shard_infos(param):
    infos = getattr(param, "_shard_param_infos", None)
    if infos is None:
        return None
    return list(infos)


def _print_grad_debug_line(root_model, name, grad, param_slice=None):
    if getattr(root_model, "_terra_debug_grad_hook_count", 0) >= DEBUG_GRAD_HOOK_PRINT_LIMIT:
        return
    seen = getattr(root_model, "_terra_debug_grad_hook_seen", None)
    if seen is None:
        seen = set()
        root_model._terra_debug_grad_hook_seen = seen

    name = _clean_grad_debug_name(name)
    if name in seen:
        return
    seen.add(name)
    root_model._terra_debug_grad_hook_count = getattr(root_model, "_terra_debug_grad_hook_count", 0) + 1

    grad_sum = grad.sum()
    print(f"[grad hook] {name} grad sum: {grad_sum.item():.20f}", grad_sum.dtype)
    if param_slice is not None:
        param_sum = param_slice.sum()
        print(f"[grad hook]      parameter sum: {param_sum.item():.20f}", param_slice.dtype, tuple(param_slice.shape))


def _make_grad_debug_hook(root_model, param_name, param, targets):
    def hook(grad):
        if not getattr(root_model, "_terra_debug_grad_hook_active", False):
            return grad
        if not getattr(root_model, "_terra_debug_grad_hook_print_this_rank", False):
            return grad

        fqns, numels = _flat_param_fqns_and_numels(param)
        if fqns is not None and numels is not None and grad.numel() == sum(numels):
            flat_grad = grad.reshape(-1)
            flat_param = param.detach().reshape(-1) if param.numel() == sum(numels) else None
            offset = 0
            for fqn, numel in zip(fqns, numels):
                next_offset = offset + numel
                if _grad_debug_name_matches(fqn, targets):
                    param_slice = flat_param[offset:next_offset] if flat_param is not None else None
                    _print_grad_debug_line(root_model, fqn, flat_grad[offset:next_offset], param_slice)
                offset = next_offset
        elif fqns is not None:
            shard_infos = _flat_param_shard_infos(param)
            if shard_infos is not None and len(shard_infos) == len(fqns):
                flat_grad = grad.reshape(-1)
                flat_param = param.detach().reshape(-1) if param.numel() == grad.numel() else None
                for fqn, shard_info in zip(fqns, shard_infos):
                    if not getattr(shard_info, "in_shard", False):
                        continue
                    numel = int(getattr(shard_info, "numel_in_shard", 0))
                    offset = int(getattr(shard_info, "offset_in_shard", 0))
                    if numel <= 0 or offset + numel > flat_grad.numel():
                        continue
                    if _grad_debug_name_matches(fqn, targets):
                        param_slice = flat_param[offset:offset + numel] if flat_param is not None else None
                        _print_grad_debug_line(root_model, f"{fqn} [local shard]", flat_grad[offset:offset + numel], param_slice)
        elif _grad_debug_name_matches(param_name, targets):
            _print_grad_debug_line(root_model, param_name, grad, param.detach())
        return grad
    return hook


def setup_debug_grad_hooks(model, task_type, manager, model_type, active):
    if not DEBUG_GRAD_HOOK:
        return

    print_this_rank = should_show_window_debug_rank(manager) if (model_type == "parallel" or model_type == "hybrid") else manager.get_rank() == 0
    if not getattr(model, "_terra_debug_grad_hooks_registered", False):
        targets = _grad_debug_targets(task_type)
        handles = []
        for name, param in model.named_parameters():
            handles.append(param.register_hook(_make_grad_debug_hook(model, name, param, targets)))
        model._terra_debug_grad_hook_handles = handles
        model._terra_debug_grad_hooks_registered = True
        if print_this_rank:
            print(f"[grad hook] registered {len(handles)} parameter hooks")

    model._terra_debug_grad_hook_print_this_rank = print_this_rank
    model._terra_debug_grad_hook_active = bool(active)
    if active:
        model._terra_debug_grad_hook_seen = set()
        model._terra_debug_grad_hook_count = 0
        if print_this_rank:
            print(" show gradient hook ---------------------------start")


def finish_debug_grad_hooks(model):
    if not DEBUG_GRAD_HOOK:
        return
    if getattr(model, "_terra_debug_grad_hook_active", False) and getattr(model, "_terra_debug_grad_hook_print_this_rank", False):
        print(" show gradient hook ---------------------------over")
    model._terra_debug_grad_hook_active = False

def show_gradient(
                  task_type,
                  my_model,
                  mode='seq',

                  cur_rank=0,
                  manager = None,
                  mp_all_reduce_list = None,
                  ):
    layer_id = 1
    if task_type=='glorys':
        show_name_list = [

                        'module.patch_embed.weight', 'patch_embed.weight',
                        'module.patch_embed.bias', 'patch_embed.bias',

                        'module.patch_embed.linear.weight', 'patch_embed.linear.weight',
                        'module.patch_embed.linear.bias', 'patch_embed.linear.bias',

                        'module.patch_recovery.weight', 'patch_recovery.weight',
                        'module.patch_recovery.bias', 'patch_recovery.bias',

                        'module.patch_recovery.linear.weight', 'patch_recovery.linear.weight',
                        'module.patch_recovery.linear.bias', 'patch_recovery.linear.bias',


                        'module.down_blk.conv.weight', 'down_blk.conv.weight',
                        'module.down_blk.conv.conv.weight', 'down_blk.conv.conv.weight',

                        'module.down_blk.conv.bias', 'down_blk.conv.bias',
                        'module.down_blk.conv.conv.bias', 'down_blk.conv.conv.bias',

                        'module.up_blk.conv.weight', 'up_blk.conv.weight',
                        'module.up_blk.conv.conv.weight', 'up_blk.conv.conv.weight',
                        'module.up_blk.conv.bias', 'up_blk.conv.bias',
                        'module.up_blk.conv.conv.bias', 'up_blk.conv.conv.bias',


                        'module.layers.blocks.0.0.norm1.weight', 'layers.blocks.0.0.norm1.weight',
                        'module.layers.blocks.0.0.norm1.bias', 'layers.blocks.0.0.norm1.bias',


                        'module.layers.blocks.0.0.attn.q_linear.weight', 'layers.blocks.0.0.attn.q_linear.weight',
                        'module.layers.blocks.0.0.attn.q_linear.bias', 'layers.blocks.0.0.attn.q_linear.bias',

                        'module.layers.blocks.0.0.attn.proj.weight', 'layers.blocks.0.0.attn.proj.weight',
                        'module.layers.blocks.0.0.attn.proj_bias', 'layers.blocks.0.0.attn.proj_bias',
                            'module.layers.blocks.0.0.attn.proj.bias', 'layers.blocks.0.0.attn.proj.bias',


                        'module.layers.blocks.0.1.fc1.weight', 'layers.blocks.0.1.fc1.weight',
                        'module.layers.blocks.0.1.fc1.bias',  'layers.blocks.0.1.fc1.bias',
                            'module.layers.blocks.0.1.linear1.weight', 'layers.blocks.0.1.linear1.weight',
                            'module.layers.blocks.0.1.linear1.bias', 'layers.blocks.0.1.linear1.bias',


#name++++++++++++++++++++++++++++++ module.layers.blocks.0.1.linear1.weight
#name++++++++++++++++++++++++++++++ module.layers.blocks.0.1.linear1.bias


                        #'layers.0.blocks.transformer.attn.proj.weight', 'module.layers.0.blocks.transformer.attn.proj.weight',
                        #'layers.1.blocks.transformer.attn.proj.weight', 'module.layers.1.blocks.transformer.attn.proj.weight',

                        #'layers.0.blocks.transformer.attn.proj.bias',


                        f'module.layers.{layer_id}.blocks.transformer.attn.q_linear.weight',
                            f'module.layers.{layer_id}.blocks.transformer.attn_list.1.q_linear.weight',

                        f'module.layers.{layer_id}.blocks.transformer.attn.q_linear.bias',
                            f'module.layers.{layer_id}.blocks.transformer.attn_list.1.q_linear.bias',


                        f'module.layers.{layer_id}.blocks.transformer.attn.k_linear.weight',
                        f'module.layers.{layer_id}.blocks.transformer.attn.k_linear.bias',
                        f'module.layers.{layer_id}.blocks.transformer.attn.v_linear.weight',
                        f'module.layers.{layer_id}.blocks.transformer.attn.proj.weight',
                            f'module.layers.{layer_id}.blocks.transformer.attn_list.1.proj.weight',


                        'module.layers.0.blocks.transformer.attn.relative_position_bias_table'
                        ]
    if cur_rank==0:
        print(' show gradient ---------------------------start')

    has_gradients = False

    def clean_name(name):
        while name.startswith("module."):
            name = name[len("module."):]
        return name.replace("_fsdp_wrapped_module.", "")

    show_name_set = set(show_name_list)
    clean_show_name_set = {clean_name(name) for name in show_name_list}


    for name, param in my_model.named_parameters():


        display_name = clean_name(name)

        if name in show_name_set or display_name in clean_show_name_set:

            if param.grad is not None:

                grad_sum = param.grad.sum()
                param_sum = param.sum()

                if mode=='seq':
                    print(f'{display_name} grad sum: {grad_sum.item():.20f}', grad_sum.dtype)
                    print(f'     parameter sum: {param_sum.item():.20f}', param.dtype, param.shape)


                else:
                    if name in mp_all_reduce_list or display_name in mp_all_reduce_list:
                        if cur_rank==0:
                            print(f'{display_name} grad1 sum: {grad_sum.item():.20f}', grad_sum.dtype)
                    else:
                        all_reduce_and_print_rank0(grad_sum,
                                            manager.get_mp_rank(),
                                            manager.model_parallel_group,
                                            description=f'{display_name} grad2 sum:')

                has_gradients = True
            else:
                print(f"Parameter {display_name} has no gradient")

    if not has_gradients:
        print("WARNING: No gradients found in any model parameters!")

    if cur_rank==0:
        print(' show gradient ---------------------------over')


def one_step_train(
                   task_type,
                   model_type,
                   model,
                   engine,
                   optimizer,

                   required_tensors,
                   task_specific_data_dict,
                   other_params,

                   precision,
                   half_model,

                   my_dtype,
                   loss_fn,
                   gscaler,
                   manager=None,
                   optimizer_state_tuple = None,
                   memory_timeline = None,
                   clip_grad=False,
                   sync_for_timing=True,
                   barrier_for_timing=True,
                   ):
    test_iter_num = task_specific_data_dict['test_iter_num']
    show_gradient_flag = task_specific_data_dict['show_gradient_flag']

    rank = manager.get_rank()
    #USE_DDP, USE_FSDP, ZERO_STAGE_NUMBER = optimizer_state_tuple
    USE_DDP, USE_FSDP, ZERO_STAGE_NUMBER, USE_DIST_OPT = optimizer_state_tuple


    global_step = optimizer.global_step
    debug_grad_hook_active = show_gradient_flag and global_step == test_iter_num - 1
    if USE_FSDP:
        setup_debug_grad_hooks(model, task_type, manager, model_type, debug_grad_hook_active)

    if sync_for_timing and torch.cuda.is_available():
        torch.cuda.synchronize()
    if barrier_for_timing and torch.distributed.is_initialized():
        torch.distributed.barrier()
    t0 = time.time()

    output_info = {}


    def mark_memory(phase):
        if phase == "before_forward":
            set_memory_timeline_context(execution_phase="forward", lead_idx=None)
        elif phase == "before_backward":
            set_memory_timeline_context(execution_phase="backward")
        elif phase == "after_optimizer":
            set_memory_timeline_context(execution_phase="optimizer")
        if memory_timeline is not None:
            memory_timeline.mark(phase, **get_memory_timeline_context())


    mark_memory("before_forward")
    if precision=='fp32':
        output_tensor_list = backbone_run(task_type, model, required_tensors, task_specific_data_dict)
        loss = cal_loss(task_type, output_tensor_list, required_tensors, task_specific_data_dict, loss_fn, output_info)

    elif precision=='fp16' or precision=='bf16':


        if USE_DDP:
            with torch.amp.autocast("cuda", dtype=my_dtype):
                output_tensor_list = backbone_run(task_type, model, required_tensors, task_specific_data_dict)
                loss = cal_loss(task_type, output_tensor_list, required_tensors, task_specific_data_dict, loss_fn, output_info)
        elif USE_FSDP:
            if precision=='fp16':


                with torch.amp.autocast("cuda", dtype=my_dtype):
                    output_tensor_list = backbone_run(task_type, model, required_tensors, task_specific_data_dict)
                    loss = cal_loss(task_type, output_tensor_list, required_tensors, task_specific_data_dict, loss_fn, output_info)
            elif precision=='bf16':
                output_tensor_list = backbone_run(task_type, model, required_tensors, task_specific_data_dict)
                loss = cal_loss(task_type, output_tensor_list, required_tensors, task_specific_data_dict, loss_fn, output_info)
        elif ZERO_STAGE_NUMBER is not None:

            output_tensor_list = backbone_run(task_type, model, required_tensors, task_specific_data_dict)
            loss = cal_loss(task_type, output_tensor_list, required_tensors, task_specific_data_dict, loss_fn, output_info)
        elif USE_DIST_OPT:
            with torch.amp.autocast("cuda", dtype=my_dtype):
                output_tensor_list = backbone_run(task_type, model, required_tensors, task_specific_data_dict)
                loss = cal_loss(task_type, output_tensor_list, required_tensors, task_specific_data_dict, loss_fn, output_info)


    embedding_parallel_type = other_params['embedding_parallel_type']

    loss = loss_post_process(loss, model_type, embedding_parallel_type, manager, task_specific_data_dict, USE_FSDP)
    mark_memory("after_forward")


    t1 = time.time()
    forward_time = t1 - t0

    t2 = time.time()


    if precision=='fp32':
        optimizer.zero_grad()
        loss.backward()
    elif precision=='fp16' or precision=='bf16':
        if USE_DDP:
            mark_memory("before_zero_grad")
            optimizer.zero_grad()
            mark_memory("before_backward")
            if precision=='fp16':
                gscaler.scale(loss).backward()
            elif precision=='bf16':
                loss.backward()
            mark_memory("after_backward")
        elif USE_FSDP:
            mark_memory("before_zero_grad")
            optimizer.zero_grad()
            mark_memory("before_backward")
            if precision=='fp16':
                gscaler.scale(loss).backward()
            elif precision=='bf16':
                loss.backward()
            mark_memory("after_backward")
        elif ZERO_STAGE_NUMBER is not None:
            engine.backward(loss)
        elif USE_DIST_OPT:
            mark_memory("before_zero_grad")
            optimizer.zero_grad()
            mark_memory("before_backward")
            if precision=='fp16':
                gscaler.scale(loss).backward()
            elif precision=='bf16':
                loss.backward()
            mark_memory("after_backward")


    test_manual_allreduce_time = False


    if test_manual_allreduce_time:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        t01 = time.time()

    mark_memory("before_grad_post_process")
    grad_post_process(model_type, optimizer_state_tuple, model, manager)
    mark_memory("after_grad_post_process")
    if USE_FSDP:
        finish_debug_grad_hooks(model)

    if clip_grad and precision == 'fp16' and (USE_DDP or USE_FSDP):
        gscaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    if test_manual_allreduce_time:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        t02 = time.time()

        ranks_per_dp = manager.world_size//manager.get_dp_group_size()
        if (rank % ranks_per_dp)==0:
            print('manual allreduce time', (t02 - t01)*1000, 'ms')


    if False:

        active_memory = memory_show(
                    manager,
                    device,
                    model,
                    optimizer,
                    )

        print('rank , active_memory', rank, active_memory)

    if global_step == test_iter_num-1 and not (USE_FSDP and DEBUG_GRAD_HOOK):
        if show_gradient_flag:

            if (model_type=='parallel' or model_type=='hybrid'):
                if embedding_parallel_type == 'window_embedding' or embedding_parallel_type == 'window_linear':
                    if should_show_window_debug_rank(manager):
                        show_gradient(task_type, model,
                                    mode='seq',
                                    cur_rank = 0,
                                    )
                elif embedding_parallel_type == 'domain_parallel' or embedding_parallel_type == 'window_domain':
                    if manager.get_dp_rank()==0 and manager.get_wp_rank()==0:
                        if USE_DDP:
                            real_model = model.module
                        else:
                            real_model = model
                        show_gradient(task_type, real_model,
                                        mode='parallel',
                                        cur_rank = manager.get_mp_rank(),
                                        manager = manager,
                                        mp_all_reduce_list = real_model.mp_all_reduce_list,
                                        )
                else:
                    print('we are here new show gradient')
                    exit(0)
            elif model_type=='sequential': # Sequential-reference mode does not emit per-parameter gradient diagnostics.
                show_gradient(task_type, model)

    mark_memory("before_optimizer_step")
    if precision=='fp32':
        optimizer.step()
        optimizer.global_step = global_step + 1
    elif precision=='fp16' or precision=='bf16':
        if USE_DDP:
            if precision=='fp16':
                gscaler.step(optimizer)
                gscaler.update()
            elif precision=='bf16':
                optimizer.step()
            optimizer.global_step = global_step + 1
        elif USE_FSDP:
            if precision=='fp16':
                gscaler.step(optimizer)
                gscaler.update()
            elif precision=='bf16':
                optimizer.step()
            optimizer.global_step = global_step + 1
        elif ZERO_STAGE_NUMBER is not None:
            engine.step()
            engine.optimizer.global_step = global_step + 1
        elif USE_DIST_OPT:
            if precision=='fp16':
                gscaler.step(optimizer)
                gscaler.update()
            elif precision=='bf16':
                optimizer.step()
            optimizer.global_step = global_step + 1


    mark_memory("after_optimizer_step")


    if sync_for_timing and torch.cuda.is_available():
        torch.cuda.synchronize()
    if barrier_for_timing and torch.distributed.is_initialized():
        torch.distributed.barrier()
    t3 = time.time()
    backward_and_optimizer_time = t3 - t2

    one_step_train_time_s = forward_time + backward_and_optimizer_time


    with torch.no_grad():
        if USE_FSDP:
            loss = loss / manager.get_wp_group_size()
            #pass

    report_loss = make_global_report_loss(loss, manager)
    return output_tensor_list[0], report_loss, one_step_train_time_s, forward_time


def one_step_train_production(*args, **kwargs):
    """Production pretrain step without per-step timing barriers."""
    kwargs.setdefault("sync_for_timing", False)
    kwargs.setdefault("barrier_for_timing", False)
    return one_step_train(*args, **kwargs)


def one_step_finetune(
                   task_type,
                   model_type,
                   model,
                   engine,
                   optimizer,

                   required_tensors,
                   task_specific_data_dict,
                   other_params,

                   precision,

                   my_dtype,
                   loss_fn,
                   gscaler,
                   manager=None,
                   optimizer_state_tuple = None,
                   memory_timeline = None,
                   lead_time = 1,
                   loss_reduction = "mean",
                   lead_callback = None,
                   external_step_start_time_s = None,
                   ):

    test_iter_num = task_specific_data_dict['test_iter_num']
    show_gradient_flag = task_specific_data_dict['show_gradient_flag']

    rank = manager.get_rank()


    USE_DDP, USE_FSDP, ZERO_STAGE_NUMBER, USE_DIST_OPT = optimizer_state_tuple

    global_step = optimizer.global_step
    debug_grad_hook_active = show_gradient_flag and global_step == test_iter_num - 1
    if USE_FSDP:
        setup_debug_grad_hooks(model, task_type, manager, model_type, debug_grad_hook_active)


    if external_step_start_time_s is None:
        torch.cuda.synchronize()
        torch.distributed.barrier()
    t0 = time.time()

    output_info = {}


    def mark_memory(phase):
        if phase == "before_forward":
            set_memory_timeline_context(execution_phase="forward", lead_idx=None)
        elif phase == "before_backward":
            set_memory_timeline_context(execution_phase="backward")
        elif phase == "after_optimizer":
            set_memory_timeline_context(execution_phase="optimizer")
        if memory_timeline is not None:
            memory_timeline.mark(phase, **get_memory_timeline_context())


    mark_memory("before_forward")
    if precision=='fp32':
        print('fp32 finetune not supported')
        exit(0)

    elif precision=='fp16' or precision=='bf16':
        if USE_DDP:
            with torch.amp.autocast("cuda", dtype=my_dtype):


                output_tensor_list, loss = funetune_run(task_type, model, required_tensors, task_specific_data_dict, loss_fn, lead_time=lead_time, loss_reduction=loss_reduction, lead_callback=lead_callback)


        elif USE_DIST_OPT:
            with torch.amp.autocast("cuda", dtype=my_dtype):
                output_tensor_list, loss = funetune_run(task_type, model, required_tensors, task_specific_data_dict, loss_fn, lead_time=lead_time, loss_reduction=loss_reduction, lead_callback=lead_callback)

        elif USE_FSDP or ZERO_STAGE_NUMBER is not None:
            with torch.amp.autocast("cuda", dtype=my_dtype):
                output_tensor_list, loss = funetune_run(task_type, model, required_tensors, task_specific_data_dict, loss_fn, lead_time=lead_time, loss_reduction=loss_reduction, lead_callback=lead_callback)

        else:
            print('unsupported optimizer for finetune')
            exit(0)


    embedding_parallel_type = other_params['embedding_parallel_type']

    loss = loss_post_process(loss, model_type, embedding_parallel_type, manager, task_specific_data_dict, USE_FSDP)
    mark_memory("after_forward")


    t1 = time.time()
    forward_time = t1 - t0

    t2 = time.time()


    if precision=='fp32':
        optimizer.zero_grad()
        loss.backward()
    elif precision=='fp16' or precision=='bf16':
        if USE_DDP:
            mark_memory("before_zero_grad")
            optimizer.zero_grad()
            mark_memory("before_backward")
            if precision=='fp16':
                gscaler.scale(loss).backward()
            elif precision=='bf16':
                loss.backward()
            mark_memory("after_backward")
        elif USE_FSDP:
            mark_memory("before_zero_grad")
            optimizer.zero_grad()
            mark_memory("before_backward")
            if precision=='fp16':
                gscaler.scale(loss).backward()
            elif precision=='bf16':
                loss.backward()
            mark_memory("after_backward")
        elif ZERO_STAGE_NUMBER is not None:
            engine.backward(loss)
        elif USE_DIST_OPT:
            mark_memory("before_zero_grad")
            optimizer.zero_grad()
            mark_memory("before_backward")
            if precision=='fp16':
                gscaler.scale(loss).backward()
            elif precision=='bf16':
                loss.backward()
            mark_memory("after_backward")


    test_manual_allreduce_time = False


    if test_manual_allreduce_time:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        t01 = time.time()

    mark_memory("before_grad_post_process")
    grad_post_process(model_type, optimizer_state_tuple, model, manager)
    mark_memory("after_grad_post_process")
    if USE_FSDP:
        finish_debug_grad_hooks(model)

    if test_manual_allreduce_time:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        t02 = time.time()

        ranks_per_dp = manager.world_size//manager.get_dp_group_size()
        if (rank % ranks_per_dp)==0:
            print('manual allreduce time', (t02 - t01)*1000, 'ms')


    if global_step == test_iter_num-1 and not (USE_FSDP and DEBUG_GRAD_HOOK):
        if show_gradient_flag:
            if (model_type=='parallel' or model_type=='hybrid'):
                if embedding_parallel_type == 'window_embedding' or embedding_parallel_type == 'window_linear':
                    ranks_per_dp = manager.world_size//manager.get_dp_group_size()
                    if (rank % ranks_per_dp)==0:
                        show_gradient(task_type, model,
                                    mode='seq',
                                    cur_rank = 0,
                                    )
                elif embedding_parallel_type == 'domain_parallel' or embedding_parallel_type == 'window_domain':
                    if manager.get_dp_rank()==0 and manager.get_wp_rank()==0:
                        if USE_DDP:
                            real_model = model.module
                        else:
                            real_model = model
                        show_gradient(task_type, real_model,
                                        mode='parallel',
                                        cur_rank = manager.get_mp_rank(),
                                        manager = manager,
                                        mp_all_reduce_list = real_model.mp_all_reduce_list,
                                        )
                else:
                    print('we are here new show gradient')
                    exit(0)
            elif model_type=='sequential': # Sequential-reference mode does not emit per-parameter gradient diagnostics.
                show_gradient(task_type, model)

    mark_memory("before_optimizer_step")
    if precision=='fp32':
        optimizer.step()
        optimizer.global_step = global_step + 1
    elif precision=='fp16' or precision=='bf16':
        if USE_DDP:
            if precision=='fp16':
                gscaler.step(optimizer)
                gscaler.update()
            elif precision=='bf16':
                optimizer.step()
            optimizer.global_step = global_step + 1
        elif USE_FSDP:
            if precision=='fp16':
                gscaler.step(optimizer)
                gscaler.update()
            elif precision=='bf16':
                optimizer.step()
            optimizer.global_step = global_step + 1
        elif ZERO_STAGE_NUMBER is not None:
            engine.step()
            engine.optimizer.global_step = global_step + 1
        elif USE_DIST_OPT:
            if precision=='fp16':
                gscaler.step(optimizer)
                gscaler.update()
            elif precision=='bf16':
                optimizer.step()
            optimizer.global_step = global_step + 1


    mark_memory("after_optimizer_step")


    torch.cuda.synchronize()
    torch.distributed.barrier()
    t3 = time.time()
    backward_and_optimizer_time = t3 - t2

    if external_step_start_time_s is None:
        one_step_train_time_s = forward_time + backward_and_optimizer_time
    else:
        # The caller starts this interval immediately before enqueueing the
        # non-blocking input copy. The final synchronize above closes an
        # end-to-end H2D + forward + backward + optimizer interval.
        one_step_train_time_s = t3 - external_step_start_time_s


    with torch.no_grad():
        if USE_FSDP:
            loss = loss / manager.get_wp_group_size()
            #pass

    report_loss = make_global_report_loss(loss, manager)
    return output_tensor_list[0], report_loss, one_step_train_time_s, forward_time
