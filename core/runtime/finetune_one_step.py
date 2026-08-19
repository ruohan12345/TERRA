import torch
from core.runtime.run_one_step import cal_loss, grad_post_process, loss_post_process


def _reduce_finetune_loss(final_loss, lead_time, loss_reduction):
    if final_loss is None:
        raise RuntimeError("finetune loss is None; check lead_time and required_tensors")
    if loss_reduction == "sum":
        return final_loss
    if loss_reduction == "mean":
        return final_loss / max(1, int(lead_time))
    raise ValueError(f"unsupported finetune_loss_reduction: {loss_reduction}")


def funetune_run(
        task_type,
        model,
        required_tensors,
        task_specific_data_dict,
        loss_fn,
        lead_time = 1,
        loss_reduction = "mean",
        lead_callback = None,
        ):

    if task_type=="glorys":
        output_info = {}
        land_sea_mask = task_specific_data_dict['land_sea_mask']
        final_loss = None
        cur_output_tensor_list = None

        for cur_lead_time in range(0, lead_time):
            cur_required_tensors = required_tensors[cur_lead_time]
            if lead_callback is not None:
                lead_callback("before_forward", cur_lead_time, None)

            if cur_lead_time == 0:
                cur_input = cur_required_tensors['input_tensor']

            cur_output = model(cur_input)
            cur_output = cur_output * land_sea_mask
            cur_output_tensor_list = [cur_output]

            cur_loss = cal_loss(
                task_type,
                cur_output_tensor_list,
                cur_required_tensors,
                task_specific_data_dict,
                loss_fn,
                output_info,
            )

            if final_loss is None:
                final_loss = cur_loss
            else:
                final_loss = final_loss + cur_loss

            if lead_callback is not None:
                lead_callback("after_loss", cur_lead_time, cur_loss)

            cur_input = cur_output
    else:
        raise ValueError("Unsupported task_type; only glorys is supported")

    final_loss = _reduce_finetune_loss(final_loss, lead_time, loss_reduction)
    return cur_output_tensor_list, final_loss


finetune_run = funetune_run


def _finetune_forward(
        task_type,
        model,
        required_tensors,
        task_specific_data_dict,
        loss_fn,
        precision,
        my_dtype,
        lead_time,
        loss_reduction,
        lead_callback=None,
        ):
    if precision == 'fp32':
        return funetune_run(
            task_type,
            model,
            required_tensors,
            task_specific_data_dict,
            loss_fn,
            lead_time=lead_time,
            loss_reduction=loss_reduction,
            lead_callback=lead_callback,
        )

    if precision == 'fp16' or precision == 'bf16':
        with torch.amp.autocast("cuda", dtype=my_dtype):
            return funetune_run(
                task_type,
                model,
                required_tensors,
                task_specific_data_dict,
                loss_fn,
                lead_time=lead_time,
                loss_reduction=loss_reduction,
                lead_callback=lead_callback,
            )

    raise RuntimeError(f"unsupported precision for finetune: {precision}")


def run_one_step_for_finetune_train(
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
        loss_fn,
        lead_time=1,
        loss_reduction="mean",
        clip_grad=False,
        ):
    USE_DDP, USE_FSDP, ZERO_STAGE_NUMBER, USE_DIST_OPT = optimizer_state_tuple

    output_tensor_list, loss = _finetune_forward(
        task_type,
        model,
        required_tensors,
        task_specific_data_dict,
        loss_fn,
        precision,
        my_dtype,
        lead_time,
        loss_reduction,
    )

    embedding_parallel_type = other_params['embedding_parallel_type']
    loss = loss_post_process(
        loss,
        model_type,
        embedding_parallel_type,
        manager,
        task_specific_data_dict,
        USE_FSDP,
    )

    if precision == 'fp32':
        optimizer.zero_grad()
        loss.backward()
    elif precision == 'fp16' or precision == 'bf16':
        if USE_DDP or USE_FSDP or USE_DIST_OPT:
            optimizer.zero_grad()
            if precision == 'fp16':
                gscaler.scale(loss).backward()
            else:
                loss.backward()
        elif ZERO_STAGE_NUMBER is not None:
            engine.backward(loss)
        else:
            raise RuntimeError("unsupported finetune optimizer state")

    grad_post_process(model_type, optimizer_state_tuple, model, manager)

    output_info = {}
    if clip_grad and precision == 'fp16' and (USE_DDP or USE_FSDP):
        gscaler.unscale_(optimizer)
        output_info['grad_norm'] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    if precision == 'fp32':
        optimizer.step()
    elif precision == 'fp16' or precision == 'bf16':
        if USE_DDP or USE_FSDP or USE_DIST_OPT:
            if precision == 'fp16':
                gscaler.step(optimizer)
                gscaler.update()
            else:
                optimizer.step()
        elif ZERO_STAGE_NUMBER is not None:
            engine.step()

    if ZERO_STAGE_NUMBER is not None:
        engine.optimizer.global_step = engine.optimizer.global_step + 1
    else:
        optimizer.global_step = optimizer.global_step + 1


    return output_info, output_tensor_list, loss


def run_one_step_for_finetune_eval(
        task_type,
        model_type,
        model,
        required_tensors,
        precision,
        my_dtype,
        task_specific_data_dict,
        other_params,
        manager,
        loss_fn,
        lead_time=1,
        loss_reduction="mean",
        ):
    output_tensor_list, loss = _finetune_forward(
        task_type,
        model,
        required_tensors,
        task_specific_data_dict,
        loss_fn,
        precision,
        my_dtype,
        lead_time,
        loss_reduction,
    )

    loss = loss_post_process(
        loss,
        model_type,
        other_params['embedding_parallel_type'],
        manager,
        task_specific_data_dict,
        USE_FSDP=False,
    )

    return {}, output_tensor_list, loss
