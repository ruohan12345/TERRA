import torch

def backbone_run(task_type, model, required_tensors, task_specific_data_dict):
    if task_type=="glorys":
        input_tensor = required_tensors['input_tensor']
        land_sea_mask = task_specific_data_dict['land_sea_mask']

        output_tensor = model(input_tensor)

        output_tensor = output_tensor * land_sea_mask

        output_tensor_list = [output_tensor]
    else:
        raise ValueError("Unsupported task_type; only glorys is supported")

    return output_tensor_list

def cal_loss(task_type, output_tensor_list, required_tensors, task_specific_data_dict, loss_fn, output_info):
    if task_type=="glorys":
        output_tensor = output_tensor_list[0]
        label_tensor = required_tensors['label_tensor']
        loss = loss_fn(output_tensor, label_tensor).type(torch.float32)
    else:
        raise ValueError("Unsupported task_type; only glorys is supported")

    return loss


def loss_post_process(loss, model_type, embedding_parallel_type, manager, task_specific_data_dict, USE_FSDP):
    if (model_type=='parallel' or model_type=='hybrid'):
        if embedding_parallel_type == 'domain_parallel':
            loss = loss / manager.get_mp_group_size()

            print('warning, domain parallel may be combined with sequence parallel')

            exit(0)
        elif embedding_parallel_type == 'window_embedding' or embedding_parallel_type == 'window_linear':
            if not USE_FSDP:
                loss = loss / manager.get_wp_group_size()
        elif embedding_parallel_type == 'window_domain':
            loss = loss / (manager.get_mp_group_size() * manager.get_wp_group_size())
        else:
            print('we are here need to divide loss for', embedding_parallel_type)
            exit(0)

        loss = loss * task_specific_data_dict['padding_loss_scale']

    return loss


def grad_post_process(model_type, optimizer_state_tuple, model, manager):


    USE_DDP, USE_FSDP, ZERO_STAGE_NUMBER, USE_DIST_OPT = optimizer_state_tuple


    if True and (model_type=='parallel' or model_type=='hybrid'):
        if USE_DDP:
            real_model = model.module
        else:
            real_model = model


        for name, param in real_model.named_parameters():


                    # mp_all_reduce_list，world_size


            if (param.grad is not None) and (ZERO_STAGE_NUMBER==None)  and (not USE_DIST_OPT) and (not USE_FSDP):
                custom_reduce_group_name = getattr(param, "terra_grad_reduce_group", None)
                if custom_reduce_group_name is not None:
                    if custom_reduce_group_name == "xfmr_tp_param_group":
                        reduce_group = getattr(manager, "xfmr_tp_param_group", None)
                    else:
                        raise ValueError(f"Unknown terra_grad_reduce_group: {custom_reduce_group_name}")

                    if reduce_group is None:
                        raise RuntimeError(f"{custom_reduce_group_name} is not initialized")

                    grad = param.grad.contiguous()
                    if torch.distributed.get_world_size(group=reduce_group) > 1:
                        torch.distributed.all_reduce(
                            grad,
                            op=torch.distributed.ReduceOp.SUM,
                            group=reduce_group,
                        )
                    param.grad = grad
                    continue


                if manager.get_wp_group_size()>1:
                    if manager.get_mp_group_size()>1:
                        if name in real_model.mp_all_reduce_list: # wp+mp
                            grad = param.grad.contiguous()
                            torch.distributed.all_reduce(grad,
                                                    op=torch.distributed.ReduceOp.SUM,
                                                    group=manager.window_parallel_group) # wp group
                            torch.distributed.all_reduce(grad,
                                                    op=torch.distributed.ReduceOp.SUM,
                                                    group=manager.model_parallel_group)
                            param.grad = grad
                        else: # wp
                            grad = param.grad.contiguous()
                            torch.distributed.all_reduce(grad,
                                                    op=torch.distributed.ReduceOp.SUM,
                                                    group=manager.window_parallel_group) # wp group
                            param.grad = grad

                    else: # wp
                        if name in real_model.mp_all_reduce_list:

                            grad = param.grad.contiguous()
                            torch.distributed.all_reduce(grad,
                                                    op=torch.distributed.ReduceOp.SUM,
                                                    group=manager.window_parallel_group) # wp group
                            param.grad = grad

                else:
                    if manager.get_mp_group_size()>1:
                        if name in real_model.mp_all_reduce_list:

                            grad = param.grad.contiguous()
                            torch.distributed.all_reduce(grad,
                                                    op=torch.distributed.ReduceOp.SUM,
                                                    group=manager.model_parallel_group)
                            param.grad = grad
