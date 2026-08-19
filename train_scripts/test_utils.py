import torch
import torch.distributed as dist

def process_loss_and_output_in_test(loss, output_tensor, model_type, other_params, manager):
    with torch.no_grad():
        if model_type != 'sequential':
            embedding_parallel_type = other_params['embedding_parallel_type']
            if embedding_parallel_type == 'domain_parallel':
                torch.distributed.all_reduce(output_tensor, op=torch.distributed.ReduceOp.SUM, group=manager.model_parallel_group)
            elif embedding_parallel_type == 'window_embedding' or embedding_parallel_type == 'window_linear':
                torch.distributed.all_reduce(output_tensor, op=torch.distributed.ReduceOp.SUM, group=manager.window_parallel_group)
            elif embedding_parallel_type == 'window_domain':
                torch.distributed.all_reduce(output_tensor, op=torch.distributed.ReduceOp.SUM, group=manager.model_parallel_group)
                torch.distributed.all_reduce(output_tensor, op=torch.distributed.ReduceOp.SUM, group=manager.window_parallel_group)
            else:
                print('unsupported embedding_parallel_type for loss all-reduce', embedding_parallel_type) # window_domain
                exit(0)

    return loss, output_tensor


def gather_scalar(x):
    t = torch.tensor(x, device=torch.cuda.current_device(), dtype=torch.float32)
    world_size = dist.get_world_size()
    out = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(out, t)
    return [v.item() for v in out]
