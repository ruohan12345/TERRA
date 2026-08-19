import torch

from torch.distributed import P2POp
def ring_transfer(tensor, buffer, manager=None,
                  ring_topo=None,
                  sync=False):


    """
    tensor: local tensor to send
    buffer: buffer to receive remote tensor
    manager: SwFormer-like parallel manager
    ring_topo: dict or list describing send mapping, e.g. {0:2, 2:3, 3:1, 1:0}
    sync: whether to synchronize CUDA after comm
    """
    wp_rank = manager.get_wp_rank()
    wp_group = manager.window_parallel_group


    if ring_topo is None:
        raise ValueError("Please provide a ring_topo, e.g. {0:2, 2:3, 3:1, 1:0}")

    if isinstance(ring_topo, (list, tuple)):
        ring_topo = {i: ring_topo[i] for i in range(len(ring_topo))}


    send_to = ring_topo[wp_rank]

    recv_from = {v: k for k, v in ring_topo.items()}[wp_rank]


    send_to_global = manager.get_global_rank(
        dp_rank = manager.get_dp_rank(),
        mp_rank = manager.get_mp_rank(),
        wp_rank = send_to)
    recv_from_global = manager.get_global_rank(
        dp_rank = manager.get_dp_rank(),
        mp_rank = manager.get_mp_rank(),
        wp_rank = recv_from)


    assert tensor.is_contiguous()
    assert buffer.is_contiguous()


    if False:
        comm_stream = torch.cuda.Stream()
        with torch.cuda.stream(comm_stream):

            ops = [
                P2POp(torch.distributed.isend, tensor.contiguous(), send_to_global, wp_group),
                P2POp(torch.distributed.irecv, buffer, recv_from_global, wp_group),
            ]
            reqs = torch.distributed.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()
        comm_stream.synchronize()
    else:


        ops = [
            P2POp(torch.distributed.isend, tensor, send_to_global, wp_group),
            P2POp(torch.distributed.irecv, buffer, recv_from_global, wp_group),
        ]
        reqs = torch.distributed.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()


    if sync:
        torch.cuda.synchronize()
