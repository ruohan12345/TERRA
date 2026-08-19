import torch
import torch.distributed as dist
from torch.distributed import P2POp
from core.parallel.window_assignment import get_window_assignment_mode, get_window_indices


def _get_grid_to_window_route(manager, local_h, local_w, window_size, device):
    data_wp_topo = getattr(manager, "data_wp_topo", (manager.get_wp_group_size(), 1))
    xfmr_wp_topo = getattr(manager, "xfmr_wp_topo", data_wp_topo)
    wp_group_size = manager.get_wp_group_size()
    wp_rank = manager.get_wp_rank()

    cache_key = (
        "grid_to_window_route",
        local_h,
        local_w,
        window_size,
        data_wp_topo,
        xfmr_wp_topo,
        get_window_assignment_mode(manager),
        str(device),
    )
    if not hasattr(manager, "_layout_transform_route_cache"):
        manager._layout_transform_route_cache = {}
    if cache_key in manager._layout_transform_route_cache:
        return manager._layout_transform_route_cache[cache_key]

    if data_wp_topo[1] != 1:
        raise ValueError(f"grid transform requires data wp_topo=(m, 1), got {data_wp_topo}")
    if data_wp_topo[0] * data_wp_topo[1] != wp_group_size:
        raise ValueError(f"data_wp_topo={data_wp_topo} does not match wp_group_size={wp_group_size}")
    if xfmr_wp_topo[0] * xfmr_wp_topo[1] != wp_group_size:
        raise ValueError(f"xfmr_wp_topo={xfmr_wp_topo} does not match wp_group_size={wp_group_size}")

    global_h = local_h * data_wp_topo[0]
    global_w = local_w
    if global_h % window_size != 0 or global_w % window_size != 0:
        raise ValueError(
            f"global token shape {(global_h, global_w)} is not divisible by window_size={window_size}"
        )

    xfmr_h, xfmr_w = xfmr_wp_topo
    num_windows_h = global_h // window_size
    num_windows_w = global_w // window_size
    rr_window_count = [0 for _ in range(wp_group_size)]
    window_to_rr_local_idx = {}

    rank_to_indices, global_to_local = get_window_indices(
        num_windows_h,
        num_windows_w,
        xfmr_h,
        xfmr_w,
        mode=get_window_assignment_mode(manager),
        debug_rank=getattr(manager, "xfmr_window_group_rank", wp_rank),
        debug_global_rank=getattr(manager, "rank", None),
    )
    for rank, indices in rank_to_indices.items():
        rr_window_count[rank] = len(indices)
    for window_idx, (rank, local_idx) in global_to_local.items():
        window_to_rr_local_idx[window_idx] = local_idx

    stripe_send_idx_by_dst = [[] for _ in range(wp_group_size)]
    stripe_recv_dst_idx_by_src = [[] for _ in range(wp_group_size)]
    rr_send_idx_by_dst = [[] for _ in range(wp_group_size)]
    rr_recv_dst_idx_by_src = [[] for _ in range(wp_group_size)]

    for gi in range(global_h):
        stripe_rank = gi // local_h
        stripe_local_i = gi % local_h
        wi = gi // window_size
        token_i = gi % window_size
        for gj in range(global_w):
            wj = gj // window_size
            rr_rank, _ = global_to_local[(wi, wj)]
            rr_local_idx = window_to_rr_local_idx[(wi, wj)]
            token_idx = token_i * window_size + (gj % window_size)
            stripe_local_idx = stripe_local_i * local_w + gj
            rr_flat_idx = rr_local_idx * window_size * window_size + token_idx

            if stripe_rank == wp_rank:
                stripe_send_idx_by_dst[rr_rank].append(stripe_local_idx)
            if rr_rank == wp_rank:
                stripe_recv_dst_idx_by_src[stripe_rank].append(rr_flat_idx)
            if rr_rank == wp_rank:
                rr_send_idx_by_dst[stripe_rank].append(rr_flat_idx)
            if stripe_rank == wp_rank:
                rr_recv_dst_idx_by_src[rr_rank].append(stripe_local_idx)

    def to_tensor_lists(lists):
        return [
            torch.tensor(item, device=device, dtype=torch.long)
            for item in lists
        ]

    route = {
        "stripe_local_count": local_h * local_w,
        "rr_local_window_count": rr_window_count[wp_rank],
        "stripe_send_idx_by_dst": to_tensor_lists(stripe_send_idx_by_dst),
        "stripe_recv_dst_idx_by_src": to_tensor_lists(stripe_recv_dst_idx_by_src),
        "rr_send_idx_by_dst": to_tensor_lists(rr_send_idx_by_dst),
        "rr_recv_dst_idx_by_src": to_tensor_lists(rr_recv_dst_idx_by_src),
    }
    manager._layout_transform_route_cache[cache_key] = route
    return route


def _p2p_relayout_flat(x_flat, manager, send_idx_by_dst, recv_dst_idx_by_src, output_count):
    B, _, hidden_dim = x_flat.shape
    wp_group_size = manager.get_wp_group_size()
    wp_rank = manager.get_wp_rank()
    wp_group = manager.window_parallel_group
    out = x_flat.new_zeros(B, output_count, hidden_dim)

    send_tensors = {}
    recv_tensors = {}
    ops = []

    for peer_rank in range(wp_group_size):
        send_idx = send_idx_by_dst[peer_rank]
        if send_idx.numel() > 0:
            send_tensors[peer_rank] = torch.index_select(
                x_flat,
                dim=1,
                index=send_idx,
            ).permute(1, 0, 2).contiguous()

        recv_idx = recv_dst_idx_by_src[peer_rank]
        if recv_idx.numel() > 0:
            recv_tensors[peer_rank] = torch.empty(
                (recv_idx.numel(), B, hidden_dim),
                dtype=x_flat.dtype,
                device=x_flat.device,
            )

    for peer_rank, send_tensor in send_tensors.items():
        if peer_rank == wp_rank:
            continue
        peer_global_rank = manager.get_global_rank(
            dp_rank=manager.get_dp_rank(),
            mp_rank=manager.get_mp_rank(),
            wp_rank=peer_rank,
        )
        ops.append(P2POp(dist.isend, send_tensor, peer_global_rank, wp_group))

    for peer_rank, recv_tensor in recv_tensors.items():
        if peer_rank == wp_rank:
            continue
        peer_global_rank = manager.get_global_rank(
            dp_rank=manager.get_dp_rank(),
            mp_rank=manager.get_mp_rank(),
            wp_rank=peer_rank,
        )
        ops.append(P2POp(dist.irecv, recv_tensor, peer_global_rank, wp_group))

    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    if wp_rank in send_tensors:
        recv_tensors[wp_rank] = send_tensors[wp_rank]

    for peer_rank, recv_tensor in recv_tensors.items():
        recv_idx = recv_dst_idx_by_src[peer_rank]
        if recv_idx.numel() == 0:
            continue
        out.index_copy_(
            1,
            recv_idx,
            recv_tensor.permute(1, 0, 2).contiguous(),
        )

    return out


def _stripe_grid_to_round_robin_windows_impl(x, manager, window_size):
    B, local_h, local_w, hidden_dim = x.shape
    route = _get_grid_to_window_route(manager, local_h, local_w, window_size, x.device)
    x_flat = x.contiguous().view(B, local_h * local_w, hidden_dim)
    out_flat = _p2p_relayout_flat(
        x_flat,
        manager,
        route["stripe_send_idx_by_dst"],
        route["stripe_recv_dst_idx_by_src"],
        route["rr_local_window_count"] * window_size * window_size,
    )
    return out_flat.view(B, route["rr_local_window_count"], window_size * window_size, hidden_dim)


def _round_robin_windows_to_stripe_grid_impl(x, manager, local_h, local_w, window_size):
    B, _, _, hidden_dim = x.shape
    route = _get_grid_to_window_route(manager, local_h, local_w, window_size, x.device)
    x_flat = x.contiguous().view(B, route["rr_local_window_count"] * window_size * window_size, hidden_dim)
    out_flat = _p2p_relayout_flat(
        x_flat,
        manager,
        route["rr_send_idx_by_dst"],
        route["rr_recv_dst_idx_by_src"],
        route["stripe_local_count"],
    )
    return out_flat.view(B, local_h, local_w, hidden_dim)


class _StripeGridToRoundRobinWindows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, manager, window_size):
        ctx.manager = manager
        ctx.local_h = x.shape[1]
        ctx.local_w = x.shape[2]
        ctx.window_size = window_size
        return _stripe_grid_to_round_robin_windows_impl(x, manager, window_size)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _round_robin_windows_to_stripe_grid_impl(
            grad_output.contiguous(),
            ctx.manager,
            ctx.local_h,
            ctx.local_w,
            ctx.window_size,
        )
        return grad_input, None, None


class _RoundRobinWindowsToStripeGrid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, manager, local_h, local_w, window_size):
        ctx.manager = manager
        ctx.local_h = local_h
        ctx.local_w = local_w
        ctx.window_size = window_size
        return _round_robin_windows_to_stripe_grid_impl(x, manager, local_h, local_w, window_size)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _stripe_grid_to_round_robin_windows_impl(
            grad_output.contiguous(),
            ctx.manager,
            ctx.window_size,
        )
        return grad_input, None, None, None, None


def stripe_grid_to_round_robin_windows(x, manager, window_size):
    data_wp_topo = getattr(manager, "data_wp_topo", (manager.get_wp_group_size(), 1))
    xfmr_wp_topo = getattr(manager, "xfmr_wp_topo", data_wp_topo)
    if data_wp_topo == xfmr_wp_topo and get_window_assignment_mode(manager) == "regular":
        B, local_h, local_w, hidden_dim = x.shape
        return x.view(B, local_h // window_size, window_size, local_w // window_size, window_size, hidden_dim).permute(
            0, 1, 3, 2, 4, 5
        ).contiguous().view(B, (local_h // window_size) * (local_w // window_size), window_size * window_size, hidden_dim)
    return _StripeGridToRoundRobinWindows.apply(x, manager, window_size)


def round_robin_windows_to_stripe_grid(x, manager, local_h, local_w, window_size):
    data_wp_topo = getattr(manager, "data_wp_topo", (manager.get_wp_group_size(), 1))
    xfmr_wp_topo = getattr(manager, "xfmr_wp_topo", data_wp_topo)
    if data_wp_topo == xfmr_wp_topo and get_window_assignment_mode(manager) == "regular":
        B, _, _, hidden_dim = x.shape
        return x.view(B, local_h // window_size, local_w // window_size, window_size, window_size, hidden_dim).permute(
            0, 1, 3, 2, 4, 5
        ).contiguous().view(B, local_h, local_w, hidden_dim)
    return _RoundRobinWindowsToStripeGrid.apply(x, manager, local_h, local_w, window_size)


def _get_ulysses_window_route(manager, local_h, local_w, window_size, shift_size, device):
    data_wp_topo = getattr(manager, "data_wp_topo", (manager.get_wp_group_size(), 1))
    xfmr_wp_topo = getattr(manager, "xfmr_wp_topo", data_wp_topo)
    wp_group_size = manager.get_wp_group_size()
    wp_rank = manager.get_wp_rank()
    sp_size = int(getattr(manager, "xfmr_sp_size", wp_group_size))
    tp_size = int(getattr(manager, "xfmr_tp_size", 1))
    window_group_size = xfmr_wp_topo[0] * xfmr_wp_topo[1]
    window_group_rank = int(getattr(manager, "xfmr_window_group_rank", wp_rank // (sp_size * tp_size)))
    tp_rank = int(getattr(manager, "xfmr_tp_rank", 0))

    cache_key = (
        "ulysses_window_route",
        local_h,
        local_w,
        window_size,
        shift_size,
        data_wp_topo,
        xfmr_wp_topo,
        sp_size,
        tp_size,
        getattr(manager, "sp_tp_placement", "tp_first"),
        get_window_assignment_mode(manager),
        str(device),
    )
    if not hasattr(manager, "_layout_transform_route_cache"):
        manager._layout_transform_route_cache = {}
    if cache_key in manager._layout_transform_route_cache:
        return manager._layout_transform_route_cache[cache_key]

    if data_wp_topo[1] != 1:
        raise ValueError(f"ulysses window transform requires data wp_topo=(m, 1), got {data_wp_topo}")
    if data_wp_topo[0] * data_wp_topo[1] != wp_group_size:
        raise ValueError(f"data_wp_topo={data_wp_topo} does not match wp_group_size={wp_group_size}")
    if window_group_size * sp_size * tp_size != wp_group_size:
        raise ValueError(
            f"xfmr_wp_topo={xfmr_wp_topo} times xfmr_sp_size={sp_size} times tensor_parallel_size={tp_size} does not match wp_group_size={wp_group_size}"
        )

    global_h = local_h * data_wp_topo[0]
    global_w = local_w
    if global_h % window_size != 0 or global_w % window_size != 0:
        raise ValueError(
            f"global token shape {(global_h, global_w)} is not divisible by window_size={window_size}"
        )

    tokens_per_window = window_size * window_size
    shard_tokens = (tokens_per_window + sp_size - 1) // sp_size
    num_windows_h = global_h // window_size
    num_windows_w = global_w // window_size
    rank_to_indices, global_to_local = get_window_indices(
        num_windows_h,
        num_windows_w,
        xfmr_wp_topo[0],
        xfmr_wp_topo[1],
        mode=get_window_assignment_mode(manager),
        debug_rank=window_group_rank,
        debug_global_rank=getattr(manager, "rank", None),
    )
    window_group_full_counts = [len(rank_to_indices[r]) for r in range(window_group_size)]
    window_group_tp_counts = [
        (count + tp_size - 1) // tp_size
        for count in window_group_full_counts
    ]
    full_window_count = window_group_full_counts[window_group_rank]
    tp_window_count = window_group_tp_counts[window_group_rank]

    stripe_send_idx_by_dst = [[] for _ in range(wp_group_size)]
    stripe_recv_dst_idx_by_src = [[] for _ in range(wp_group_size)]
    shard_send_idx_by_dst = [[] for _ in range(wp_group_size)]
    shard_recv_dst_idx_by_src = [[] for _ in range(wp_group_size)]

    for gi in range(global_h):
        stripe_rank = gi // local_h
        stripe_local_i = gi % local_h
        shifted_i = (gi - shift_size) % global_h
        wi = shifted_i // window_size
        token_i = shifted_i % window_size
        for gj in range(global_w):
            shifted_j = (gj - shift_size) % global_w
            wj = shifted_j // window_size
            token_j = shifted_j % window_size
            token_idx = token_i * window_size + token_j
            sp_rank = token_idx // shard_tokens
            shard_local_idx = token_idx % shard_tokens
            token_window_group_rank, local_window_idx = global_to_local[(wi, wj)]
            dst_tp_window_count = window_group_tp_counts[token_window_group_rank]
            if dst_tp_window_count <= 0:
                raise RuntimeError(f"window group {token_window_group_rank} has no windows")
            window_tp_rank = local_window_idx // dst_tp_window_count
            tp_local_window_idx = local_window_idx - window_tp_rank * dst_tp_window_count
            stripe_local_idx = stripe_local_i * local_w + gj
            shard_flat_idx = tp_local_window_idx * shard_tokens + shard_local_idx

            if hasattr(manager, "xfmr_coord_to_wp_rank"):
                dst_shard_rank = manager.xfmr_coord_to_wp_rank(
                    token_window_group_rank,
                    sp_rank,
                    window_tp_rank,
                )
            else:
                dst_shard_rank = token_window_group_rank * (sp_size * tp_size) + sp_rank * tp_size + window_tp_rank

            if stripe_rank == wp_rank:
                stripe_send_idx_by_dst[dst_shard_rank].append(stripe_local_idx)
            if dst_shard_rank == wp_rank:
                stripe_recv_dst_idx_by_src[stripe_rank].append(shard_flat_idx)
            if dst_shard_rank == wp_rank:
                shard_send_idx_by_dst[stripe_rank].append(shard_flat_idx)
            if stripe_rank == wp_rank:
                shard_recv_dst_idx_by_src[dst_shard_rank].append(stripe_local_idx)

    def to_tensor_lists(lists):
        return [
            torch.tensor(item, device=device, dtype=torch.long)
            for item in lists
        ]

    route = {
        "stripe_local_count": local_h * local_w,
        "full_window_count": full_window_count,
        "local_window_count": tp_window_count,
        "shard_tokens": shard_tokens,
        "stripe_send_idx_by_dst": to_tensor_lists(stripe_send_idx_by_dst),
        "stripe_recv_dst_idx_by_src": to_tensor_lists(stripe_recv_dst_idx_by_src),
        "shard_send_idx_by_dst": to_tensor_lists(shard_send_idx_by_dst),
        "shard_recv_dst_idx_by_src": to_tensor_lists(shard_recv_dst_idx_by_src),
    }
    manager._layout_transform_route_cache[cache_key] = route
    return route


def _stripe_grid_to_ulysses_windows_impl(x, manager, window_size, shift_size):
    B, local_h, local_w, hidden_dim = x.shape
    route = _get_ulysses_window_route(manager, local_h, local_w, window_size, shift_size, x.device)
    x_flat = x.contiguous().view(B, local_h * local_w, hidden_dim)
    out_flat = _p2p_relayout_flat(
        x_flat,
        manager,
        route["stripe_send_idx_by_dst"],
        route["stripe_recv_dst_idx_by_src"],
        route["local_window_count"] * route["shard_tokens"],
    )
    return out_flat.view(B, route["local_window_count"], route["shard_tokens"], hidden_dim)


def _ulysses_windows_to_stripe_grid_impl(x, manager, local_h, local_w, window_size, shift_size):
    B, _, _, hidden_dim = x.shape
    route = _get_ulysses_window_route(manager, local_h, local_w, window_size, shift_size, x.device)
    x_flat = x.contiguous().view(B, route["local_window_count"] * route["shard_tokens"], hidden_dim)
    out_flat = _p2p_relayout_flat(
        x_flat,
        manager,
        route["shard_send_idx_by_dst"],
        route["shard_recv_dst_idx_by_src"],
        route["stripe_local_count"],
    )
    return out_flat.view(B, local_h, local_w, hidden_dim)


def _get_ulysses_to_ulysses_route(manager, local_h, local_w, window_size, src_shift_size, dst_shift_size, device):
    data_wp_topo = getattr(manager, "data_wp_topo", (manager.get_wp_group_size(), 1))
    xfmr_wp_topo = getattr(manager, "xfmr_wp_topo", data_wp_topo)
    wp_group_size = manager.get_wp_group_size()
    wp_rank = manager.get_wp_rank()
    sp_size = int(getattr(manager, "xfmr_sp_size", wp_group_size))
    tp_size = int(getattr(manager, "xfmr_tp_size", 1))
    window_group_size = xfmr_wp_topo[0] * xfmr_wp_topo[1]
    window_group_rank = int(getattr(manager, "xfmr_window_group_rank", wp_rank // (sp_size * tp_size)))

    cache_key = (
        "ulysses_to_ulysses_route",
        local_h,
        local_w,
        window_size,
        src_shift_size,
        dst_shift_size,
        data_wp_topo,
        xfmr_wp_topo,
        sp_size,
        tp_size,
        getattr(manager, "sp_tp_placement", "tp_first"),
        get_window_assignment_mode(manager),
        str(device),
    )
    if not hasattr(manager, "_layout_transform_route_cache"):
        manager._layout_transform_route_cache = {}
    if cache_key in manager._layout_transform_route_cache:
        return manager._layout_transform_route_cache[cache_key]

    if data_wp_topo[1] != 1:
        raise ValueError(f"ulysses direct transform requires data wp_topo=(m, 1), got {data_wp_topo}")
    if data_wp_topo[0] * data_wp_topo[1] != wp_group_size:
        raise ValueError(f"data_wp_topo={data_wp_topo} does not match wp_group_size={wp_group_size}")
    if window_group_size * sp_size * tp_size != wp_group_size:
        raise ValueError(
            f"xfmr_wp_topo={xfmr_wp_topo} times xfmr_sp_size={sp_size} times tensor_parallel_size={tp_size} does not match wp_group_size={wp_group_size}"
        )

    global_h = local_h * data_wp_topo[0]
    global_w = local_w
    if global_h % window_size != 0 or global_w % window_size != 0:
        raise ValueError(
            f"global token shape {(global_h, global_w)} is not divisible by window_size={window_size}"
        )

    tokens_per_window = window_size * window_size
    shard_tokens = (tokens_per_window + sp_size - 1) // sp_size
    num_windows_h = global_h // window_size
    num_windows_w = global_w // window_size
    rank_to_indices, global_to_local = get_window_indices(
        num_windows_h,
        num_windows_w,
        xfmr_wp_topo[0],
        xfmr_wp_topo[1],
        mode=get_window_assignment_mode(manager),
        debug_rank=window_group_rank,
        debug_global_rank=getattr(manager, "rank", None),
    )
    window_group_full_counts = [len(rank_to_indices[r]) for r in range(window_group_size)]
    window_group_tp_counts = [
        (count + tp_size - 1) // tp_size
        for count in window_group_full_counts
    ]
    local_window_count = window_group_tp_counts[window_group_rank]

    def token_owner_and_flat_idx(gi, gj, shift_size):
        shifted_i = (gi - shift_size) % global_h
        shifted_j = (gj - shift_size) % global_w
        wi = shifted_i // window_size
        wj = shifted_j // window_size
        token_i = shifted_i % window_size
        token_j = shifted_j % window_size
        token_idx = token_i * window_size + token_j
        sp_rank = token_idx // shard_tokens
        shard_local_idx = token_idx % shard_tokens
        token_window_group_rank, local_window_idx = global_to_local[(wi, wj)]
        tp_window_count = window_group_tp_counts[token_window_group_rank]
        if tp_window_count <= 0:
            raise RuntimeError(f"window group {token_window_group_rank} has no windows")
        window_tp_rank = local_window_idx // tp_window_count
        tp_local_window_idx = local_window_idx - window_tp_rank * tp_window_count
        if hasattr(manager, "xfmr_coord_to_wp_rank"):
            rank = manager.xfmr_coord_to_wp_rank(
                token_window_group_rank,
                sp_rank,
                window_tp_rank,
            )
        else:
            rank = token_window_group_rank * (sp_size * tp_size) + sp_rank * tp_size + window_tp_rank
        flat_idx = tp_local_window_idx * shard_tokens + shard_local_idx
        return rank, flat_idx

    src_send_idx_by_dst = [[] for _ in range(wp_group_size)]
    dst_recv_idx_by_src = [[] for _ in range(wp_group_size)]

    for gi in range(global_h):
        for gj in range(global_w):
            src_rank, src_flat_idx = token_owner_and_flat_idx(gi, gj, src_shift_size)
            dst_rank, dst_flat_idx = token_owner_and_flat_idx(gi, gj, dst_shift_size)
            if src_rank == wp_rank:
                src_send_idx_by_dst[dst_rank].append(src_flat_idx)
            if dst_rank == wp_rank:
                dst_recv_idx_by_src[src_rank].append(dst_flat_idx)

    def to_tensor_lists(lists):
        return [
            torch.tensor(item, device=device, dtype=torch.long)
            for item in lists
        ]

    route = {
        "local_window_count": local_window_count,
        "shard_tokens": shard_tokens,
        "send_idx_by_dst": to_tensor_lists(src_send_idx_by_dst),
        "recv_dst_idx_by_src": to_tensor_lists(dst_recv_idx_by_src),
    }
    manager._layout_transform_route_cache[cache_key] = route
    return route


def _ulysses_windows_to_ulysses_windows_impl(x, manager, local_h, local_w, window_size, src_shift_size, dst_shift_size):
    B, local_window_count, shard_tokens, hidden_dim = x.shape
    route = _get_ulysses_to_ulysses_route(
        manager,
        local_h,
        local_w,
        window_size,
        src_shift_size,
        dst_shift_size,
        x.device,
    )
    if local_window_count != route["local_window_count"] or shard_tokens != route["shard_tokens"]:
        raise RuntimeError(
            f"ulysses direct transform input shape mismatch: got local_window_count={local_window_count}, "
            f"shard_tokens={shard_tokens}, expected local_window_count={route['local_window_count']}, "
            f"shard_tokens={route['shard_tokens']}"
        )
    x_flat = x.contiguous().view(B, local_window_count * shard_tokens, hidden_dim)
    out_flat = _p2p_relayout_flat(
        x_flat,
        manager,
        route["send_idx_by_dst"],
        route["recv_dst_idx_by_src"],
        route["local_window_count"] * route["shard_tokens"],
    )
    return out_flat.view(B, route["local_window_count"], route["shard_tokens"], hidden_dim)


class _StripeGridToUlyssesWindows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, manager, window_size, shift_size):
        ctx.manager = manager
        ctx.local_h = x.shape[1]
        ctx.local_w = x.shape[2]
        ctx.window_size = window_size
        ctx.shift_size = shift_size
        return _stripe_grid_to_ulysses_windows_impl(x, manager, window_size, shift_size)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _ulysses_windows_to_stripe_grid_impl(
            grad_output.contiguous(),
            ctx.manager,
            ctx.local_h,
            ctx.local_w,
            ctx.window_size,
            ctx.shift_size,
        )
        return grad_input, None, None, None


class _UlyssesWindowsToStripeGrid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, manager, local_h, local_w, window_size, shift_size):
        ctx.manager = manager
        ctx.local_h = local_h
        ctx.local_w = local_w
        ctx.window_size = window_size
        ctx.shift_size = shift_size
        return _ulysses_windows_to_stripe_grid_impl(x, manager, local_h, local_w, window_size, shift_size)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _stripe_grid_to_ulysses_windows_impl(
            grad_output.contiguous(),
            ctx.manager,
            ctx.window_size,
            ctx.shift_size,
        )
        return grad_input, None, None, None, None, None


class _UlyssesWindowsToUlyssesWindows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, manager, local_h, local_w, window_size, src_shift_size, dst_shift_size):
        ctx.manager = manager
        ctx.local_h = local_h
        ctx.local_w = local_w
        ctx.window_size = window_size
        ctx.src_shift_size = src_shift_size
        ctx.dst_shift_size = dst_shift_size
        return _ulysses_windows_to_ulysses_windows_impl(
            x,
            manager,
            local_h,
            local_w,
            window_size,
            src_shift_size,
            dst_shift_size,
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _ulysses_windows_to_ulysses_windows_impl(
            grad_output.contiguous(),
            ctx.manager,
            ctx.local_h,
            ctx.local_w,
            ctx.window_size,
            ctx.dst_shift_size,
            ctx.src_shift_size,
        )
        return grad_input, None, None, None, None, None, None


def stripe_grid_to_ulysses_windows(x, manager, window_size, shift_size=0):
    return _StripeGridToUlyssesWindows.apply(x, manager, window_size, shift_size)


def ulysses_windows_to_stripe_grid(x, manager, local_h, local_w, window_size, shift_size=0):
    return _UlyssesWindowsToStripeGrid.apply(x, manager, local_h, local_w, window_size, shift_size)


def ulysses_windows_to_ulysses_windows(x, manager, local_h, local_w, window_size, src_shift_size=0, dst_shift_size=0):
    return _UlyssesWindowsToUlyssesWindows.apply(
        x,
        manager,
        local_h,
        local_w,
        window_size,
        src_shift_size,
        dst_shift_size,
    )
