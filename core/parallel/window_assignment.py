TERRA_M1_ASSIGNMENT_MODES = {
    "terra_m1_regular",
    "terra_m1_ragged_row",
    "terra_m1_ragged_row_major",
    "terra_m1_ragged_auto",
}

_TERRA_M1_DEBUG_PRINTED = set()


def is_terra_m1_assignment_mode(mode):
    return mode in TERRA_M1_ASSIGNMENT_MODES


def get_window_assignment_mode(manager):
    return getattr(manager, "window_assignment_mode", "regular")


def get_window_owner(ii, jj, num_windows_w, wp_group_h, wp_group_w, mode="regular"):
    if mode == "regular":
        return (jj % wp_group_w) + (ii % wp_group_h) * wp_group_w
    if mode == "ragged_round_robin":
        return (ii * num_windows_w + jj) % (wp_group_h * wp_group_w)
    raise ValueError(f"Unsupported window_assignment_mode: {mode}")


def _append_window(rank_to_indices, global_to_local, rank, ii, jj):
    local_idx = len(rank_to_indices[rank])
    rank_to_indices[rank].append((ii, jj))
    global_to_local[(ii, jj)] = (rank, local_idx)


def _assign_terra_m1_regular(rank_to_indices, global_to_local, num_windows_h, num_windows_w, wp_group_h):
    if num_windows_h % wp_group_h != 0:
        raise ValueError(
            f"terra_m1_regular requires num_windows_h={num_windows_h} divisible by m={wp_group_h}"
        )
    rows_per_rank = num_windows_h // wp_group_h
    for rank in range(wp_group_h):
        row_start = rank * rows_per_rank
        row_end = row_start + rows_per_rank
        for ii in range(row_start, row_end):
            for jj in range(num_windows_w):
                _append_window(rank_to_indices, global_to_local, rank, ii, jj)


def _assign_terra_m1_ragged_row(rank_to_indices, global_to_local, num_windows_h, num_windows_w, wp_group_h):
    if num_windows_h < wp_group_h:
        raise ValueError(
            f"terra_m1_ragged_row requires num_windows_h={num_windows_h} >= m={wp_group_h}"
        )
    base_rows = num_windows_h // wp_group_h
    extra_rows = num_windows_h % wp_group_h
    row_start = 0
    for rank in range(wp_group_h):
        row_count = base_rows + (1 if rank < extra_rows else 0)
        row_end = row_start + row_count
        for ii in range(row_start, row_end):
            for jj in range(num_windows_w):
                _append_window(rank_to_indices, global_to_local, rank, ii, jj)
        row_start = row_end


def _assign_terra_m1_ragged_row_major(rank_to_indices, global_to_local, num_windows_h, num_windows_w, wp_group_h):
    total_windows = num_windows_h * num_windows_w
    if total_windows < wp_group_h:
        raise ValueError(
            f"terra_m1_ragged_row_major requires total_windows={total_windows} >= m={wp_group_h}"
        )
    for rank in range(wp_group_h):
        start = (rank * total_windows) // wp_group_h
        end = ((rank + 1) * total_windows) // wp_group_h
        for flat_idx in range(start, end):
            ii = flat_idx // num_windows_w
            jj = flat_idx % num_windows_w
            _append_window(rank_to_indices, global_to_local, rank, ii, jj)


def _maybe_print_terra_m1_auto_assignment(
    rank_to_indices,
    num_windows_h,
    num_windows_w,
    wp_group_h,
    mode,
    resolved_mode,
    debug_rank,
    debug_global_rank,
):
    if mode != "terra_m1_ragged_auto" or debug_rank is None:
        return
    if debug_rank < 0 or debug_rank >= wp_group_h:
        return

    indices = rank_to_indices[debug_rank]
    ids = [ii * num_windows_w + jj for ii, jj in indices]
    key = (
        debug_global_rank,
        debug_rank,
        num_windows_h,
        num_windows_w,
        wp_group_h,
        tuple(ids),
    )
    if key in _TERRA_M1_DEBUG_PRINTED:
        return
    _TERRA_M1_DEBUG_PRINTED.add(key)

    print(
        "[terra_m1_ragged_auto_assignment] "
        f"rank={debug_global_rank} window_rank={debug_rank} "
        f"grid=({num_windows_h},{num_windows_w}) m={wp_group_h} "
        f"resolved={resolved_mode} count={len(ids)} ids={ids} coords={indices}",
        flush=True,
    )


def get_window_indices(
    num_windows_h,
    num_windows_w,
    wp_group_h,
    wp_group_w,
    mode="regular",
    debug_rank=None,
    debug_global_rank=None,
):
    wp_size = wp_group_h * wp_group_w
    rank_to_indices = {rank: [] for rank in range(wp_size)}
    global_to_local = {}

    if is_terra_m1_assignment_mode(mode):
        if wp_group_w != 1:
            raise ValueError(f"{mode} requires xfmr_wp_topo=(m, 1), got ({wp_group_h}, {wp_group_w})")
        resolved_mode = mode
        if mode == "terra_m1_ragged_auto":
            resolved_mode = "terra_m1_ragged_row_major"
        if resolved_mode == "terra_m1_regular":
            _assign_terra_m1_regular(rank_to_indices, global_to_local, num_windows_h, num_windows_w, wp_group_h)
        elif resolved_mode == "terra_m1_ragged_row":
            _assign_terra_m1_ragged_row(rank_to_indices, global_to_local, num_windows_h, num_windows_w, wp_group_h)
        elif resolved_mode == "terra_m1_ragged_row_major":
            _assign_terra_m1_ragged_row_major(rank_to_indices, global_to_local, num_windows_h, num_windows_w, wp_group_h)
        else:
            raise ValueError(f"Unsupported window_assignment_mode: {mode}")
        _maybe_print_terra_m1_auto_assignment(
            rank_to_indices,
            num_windows_h,
            num_windows_w,
            wp_group_h,
            mode,
            resolved_mode,
            debug_rank,
            debug_global_rank,
        )
        return rank_to_indices, global_to_local

    if mode == "ragged_round_robin" and wp_group_w == 1:
        base_rows = num_windows_h // wp_group_h
        extra_rows = num_windows_h % wp_group_h
        row_start = 0
        for rank in range(wp_group_h):
            row_count = base_rows + (1 if rank < extra_rows else 0)
            row_end = row_start + row_count
            for ii in range(row_start, row_end):
                for jj in range(num_windows_w):
                    local_idx = len(rank_to_indices[rank])
                    rank_to_indices[rank].append((ii, jj))
                    global_to_local[(ii, jj)] = (rank, local_idx)
            row_start = row_end
        return rank_to_indices, global_to_local

    for ii in range(num_windows_h):
        for jj in range(num_windows_w):
            rank = get_window_owner(ii, jj, num_windows_w, wp_group_h, wp_group_w, mode)
            local_idx = len(rank_to_indices[rank])
            rank_to_indices[rank].append((ii, jj))
            global_to_local[(ii, jj)] = (rank, local_idx)

    return rank_to_indices, global_to_local
