import os
import sys

def _redirect_to_devnull():
    devnull = open(os.devnull, "w", buffering=1, encoding="utf-8")
    sys.stdout = devnull
    sys.stderr = devnull


def _set_dp_rank_print_redirect_legacy(
                               rank,
                               dp_rank,
                               mode,
                               embedding_parallel_type,
                               optimizer_state_tuple):
    #USE_DDP, USE_FSDP, ZERO_STAGE_NUMBER = optimizer_state_tuple
    USE_DDP, USE_FSDP, ZERO_STAGE_NUMBER, USE_DIST_OPT = optimizer_state_tuple


    if mode == 'sequential':
        mode = 'seq'
    elif mode == 'parallel' or mode == 'hybrid':
        mode = 'para'

    assert mode in ['seq', 'para']

    if mode == 'seq':
        log_dir = "./log"
    elif mode == 'para':
        log_dir = "./log/" + embedding_parallel_type

    if rank==0:
        os.makedirs(log_dir, exist_ok=True)

    os.makedirs(log_dir, exist_ok=True)
    if USE_DDP:
        log_path = os.path.join(log_dir, f"ddp_{mode}_dp_rank{dp_rank}.txt")
    elif USE_FSDP:
        log_path = os.path.join(log_dir, f"fsdp_{mode}_dp_rank{dp_rank}.txt")
    elif ZERO_STAGE_NUMBER is not None:
        log_path = os.path.join(log_dir, f"zero{ZERO_STAGE_NUMBER}_{mode}_dp_rank{dp_rank}.txt")
    elif USE_DIST_OPT:
        log_path = os.path.join(log_dir, f"my_dist_optim_{mode}_dp_rank{dp_rank}.txt")


    sys.stdout = open(log_path, "w", buffering=1, encoding="utf-8")
    sys.stderr = sys.stdout


def set_dp_rank_print_redirect(
                               rank,
                               dp_rank,
                               mode,
                               embedding_parallel_type,
                               optimizer_state_tuple,
                               manager=None,
                               only_wp_rank=None,
                               only_wp_ranks=None,
                               append=False):
    USE_DDP, USE_FSDP, ZERO_STAGE_NUMBER, USE_DIST_OPT = optimizer_state_tuple

    if mode == 'sequential':
        mode = 'seq'
    elif mode == 'parallel' or mode == 'hybrid':
        mode = 'para'

    assert mode in ['seq', 'para']

    if mode == 'seq':
        log_dir = "./log"
    elif mode == 'para':
        log_dir = "./log/" + embedding_parallel_type

    os.makedirs(log_dir, exist_ok=True)

    wp_rank = None
    if manager is not None:
        wp_rank = manager.get_wp_rank()
        if only_wp_ranks is not None:
            only_wp_ranks = set(only_wp_ranks)
            if wp_rank not in only_wp_ranks:
                _redirect_to_devnull()
                return False
        elif only_wp_rank is not None and wp_rank != only_wp_rank:
            _redirect_to_devnull()
            return False

    suffix_parts = [f"dp_rank{dp_rank}"]
    if manager is not None:
        suffix_parts.extend([
            f"wp_rank{wp_rank}",
            f"wg{getattr(manager, 'xfmr_window_group_rank', 0)}",
            f"sp{getattr(manager, 'xfmr_sp_rank', 0)}",
            f"tp{getattr(manager, 'xfmr_tp_rank', 0)}",
        ])
    log_suffix = "_".join(suffix_parts)

    if USE_DDP:
        log_path = os.path.join(log_dir, f"ddp_{mode}_{log_suffix}.txt")
    elif USE_FSDP:
        log_path = os.path.join(log_dir, f"fsdp_{mode}_{log_suffix}.txt")
    elif ZERO_STAGE_NUMBER is not None:
        log_path = os.path.join(log_dir, f"zero{ZERO_STAGE_NUMBER}_{mode}_{log_suffix}.txt")
    elif USE_DIST_OPT:
        log_path = os.path.join(log_dir, f"my_dist_optim_{mode}_{log_suffix}.txt")
    else:
        log_path = os.path.join(log_dir, f"rank{rank}_{mode}_{log_suffix}.txt")

    file_mode = "a" if append else "w"
    sys.stdout = open(log_path, file_mode, buffering=1, encoding="utf-8")
    sys.stderr = sys.stdout
    return True
