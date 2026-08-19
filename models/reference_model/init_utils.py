import contextlib
import torch


REFERENCE_INIT_SEEDS = {
    "layers": 2026061001,
    "patch_embed": 2026061002,
    "patch_recovery": 2026061003,
    "down_blk": 2026061004,
    "up_blk": 2026061005,
    "attention": 2026061006,
    "mlp": 2026061007,
}


@contextlib.contextmanager
def deterministic_rng(seed):
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def make_with_seed(seed, factory):
    with deterministic_rng(seed):
        return factory()
