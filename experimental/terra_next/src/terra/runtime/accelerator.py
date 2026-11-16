"""Explicit CPU/CUDA/Ascend runtime operations.

Importing this module on CPU/CUDA does not import torch_npu or initialize an NPU.
No CUDA-to-NPU monkeypatching or process-global ``transfer_to_npu`` is used.
"""
import importlib
import os

import torch
import torch.distributed as dist


def _kind(device):
    return str(device).split(':', 1)[0]


def npu_module():
    try:
        package = importlib.import_module('torch_npu')
    except (ImportError, OSError) as exc:
        raise RuntimeError('NPU requested: install a matching torch/torch_npu/CANN environment') from exc
    return getattr(torch, 'npu', None) or package.npu


def as_device(requested):
    if _kind(requested) == 'npu':
        npu_module()  # Register PrivateUse1 before parsing torch.device('npu').
    device = torch.device(requested)
    if device.type not in ('cpu', 'cuda', 'npu'):
        raise NotImplementedError('Supported devices are cpu, cuda and npu')
    return device


def device_module(device):
    kind = _kind(device)
    if kind == 'cuda':
        return torch.cuda
    if kind == 'npu':
        return npu_module()
    if kind == 'cpu':
        return None
    raise NotImplementedError('Unsupported device: '+str(device))


def resolve_device(requested, *, local_rank=None, distributed=False):
    device = as_device(requested)
    module = device_module(device)
    if module is None:
        return device
    if not module.is_available():
        raise RuntimeError(device.type.upper()+' requested but unavailable')
    local_rank = int(os.environ.get('LOCAL_RANK', '0')) if local_rank is None else int(local_rank)
    index = device.index if device.index is not None else local_rank
    if distributed and index != local_rank:
        raise ValueError('Accelerator device index must match LOCAL_RANK')
    device = torch.device(device.type, index)
    module.set_device(device)
    return device


def distributed_backend(device, requested=None):
    expected = {'cpu': 'gloo', 'cuda': 'nccl', 'npu': 'hccl'}[as_device(device).type]
    backend = expected if requested is None else str(requested).lower()
    # MPI remains available for the explicit CPU debug path.
    if backend != expected and not (expected == 'gloo' and backend == 'mpi'):
        raise ValueError('Device %s requires backend %s, got %s' % (device, expected, backend))
    return backend


def control_device(backend):
    backend = str(backend).lower()
    if backend == 'nccl':
        return torch.device('cuda', torch.cuda.current_device())
    if backend == 'hccl':
        api = npu_module()
        return torch.device('npu', api.current_device())
    return torch.device('cpu')


def synchronize(device):
    module = device_module(device)
    if module is not None:
        module.synchronize(device)


def manual_seed_all(device, seed):
    module = device_module(device)
    if module is not None:
        module.manual_seed_all(seed)


def peak_memory(device, *, reset=False):
    module = device_module(device)
    if module is None:
        return None
    if reset:
        module.reset_peak_memory_stats(device)
    return int(module.max_memory_allocated(device))


def get_rng_state(device):
    module = device_module(device)
    return None if module is None else module.get_rng_state(device).cpu()


def set_rng_state(device, state):
    module = device_module(device)
    if module is not None:
        module.set_rng_state(state.cpu(), device)


def stream(device):
    module = device_module(device)
    if module is None:
        raise NotImplementedError('CPU has no accelerator stream')
    return module.Stream(device=device)


def event(device, *, enable_timing=False):
    module = device_module(device)
    if module is None:
        raise NotImplementedError('CPU has no accelerator event')
    return module.Event(enable_timing=enable_timing)


def stream_context(device, value):
    module = device_module(device)
    if module is None:
        raise NotImplementedError('CPU has no accelerator stream context')
    return module.stream(value)


def metric_device(device):
    # Ascend 910B does not execute FP64. Preserve FP64 metric semantics on host.
    device = as_device(device)
    return torch.device('cpu') if device.type == 'npu' else device


def sum_tensor(value, group=None):
    """SUM with a precision-preserving host path for HCCL CPU metrics.

    HCCL cannot reduce FP64 on 910B. Host FP64 values are transported as
    objects and summed on CPU, never cast to a lower precision on the wire.
    This explicit correctness path is synchronous and has extra overhead.
    """
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return value
    if str(dist.get_backend(group)).lower() == 'hccl' and value.device.type == 'cpu':
        values = [None] * dist.get_world_size(group)
        dist.all_gather_object(values, value, group=group)
        result = torch.zeros_like(value)
        for item in values:
            result.add_(item)
        return result
    dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
    return value


def capability_report(device):
    device = as_device(device)
    module = device_module(device)
    return {'device': str(device), 'torch': torch.__version__,
            'backend': distributed_backend(device),
            'available': True if module is None else bool(module.is_available()),
            'native_fp64': device.type != 'npu',
            'metric_precision': 'host_fp64_with_object_transport' if device.type == 'npu' else 'native_fp64',
            'gradient_norm_policy': 'device_fp32_v1',
            'gradient_norm_accumulation': 'parameter_device_float32; float64 debug oracle preserves float64',
            'hccl_zero_split_strategy': 'pad_empty_peer_payload_to_one_element' if device.type == 'npu' else None,
            'hccl_halo_p2p_strategy': 'native_ProcessGroupHCCL_batch_send_recv' if device.type == 'npu' else None,
            'logical_memory_format': 'contiguous_nchw' if device.type == 'npu' else 'preserve_cpu_cuda_policy',
            'synchronous_transfer': True, 'async_stream_event_primitives': module is not None,
            'asynchronous_offload_pipeline': False,
            'validated_collectives': 'requires platform smoke test; presence is not validation',
            'cross_device_exact_resume': False}
