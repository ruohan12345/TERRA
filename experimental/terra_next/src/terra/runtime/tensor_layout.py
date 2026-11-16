"""Logical tensor layouts supported by the selected execution backend."""
import torch


def has_channels_last_layout(tensor):
    # TorchNPU does not support the channels_last memory-format API. Avoid
    # even querying it: NPU model tensors use ordinary contiguous NCHW.
    return (tensor.device.type != 'npu' and tensor.ndim == 4
            and tensor.is_contiguous(memory_format=torch.channels_last))


def contiguous_for_device(tensor, *, channels_last=False):
    """Preserve CPU/CUDA policy; use supported contiguous NCHW on NPU."""
    if tensor.device.type == 'npu':
        return tensor.contiguous()
    return tensor.contiguous(memory_format=(torch.channels_last if channels_last
                                            else torch.contiguous_format))


def staging_memory_format(device):
    """Choose CPU staging strides before a preserving host-to-device copy."""
    kind = device.type if hasattr(device, 'type') else str(device).split(':', 1)[0]
    return torch.contiguous_format if kind == 'npu' else torch.channels_last
