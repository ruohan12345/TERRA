import torch


_prealloc_buffers = {}


def get_preallocated_buffer(shape, dtype, device, rank, index=0):
    """
    Retrieve or create a preallocated buffer with the specified shape, dtype, and device.

    Parameters
    ----------
    shape : tuple
        The shape of the buffer to be created or retrieved.
    dtype : torch.dtype
        The data type of the buffer.
    device : torch.device
        The device on which the buffer will be allocated.
    rank : int
        The rank of the process requesting the buffer.
    index : int, optional
        An additional index to differentiate buffers (default is 0).

    Returns
    -------
    torch.Tensor
        A tensor with the specified shape, dtype, and device, either retrieved from the preallocated buffers or newly created.
    """
    key = (dtype, rank, index)
    if key not in _prealloc_buffers:
        _prealloc_buffers[key] = (None, 0)

    buf, buf_numel = _prealloc_buffers[key]

    needed = 1

    for dim in shape:
        needed *= dim

    # Reallocate if buffer is None or too small
    if buf is None or needed > buf_numel:
        buf = torch.zeros(needed, dtype=dtype, device=device)
        buf_numel = needed
        _prealloc_buffers[key] = (buf, buf_numel)
        return buf.view(*shape)
    else:
        out, _ = _prealloc_buffers[key]
        out = out[:needed].view(*shape)

        return out
