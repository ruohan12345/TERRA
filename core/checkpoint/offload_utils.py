import torch


OFFLOAD_MIN_NUMEL = 1024 * 1024
OFFLOAD_PIN_MEMORY = True
OFFLOAD_NON_BLOCKING = True
OFFLOAD_MODE = "async_d2h_h2d"

def new_cpu_buffer_like(tensor, pin_memory=OFFLOAD_PIN_MEMORY):
    if pin_memory:
        try:
            return torch.empty_like(tensor, device=torch.device("cpu"), pin_memory=True)
        except TypeError:
            return torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                device=torch.device("cpu"),
                pin_memory=True,
            )
    return torch.empty_like(tensor, device=torch.device("cpu"))


def get_cuda_stream(stream_table, device):
    device = torch.device(device)
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    if device_index not in stream_table:
        with torch.cuda.device(device_index):
            stream_table[device_index] = torch.cuda.Stream()
    return stream_table[device_index]
