import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from core.checkpoint.offload_utils import (
    OFFLOAD_MIN_NUMEL,
    OFFLOAD_MODE,
    OFFLOAD_NON_BLOCKING,
    OFFLOAD_PIN_MEMORY,
    get_cuda_stream,
    new_cpu_buffer_like,
)


_OFFLOAD_STREAMS = {}
_PREFETCH_STREAMS = {}
_BOUNDARY_OFFLOAD_STATS = {
    "pack_calls": 0,
    "unpack_calls": 0,
    "offloaded_tensors": 0,
    "restored_tensors": 0,
    "d2h_bytes": 0,
    "h2d_bytes": 0,
    "inline_tensors": 0,
    "cpu_tensors": 0,
    "sampling_offloaded_tensors": 0,
    "sampling_restored_tensors": 0,
    "sampling_d2h_bytes": 0,
    "sampling_h2d_bytes": 0,
    "transformer_offloaded_tensors": 0,
    "transformer_restored_tensors": 0,
    "transformer_d2h_bytes": 0,
    "transformer_h2d_bytes": 0,
}


def _is_boundary_activation(tensor, min_numel=OFFLOAD_MIN_NUMEL):
    if not torch.is_tensor(tensor):
        return False
    if not tensor.is_floating_point():
        return False
    if tensor.numel() < min_numel:
        return False
    if hasattr(tensor, "no_checkpointing") and tensor.no_checkpointing:
        return False
    return True


def _get_autocast_dtype():
    if hasattr(torch, "get_autocast_dtype"):
        return torch.get_autocast_dtype("cuda")
    return torch.get_autocast_gpu_dtype()


def _get_autocast_cache_enabled():
    if hasattr(torch, "is_autocast_cache_enabled"):
        return torch.is_autocast_cache_enabled()
    return True


def get_boundary_offload_stats():
    return dict(_BOUNDARY_OFFLOAD_STATS)


def reset_boundary_offload_stats():
    for key in _BOUNDARY_OFFLOAD_STATS:
        _BOUNDARY_OFFLOAD_STATS[key] = 0


def _tensor_nbytes(tensor):
    return int(tensor.numel()) * int(tensor.element_size())


class _BoundaryTensorStore:
    def __init__(self, offload_config=None):
        offload_config = offload_config or {}
        scope = str(offload_config.get("scope", "other")).strip().lower()
        self.scope = scope if scope in ("sampling", "transformer") else "other"
        self.min_numel = int(offload_config.get("min_numel", OFFLOAD_MIN_NUMEL))
        self.pin_memory = bool(offload_config.get("pin_memory", OFFLOAD_PIN_MEMORY))
        self.non_blocking = bool(offload_config.get("non_blocking", OFFLOAD_NON_BLOCKING))
        self.mode = offload_config.get("mode", OFFLOAD_MODE)

    def pack(self, tensor):
        _BOUNDARY_OFFLOAD_STATS["pack_calls"] += 1
        if not _is_boundary_activation(tensor, min_numel=self.min_numel):
            _BOUNDARY_OFFLOAD_STATS["inline_tensors"] += 1
            return {
                "kind": "tensor_inline",
                "tensor": tensor.detach(),
                "requires_grad": tensor.requires_grad,
            }

        if not tensor.is_cuda:
            _BOUNDARY_OFFLOAD_STATS["cpu_tensors"] += 1
            return {
                "kind": "tensor_cpu",
                "cpu_tensor": tensor.detach().cpu(),
                "device": tensor.device,
                "dtype": tensor.dtype,
                "requires_grad": tensor.requires_grad,
            }

        cpu_tensor = new_cpu_buffer_like(tensor, pin_memory=self.pin_memory)
        current_stream = torch.cuda.current_stream(tensor.device)
        d2h_event = torch.cuda.Event()

        if self.mode in ("async_d2h", "async_d2h_h2d"):
            offload_stream = get_cuda_stream(_OFFLOAD_STREAMS, tensor.device)
            with torch.cuda.stream(offload_stream):
                offload_stream.wait_stream(current_stream)
                tensor.record_stream(offload_stream)
                cpu_tensor.copy_(tensor.detach(), non_blocking=self.non_blocking)
                d2h_event.record(offload_stream)
        elif self.mode in ("sync", "strict_sync"):
            cpu_tensor.copy_(tensor.detach(), non_blocking=self.non_blocking)
            d2h_event.record(current_stream)
        else:
            raise ValueError(f"Unsupported boundary offload mode={self.mode}")

        _BOUNDARY_OFFLOAD_STATS["offloaded_tensors"] += 1
        _BOUNDARY_OFFLOAD_STATS["d2h_bytes"] += _tensor_nbytes(tensor)
        if self.scope != "other":
            _BOUNDARY_OFFLOAD_STATS[f"{self.scope}_offloaded_tensors"] += 1
            _BOUNDARY_OFFLOAD_STATS[f"{self.scope}_d2h_bytes"] += _tensor_nbytes(tensor)
        return {
            "kind": "tensor_offloaded",
            "cpu_tensor": cpu_tensor,
            "device": tensor.device,
            "dtype": tensor.dtype,
            "size": tuple(tensor.size()),
            "stride": tuple(tensor.stride()),
            "layout": tensor.layout,
            "requires_grad": tensor.requires_grad,
            "d2h_event": d2h_event,
        }

    def unpack(self, packed):
        _BOUNDARY_OFFLOAD_STATS["unpack_calls"] += 1
        kind = packed["kind"]
        if kind == "tensor_inline":
            tensor = packed["tensor"]
        elif kind == "tensor_cpu":
            tensor = packed["cpu_tensor"].to(device=packed["device"], dtype=packed["dtype"])
        elif kind == "tensor_offloaded":
            tensor = self._restore_offloaded(packed)
            _BOUNDARY_OFFLOAD_STATS["restored_tensors"] += 1
            restored_bytes = _tensor_nbytes(packed["cpu_tensor"])
            _BOUNDARY_OFFLOAD_STATS["h2d_bytes"] += restored_bytes
            if self.scope != "other":
                _BOUNDARY_OFFLOAD_STATS[f"{self.scope}_restored_tensors"] += 1
                _BOUNDARY_OFFLOAD_STATS[f"{self.scope}_h2d_bytes"] += restored_bytes
        else:
            raise ValueError(f"Unsupported packed boundary tensor kind={kind}")

        tensor = tensor.detach()
        requires_grad = bool(packed["requires_grad"])
        tensor.requires_grad_(requires_grad)
        return tensor

    def _restore_offloaded(self, packed):
        device = packed["device"]
        current_stream = torch.cuda.current_stream(device)
        d2h_event = packed.get("d2h_event", None)
        cpu_tensor = packed["cpu_tensor"]
        dtype = packed["dtype"]

        def new_gpu_tensor():
            if packed.get("layout", torch.strided) == torch.strided:
                return torch.empty_strided(
                    packed["size"],
                    packed["stride"],
                    dtype=dtype,
                    device=device,
                )
            return torch.empty(
                packed["size"],
                dtype=dtype,
                device=device,
            )

        if self.mode == "async_d2h_h2d":
            prefetch_stream = get_cuda_stream(_PREFETCH_STREAMS, device)
            h2d_event = torch.cuda.Event()
            with torch.cuda.stream(prefetch_stream):
                if d2h_event is not None:
                    prefetch_stream.wait_event(d2h_event)
                gpu_tensor = new_gpu_tensor()
                gpu_tensor.copy_(cpu_tensor, non_blocking=self.non_blocking)
                h2d_event.record(prefetch_stream)
            current_stream.wait_event(h2d_event)
            gpu_tensor.record_stream(current_stream)
            return gpu_tensor

        if d2h_event is not None:
            current_stream.wait_event(d2h_event)
        gpu_tensor = new_gpu_tensor()
        gpu_tensor.copy_(cpu_tensor, non_blocking=self.non_blocking)
        gpu_tensor.record_stream(current_stream)
        return gpu_tensor


def _split_tensor_args(args):
    tensor_args = []
    non_tensor_args = []
    tensor_flags = []
    for arg in args:
        if torch.is_tensor(arg):
            tensor_args.append(arg)
            tensor_flags.append(True)
        else:
            non_tensor_args.append(arg)
            tensor_flags.append(False)
    return tuple(tensor_args), tuple(non_tensor_args), tuple(tensor_flags)


def _merge_args(tensor_args, non_tensor_args, tensor_flags):
    merged = []
    tensor_idx = 0
    non_tensor_idx = 0
    for is_tensor in tensor_flags:
        if is_tensor:
            merged.append(tensor_args[tensor_idx])
            tensor_idx += 1
        else:
            merged.append(non_tensor_args[non_tensor_idx])
            non_tensor_idx += 1
    return tuple(merged)


def _first_tensor_device(args):
    for arg in args:
        if torch.is_tensor(arg):
            return arg.device
    return None


def _outputs_to_tuple(outputs):
    if torch.is_tensor(outputs):
        return (outputs,), True
    if isinstance(outputs, tuple) and all(torch.is_tensor(item) for item in outputs):
        return outputs, False
    raise RuntimeError(
        "Boundary CPU offload checkpoint only supports tensor outputs or tuple of tensor outputs."
    )


class _BoundaryOffloadCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, offload_config, tensor_flags, non_tensor_args, *tensor_args):
        ctx.run_function = run_function
        ctx.tensor_flags = tensor_flags
        ctx.non_tensor_args = non_tensor_args

        store = _BoundaryTensorStore(offload_config)
        ctx.store = store

        device = _first_tensor_device(tensor_args)
        ctx.device = device
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.has_cuda_rng_state = device is not None and torch.device(device).type == "cuda" and torch.cuda.is_available()
        if ctx.has_cuda_rng_state:
            ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state(device)

        ctx.use_autocast = torch.is_autocast_enabled()
        ctx.autocast_dtype = _get_autocast_dtype()
        ctx.autocast_cache_enabled = _get_autocast_cache_enabled()

        args = _merge_args(tensor_args, non_tensor_args, tensor_flags)
        with torch.no_grad():
            outputs = run_function(*args)
        output_tensors, is_single_tensor = _outputs_to_tuple(outputs)

        # Schedule boundary offload only after the forward kernels have been
        # queued. This avoids racing an async D2H copy with kernels inside the
        # checkpointed block that may read or alias the same input tensor.
        ctx.packed_tensor_args = [store.pack(tensor) for tensor in tensor_args]
        ctx.is_single_tensor = is_single_tensor
        return output_tensors[0] if is_single_tensor else output_tensors

    @staticmethod
    def backward(ctx, *grad_outputs):
        tensor_args = tuple(ctx.store.unpack(packed) for packed in ctx.packed_tensor_args)
        detached_args = _merge_args(tensor_args, ctx.non_tensor_args, ctx.tensor_flags)

        bwd_cpu_rng_state = torch.get_rng_state()
        if ctx.has_cuda_rng_state:
            bwd_cuda_rng_state = torch.cuda.get_rng_state(ctx.device)

        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        if ctx.has_cuda_rng_state:
            torch.cuda.set_rng_state(ctx.fwd_cuda_rng_state, ctx.device)

        with torch.enable_grad():
            if ctx.use_autocast:
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=ctx.autocast_dtype,
                    cache_enabled=ctx.autocast_cache_enabled,
                ):
                    outputs = ctx.run_function(*detached_args)
            else:
                outputs = ctx.run_function(*detached_args)

        torch.set_rng_state(bwd_cpu_rng_state)
        if ctx.has_cuda_rng_state:
            torch.cuda.set_rng_state(bwd_cuda_rng_state, ctx.device)

        output_tensors, _ = _outputs_to_tuple(outputs)
        output_tensors_for_backward = []
        grad_tensors_for_backward = []
        for output, grad in zip(output_tensors, grad_outputs):
            if output.requires_grad:
                output_tensors_for_backward.append(output)
                grad_tensors_for_backward.append(grad)

        if output_tensors_for_backward:
            torch.autograd.backward(output_tensors_for_backward, grad_tensors_for_backward)

        grad_tensor_args = []
        for arg in tensor_args:
            grad_tensor_args.append(arg.grad if arg.requires_grad else None)

        return (None, None, None, None, *grad_tensor_args)


class _BoundaryGradientIdentity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _BoundaryOffloadRecomputeRunner:
    def __init__(self, function, args, store):
        self.function = function
        self.store = store
        tensor_args, non_tensor_args, tensor_flags = _split_tensor_args(args)
        self.tensor_flags = tensor_flags
        self.non_tensor_args = non_tensor_args
        self.forward_tensor_args = tuple(
            _BoundaryGradientIdentity.apply(tensor)
            if tensor.requires_grad
            else tensor
            for tensor in tensor_args
        )
        self.packed_tensor_args = None

    def __call__(self, _dummy):
        if self.forward_tensor_args is not None:
            tensor_args = self.forward_tensor_args
            args = _merge_args(
                tensor_args,
                self.non_tensor_args,
                self.tensor_flags,
            )
            try:
                return self.function(*args)
            finally:
                # Queue D2H only after the checkpoint body has queued all of
                # its kernels, then release the runner's GPU references.
                self.packed_tensor_args = [
                    self.store.pack(tensor) for tensor in tensor_args
                ]
                self.forward_tensor_args = None

        if self.packed_tensor_args is None:
            raise RuntimeError("boundary offload recompute ran before forward pack")
        restored_tensor_args = tuple(
            self.store.unpack(packed) for packed in self.packed_tensor_args
        )
        # Match the original forward inputs exactly: forward_tensor_args are
        # outputs of this identity and therefore non-leaf tensors. Feeding
        # restored leaves directly can change saved-tensor order under FSDP.
        tensor_args = tuple(
            _BoundaryGradientIdentity.apply(tensor)
            if tensor.requires_grad
            else tensor
            for tensor in restored_tensor_args
        )
        args = _merge_args(
            tensor_args,
            self.non_tensor_args,
            self.tensor_flags,
        )
        return self.function(*args)


def checkpoint(function, *args, offload_config=None):
    device = _first_tensor_device(args)
    if device is None:
        return function(*args)
    store = _BoundaryTensorStore(offload_config)
    # This outer hook is active while non-reentrant checkpoint creates its
    # _NoopSaveInputs node, so only the real checkpoint boundary inputs are
    # packed here. During the model body, PyTorch's inner checkpoint hook takes
    # precedence and preserves the standard FSDP-compatible recompute path.
    with torch.autograd.graph.saved_tensors_hooks(store.pack, store.unpack):
        return torch_checkpoint(
            function,
            *args,
            use_reentrant=False,
            preserve_rng_state=True,
        )
