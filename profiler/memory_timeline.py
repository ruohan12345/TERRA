import csv
import itertools
import os
import time
from contextlib import ContextDecorator
from pathlib import Path

import torch


TIMELINE_OUTPUT_START_ITER = int(os.environ.get("MEMORY_TIMELINE_OUTPUT_START_ITER", "7"))
TIMELINE_OUTPUT_MAX_ITERS = int(os.environ.get("MEMORY_TIMELINE_OUTPUT_MAX_ITERS", "5"))
TIMELINE_WRITE_CSV = int(os.environ.get("MEMORY_TIMELINE_WRITE_CSV", "1"))
TIMELINE_WRITE_SUMMARY = int(os.environ.get("MEMORY_TIMELINE_WRITE_SUMMARY", "1"))
TIMELINE_WRITE_EVENT_TABLE = int(
    os.environ.get("MEMORY_TIMELINE_WRITE_EVENT_TABLE", "1")
)
TIMELINE_WRITE_PLOT = int(os.environ.get("MEMORY_TIMELINE_WRITE_PLOT", "1"))
TIMELINE_ANNOTATE_EVENTS = int(os.environ.get("MEMORY_TIMELINE_ANNOTATE_EVENTS", "1"))
TIMELINE_ANNOTATE_MODULE_HOOKS = int(os.environ.get("MEMORY_TIMELINE_ANNOTATE_MODULE_HOOKS", "0"))
TIMELINE_RECORD_MODULE_HOOKS = int(os.environ.get("MEMORY_TIMELINE_RECORD_MODULE_HOOKS", "1"))
TIMELINE_MAX_EVENT_ANNOTATIONS = int(os.environ.get("MEMORY_TIMELINE_MAX_EVENT_ANNOTATIONS", "120"))
TIMELINE_X_AXIS = os.environ.get("MEMORY_TIMELINE_X_AXIS", "time").strip().lower()
TIMELINE_LABEL_MODULE_HOOKS = int(os.environ.get("MEMORY_TIMELINE_LABEL_MODULE_HOOKS", "0"))
TIMELINE_SHOW_MODULE_HOOKS = int(os.environ.get("MEMORY_TIMELINE_SHOW_MODULE_HOOKS", "0"))
TIMELINE_SHOW_SAMPLING_INTERNALS = int(os.environ.get("MEMORY_TIMELINE_SHOW_SAMPLING_INTERNALS", "0"))
TIMELINE_SHOW_CHECKPOINT_BODY = int(os.environ.get("MEMORY_TIMELINE_SHOW_CHECKPOINT_BODY", "0"))
TIMELINE_COMPACT_CHECKPOINT_EVENTS = int(os.environ.get("MEMORY_TIMELINE_COMPACT_CHECKPOINT_EVENTS", "1"))
TIMELINE_CHECKPOINT_EVENT_LINES = int(os.environ.get("MEMORY_TIMELINE_CHECKPOINT_EVENT_LINES", "0"))
TIMELINE_LABEL_CHECKPOINT_EVENTS = int(os.environ.get("MEMORY_TIMELINE_LABEL_CHECKPOINT_EVENTS", "0"))
TIMELINE_KEY_BOUNDARY_PHASES = {"before_forward", "after_forward", "after_backward"}
TIMELINE_COARSE_PHASES = {
    "iter_start",
    "before_forward",
    "after_forward",
    "before_backward",
    "after_backward",
    "after_optimizer",
    "iter_end",
}
TIMELINE_SAMPLING_BOUNDARY_PHASES = {
    "sampling_down_enter",
    "sampling_down_exit",
    "sampling_up_enter",
    "sampling_up_exit",
}
TIMELINE_CHECKPOINT_BOUNDARY_PHASES = {
    "checkpoint_edge_pre",
    "checkpoint_edge_post",
    "transformer_segment_pre",
    "transformer_segment_post",
    "transformer_segment_checkpoint_pre",
    "transformer_segment_checkpoint_post",
    "primitive_forward_pre",
    "primitive_forward_post",
    "primitive_backward_pre",
    "primitive_backward_post",
}

_ACTIVE_TIMELINE = None
_MEMORY_TIMELINE_CONTEXT = {}
_MEMORY_TIMELINE_OCCURRENCE = itertools.count()

_EVENT_METADATA_DEFAULTS = {
    "execution_phase": "",
    "lead_idx": "",
    "primitive_kind": "",
    "primitive_mode": "",
    "primitive_part": "",
    "primitive_id": "",
    "occurrence_id": "",
    "segment_start": "",
    "segment_end": "",
    "tensor_mib": "",
    "interval_kind": "",
    "interval_start_allocated_mib": "",
    "interval_peak_allocated_mib": "",
    "interval_peak_rise_mib": "",
}


def memory_timeline_iteration_enabled(iter_idx):
    """Return whether this iteration is inside the requested capture window."""
    return (
        TIMELINE_OUTPUT_MAX_ITERS > 0
        and int(iter_idx) >= TIMELINE_OUTPUT_START_ITER
        and int(iter_idx)
        < TIMELINE_OUTPUT_START_ITER + TIMELINE_OUTPUT_MAX_ITERS
    )


def set_memory_timeline_context(**context):
    """Set semantic context copied into subsequent memory events."""
    for key, value in context.items():
        if value is None:
            _MEMORY_TIMELINE_CONTEXT.pop(key, None)
        else:
            _MEMORY_TIMELINE_CONTEXT[key] = value


def get_memory_timeline_context():
    return dict(_MEMORY_TIMELINE_CONTEXT)


def new_memory_timeline_occurrence(prefix):
    lead_idx = _MEMORY_TIMELINE_CONTEXT.get("lead_idx", "na")
    return f"{prefix}:lead{lead_idx}:occ{next(_MEMORY_TIMELINE_OCCURRENCE)}"


def mark_memory_timeline(phase, label="", **metadata):
    tracer = _ACTIVE_TIMELINE
    if tracer is not None:
        merged_metadata = get_memory_timeline_context()
        merged_metadata.update(metadata)
        tracer.mark(phase, label, **merged_metadata)


def get_memory_timeline_peak_mib():
    """Return the true run-wide allocated peak after interval peak resets."""
    if not torch.cuda.is_available():
        return 0.0
    current_peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    tracer = _ACTIVE_TIMELINE
    if tracer is None:
        return current_peak_mib
    return max(current_peak_mib, tracer.overall_peak_allocated_mib)


class _MemoryTimelineBackwardBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, token, phase, label, metadata):
        ctx.phase = phase
        ctx.label = label
        ctx.boundary_metadata = metadata
        return tensor

    @staticmethod
    def backward(ctx, tensor_grad):
        mark_memory_timeline(ctx.phase, ctx.label, **ctx.boundary_metadata)
        return tensor_grad, None, None, None, None


def profile_memory_backward_boundary(tensor, phase, label="", **metadata):
    """Attach an exact backward boundary without changing model numerics."""
    tracer = _ACTIVE_TIMELINE
    if tracer is None or not torch.is_grad_enabled():
        return tensor
    token = tracer.backward_boundary_token(tensor.device)
    return _MemoryTimelineBackwardBoundary.apply(
        tensor, token, phase, label, dict(metadata)
    )


def _compact_label(label, max_len=28):
    label = str(label or "")
    if not label:
        return ""
    if ":" in label:
        name, cls_name = label.split(":", 1)
        parts = [part for part in name.split(".") if part]
        if len(parts) >= 2:
            label = f"{parts[-2]}.{parts[-1]}:{cls_name}"
        elif parts:
            label = f"{parts[-1]}:{cls_name}"
        else:
            label = cls_name
    if len(label) > max_len:
        return label[: max_len - 3] + "..."
    return label


class CudaMemoryTimeline(ContextDecorator):
    """Record a per-iteration CUDA memory timeline with module hooks.

    The tracer is intentionally lightweight: it samples memory at module
    forward/backward boundaries and at manually inserted marks. It does not
    synchronize by default, so timings reflect CPU launch order while memory
    values reflect the CUDA caching allocator state visible at the hook point.
    Use ``synchronize=True`` for more precise but slower traces.
    """

    def __init__(
        self,
        model,
        device=None,
        output_dir="./log/memory_timeline",
        model_type="seq",
        rank=0,
        dp_rank=-1,
        wp_rank=-1,
        iter_idx=0,
        name="iter",
        module_filter=None,
        max_depth=3,
        record_leaf_only=False,
        synchronize=False,
        reset_peak=True,
        enabled=True,
    ):
        self.model = model
        self.device = torch.device(device if device is not None else torch.cuda.current_device())

        self.output_dir = Path(output_dir + "/" + model_type + "/dp_rank_" + str(dp_rank))
        self.rank = rank
        self.wp_rank = wp_rank

        self.iter_idx = iter_idx
        self.name = name
        self.module_filter = module_filter
        self.max_depth = max_depth
        self.record_leaf_only = record_leaf_only
        self.synchronize = synchronize
        self.reset_peak = reset_peak
        # Do not collect events outside the output window. Previously those
        # events were discarded at exit but still slowed every training step.
        self.enabled = (
            enabled
            and torch.cuda.is_available()
            and memory_timeline_iteration_enabled(iter_idx)
        )
        self.events = []
        self._handles = []
        self._module_names = {}
        self._module_depth = {}
        self._start = None
        self._max_driver_used_mib = 0.0
        self._max_external_mib = 0.0
        self._overall_peak_allocated_mib = 0.0
        self._overall_peak_reserved_mib = 0.0
        self._interval_starts = {}
        self._backward_tokens = {}
        self.isolate_primitive_peaks = (
            os.environ.get("MEMORY_TIMELINE_ISOLATE_PRIMITIVE_PEAKS", "0") == "1"
        )

    @property
    def overall_peak_allocated_mib(self):
        return self._overall_peak_allocated_mib

    def backward_boundary_token(self, device):
        device_key = str(device)
        if device_key not in self._backward_tokens:
            self._backward_tokens[device_key] = torch.zeros(
                (), device=device, requires_grad=True
            )
        return self._backward_tokens[device_key]

    def __enter__(self):
        global _ACTIVE_TIMELINE
        if not self.enabled:
            return self
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self._start is None:
            self._start = time.perf_counter()
            if self.reset_peak:
                torch.cuda.reset_peak_memory_stats(self.device)
        if TIMELINE_RECORD_MODULE_HOOKS:
            self._index_modules()
            self._register_hooks()
        self._previous_active_timeline = _ACTIVE_TIMELINE
        _ACTIVE_TIMELINE = self
        self.mark("iter_start")
        return self

    def __exit__(self, exc_type, exc, tb):
        global _ACTIVE_TIMELINE
        if not self.enabled:
            return False
        self.mark("iter_end")
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        _ACTIVE_TIMELINE = getattr(self, "_previous_active_timeline", None)

        if self._should_write_output():
            if TIMELINE_WRITE_CSV:
                self.write_csv()
            if TIMELINE_WRITE_SUMMARY:
                self.write_summary()
            if TIMELINE_WRITE_EVENT_TABLE:
                self.write_event_table()
            if TIMELINE_WRITE_PLOT:
                self.plot()
        return False

    def _should_write_output(self):
        return memory_timeline_iteration_enabled(self.iter_idx)

    def _index_modules(self):
        for name, module in self.model.named_modules():
            if name == "":
                continue
            self._module_names[module] = name
            self._module_depth[module] = name.count(".") + 1

    def _should_trace(self, module):
        name = self._module_names.get(module, "")
        if not name:
            return False
        if self.max_depth is not None and self._module_depth.get(module, 999) > self.max_depth:
            return False
        if self.record_leaf_only and any(True for _ in module.children()):
            return False
        if self.module_filter is None:
            return True
        if callable(self.module_filter):
            return bool(self.module_filter(name, module))
        return any(token in name or token in module.__class__.__name__ for token in self.module_filter)

    def _register_hooks(self):
        if not TIMELINE_RECORD_MODULE_HOOKS:
            return
        for module in self._module_names:
            if not self._should_trace(module):
                continue
            name = self._module_names[module]
            cls_name = module.__class__.__name__
            label = f"{name}:{cls_name}"
            self._handles.append(module.register_forward_pre_hook(lambda m, inputs, label=label: self.mark("fwd_pre", label)))
            self._handles.append(module.register_forward_hook(lambda m, inputs, output, label=label: self.mark("fwd_post", label)))
            if hasattr(module, "register_full_backward_pre_hook"):
                self._handles.append(module.register_full_backward_pre_hook(lambda m, grad_output, label=label: self.mark("bwd_pre", label)))
            self._handles.append(module.register_full_backward_hook(lambda m, grad_input, grad_output, label=label: self.mark("bwd_post", label)))

    def mark(self, phase, label="", **metadata):
        if not self.enabled:
            return
        if self._start is None:
            # Allow passing the tracer directly into instrumented code even if
            # the caller forgot to use it as a context manager.
            self._start = time.perf_counter()
            if self.reset_peak:
                torch.cuda.reset_peak_memory_stats(self.device)
        if self.synchronize:
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()

        allocated_mib = torch.cuda.memory_allocated(self.device) / 1024**2
        reserved_mib = torch.cuda.memory_reserved(self.device) / 1024**2
        max_allocated_mib = torch.cuda.max_memory_allocated(self.device) / 1024**2
        max_reserved_mib = torch.cuda.max_memory_reserved(self.device) / 1024**2
        self._overall_peak_allocated_mib = max(
            self._overall_peak_allocated_mib, max_allocated_mib
        )
        self._overall_peak_reserved_mib = max(
            self._overall_peak_reserved_mib, max_reserved_mib
        )

        driver_used_mib, external_mib = self._driver_memory_mib(reserved_mib)
        if driver_used_mib is not None:
            self._max_driver_used_mib = max(self._max_driver_used_mib, driver_used_mib)
        if external_mib is not None:
            self._max_external_mib = max(self._max_external_mib, external_mib)

        event_metadata = dict(_EVENT_METADATA_DEFAULTS)
        event_metadata.update(metadata)
        occurrence_id = str(event_metadata.get("occurrence_id", ""))

        interval_kind = ""
        if phase.endswith("_forward_pre"):
            interval_kind = "forward"
        elif phase.endswith("_backward_pre"):
            interval_kind = "backward"
        if interval_kind and occurrence_id:
            interval_key = f"{interval_kind}:{occurrence_id}"
            self._interval_starts[interval_key] = allocated_mib
            event_metadata["interval_kind"] = interval_kind

        interval_post_kind = ""
        if phase.endswith("_forward_post"):
            interval_post_kind = "forward"
        elif phase.endswith("_backward_post"):
            interval_post_kind = "backward"
        if interval_post_kind and occurrence_id:
            interval_key = f"{interval_post_kind}:{occurrence_id}"
            interval_start = self._interval_starts.pop(interval_key, None)
            if interval_start is not None:
                event_metadata["interval_kind"] = interval_post_kind
                event_metadata["interval_start_allocated_mib"] = interval_start
                event_metadata["interval_peak_allocated_mib"] = max_allocated_mib
                event_metadata["interval_peak_rise_mib"] = max(
                    0.0, max_allocated_mib - interval_start
                )

        event = {
            "idx": len(self.events),
            "time_s": now - self._start,
            "phase": phase,
            "label": label,
            "allocated_mib": allocated_mib,
            "reserved_mib": reserved_mib,
            "max_allocated_mib": max_allocated_mib,
            "max_reserved_mib": max_reserved_mib,
            "overall_max_allocated_mib": self._overall_peak_allocated_mib,
            "overall_max_reserved_mib": self._overall_peak_reserved_mib,
            "driver_used_mib": driver_used_mib,
            "external_mib": external_mib,
            "max_driver_used_mib": self._max_driver_used_mib if driver_used_mib is not None else None,
            "max_external_mib": self._max_external_mib if external_mib is not None else None,
        }
        event.update(event_metadata)
        self.events.append(event)

        if self.isolate_primitive_peaks and interval_kind and occurrence_id:
            torch.cuda.reset_peak_memory_stats(self.device)

    def _driver_memory_mib(self, reserved_mib):
        if not hasattr(torch.cuda, "mem_get_info"):
            return None, None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        except TypeError:
            try:
                with torch.cuda.device(self.device):
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
            except Exception:
                return None, None
        except Exception:
            return None, None
        driver_used_mib = (total_bytes - free_bytes) / 1024**2
        external_mib = max(0.0, driver_used_mib - reserved_mib)
        return driver_used_mib, external_mib

    def csv_path(self):
        return self.output_dir / f"{self.name}_wp_rank{self.wp_rank}_iter{self.iter_idx}.csv"

    def png_path(self):
        return self.output_dir / f"{self.name}_wp_rank{self.wp_rank}_iter{self.iter_idx}.png"

    def summary_path(self):
        return self.output_dir / f"{self.name}_wp_rank{self.wp_rank}_iter{self.iter_idx}_summary.csv"

    def event_table_path(self):
        return self.output_dir / f"{self.name}_wp_rank{self.wp_rank}_iter{self.iter_idx}_events.csv"

    def write_csv(self):
        if not self.events:
            return
        with self.csv_path().open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.events[0].keys()))
            writer.writeheader()
            writer.writerows(self.events)

    def write_summary(self):
        if not self.events:
            return

        def max_metric(key):
            vals = [e[key] for e in self.events if e.get(key) is not None]
            return max(vals) if vals else None

        final = self.events[-1]
        summary = {
            "name": self.name,
            "rank": self.rank,
            "wp_rank": self.wp_rank,
            "iter_idx": self.iter_idx,
            "duration_s": final["time_s"],
            "peak_allocated_mib": max_metric("allocated_mib"),
            "peak_reserved_mib": max_metric("reserved_mib"),
            "peak_max_allocated_mib": self._overall_peak_allocated_mib,
            "peak_max_reserved_mib": self._overall_peak_reserved_mib,
            "peak_driver_used_mib": max_metric("driver_used_mib"),
            "peak_external_mib": max_metric("external_mib"),
            "final_allocated_mib": final.get("allocated_mib"),
            "final_reserved_mib": final.get("reserved_mib"),
            "final_driver_used_mib": final.get("driver_used_mib"),
            "final_external_mib": final.get("external_mib"),
        }
        with self.summary_path().open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)

    def _is_module_hook_event(self, phase):
        return phase in {"fwd_pre", "fwd_post", "bwd_pre", "bwd_post"}

    def _is_sampling_internal_event(self, phase):
        return phase.startswith("sampling_") and phase not in TIMELINE_SAMPLING_BOUNDARY_PHASES

    def _is_checkpoint_body_event(self, phase):
        return phase in {
            "checkpoint_body_pre",
            "checkpoint_body_post",
            "transformer_segment_body_pre",
            "transformer_segment_body_post",
            "primitive_body_pre",
            "primitive_body_post",
        }

    def _is_forward_checkpoint_plot_event(self, event):
        phase = event.get("phase", "")
        return phase in {
            "checkpoint_edge_post",
            "transformer_segment_checkpoint_post",
            "primitive_forward_post",
        }

    def _is_backward_checkpoint_plot_event(self, event, in_backward_region):
        if not in_backward_region:
            return False
        phase = event.get("phase", "")
        label = str(event.get("label", ""))
        if phase == "transformer_segment_body_pre" and ":checkpoint" in label:
            return True
        if phase in {"checkpoint_body_pre", "primitive_backward_pre"}:
            return True
        return False

    def _is_checkpoint_boundary_event(self, event):
        phase = event.get("phase", "")
        if phase in TIMELINE_SAMPLING_BOUNDARY_PHASES:
            return True
        return phase in TIMELINE_CHECKPOINT_BOUNDARY_PHASES

    def _should_plot_event(self, event, in_backward_region=False):
        phase = event.get("phase", "")
        if phase in TIMELINE_KEY_BOUNDARY_PHASES:
            return True
        if TIMELINE_COMPACT_CHECKPOINT_EVENTS:
            if phase in TIMELINE_SAMPLING_BOUNDARY_PHASES:
                return True
            if self._is_forward_checkpoint_plot_event(event):
                return True
            if self._is_backward_checkpoint_plot_event(event, in_backward_region):
                return True
            return phase in TIMELINE_COARSE_PHASES
        if self._is_checkpoint_boundary_event(event):
            return True
        if self._is_module_hook_event(phase):
            return bool(TIMELINE_SHOW_MODULE_HOOKS)
        if self._is_sampling_internal_event(phase):
            return bool(TIMELINE_SHOW_SAMPLING_INTERNALS)
        if self._is_checkpoint_body_event(phase):
            return bool(TIMELINE_SHOW_CHECKPOINT_BODY)
        return phase in TIMELINE_COARSE_PHASES

    def _is_annotation_event(self, event):
        phase = event.get("phase", "")
        if self._is_module_hook_event(phase):
            return bool(TIMELINE_ANNOTATE_MODULE_HOOKS)
        return True

    def _short_event_label(self, event):
        phase = event.get("phase", "")
        label = _compact_label(event.get("label", ""), max_len=18 if self._is_module_hook_event(phase) else 28)
        allocated = event.get("allocated_mib", None)
        mem = "" if allocated is None else f"{allocated / 1024.0:.1f}G"
        if label:
            return f"{phase}\n{label}\n{mem}"
        return f"{phase}\n{mem}"

    def _event_x(self, event, use_event_axis):
        return event["idx"] if use_event_axis else event["time_s"]

    def _first_phase_x(self, phase, use_event_axis):
        for event in self.events:
            if event.get("phase") == phase:
                return self._event_x(event, use_event_axis)
        return None

    def write_event_table(self):
        if not self.events:
            return
        rows = []
        for event in self.events:
            if not self._is_annotation_event(event):
                continue
            rows.append(
                {
                    "idx": event.get("idx", ""),
                    "time_s": event.get("time_s", ""),
                    "phase": event.get("phase", ""),
                    "label": event.get("label", ""),
                    "allocated_mib": event.get("allocated_mib", ""),
                    "allocated_gib": "" if event.get("allocated_mib") is None else event.get("allocated_mib") / 1024.0,
                    "reserved_mib": event.get("reserved_mib", ""),
                    "reserved_gib": "" if event.get("reserved_mib") is None else event.get("reserved_mib") / 1024.0,
                    "max_allocated_mib": event.get("max_allocated_mib", ""),
                    "driver_used_mib": event.get("driver_used_mib", ""),
                }
            )
        if not rows:
            return
        with self.event_table_path().open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def plot(self):
        if not self.events:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        use_event_axis = TIMELINE_X_AXIS in {"event", "events", "idx", "index", "order"}
        xs = [e["idx"] if use_event_axis else e["time_s"] for e in self.events]
        allocated = [e["allocated_mib"] for e in self.events]
        reserved = [e["reserved_mib"] for e in self.events]
        max_allocated = [e["max_allocated_mib"] for e in self.events]
        max_reserved = [e["max_reserved_mib"] for e in self.events]
        driver_used = [e["driver_used_mib"] for e in self.events]
        external = [e["external_mib"] for e in self.events]
        has_driver = any(v is not None for v in driver_used)
        before_backward_x = self._first_phase_x("before_backward", use_event_axis)
        after_backward_x = self._first_phase_x("after_backward", use_event_axis)

        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6.2), dpi=180, sharex=True)

        ax.plot(xs, allocated, label="allocated (live tensors)", linewidth=1.6)
        ax.plot(xs, reserved, label="reserved (PyTorch cache)", linewidth=1.2, alpha=0.8)
        if has_driver:
            ax.plot(xs, driver_used, label="driver used (total CUDA)", linewidth=1.1, alpha=0.85)
        ax.set_ylabel("current memory (MiB)")
        ax.set_title(f"CUDA memory timeline: {self.name}, rank {self.rank}, iter {self.iter_idx}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7)

        ax2.plot(xs, max_allocated, label="max allocated", linewidth=1.1, linestyle="--", alpha=0.85)
        ax2.plot(xs, max_reserved, label="max reserved", linewidth=1.1, linestyle="--", alpha=0.85)
        if has_driver:
            ax2.plot(xs, external, label="external = driver used - reserved", linewidth=1.0, alpha=0.8)
        ax2.set_xlabel("event order in one training iteration" if use_event_axis else "time in one training iteration (s)")
        ax2.set_ylabel("peak / external (MiB)")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best", fontsize=7)

        if TIMELINE_ANNOTATE_EVENTS:
            labeled = 0
            y_top = ax.get_ylim()[1]
            y_min = ax.get_ylim()[0]
            y_span = max(1.0, y_top - y_min)
            levels = [0.98, 0.90, 0.82, 0.74, 0.66, 0.58]
            for e in self.events:
                x = self._event_x(e, use_event_axis)
                in_backward_region = (
                    before_backward_x is not None
                    and x >= before_backward_x
                    and (after_backward_x is None or x <= after_backward_x)
                )
                if not self._is_annotation_event(e) or not self._should_plot_event(e, in_backward_region):
                    continue
                y = e.get("allocated_mib", None)
                if y is None:
                    continue
                phase = e.get("phase", "")
                is_key_boundary = phase in TIMELINE_KEY_BOUNDARY_PHASES
                is_coarse = phase in TIMELINE_COARSE_PHASES
                is_module_hook = self._is_module_hook_event(phase)
                is_backward_checkpoint = self._is_backward_checkpoint_plot_event(e, in_backward_region)
                is_checkpoint = (
                    self._is_checkpoint_boundary_event(e)
                    or self._is_forward_checkpoint_plot_event(e)
                    or is_backward_checkpoint
                )
                is_sampling = phase in TIMELINE_SAMPLING_BOUNDARY_PHASES
                color = (
                    "#d62728" if is_key_boundary
                    else ("#2ca02c" if is_backward_checkpoint
                          else ("#9467bd" if is_checkpoint
                                else ("0.45" if is_coarse else ("#1f77b4" if is_module_hook else "#d62728"))))
                )
                alpha = (
                    0.92 if is_key_boundary
                    else (0.88 if is_backward_checkpoint
                          else (0.78 if is_checkpoint
                                else (0.35 if is_coarse else (0.24 if is_module_hook else 0.45))))
                )
                linewidth = (
                    1.5 if is_key_boundary
                    else (0.45 if is_sampling
                          else (0.22 if is_checkpoint and not TIMELINE_CHECKPOINT_EVENT_LINES
                                else (0.8 if is_checkpoint else (0.6 if is_coarse else (0.22 if is_module_hook else 0.45)))))
                )
                zorder = 6 if is_key_boundary else 1
                draw_event_line = (
                    is_key_boundary
                    or is_sampling
                    or (is_checkpoint and bool(TIMELINE_CHECKPOINT_EVENT_LINES))
                    or (is_coarse and not is_checkpoint)
                )
                if draw_event_line:
                    ax.axvline(x, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
                    ax2.axvline(x, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
                marker = "^" if is_backward_checkpoint else ("D" if is_checkpoint else "o")
                marker_size = (
                    34 if is_backward_checkpoint
                    else (28 if is_checkpoint else (18 if is_key_boundary else (8 if is_module_hook else 12)))
                )
                ax.scatter(
                    [x],
                    [y],
                    s=marker_size,
                    marker=marker,
                    color=color,
                    edgecolors="white" if is_checkpoint else "none",
                    linewidths=0.45 if is_checkpoint else 0.0,
                    alpha=0.95 if is_checkpoint else 0.85,
                    zorder=7 if is_checkpoint else 5,
                )
                should_label = (
                    is_key_boundary
                    or (is_checkpoint and bool(TIMELINE_LABEL_CHECKPOINT_EVENTS))
                    or ((not is_module_hook and not is_checkpoint) and not TIMELINE_COMPACT_CHECKPOINT_EVENTS)
                    or (is_module_hook and bool(TIMELINE_LABEL_MODULE_HOOKS))
                )
                if not should_label or labeled >= TIMELINE_MAX_EVENT_ANNOTATIONS:
                    continue
                text_y = y_min + y_span * levels[labeled % len(levels)]
                ax.text(
                    x,
                    text_y,
                    self._short_event_label(e),
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=3.2 if is_module_hook else 4.8,
                    color=color,
                    alpha=0.95,
                )
                labeled += 1

        fig.tight_layout()
        fig.savefig(self.png_path())
        plt.close(fig)
