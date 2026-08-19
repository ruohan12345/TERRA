from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class FlopsSummary:
    enabled: bool
    local_flops_per_step: float = 0.0
    world_flops_per_step: float = 0.0
    world_avg_flops_per_step: float = 0.0
    active_steps: int = 0
    event_count: int = 0

    def local_tflops(self, step_time_s):
        if step_time_s <= 0:
            return 0.0
        return self.local_flops_per_step / step_time_s / 1e12

    def world_tflops(self, step_time_s):
        if step_time_s <= 0:
            return 0.0
        return self.world_flops_per_step / step_time_s / 1e12


class TorchFlopsProfiler:
    def __init__(
        self,
        enabled,
        rank,
        schedule_steps,
        trace_dir="./log/profiler/rank0",
        write_trace=False,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ):
        self.enabled = bool(enabled)
        self.rank = rank
        self.schedule_steps = schedule_steps
        self.trace_dir = trace_dir
        self.write_trace = bool(write_trace)
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.prof = None
        self.summary_cache = None

        if len(schedule_steps) != 4:
            raise ValueError(f"schedule_steps must be [wait, warmup, active, repeat], got {schedule_steps}")
        self.wait_iters, self.warmup_iters, self.active_iters, self.repeat_iters = schedule_steps

    def start(self):
        if not self.enabled:
            return

        handler = None
        if self.write_trace and self.rank == 0:
            handler = torch.profiler.tensorboard_trace_handler(self.trace_dir)

        self.prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=self.wait_iters,
                warmup=self.warmup_iters,
                active=self.active_iters,
                repeat=self.repeat_iters,
            ),
            on_trace_ready=handler,
            with_flops=True,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        )
        self.prof.start()

    def step(self):
        if self.prof is not None:
            self.prof.step()

    def stop(self):
        if self.prof is not None:
            self.prof.stop()

    def _local_total_flops(self):
        if self.prof is None:
            return 0.0, 0

        total_flops = 0.0
        event_count = 0
        for evt in self.prof.key_averages():
            flops = getattr(evt, "flops", None)
            if flops is not None:
                total_flops += float(flops)
                if flops > 0:
                    event_count += 1
        return total_flops, event_count

    def summarize(self):
        if self.summary_cache is not None:
            return self.summary_cache

        if not self.enabled or self.prof is None:
            self.summary_cache = FlopsSummary(enabled=False)
            return self.summary_cache

        local_total_flops, event_count = self._local_total_flops()
        active_steps = max(1, int(self.active_iters) * int(self.repeat_iters))
        local_flops_per_step = local_total_flops / active_steps

        world_flops_per_step = local_flops_per_step
        world_avg_flops_per_step = local_flops_per_step
        if dist.is_available() and dist.is_initialized():
            tensor = torch.tensor(
                [local_flops_per_step, 1.0],
                dtype=torch.float64,
                device=torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu"),
            )
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            world_flops_per_step = float(tensor[0].item())
            world_avg_flops_per_step = float((tensor[0] / tensor[1]).item())

        self.summary_cache = FlopsSummary(
            enabled=True,
            local_flops_per_step=local_flops_per_step,
            world_flops_per_step=world_flops_per_step,
            world_avg_flops_per_step=world_avg_flops_per_step,
            active_steps=active_steps,
            event_count=event_count,
        )
        return self.summary_cache

    def print_top_ops(self, row_limit=20):
        if not self.enabled or self.prof is None:
            return

        events = [e for e in self.prof.key_averages() if e.key.startswith("aten::")]
        events = sorted(
            events,
            key=lambda e: getattr(e, "device_time_total", e.cpu_time_total),
            reverse=True,
        )

        print("-----------------------------Top CUDA ops------------------------------")
        for evt in events[:row_limit]:
            time_ms = getattr(evt, "device_time_total", evt.cpu_time_total) / 1000
            flops = getattr(evt, "flops", 0) or 0
            print(f"{evt.key:30s} {time_ms:8.3f} ms  calls={evt.count}  flops={flops:.3e}")
