import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist


class DistributedAdamW(torch.optim.Optimizer):
    """ZeRO-1-style AdamW with sharded optimizer states.

    Each replica group owns a balanced subset of parameters. Gradients are
    reduced over the parameter-specific replica group, only the owner rank keeps
    AdamW states and applies the update, and the updated parameter is broadcast
    back to the rest of that group.

    This optimizer intentionally does not wrap the model and does not alter
    forward precision. It is designed to compose with custom activation
    checkpointing/offload and TERRA's model/window-parallel gradient handling.
    """

    def __init__(
        self,
        named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        process_group=None,
        average_grad: bool = True,
        state_dtype: torch.dtype = torch.float32,
        balance_by_numel: bool = True,
        bucket_cap_mb: float = 50.0,

        manager = None,
    ):
        self.manager = manager

        named_params = [(name, p) for name, p in named_parameters if p.requires_grad]
        if not named_params:
            raise ValueError("DistributedAdamW received no trainable parameters")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            average_grad=average_grad,
            state_dtype=state_dtype,
        )
        super().__init__([p for _, p in named_params], defaults)

        self.named_params = named_params
        self.process_group = process_group
        self.global_step = 0
        self.bucket_cap_bytes = int(bucket_cap_mb * 1024 * 1024)

        self.param_comm_label = {}
        self.comm_groups = {}
        for _, p in named_params:
            label, group = self._resolve_param_group(p)
            self.param_comm_label[p] = label
            self._ensure_comm_group(label, group)

        if self.comm_groups:
            self.group_size = max(info["size"] for info in self.comm_groups.values())
            self.group_rank = 0
            self.group_ranks = []
        else:
            self.group_size = 1
            self.group_rank = 0
            self.group_ranks = [0]

        self.owner_by_param = self._assign_owners(named_params, balance_by_numel)

    def _resolve_param_group(self, p):
        reduce_group_name = getattr(p, "terra_grad_reduce_group", None)
        if self.manager is not None and dist.is_available() and dist.is_initialized():
            if reduce_group_name == "xfmr_tp_param_group":
                group = getattr(self.manager, "xfmr_tp_param_replica_group", None)
                if group is None:
                    raise RuntimeError("manager.xfmr_tp_param_replica_group is not initialized")
                return "xfmr_tp_param_replica_group", group
            if reduce_group_name is not None:
                raise ValueError(f"Unknown terra_grad_reduce_group: {reduce_group_name}")

            group = getattr(self.manager, "full_param_replica_group", None)
            if group is None:
                raise RuntimeError("manager.full_param_replica_group is not initialized")
            return "full_param_replica_group", group

        return "default", self.process_group

    def _ensure_comm_group(self, label, group):
        if label in self.comm_groups:
            return
        if dist.is_available() and dist.is_initialized():
            self.comm_groups[label] = {
                "group": group,
                "size": dist.get_world_size(group=group),
                "rank": dist.get_rank(group=group),
                "ranks": self._get_group_ranks(group),
            }
        else:
            self.comm_groups[label] = {
                "group": group,
                "size": 1,
                "rank": 0,
                "ranks": [0],
            }

    def _get_group_ranks(self, group) -> List[int]:
        if group is None:
            return list(range(dist.get_world_size()))
        if hasattr(dist, "get_process_group_ranks"):
            return list(dist.get_process_group_ranks(group))

        rank = torch.tensor([dist.get_rank()], device=torch.cuda.current_device())
        gathered = [torch.zeros_like(rank) for _ in range(dist.get_world_size(group=group))]
        dist.all_gather(gathered, rank, group=group)
        return [int(x.item()) for x in gathered]

    def _assign_owners(self, named_params, balance_by_numel):
        owner_by_param = {}
        params_by_label = {}
        for name, p in named_params:
            params_by_label.setdefault(self.param_comm_label[p], []).append((name, p))

        for label, label_named_params in params_by_label.items():
            group_size = self.comm_groups[label]["size"]
            if group_size == 1:
                for _, p in label_named_params:
                    owner_by_param[p] = 0
                continue

            if not balance_by_numel:
                for i, (_, p) in enumerate(label_named_params):
                    owner_by_param[p] = i % group_size
                continue

            loads = [0 for _ in range(group_size)]
            # Large tensors first gives a better memory balance across ranks.
            for _, p in sorted(label_named_params, key=lambda item: item[1].numel(), reverse=True):
                owner = min(range(group_size), key=lambda r: loads[r])
                owner_by_param[p] = owner
                loads[owner] += p.numel()
        return owner_by_param

    def _is_owner(self, p):
        label = self.param_comm_label[p]
        return self.owner_by_param[p] == self.comm_groups[label]["rank"]

    def _owner_global_rank(self, p):
        label = self.param_comm_label[p]
        return self.comm_groups[label]["ranks"][self.owner_by_param[p]]

    def _group_params_by_comm_label(self, params):
        grouped = {}
        for p in params:
            grouped.setdefault(self.param_comm_label[p], []).append(p)
        return grouped

    def _grad_average_divisor(self):
        if self.manager is not None:
            return max(1, int(self.manager.get_dp_group_size()))
        return max(1, self.group_size)

    def _make_buckets(self, params):
        buckets = []
        cur = []
        cur_bytes = 0
        cur_key = None

        for p in params:
            key = (p.device, p.dtype)
            nbytes = p.numel() * p.element_size()
            flush = (
                cur
                and (
                    key != cur_key
                    or (cur_bytes + nbytes > self.bucket_cap_bytes and cur_bytes > 0)
                )
            )
            if flush:
                buckets.append(cur)
                cur = []
                cur_bytes = 0

            cur.append(p)
            cur_bytes += nbytes
            cur_key = key

            if cur_bytes >= self.bucket_cap_bytes:
                buckets.append(cur)
                cur = []
                cur_bytes = 0
                cur_key = None

        if cur:
            buckets.append(cur)
        return buckets

    def _all_reduce_grad_buckets(self, params, average_grad):
        params_with_grad = [p for p in params if p.grad is not None]
        for label, label_params in self._group_params_by_comm_label(params_with_grad).items():
            comm_info = self.comm_groups[label]
            if comm_info["size"] <= 1:
                continue

            for bucket in self._make_buckets(label_params):
                flat = torch.empty(
                    sum(p.grad.numel() for p in bucket),
                    device=bucket[0].grad.device,
                    dtype=bucket[0].grad.dtype,
                )
                offset = 0
                for p in bucket:
                    numel = p.grad.numel()
                    flat[offset:offset + numel].copy_(p.grad.detach().reshape(-1))
                    offset += numel

                dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=comm_info["group"])

                if average_grad:
                    flat.div_(self._grad_average_divisor())

                offset = 0
                for p in bucket:
                    numel = p.grad.numel()
                    p.grad.copy_(flat[offset:offset + numel].view_as(p.grad))
                    offset += numel

    def _broadcast_param_buckets(self, params):
        for label, label_params in self._group_params_by_comm_label(params).items():
            comm_info = self.comm_groups[label]
            if comm_info["size"] <= 1:
                continue

            for owner in range(comm_info["size"]):
                owner_params = [p for p in label_params if self.owner_by_param[p] == owner]
                if not owner_params:
                    continue
                src = comm_info["ranks"][owner]
                for bucket in self._make_buckets(owner_params):
                    flat = torch.empty(
                        sum(p.numel() for p in bucket),
                        device=bucket[0].device,
                        dtype=bucket[0].dtype,
                    )

                    if comm_info["rank"] == owner:
                        offset = 0
                        for p in bucket:
                            numel = p.numel()
                            flat[offset:offset + numel].copy_(p.detach().reshape(-1))
                            offset += numel

                    dist.broadcast(flat, src=src, group=comm_info["group"])

                    if comm_info["rank"] != owner:
                        offset = 0
                        for p in bucket:
                            numel = p.numel()
                            p.copy_(flat[offset:offset + numel].view_as(p))
                            offset += numel

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            average_grad = group["average_grad"]
            state_dtype = group["state_dtype"]
            params = [p for p in group["params"] if p.grad is not None]

            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("DistributedAdamW does not support sparse gradients")

            self._all_reduce_grad_buckets(params, average_grad)

            for p in params:
                if self._is_owner(p):
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(
                            p, dtype=state_dtype, memory_format=torch.preserve_format
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, dtype=state_dtype, memory_format=torch.preserve_format
                        )

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    state["step"] += 1
                    step = state["step"]

                    grad_for_update = p.grad.detach()
                    if grad_for_update.dtype != state_dtype:
                        grad_for_update = grad_for_update.to(dtype=state_dtype)

                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)

                    exp_avg.mul_(beta1).add_(grad_for_update, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad_for_update, grad_for_update, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    step_size = lr / bias_correction1
                    denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)

                    update = exp_avg / denom
                    if update.dtype != p.dtype:
                        update = update.to(dtype=p.dtype)
                    p.add_(update, alpha=-step_size)

            self._broadcast_param_buckets(params)

        return loss

    def local_state_numel(self):
        total = 0
        for p, state in self.state.items():
            if self._is_owner(p):
                total += sum(v.numel() for v in state.values() if torch.is_tensor(v))
        return total
