"""Process groups for fine DMP storage and per-convolution compute layouts.

Ranks own consecutive spatial strips using explicit global ownership ranges.
A compute team of Ci*Co adjacent ranks temporarily handles one larger strip.
DP is the outer sample-replica axis. Sampling and Transformer factorizations
reuse exactly the same per-replica ranks, rather than multiplying their sizes.
"""
import torch
import torch.distributed as dist
from terra.runtime.accelerator import control_device


class DMPMesh:
    def __init__(self, ci_parts=1, co_parts=1, group=None, *, dp_size=1,
                 wp_size=None, sp_size=1, tp_size=1, sp_tp_placement="tp_first"):
        from terra.core.parallel_manager.rank_manager import ParallelManager
        for name, value in (("ci_parts", ci_parts), ("co_parts", co_parts),
                            ("dp_size", dp_size), ("sp_size", sp_size), ("tp_size", tp_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.group = group
        self._owned_groups = []
        self._closed = False
        self.ci_parts, self.co_parts = ci_parts, co_parts
        self.tp_size = ci_parts * co_parts
        self.records = []
        self.stats = {"wire_dtypes": set(), "logical_dtypes": set(),
                      "logical_memory_formats": set(), "all_to_all_calls": 0,
                      "all_to_all_backward_calls": 0, "halo_calls": 0,
                      "halo_backward_calls": 0, "max_boundary_input_numel": 0,
                      "max_compute_input_numel": 0, "max_partial_output_numel": 0,
                      "max_boundary_output_numel": 0, "max_packed_exchange_numel": 0,
                      "max_halo_output_numel": 0}
        initialized = dist.is_available() and dist.is_initialized()
        if initialized:
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            signatures = [None] * self.world_size
            signature=(ci_parts,co_parts,dp_size,wp_size,sp_size,tp_size,sp_tp_placement)
            dist.all_gather_object(signatures, signature)
            if any(value != signatures[0] for value in signatures):
                raise ValueError("All WORLD ranks must agree on the hybrid topology")
            if group is not None and group is not dist.group.WORLD:
                raise ValueError("Pass dp_size and former sizes; arbitrary DMP groups are not supported")
            if self.world_size % dp_size:
                raise ValueError("WORLD size must be divisible by dp_size")
            self.size = self.world_size // dp_size
            self.backend = str(dist.get_backend())
        else:
            if group is not None:
                raise RuntimeError("Initialize torch.distributed before passing a process group")
            self.global_rank = self.rank = 0
            self.size = self.world_size = 1
            self.backend = "local"
        if self.size % (sp_size * tp_size):
            raise ValueError("DMP degree must be divisible by FORMER SP * TP")
        if wp_size is None:
            wp_size=self.size // (sp_size * tp_size)
        self.former=ParallelManager(rank=self.global_rank,world_size=self.world_size,
            dp_size=dp_size,wp_size=wp_size,sp_size=sp_size,tp_size=tp_size,
            sp_tp_placement=sp_tp_placement)
        self.dp_size,self.dp_rank=dp_size,self.former.get_rank("dp")
        self.rank=self.former.get_rank("dmp")
        self.group_ranks=list(range(self.dp_rank*self.size,(self.dp_rank+1)*self.size))
        if initialized:
            self.former.create_process_groups()
            self.group=self.former.get_group("dmp")
            self.dp_group=self.former.get_group("dp")
        else:
            self.group=self.dp_group=None
        self.dmp_degree = self.size
        if self.size % self.tp_size:
            raise ValueError("dmp_degree must be divisible by ci_parts * co_parts")
        self.spatial_size = self.size // self.tp_size
        self.spatial_rank, self.tp_rank = divmod(self.rank, self.tp_size)
        self.co_rank, self.ci_rank = divmod(self.tp_rank, self.ci_parts)
        self.tp_ranks = self.group_ranks[
            self.spatial_rank * self.tp_size:(self.spatial_rank + 1) * self.tp_size]
        self.spatial_ranks = [self.group_ranks[s * self.tp_size + self.tp_rank]
                              for s in range(self.spatial_size)]
        self.tp_group = self.spatial_group = self.group
        # WP x SP replicas share each TP parameter shard. LayerNorm parameters
        # instead use all DMP ranks, because TP boundaries shard tokens too.
        f_tp=self.former.get_rank("tp")
        self.former_replica_ranks=[r for r in self.group_ranks
                                  if self.former.get_rank("tp",global_rank=r)==f_tp]
        self.former_replica_group=self.group
        if initialized:
            cache = {tuple(range(self.world_size)): dist.group.WORLD}
            for dp in range(dp_size):
                ranks=tuple(range(dp*self.size,(dp+1)*self.size))
                cache[ranks]=self.group if dp==self.dp_rank else dist.GroupMember.NON_GROUP_MEMBER
            specifications=[]
            # Every WORLD rank creates every replica's groups in the same order.
            for dp in range(dp_size):
                base=dp*self.size
                specifications += [tuple(base+s*self.tp_size+t for t in range(self.tp_size))
                                   for s in range(self.spatial_size)]
                specifications += [tuple(base+s*self.tp_size+t for s in range(self.spatial_size))
                                   for t in range(self.tp_size)]
                specifications += [tuple(r for r in range(base,base+self.size)
                                    if self.former.get_rank("tp",global_rank=r)==t)
                                   for t in range(tp_size)]
            for ranks in specifications:
                if ranks not in cache:
                    handle = dist.new_group(ranks=list(ranks), backend=self.backend)
                    cache[ranks] = handle
                    if self.global_rank in ranks:
                        self._owned_groups.append(handle)
            self.tp_group = cache[tuple(self.tp_ranks)]
            self.spatial_group = cache[tuple(self.spatial_ranks)]
            self.former_replica_group=cache[tuple(self.former_replica_ranks)]

    def check_open(self):
        if self._closed:
            raise RuntimeError("DMP mesh is closed")

    def record(self, kind, **values):
        self.records.append({"kind": kind, **values})
        if len(self.records)>512:
            del self.records[:-512]

    def peak(self, name, value):
        self.stats[name] = max(self.stats.get(name, 0), int(value))

    def all_finite(self, local_finite):
        self.check_open()
        if self.world_size == 1:
            return bool(local_finite)
        device = control_device(self.backend)
        value = torch.tensor(int(bool(local_finite)), dtype=torch.int32, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
        return bool(value.item())

    def describe(self):
        return {"rank": self.rank, "global_rank": self.global_rank, "dmp_degree": self.size,
                "world_size":self.world_size,"dp_size":self.dp_size,"dp_rank":self.dp_rank,
                "former":{"wp_size":self.former.get_group_size("wp"),
                          "sp_size":self.former.get_group_size("sp"),
                          "tp_size":self.former.get_group_size("tp"),
                          "wp_rank":self.former.get_rank("wp"),
                          "sp_rank":self.former.get_rank("sp"),
                          "tp_rank":self.former.get_rank("tp")},
                "ci_parts": self.ci_parts, "co_parts": self.co_parts,
                "tp_size": self.tp_size, "spatial_size": self.spatial_size,
                "ci_rank": self.ci_rank, "co_rank": self.co_rank,
                "spatial_rank": self.spatial_rank, "tp_ranks": list(self.tp_ranks),
                "spatial_ranks": list(self.spatial_ranks), "backend": self.backend,
                "boundary_layout": "full_channels_owned_height_ranges",
                "cpu_low_precision_wire": "float32", "cuda_wire": "native", "npu_wire": "native_with_empty_peer_padding",
                "channel_reduction": "float32_for_fp16_bf16",
                "statistics": {key: sorted(value) if isinstance(value, set) else value
                               for key, value in self.stats.items()},
                "temporary_buffers_are_not_bounded_by_1_over_dmp": True}

    def close(self):
        """Call on all ranks after outstanding forward/backward work completes."""
        if self._closed:
            return
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            for handle in reversed(self._owned_groups):
                dist.destroy_process_group(handle)
            self.former.destroy_process_groups()
        self._owned_groups.clear()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
