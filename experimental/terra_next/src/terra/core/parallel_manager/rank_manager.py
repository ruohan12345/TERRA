"""Parallel-rank queries and optional torch.distributed process groups.

Logical queries do not import torch. Communication groups are created only by
an explicit collective call after external distributed initialization.
All ranks refer to processes, not physical machines. No device selection or PP.
"""


class ParallelManager:
    """Query a process's rank and size in its own logical parallel groups.

    Transformer layout: DP x WP x SP x TP (TP-fastest by default).
    Sampling layout:    DP x DMP, where DMP = WP * SP * TP.
    DMP reuses the same ranks; it is not another independent product axis.
    DMPMesh additionally factorizes these DMP ranks into convolution Ci/Co/S.

    ``sp_tp_placement='sp_first'`` instead uses DP x WP x TP x SP.
    Within each axis group, ranks are ordered by increasing global rank.
    WP is complete-window ownership, not the old aggregate ``wp_size``.

    Only contiguous global ranks [0, world_size) are represented. Sizes must
    agree with the caller's launcher. For real communication, every WORLD rank
    must call create_process_groups() with the same sizes and placement.
    """

    def __init__(
        self,
        rank=0,
        world_size=None,
        *,
        dp_size=1,
        wp_size=1,
        sp_size=1,
        tp_size=1,
        sp_tp_placement="tp_first",
    ):
        sizes = {"dp": dp_size, "wp": wp_size, "sp": sp_size, "tp": tp_size}
        for name, size in sizes.items():
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(f"{name}_size must be a positive integer")
        if sp_tp_placement not in ("tp_first", "sp_first"):
            raise ValueError("sp_tp_placement must be 'tp_first' or 'sp_first'")

        sizes["dmp"] = wp_size * sp_size * tp_size
        sizes["global"] = dp_size * sizes["dmp"]
        if world_size is not None:
            if (isinstance(world_size, bool) or not isinstance(world_size, int)
                    or world_size != sizes["global"]):
                raise ValueError(
                    f"world_size must equal dp_size * wp_size * sp_size * tp_size "
                    f"= {sizes['global']}, got {world_size!r}"
                )
        self._sizes = sizes
        self._strides = {"global": 1, "dp": sizes["dmp"], "dmp": 1,
                         "wp": sp_size * tp_size}
        if sp_tp_placement == "tp_first":
            self._strides.update(sp=tp_size, tp=1)
        else:
            self._strides.update(sp=1, tp=sp_size)
        self._validate_rank(rank)
        self._rank = rank
        self._groups = None
        self._owned_groups = []

    def _validate_rank(self, rank):
        if (isinstance(rank, bool) or not isinstance(rank, int)
                or not 0 <= rank < self._sizes["global"]):
            raise ValueError(f"global rank must be an integer in [0, {self._sizes['global']})")

    def _group_name(self, group):
        if not isinstance(group, str) or group.lower() not in self._sizes:
            raise ValueError("group must be one of: global, dp, dmp, wp, sp, tp")
        return group.lower()

    def get_rank(self, group="global", *, global_rank=None):
        """Return the rank within the queried process's own group.

        Omit global_rank to query this manager's process. For another process,
        pass its global rank: get_rank('tp', global_rank=13). This does not
        query membership in the current process's particular TP group.
        """
        group = self._group_name(group)
        rank = self._rank if global_rank is None else global_rank
        self._validate_rank(rank)
        return (rank // self._strides[group]) % self._sizes[group]

    def get_group_size(self, group="global"):
        """Return group size; all groups of the same kind have equal size."""
        return self._sizes[self._group_name(group)]

    def create_process_groups(self):
        """Collectively create DP/DMP/WP/SP/TP groups on all WORLD ranks.

        The caller first initializes torch.distributed and, for NCCL, sets the
        current CUDA device. Groups inherit WORLD's backend. All ranks perform
        a configuration check and then call new_group in identical order,
        including groups they do not belong to. Identical membership reuses
        one handle; a full-world group reuses WORLD. Do not call only on rank 0.

        This method must be called once per manager lifecycle, synchronously
        with other group creation. Complete outstanding collectives before
        destroying/recreating groups. It does not initialize WORLD itself.
        """
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Call torch.distributed.init_process_group() first")
        world_size = dist.get_world_size()
        names = ("dp", "dmp", "wp", "sp", "tp", "global")
        signature = (tuple(self._sizes[n] for n in names),
                     tuple(self._strides[n] for n in names))
        specifications = [None] * world_size
        already_created = self._groups is not None or bool(self._owned_groups)
        dist.all_gather_object(specifications, (self._rank, signature, already_created))
        if any(created for _, _, created in specifications):
            raise RuntimeError("Groups already exist on some ranks; destroy_process_groups() first")
        if (self._sizes["global"] != world_size
                or any(rank != index or spec != signature
                       for index, (rank, spec, _) in enumerate(specifications))):
            raise ValueError(
                "All ranks must agree on topology, configured world_size must "
                "match WORLD, and each configured rank must match its actual rank"
            )

        groups = {"global": dist.group.WORLD}
        cache = {tuple(range(world_size)): dist.group.WORLD}
        for name in ("dp", "dmp", "wp", "sp", "tp"):
            size, stride = self._sizes[name], self._strides[name]
            for base in range(0, world_size, size * stride):
                for offset in range(stride):
                    ranks = tuple(base + offset + index * stride for index in range(size))
                    if ranks not in cache:
                        handle = dist.new_group(ranks=list(ranks))
                        cache[ranks] = handle
                        if self._rank in ranks:
                            self._owned_groups.append(handle)
                    if self._rank in ranks:
                        groups[name] = cache[ranks]
        self._groups = groups

    def get_group(self, group="global"):
        """Return this process's real ProcessGroup for a named dimension.

        Raises before group creation, rather than returning None (which would
        silently select WORLD in many torch.distributed collective APIs).
        """
        name = self._group_name(group)
        if self._groups is None:
            raise RuntimeError("Call create_process_groups() on every rank first")
        return self._groups[name]

    def destroy_process_groups(self):
        """Release owned subgroups, never WORLD; call on every rank.

        Finish outstanding work first. Invoke before destroying WORLD. Reverse
        creation order is consistent across overlapping subgroups. Repeated
        calls are harmless; logical rank/size queries remain available.
        """
        if self._groups is None and not self._owned_groups:
            return
        import torch.distributed as dist

        if dist.is_initialized():
            for group in reversed(self._owned_groups):
                dist.destroy_process_group(group)
        self._owned_groups.clear()
        self._groups = None
