import torch
import torch.distributed as dist
import os


class ParallelManager:
    def __init__(self,
                dp_size=1,
                mp_size=1,
                wp_topo = (1, 1),
                xfmr_wp_topo = None,

                domain_topo = (1, 1),
                rank=-1,
                world_size=-1,
                device = None,
                window_assignment_mode = "regular",
                xfmr_sp_size = 1,
                tensor_parallel_size = 1,
                sp_tp_placement = "tp_first",
                ):


        self.rank = rank
        self.world_size = world_size

        self.dp_size = dp_size
        self.mp_size = mp_size
        wp_group_h, wp_group_w = wp_topo

        xfmr_sp_size = int(xfmr_sp_size)
        if xfmr_sp_size < 1:
            raise ValueError(f"xfmr_sp_size must be >= 1, got {xfmr_sp_size}")
        tensor_parallel_size = int(tensor_parallel_size)
        if tensor_parallel_size < 1:
            raise ValueError(f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}")

        xfmr_wp_topo = xfmr_wp_topo if xfmr_wp_topo is not None else wp_topo
        xfmr_wp_group_h, xfmr_wp_group_w = xfmr_wp_topo

        self.wp_size = wp_group_h * wp_group_w
        self._validate_node_aligned_wp_groups()
        self.xfmr_sp_size = xfmr_sp_size
        self.xfmr_tp_size = tensor_parallel_size
        self.xfmr_window_group_size = xfmr_wp_group_h * xfmr_wp_group_w
        if sp_tp_placement not in ("tp_first", "sp_first"):
            raise ValueError(
                f"sp_tp_placement must be 'tp_first' or 'sp_first', got {sp_tp_placement}"
            )
        self.sp_tp_placement = sp_tp_placement

        if self.xfmr_window_group_size * self.xfmr_sp_size * self.xfmr_tp_size != self.wp_size:
            raise ValueError(
                f"xfmr_wp_topo={xfmr_wp_topo} product times xfmr_sp_size={self.xfmr_sp_size} "
                f"times tensor_parallel_size={self.xfmr_tp_size} "
                f"must equal wp_topo={wp_topo} product"
            )


        self.data_wp_topo = wp_topo
        self.data_wp_group_h = wp_group_h
        self.data_wp_group_w = wp_group_w

        self.xfmr_wp_topo = xfmr_wp_topo
        self.xfmr_wp_group_h = xfmr_wp_group_h
        self.xfmr_wp_group_w = xfmr_wp_group_w
        self.window_assignment_mode = window_assignment_mode
        self.xfmr_window_group_rank = -1
        self.xfmr_sp_rank = -1
        self.xfmr_tp_rank = -1
        self.xfmr_sp_group = None
        self.xfmr_tp_group = None
        self.xfmr_tp_param_group = None
        self.full_param_replica_group = None
        self.xfmr_tp_param_replica_group = None


        self.wp_group_h = wp_group_h
        self.wp_group_w = wp_group_w


        self.num_windows_h = -1
        self.num_windows_w = -1


        self.domain_topo = domain_topo

        self.device = device


        self._setup_parallel_groups()
        self._setup_factorized_window_groups()


        self.domain_parallel_size = dist.get_world_size(self.window_parallel_group)
        self._domain_group_idx = self.rank // self.domain_parallel_size
        self._domain_rank = self.rank % self.domain_parallel_size

    def _validate_node_aligned_wp_groups(self):
        if os.environ.get("TERRA_ALLOW_UNALIGNED_CROSS_NODE_WP", "0") == "1":
            return

        local_world_size = int(
            os.environ.get("LOCAL_WORLD_SIZE")
            or os.environ.get("NPROC_PER_NODE")
            or 0
        )
        if local_world_size <= 1 or self.wp_size <= 1:
            return
        if self.world_size > 0 and self.world_size <= local_world_size:
            return

        node_local = local_world_size % self.wp_size == 0
        whole_node_span = self.wp_size % local_world_size == 0
        if node_local or whole_node_span:
            return

        raise RuntimeError(
            "[TERRA WARNING] unsupported cross-node window_parallel_group placement: "
            f"wp_size={self.wp_size}, local_world_size={local_world_size}, world_size={self.world_size}. "
            "The current contiguous rank layout would create WP groups that cross node boundaries "
            "without using complete nodes, e.g. 2+4 ranks across two 8-GPU nodes. "
            "This topology has triggered unstable NCCL subgroup collectives in scaling tests. "
            "Use a node-aligned wp_size such as 2, 4, 8, 16, or 24 on 8-GPU nodes, "
            "or set TERRA_ALLOW_UNALIGNED_CROSS_NODE_WP=1 only for manual debugging."
        )


    def _setup_factorized_window_groups(self):
        (
            self.xfmr_window_group_rank,
            self.xfmr_sp_rank,
            self.xfmr_tp_rank,
        ) = self.wp_rank_to_xfmr_coord(self.wp_rank)

        sp_groups = []
        tp_groups = []
        tp_param_groups = []
        full_param_replica_groups = []
        tp_param_replica_groups = []
        for dp_r in range(self.dp_size):
            for mp_r in range(self.mp_size):
                for wg_r in range(self.xfmr_window_group_size):
                    for tp_r in range(self.xfmr_tp_size):
                        group = [
                            self.xfmr_coord_to_global_rank(dp_r, mp_r, wg_r, sp_r, tp_r)
                            for sp_r in range(self.xfmr_sp_size)
                        ]
                        sp_groups.append(group)
                    for sp_r in range(self.xfmr_sp_size):
                        group = [
                            self.xfmr_coord_to_global_rank(dp_r, mp_r, wg_r, sp_r, tp_r)
                            for tp_r in range(self.xfmr_tp_size)
                        ]
                        tp_groups.append(group)
                for tp_r in range(self.xfmr_tp_size):
                    group = [
                        self.xfmr_coord_to_global_rank(dp_r, mp_r, wg_r, sp_r, tp_r)
                        for wg_r in range(self.xfmr_window_group_size)
                        for sp_r in range(self.xfmr_sp_size)
                    ]
                    tp_param_groups.append(group)

        for mp_r in range(self.mp_size):
            full_param_replica_groups.append([
                dp_r * (self.mp_size * self.wp_size) + mp_r * self.wp_size + wp_r
                for dp_r in range(self.dp_size)
                for wp_r in range(self.wp_size)
            ])
            for tp_r in range(self.xfmr_tp_size):
                tp_param_replica_groups.append([
                    self.xfmr_coord_to_global_rank(dp_r, mp_r, wg_r, sp_r, tp_r)
                    for dp_r in range(self.dp_size)
                    for wg_r in range(self.xfmr_window_group_size)
                    for sp_r in range(self.xfmr_sp_size)
                ])

        sp_group_handles = [dist.new_group(ranks=g) for g in sp_groups]
        tp_group_handles = [dist.new_group(ranks=g) for g in tp_groups]
        tp_param_group_handles = [dist.new_group(ranks=g) for g in tp_param_groups]
        full_param_replica_group_handles = [dist.new_group(ranks=g) for g in full_param_replica_groups]
        tp_param_replica_group_handles = [dist.new_group(ranks=g) for g in tp_param_replica_groups]
        for h, ranks in zip(sp_group_handles, sp_groups):
            if self.rank in ranks:
                self.xfmr_sp_group = h
                break
        for h, ranks in zip(tp_group_handles, tp_groups):
            if self.rank in ranks:
                self.xfmr_tp_group = h
                break
        for h, ranks in zip(tp_param_group_handles, tp_param_groups):
            if self.rank in ranks:
                self.xfmr_tp_param_group = h
                break
        for h, ranks in zip(full_param_replica_group_handles, full_param_replica_groups):
            if self.rank in ranks:
                self.full_param_replica_group = h
                break
        for h, ranks in zip(tp_param_replica_group_handles, tp_param_replica_groups):
            if self.rank in ranks:
                self.xfmr_tp_param_replica_group = h
                break

        if self.xfmr_sp_group is None:
            raise RuntimeError(f"rank {self.rank} did not find an xfmr_sp_group")
        if self.xfmr_tp_group is None:
            raise RuntimeError(f"rank {self.rank} did not find an xfmr_tp_group")
        if self.xfmr_tp_param_group is None:
            raise RuntimeError(f"rank {self.rank} did not find an xfmr_tp_param_group")
        if self.full_param_replica_group is None:
            raise RuntimeError(f"rank {self.rank} did not find a full_param_replica_group")
        if self.xfmr_tp_param_replica_group is None:
            raise RuntimeError(f"rank {self.rank} did not find an xfmr_tp_param_replica_group")


    def xfmr_coord_to_wp_rank(self, window_group_rank, sp_rank, tp_rank):
        if not (0 <= window_group_rank < self.xfmr_window_group_size):
            raise ValueError(f"invalid window_group_rank={window_group_rank}")
        if not (0 <= sp_rank < self.xfmr_sp_size):
            raise ValueError(f"invalid sp_rank={sp_rank}")
        if not (0 <= tp_rank < self.xfmr_tp_size):
            raise ValueError(f"invalid tp_rank={tp_rank}")

        base = window_group_rank * (self.xfmr_sp_size * self.xfmr_tp_size)
        if self.sp_tp_placement == "tp_first":
            return base + sp_rank * self.xfmr_tp_size + tp_rank
        return base + tp_rank * self.xfmr_sp_size + sp_rank

    def wp_rank_to_xfmr_coord(self, wp_rank):
        if not (0 <= wp_rank < self.wp_size):
            raise ValueError(f"invalid wp_rank={wp_rank}")

        group_size = self.xfmr_sp_size * self.xfmr_tp_size
        window_group_rank = wp_rank // group_size
        inner_rank = wp_rank % group_size
        if self.sp_tp_placement == "tp_first":
            sp_rank = inner_rank // self.xfmr_tp_size
            tp_rank = inner_rank % self.xfmr_tp_size
        else:
            tp_rank = inner_rank // self.xfmr_sp_size
            sp_rank = inner_rank % self.xfmr_sp_size
        return window_group_rank, sp_rank, tp_rank

    def xfmr_coord_to_global_rank(self, dp_rank, mp_rank, window_group_rank, sp_rank, tp_rank):
        wp_rank = self.xfmr_coord_to_wp_rank(window_group_rank, sp_rank, tp_rank)
        return self.get_global_rank(dp_rank, mp_rank, wp_rank)


    def update_window_info(self, num_windows_h, num_windows_w):
        self.num_windows_h = num_windows_h
        self.num_windows_w = num_windows_w


    def _setup_parallel_groups(self):
        self.global_rank = self.rank


        denom = self.mp_size * self.wp_size
        self.dp_rank = self.rank // denom
        rem = self.rank % denom
        self.mp_rank = rem // self.wp_size
        self.wp_rank = rem % self.wp_size


        (self.data_parallel_group,
         self.model_parallel_group,
         self.window_parallel_group) = self._create_parallel_groups()


        self.mp_group_size = dist.get_world_size(group=self.model_parallel_group)
        self.dp_group_size = dist.get_world_size(group=self.data_parallel_group)
        self.wp_group_size = dist.get_world_size(group=self.window_parallel_group)


    def _create_parallel_groups(self):
        """
        Create all groups in the same order for all ranks to avoid NCCL deadlock.

        We create:
          - data parallel groups: same (mp_rank, wp_rank), varying dp_rank
          - model parallel groups: same (dp_rank, wp_rank), varying mp_rank
          - wp parallel groups:    same (dp_rank, mp_rank), varying wp_rank

        Important: iterate in a deterministic, total order so every rank calls new_group()
        in the same sequence.
        """
        data_parallel_groups = []
        model_parallel_groups = []
        wp_parallel_groups = []

        # --- Data Parallel Groups (vary dp_rank; keep mp, wp fixed) ---
        # iterate mp_rank outer, wp_rank inner to fix ordering
        for mp_r in range(self.mp_size):
            for wp_r in range(self.wp_size):
                group = [

                    dp_r * (self.mp_size * self.wp_size) + mp_r * self.wp_size + wp_r
                    for dp_r in range(self.dp_size)
                ]
                data_parallel_groups.append(group)

        # --- Model Parallel Groups (vary mp_rank; keep dp, wp fixed) ---
        for dp_r in range(self.dp_size):
            for wp_r in range(self.wp_size):
                group = [
                    dp_r * (self.mp_size * self.wp_size) + mp_r * self.wp_size + wp_r
                    for mp_r in range(self.mp_size)
                ]
                model_parallel_groups.append(group)

        # --- WP Parallel Groups (vary wp_rank; keep dp, mp fixed) ---
        for dp_r in range(self.dp_size):
            for mp_r in range(self.mp_size):
                group = [
                    dp_r * (self.mp_size * self.wp_size) + mp_r * self.wp_size + wp_r
                    for wp_r in range(self.wp_size)
                ]
                wp_parallel_groups.append(group)

        # Now create group handles in the exact same order
        dp_group_handles = [dist.new_group(ranks=g) for g in data_parallel_groups]
        mp_group_handles = [dist.new_group(ranks=g) for g in model_parallel_groups]
        wp_group_handles = [dist.new_group(ranks=g) for g in wp_parallel_groups]

        # Find the group handle that contains current rank
        data_parallel_group = None
        model_parallel_group = None
        wp_parallel_group = None

        for h, ranks in zip(dp_group_handles, data_parallel_groups):
            if self.rank in ranks:
                data_parallel_group = h
                break

        for h, ranks in zip(mp_group_handles, model_parallel_groups):
            if self.rank in ranks:
                model_parallel_group = h
                break

        for h, ranks in zip(wp_group_handles, wp_parallel_groups):
            if self.rank in ranks:
                wp_parallel_group = h
                break

        return data_parallel_group, model_parallel_group, wp_parallel_group


    def get_rank(self):
        return self.rank


    def get_dp_rank(self):
        return self.dp_rank

    def get_mp_rank(self):
        return self.mp_rank

    def get_wp_rank(self):
        return self.wp_rank


    def get_dp_group_size(self):
        return self.dp_group_size

    def get_mp_group_size(self):
        return self.mp_group_size

    def get_wp_group_size(self):
        return self.wp_group_size


    # dp_rank are same for all processes in a mp group, so we use get_mp_rank is enough
    # then we can use mp_rank to calculate the global rank
    def get_global_rank(self, dp_rank, mp_rank, wp_rank):


        return dp_rank * (self.mp_size * self.wp_size) + mp_rank * self.wp_size + wp_rank

    def domain_rank(self):
        return self._domain_rank

    def is_first_domain_rank(self):
        return self._domain_rank == 0

    def is_last_domain_rank(self):
        return self._domain_rank == self.domain_parallel_size - 1

    def get_neighbor_global_rank(self, group, dp_rank, mp_rank, wp_rank):
        base = self._domain_group_idx * self.domain_parallel_size
        prev_rank = (base + self._domain_rank - 1) if self._domain_rank > 0 else None
        next_rank = (
            base + self._domain_rank + 1
            if self._domain_rank < self.domain_parallel_size - 1
            else None
        )
        return prev_rank, next_rank
