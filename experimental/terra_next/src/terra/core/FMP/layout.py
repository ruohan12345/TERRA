"""Nonredundant spatial/window token layouts and differentiable permutations.

All plans contain integer metadata for this rank's source/destination tokens.
No rank materializes the global floating-point feature map. WP supports balanced
row/column scan assignments and cyclic ownership, including empty owners. A
stage-entry route inserts model-padding tokens once; all internal routes keep
them, and the stage-exit route crops them.
"""
import hashlib
import json

import torch

from terra.core.spatial import balanced_ranges
from .communication import permute_tokens, _inverse_full_permutation, _validate_route_transport


def _validate_window_assignment(value, window_topology, wp_size):
    """Validate physical window ownership without changing the model domain."""
    if not isinstance(value, str) or value not in ("contiguous", "column_contiguous", "aeris"):
        raise ValueError("window_assignment must be 'contiguous', 'column_contiguous' or 'aeris'")
    if value in ("contiguous", "column_contiguous"):
        if window_topology is not None:
            raise ValueError(f"{value} window assignment does not take window_topology")
        return value, None
    if (not isinstance(window_topology, (tuple, list)) or len(window_topology) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in window_topology)
            or window_topology[0]*window_topology[1] != wp_size):
        raise ValueError("aeris window_topology must contain two positive integers whose product is WP")
    return value, tuple(window_topology)


def _ids_sha256(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("utf-8")).hexdigest()


class TokenLayout:
    def __init__(self, mesh, height, width, window_size, *, shift=None,
                 spatial_shape=None, spatial_ranges=None,
                 window_assignment="contiguous", window_topology=None):
        """Describe one layout in a common, stage-padded global token domain.

        ``height,width`` always describe the Transformer model domain. A
        spatial layout may store only ``spatial_shape``: entry routing then
        creates the missing model-padding tokens once, while exit routing
        crops them. Window-to-window routes include all model-domain tokens,
        preserving their evolving values across every shifted block. Only
        excess SP capacity has token ID -1.

        Balanced ownership uses the legacy prefix-remainder split in a
        row-major (``contiguous``) or column-major (``column_contiguous``)
        scan. ``starts`` and ``ends`` index that scan. Global window IDs
        always retain their original row-major coordinate numbering.
        """
        self.mesh, self.height, self.width, self.window_size = mesh, height, width, window_size
        self.shift = shift
        self.wp_size = mesh.former.get_group_size("wp")
        self.sp_size = mesh.former.get_group_size("sp")
        self.tp_size = mesh.former.get_group_size("tp")
        self.window_assignment, self.window_topology = _validate_window_assignment(
            window_assignment, window_topology, self.wp_size)
        self.tokens = window_size ** 2
        if min(height, width, window_size, mesh.size) < 1:
            raise ValueError("Transformer layout dimensions must be positive")
        if height % window_size or width % window_size:
            raise ValueError("Transformer model domain must contain complete windows")
        self.spatial_shape = tuple(spatial_shape or (height, width))
        real_h, real_w = self.spatial_shape
        if min(real_h, real_w) < 1 or real_h > height or real_w > width:
            raise ValueError("Spatial domain must be a nonempty subset of the stage domain")
        self.spatial_ranges = tuple(tuple(v) for v in (
            spatial_ranges if spatial_ranges is not None else balanced_ranges(real_h, mesh.size)))
        if len(self.spatial_ranges) != mesh.size:
            raise ValueError("Spatial ownership must describe every model-parallel rank")
        cursor = 0
        for start, stop in self.spatial_ranges:
            if start != cursor or stop < start:
                raise ValueError("Spatial ownership must be contiguous, ordered and nonoverlapping")
            cursor = stop
        if cursor != real_h:
            raise ValueError("Spatial ownership must cover the complete real height")
        self._spatial_starts = torch.tensor([a for a, _ in self.spatial_ranges], dtype=torch.int64)
        self._spatial_counts = torch.tensor([b-a for a, b in self.spatial_ranges], dtype=torch.int64)
        self._spatial_ends = torch.tensor([b for _, b in self.spatial_ranges[:-1]], dtype=torch.int64)
        self.local_tokens = (self.tokens+self.sp_size-1) // self.sp_size
        self.window_grid = (height // window_size, width // window_size)
        self.total_windows = self.window_grid[0]*self.window_grid[1]
        global_windows = torch.arange(self.total_windows, dtype=torch.int64)
        if self.window_assignment in ("contiguous", "column_contiguous"):
            # Both scans retain the legacy prefix-remainder balance. starts
            # and ends are intervals in the selected scan, NOT necessarily
            # intervals of global IDs. A global window ID is always i*C+j;
            # column_contiguous scans k=j*R+i and stores those original IDs.
            ordered_windows = global_windows
            if self.window_assignment == "column_contiguous":
                ordered_windows = global_windows.reshape(self.window_grid).t().reshape(-1)
            q, r = divmod(self.total_windows, self.wp_size)
            self.counts = [q + (i < r) for i in range(self.wp_size)]
            self.starts = [i*q + min(i, r) for i in range(self.wp_size)]
            self.ends = [s+n for s,n in zip(self.starts,self.counts)]
            self._window_ids_by_wp = tuple(ordered_windows[a:b] for a, b in zip(self.starts, self.ends))
        else:
            rows, columns = self.window_topology
            # Aeris regular ownership is a two-dimensional cyclic pattern,
            # not rectangular blocks or flattened round-robin. Keep the
            # existing complete-window domain even when counts are unequal.
            owner = ((global_windows//self.window_grid[1]) % rows)*columns + global_windows % self.window_grid[1] % columns
            self._window_ids_by_wp = tuple(global_windows[owner == rank] for rank in range(self.wp_size))
            self.counts = [ids.numel() for ids in self._window_ids_by_wp]
            # A cyclic owner has no contiguous interval of global window IDs.
            self.starts = self.ends = None
        self._window_owner = torch.empty(self.total_windows, dtype=torch.int64)
        self._window_ordinal = torch.empty(self.total_windows, dtype=torch.int64)
        for rank, ids in enumerate(self._window_ids_by_wp):
            self._window_owner[ids] = rank
            self._window_ordinal[ids] = torch.arange(ids.numel(), dtype=torch.int64)
        self.tp_counts = [[n//self.tp_size + (t < n%self.tp_size) for t in range(self.tp_size)] for n in self.counts]
        self.tp_starts = [[t*(n//self.tp_size)+min(t,n%self.tp_size) for t in range(self.tp_size)] for n in self.counts]
        self.coords = []
        self.ranks_by_coord = torch.empty((self.wp_size,self.sp_size,self.tp_size), dtype=torch.int64)
        for rank, global_rank in enumerate(mesh.group_ranks):
            coord = tuple(mesh.former.get_rank(axis, global_rank=global_rank) for axis in ("wp","sp","tp"))
            self.coords.append(coord)
            self.ranks_by_coord[coord] = rank

    def window_ids(self, wp, tp=None, *, device=None):
        """Return ordered global window IDs, optionally one TP owner's slice.

        Global IDs always use row-major coordinates, even when ownership
        follows a column-major scan. Concatenating TP slices in TP-rank order
        reproduces the full WP list in the selected ownership order.
        The returned tensor is detached storage so callers cannot mutate the
        layout's ownership tables through this inspection interface.
        """
        if isinstance(wp, bool) or not isinstance(wp, int) or not 0 <= wp < self.wp_size:
            raise ValueError("Window WP coordinate is outside the layout")
        ids = self._window_ids_by_wp[wp]
        if tp is not None:
            if isinstance(tp, bool) or not isinstance(tp, int) or not 0 <= tp < self.tp_size:
                raise ValueError("Window TP coordinate is outside the layout")
            ids = ids.narrow(0, self.tp_starts[wp][tp], self.tp_counts[wp][tp])
        result = ids.clone()
        return result if device is None else result.to(device)

    def window_assignment_audit(self):
        ids = [value.tolist() for value in self._window_ids_by_wp]
        return {"mode": self.window_assignment,
                "topology": None if self.window_topology is None else list(self.window_topology),
                "model_domain_hw": [self.height, self.width], "window_grid_hw": list(self.window_grid),
                "counts": list(self.counts),
                "scan_order": "column_major" if self.window_assignment == "column_contiguous" else "row_major",
                "window_id_order": "row_major",
                "ownership_intervals_scan_index": None if self.starts is None else [
                    [start, end] for start, end in zip(self.starts, self.ends)],
                "ordered_window_ids_sha256": [_ids_sha256(values) for values in ids],
                "ordered_window_ids_samples": [{"first": values[:8], "last": values[-8:]} for values in ids],
                "hash_format": "SHA256 of compact JSON integer list with separators ',' and ':'"}

    def shape(self, batch, channels):
        if self.shift is None:
            start, stop = self.spatial_ranges[self.mesh.rank]
            return batch, stop-start, self.spatial_shape[1], channels
        wp, _, tp = self.coords[self.mesh.rank]
        return batch, self.tp_counts[wp][tp], self.local_tokens, channels

    def token_ids(self, batch):
        """Stage-global coordinate IDs in this rank's local tensor order."""
        if self.shift is None:
            start, stop = self.spatial_ranges[self.mesh.rank]
            rows = torch.arange(start, stop)[:, None]
            columns = torch.arange(self.spatial_shape[1])[None, :]
            spatial = (rows*self.width+columns).reshape(-1)
        else:
            wp, sp, tp = self.coords[self.mesh.rank]
            windows = self.window_ids(wp, tp)[:,None]
            start = sp*self.local_tokens
            tokens = torch.arange(start,start+self.local_tokens)[None,:]
            y = (windows // (self.width//self.window_size))*self.window_size + tokens//self.window_size
            x = (windows % (self.width//self.window_size))*self.window_size + tokens%self.window_size
            spatial = (((y+self.shift)%self.height)*self.width + (x+self.shift)%self.width).reshape(-1)
            valid=(tokens.expand(len(windows),-1)<self.tokens).reshape(-1)
            spatial=spatial.masked_fill(~valid,-1)
        ids=torch.arange(batch)[:,None]*(self.height*self.width)+spatial[None,:]
        return ids.masked_fill(spatial[None,:]<0,-1).reshape(-1)

    def contains(self, ids):
        """Whether global IDs have storage in this layout, on any rank."""
        valid = ids >= 0
        if self.shift is None:
            spatial = ids % (self.height*self.width)
            y, x = spatial//self.width, spatial%self.width
            valid = valid & (y < self.spatial_shape[0]) & (x < self.spatial_shape[1])
        return valid

    def owner_and_index(self, ids):
        batch, spatial = torch.div(ids,self.height*self.width,rounding_mode="floor"), ids%(self.height*self.width)
        y, x = spatial//self.width, spatial%self.width
        if self.shift is None:
            owner = torch.bucketize(y, self._spatial_ends, right=True)
            count, start = self._spatial_counts[owner], self._spatial_starts[owner]
            return owner, (batch*count+y-start)*self.spatial_shape[1]+x
        y, x = (y-self.shift)%self.height, (x-self.shift)%self.width
        windows = (y//self.window_size)*(self.width//self.window_size)+x//self.window_size
        wp = self._window_owner[windows]
        token = (y%self.window_size)*self.window_size+x%self.window_size
        sp = token//self.local_tokens
        offset = self._window_ordinal[windows]
        tp=torch.empty_like(wp)
        local_window=torch.empty_like(wp)
        local_count=torch.empty_like(wp)
        for w in range(self.wp_size):
            selected=wp==w
            ends=torch.tensor([s+n for s,n in zip(self.tp_starts[w],self.tp_counts[w])][:-1])
            t=torch.bucketize(offset[selected],ends,right=True)
            tp[selected]=t
            local_window[selected]=offset[selected]-torch.tensor(self.tp_starts[w])[t]
            local_count[selected]=torch.tensor(self.tp_counts[w])[t]
        index = (batch*local_count+local_window)*self.local_tokens+token%self.local_tokens
        return self.ranks_by_coord[wp,sp,tp], index


class TokenRoute:
    def __init__(self, source, destination, *, restore_implementation="scatter",
                 transport_implementation="all_to_all", window_shift="generic"):
        if (source.height,source.width)!=(destination.height,destination.width):
            raise ValueError("Token routes require one common stage-global coordinate domain")
        if source.mesh.size != destination.mesh.size or source.mesh.rank != destination.mesh.rank:
            raise ValueError("Token routes require matching model-parallel rank ownership")
        self.source, self.destination = source, destination
        self._plans = {}
        self._inverse_plans = {}
        self._inverse_device_cache = None
        self._m1_device_cache = None
        self._window_shift = "generic"
        self._shift_stats = {}
        self._shift_forward_calls = 0
        self._shift_last_backend = None
        self._shift_last_transport = None
        self.restore_implementation = restore_implementation
        self.transport_implementation = transport_implementation
        self.window_shift = window_shift

    @property
    def window_shift(self):
        return self._window_shift

    @window_shift.setter
    def window_shift(self, value):
        if not isinstance(value, str) or value not in ("generic", "m1"):
            raise ValueError("window_shift must be 'generic' or 'm1'")
        if value == "m1" and any(layout.mesh.former.get_group_size(axis) != 1
                                 for layout in (self.source, self.destination)
                                 for axis in ("sp", "tp")):
            raise ValueError("m1 window shift requires pure WP (SP=TP=1)")
        if value != self._window_shift:
            self._m1_device_cache = None
            self._shift_stats = {}
            self._shift_forward_calls = 0
            self._shift_last_backend = None
            self._shift_last_transport = None
        self._window_shift = value

    def _shift_backend(self):
        source, dest = self.source, self.destination
        if self.window_shift == "generic":
            return "generic", "policy_generic"
        if source.shift is None:
            return "generic", "stage_entry"
        if dest.shift is None:
            return "generic", "stage_exit"
        half = source.window_size // 2
        if (source.window_size == dest.window_size and half > 0
                and {source.shift, dest.shift} == {0, half}):
            return "m1", "internal_half_window_shift"
        return "generic", "not_half_window_transition"

    def window_shift_audit(self):
        """Report executed routing separately from a requested shift policy."""
        selected, reason = self._shift_backend()
        cached = self._m1_device_cache
        plan = next(iter(self._plans.values()), None)
        rank = self.source.mesh.rank
        counts = {} if plan is None else {
            "local_tokens": plan[2][rank],
            "remote_send_tokens": sum(plan[2])-plan[2][rank],
            "remote_receive_tokens": sum(plan[3])-plan[3][rank],
            "send_peers": [p for p, n in enumerate(plan[2]) if p != rank and n],
            "receive_peers": [p for p, n in enumerate(plan[3]) if p != rank and n]}
        return {"requested": self.window_shift, "backend": self._shift_last_backend,
                "eligible_backend": selected, "reason": reason,
                "forward_calls": self._shift_forward_calls,
                "backward_calls": self._shift_stats.get("backward_calls"),
                "execution": dict(self._shift_stats),
                "compiled_batch": None if cached is None else cached[0][0],
                "compiled_device": None if cached is None else str(cached[0][1]),
                "compiled_plan": None if cached is None else cached[1].describe(),
                "actual_transport": self._shift_last_transport,
                "counts_scope": "Current cached geometry since its last policy/batch/device change",
                **counts}

    def _m1_plan(self, batch, device, source_rows):
        from .window_shift import compile_shift_plan
        key = (batch, device)
        cached = self._m1_device_cache
        if cached is None or cached[0] != key:
            self._shift_stats = {}
            self._shift_forward_calls = 0
            self._shift_last_backend = None
            self._shift_last_transport = None
            compiled = compile_shift_plan(self._plans[batch], self.source.mesh.rank,
                                          self.source.mesh.size, source_rows, self.source.tokens)
            compiled.device_plan(device)
            cached = (key, compiled)
            self._m1_device_cache = cached
        return cached[1]

    @property
    def transport_implementation(self):
        return self._transport_implementation

    @transport_implementation.setter
    def transport_implementation(self, value):
        self._transport_implementation = _validate_route_transport(value)

    @property
    def restore_implementation(self):
        return self._restore_implementation

    @restore_implementation.setter
    def restore_implementation(self, value):
        if not isinstance(value, str) or value not in ("scatter", "inverse_gather"):
            raise ValueError("Token route restore_implementation must be 'scatter' or 'inverse_gather'")
        self._restore_implementation = value

    def _inverse_device_plan(self, batch, device, source_rows):
        if batch not in self._inverse_plans:
            plan = self._plans[batch]
            self._inverse_plans = {batch: (
                _inverse_full_permutation(plan[0], source_rows),
                _inverse_full_permutation(plan[1], plan[4]))}
        key = (batch, device)
        cached = self._inverse_device_cache
        if cached is None or cached[0] != key:
            send, receive, send_counts, receive_counts, rows = self._plans[batch]
            device_plan = (send.to(device), receive.to(device), send_counts, receive_counts, rows)
            inverses = tuple(None if value is None else value.to(device)
                             for value in self._inverse_plans[batch])
            # Keep only one device and batch geometry, including after moves
            # or iteration tails. The legacy strategy retains its old path.
            cached = (key, device_plan, inverses)
            self._inverse_device_cache = cached
        return cached[1], cached[2]

    def _plan(self,batch):
        source, dest = self.source,self.destination
        source_all, dest_all = source.token_ids(batch), dest.token_ids(batch)
        # A model-stage padding insertion/crop is a partial permutation.
        # Missing entry tokens are initialized to zero by permute_tokens;
        # block-to-block routes are complete permutations of the padded model
        # domain and must never recreate/reset those tokens.
        source_indexes=torch.nonzero((source_all>=0)&dest.contains(source_all),as_tuple=False).flatten()
        destination_indexes=torch.nonzero((dest_all>=0)&source.contains(dest_all),as_tuple=False).flatten()
        source_ids,dest_ids=source_all[source_indexes],dest_all[destination_indexes]
        owners, indexes = dest.owner_and_index(source_ids)
        # Lexicographic sorting does not rely on equal or bounded local sizes.
        by_index = torch.argsort(indexes,stable=True)
        send_order = source_indexes[by_index[torch.argsort(owners[by_index],stable=True)]]
        send_counts = torch.bincount(owners,minlength=source.mesh.size).tolist()
        owners,_ = source.owner_and_index(dest_ids)
        receive_order = destination_indexes[torch.argsort(owners,stable=True)]
        receive_counts = torch.bincount(owners,minlength=source.mesh.size).tolist()
        return send_order,receive_order,send_counts,receive_counts,len(dest_all)

    def __call__(self,x):
        source,dest=self.source,self.destination
        expected=source.shape(x.shape[0],x.shape[-1])
        if tuple(x.shape)!=expected:
            raise ValueError(f"Token route expected {expected}, got {tuple(x.shape)}")
        batch=x.shape[0]
        if batch not in self._plans:
            # Keep one batch geometry; iteration tails must not grow this cache.
            self._plans={batch:self._plan(batch)}
            self._inverse_plans = {}
            self._inverse_device_cache = None
            self._m1_device_cache = None
            self._shift_stats = {}
            self._shift_forward_calls = 0
            self._shift_last_backend = None
            self._shift_last_transport = None
        flattened = x.reshape(-1, x.shape[-1])
        backend, _ = self._shift_backend()
        if backend == "m1":
            from .window_shift import permute_shift
            compiled = self._m1_plan(batch, x.device, flattened.shape[0])
            output = permute_shift(flattened, compiled, source.mesh.group, source.mesh.size,
                                   stats=self._shift_stats)
        elif self.restore_implementation == "inverse_gather":
            plan, inverses = self._inverse_device_plan(batch, x.device, flattened.shape[0])
            output = permute_tokens(flattened, plan, source.mesh.group, source.mesh.size,
                                    inverse_orders=inverses,
                                    transport_implementation=self.transport_implementation)
        else:
            output=permute_tokens(flattened,self._plans[batch],source.mesh.group,source.mesh.size,
                                  transport_implementation=self.transport_implementation)
        self._shift_last_backend = backend
        self._shift_last_transport = "p2p" if backend == "m1" else self.transport_implementation
        self._shift_forward_calls += 1
        source.mesh.record("former_token_route",source_shift=source.shift,destination_shift=dest.shift,
                           input_shape=list(x.shape),output_shape=list(dest.shape(batch,x.shape[-1])) )
        return output.reshape(dest.shape(batch,x.shape[-1]))
