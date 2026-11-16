"""Remote-boundary-only window routing with the existing token ownership.

The CPU plan is compiled from TokenRoute's complete permutation, rather than
assuming whole window-row stripes. Ragged stripes may communicate with more
than two peers. Forward and its adjoint use exactly the original peer wires.
"""
from dataclasses import dataclass

import torch

from . import communication
from terra.runtime.profiling import profile_range


@dataclass(frozen=True)
class _CopyBlock:
    wire_start: int
    window_start: int
    windows: int
    pattern_rows: int
    # (destination token start, source pattern offset, contiguous token count)
    spans: tuple


@dataclass(frozen=True)
class _Direction:
    local_order: torch.Tensor
    remote_order: torch.Tensor
    remote_restore_order: object
    send_counts: tuple
    receive_counts: tuple
    copy_blocks: tuple
    rows: int
    local_tokens: int

    def describe(self):
        return {"local_tokens": self.local_tokens,
                "remote_send_tokens": sum(self.send_counts),
                "remote_recv_tokens": sum(self.receive_counts),
                "send_peers": sum(n > 0 for n in self.send_counts),
                "recv_peers": sum(n > 0 for n in self.receive_counts),
                "remote_restore_gather_tokens": (0 if self.remote_restore_order is None
                                                  else self.remote_restore_order.numel()),
                "strided_copy_blocks": sum(len(b.spans) for b in self.copy_blocks),
                "window_pattern_chunks": len(self.copy_blocks)}


@dataclass(frozen=True)
class _DeviceDirection:
    local_order: torch.Tensor
    remote_order: torch.Tensor
    remote_restore_order: object


class CompiledShiftPlan:
    """CPU metadata plus one bounded device-index cache; no floating buffers."""

    def __init__(self, forward, backward, rank, size, window_tokens):
        self.forward, self.backward = forward, backward
        self.rank, self.size, self.window_tokens = rank, size, window_tokens
        self._device_cache = None

    def describe(self):
        return {"rank": self.rank, "size": self.size,
                "source_rows": self.backward.rows, "destination_rows": self.forward.rows,
                "window_tokens": self.window_tokens,
                "forward": self.forward.describe(), "backward": self.backward.describe()}

    def device_plan(self, device):
        device = torch.device(device)
        cached = self._device_cache
        if cached is None or cached[0] != device:
            directions = tuple(_DeviceDirection(p.local_order.to(device), p.remote_order.to(device),
                               None if p.remote_restore_order is None else p.remote_restore_order.to(device))
                               for p in (self.forward, self.backward))
            cached = (device, directions)
            self._device_cache = cached
        return cached[1]


def _spans(pattern):
    spans = []
    start = 0
    while start < len(pattern):
        stop = start + 1
        while stop < len(pattern) and pattern[stop] == pattern[stop - 1] + 1:
            stop += 1
        spans.append((pattern[start], start, stop - start))
        start = stop
    return tuple(spans)


def _peer_blocks(destinations, wire_start, window_tokens):
    """Merge adjacent windows with identical wire-ordered boundary patterns.

    Only consecutive window IDs with identical boundary patterns merge. The
    optional remote-only gather has already canonicalized their wire order.
    """
    blocks = []
    cursor = 0
    pending = None
    while cursor < len(destinations):
        window = destinations[cursor] // window_tokens
        stop = cursor + 1
        while stop < len(destinations) and destinations[stop] // window_tokens == window:
            stop += 1
        pattern = tuple(index % window_tokens for index in destinations[cursor:stop])
        if (pending is not None and window == pending[1] + pending[2]
                and pattern == pending[3]):
            pending = (pending[0], pending[1], pending[2] + 1, pattern)
        else:
            if pending is not None:
                start, first_window, count, old_pattern = pending
                blocks.append(_CopyBlock(start, first_window, count, len(old_pattern), _spans(old_pattern)))
            pending = (wire_start + cursor, window, 1, pattern)
        cursor = stop
    if pending is not None:
        start, first_window, count, pattern = pending
        blocks.append(_CopyBlock(start, first_window, count, len(pattern), _spans(pattern)))
    return blocks


def _direction(send_order, receive_order, send_counts, receive_counts, rank, rows, window_tokens):
    # Remote holes temporarily select a legal source row. Every such hole is
    # overwritten below; a zero-window owner has no holes and an empty gather.
    local_order = [0] * rows
    remote_order, restore_order, blocks = [], [], []
    send_start = receive_start = remote_receive_start = 0
    for peer, (sent, received) in enumerate(zip(send_counts, receive_counts)):
        source = send_order[send_start:send_start + sent]
        destination = receive_order[receive_start:receive_start + received]
        if peer == rank:
            for src, dst in zip(source, destination):
                local_order[dst] = src
        else:
            remote_order.extend(source)
            # The adjoint's old wire is ordered by the original destination,
            # and can interleave hundreds of tiny local spans. Canonicalize
            # only its remote rows, retaining the exact P2P wire on the network.
            sorting = sorted(range(received), key=destination.__getitem__)
            restore_order.extend(remote_receive_start + i for i in sorting)
            ordered_destination = [destination[i] for i in sorting]
            blocks.extend(_peer_blocks(ordered_destination, remote_receive_start, window_tokens))
            remote_receive_start += received
        send_start += sent
        receive_start += received
    sends, receives = list(send_counts), list(receive_counts)
    local_tokens = sends[rank]
    sends[rank] = receives[rank] = 0
    restore = (None if restore_order == list(range(remote_receive_start))
               else torch.tensor(restore_order, dtype=torch.int64))
    return _Direction(torch.tensor(local_order, dtype=torch.int64),
                      torch.tensor(remote_order, dtype=torch.int64), restore, tuple(sends), tuple(receives),
                      tuple(blocks), rows, local_tokens)


def compile_shift_plan(plan, rank, size, source_rows, window_tokens):
    """Compile a complete, equal-capacity internal window permutation on CPU.

    Entry/crop routes, SP excess capacity, duplicates and partial permutations
    are rejected. Eligibility (pure WP and 0/half shift) belongs to TokenRoute.
    """
    for name, value, minimum in (("size", size, 1), ("rank", rank, 0),
                                  ("source_rows", source_rows, 0), ("window_tokens", window_tokens, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"m1 {name} must be an integer >= {minimum}")
    if rank >= size or not isinstance(plan, (tuple, list)) or len(plan) != 5:
        raise ValueError("m1 requires a valid rank and a five-element token route plan")
    send, receive, sends, receives, rows = plan
    if (isinstance(rows, bool) or not isinstance(rows, int) or rows != source_rows
            or rows % window_tokens):
        raise ValueError("m1 requires equal source/destination rows containing complete windows")
    for counts in (sends, receives):
        if (not isinstance(counts, (tuple, list)) or len(counts) != size
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts)
                or sum(counts) != rows):
            raise ValueError("m1 requires complete nonnegative per-peer split sizes")
    if sends[rank] != receives[rank]:
        raise ValueError("m1 self send/receive counts must agree")
    orders = []
    for order in (send, receive):
        if (not isinstance(order, torch.Tensor) or order.device.type != "cpu"
                or order.dtype != torch.int64 or order.ndim != 1 or order.numel() != rows):
            raise ValueError("m1 planning requires complete one-dimensional CPU int64 orders")
        values = order.tolist()
        if sorted(values) != list(range(rows)):
            raise ValueError("m1 route orders must each be a complete permutation")
        orders.append(values)
    forward = _direction(*orders, sends, receives, rank, rows, window_tokens)
    # A permutation's cotangent is its transpose, including peer wire order.
    backward = _direction(orders[1], orders[0], receives, sends, rank, rows, window_tokens)
    return CompiledShiftPlan(forward, backward, rank, size, window_tokens)


def _exchange_shift(received, packed, send_counts, receive_counts, group, size, *, direction):
    # Dynamic lookup preserves the existing warmup precision observer. The
    # shared helper supplies native HCCL batching, subgroup peer mapping and
    # waits while keeping every buffer/view/Work alive. Self counts are zero.
    with profile_range("terra/comm/fmp/token_route", direction=direction,
                       transport="p2p", implementation="m1", group_size=size):
        return communication._p2p_route_exchange(received, packed, send_counts, receive_counts, group, size)


def _execute(x, compiled, direction, group, size, stats):
    if (isinstance(size, bool) or not isinstance(size, int) or size != compiled.size
            or x.ndim != 2 or x.shape[0] != compiled.forward.rows or x.shape[1] < 1):
        raise ValueError("m1 input or group size differs from its compiled plan")
    if size > 1 and (communication.dist.get_world_size(group) != size
                     or communication.dist.get_rank(group) != compiled.rank):
        raise ValueError("m1 compiled rank/size differs from its process group")
    forward = direction == "forward"
    plan = compiled.forward if forward else compiled.backward
    device = compiled.device_plan(x.device)[0 if forward else 1]
    # Select first: a noncontiguous input/cotangent must not incur a separate
    # full-source contiguous copy before the gather already materializes it.
    output = communication._wire(x.index_select(0, device.local_order))
    if sum(plan.send_counts) or sum(plan.receive_counts):
        packed = communication._wire(x.index_select(0, device.remote_order))
        received = output.new_empty((sum(plan.receive_counts), x.shape[1]))
        _exchange_shift(received, packed, plan.send_counts, plan.receive_counts,
                        group, size, direction=direction)
        if device.remote_restore_order is not None:
            received = received.index_select(0, device.remote_restore_order)
        windows = output.view(-1, compiled.window_tokens, x.shape[1])
        for block in plan.copy_blocks:
            incoming = received.narrow(0, block.wire_start, block.windows * block.pattern_rows)
            incoming = incoming.view(block.windows, block.pattern_rows, x.shape[1])
            for destination, source, length in block.spans:
                windows[block.window_start:block.window_start + block.windows,
                        destination:destination + length].copy_(incoming[:, source:source + length])
    if stats is not None:
        key = direction + "_calls"
        stats[key] = stats.get(key, 0) + 1
        if "plan" not in stats:
            stats["plan"] = compiled.describe()
    return output.to(dtype=x.dtype)


class _Shift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, compiled, group, size, stats):
        ctx.compiled, ctx.group, ctx.size, ctx.stats = compiled, group, size, stats
        return _execute(x, compiled, "forward", group, size, stats)

    @staticmethod
    def backward(ctx, gradient):
        return _execute(gradient, ctx.compiled, "backward", ctx.group, ctx.size, ctx.stats), None, None, None, None


def permute_shift(x, compiled, group, size, *, stats=None):
    """Route internal shifted windows, preserving dtype and an exact adjoint."""
    if not isinstance(compiled, CompiledShiftPlan):
        raise TypeError("m1 requires a CompiledShiftPlan")
    if stats is not None and not isinstance(stats, dict):
        raise TypeError("m1 stats must be a mutable dict or None")
    return _Shift.apply(x, compiled, group, size, stats)
