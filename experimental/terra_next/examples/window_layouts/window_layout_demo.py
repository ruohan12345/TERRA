#!/usr/bin/env python3
"""Public, standard-library-only window ownership and shift-plan demonstration.

Python >= 3.8. This is a coordinate/communication-plan demo, not a distributed
P2P implementation or a hardware benchmark. All model-domain padding tokens
are retained. Each remote token row is counted once at its sender.

Global window IDs always equal row * columns + column. Changing scan order
changes rank ownership and local window order, not the model's coordinates.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Tuple

POLICIES = ('paper_floor', 'prefix_remainder')
ORDERS = ('row', 'column')


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return value


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(name + ' must be an integer')
    return value


@dataclass(frozen=True)
class WindowLayout:
    rows: int
    columns: int
    ranks: int
    order: str
    cut_policy: str
    cuts: Tuple[int, ...]
    owner_grid: Tuple[Tuple[int, ...], ...]
    window_ids_by_rank: Tuple[Tuple[int, ...], ...]

    @property
    def counts(self):
        return tuple(len(ids) for ids in self.window_ids_by_rank)

    def owner_of_window(self, window_id):
        _integer(window_id, 'window_id')
        if not 0 <= window_id < self.rows * self.columns:
            raise ValueError('window_id is outside the global window grid')
        row, column = divmod(window_id, self.columns)
        return self.owner_grid[row][column]

    def to_dict(self):
        return {
            'format': 'terra_window_layout_demo_v1',
            'window_grid_hw': [self.rows, self.columns],
            'ranks': self.ranks,
            'order': self.order,
            'cut_policy': self.cut_policy,
            'cuts': list(self.cuts),
            'window_counts_by_rank': list(self.counts),
            'owner_grid': [list(row) for row in self.owner_grid],
            'window_ids_by_rank': [list(ids) for ids in self.window_ids_by_rank],
            'global_window_ids': [[i * self.columns + j for j in range(self.columns)]
                                  for i in range(self.rows)],
            'window_id_convention': 'row * columns + column, unchanged by scan order',
        }


@dataclass(frozen=True)
class ShiftPlan:
    window_size: int
    token_grid_hw: Tuple[int, int]
    shift_y: int
    shift_x: int
    normalized_shift_y: int
    normalized_shift_x: int
    send_matrix: Tuple[Tuple[int, ...], ...]

    @property
    def remote_send_by_rank(self):
        return tuple(sum(row) - row[i] for i, row in enumerate(self.send_matrix))

    @property
    def remote_receive_by_rank(self):
        return tuple(sum(row[j] for row in self.send_matrix) - self.send_matrix[j][j]
                     for j in range(len(self.send_matrix)))

    @property
    def local_tokens_by_rank(self):
        return tuple(row[i] for i, row in enumerate(self.send_matrix))

    @property
    def remote_tokens(self):
        return sum(self.remote_send_by_rank)

    @property
    def max_rank_send(self):
        return max(self.remote_send_by_rank)

    @property
    def max_rank_receive(self):
        return max(self.remote_receive_by_rank)

    @property
    def peer_pairs(self):
        return tuple((i, j) for i, row in enumerate(self.send_matrix)
                     for j, tokens in enumerate(row) if i != j and tokens)

    def to_dict(self):
        return {
            'format': 'terra_window_shift_plan_demo_v1',
            'kind': 'coordinate plan only; no communication or training is executed',
            'window_size': self.window_size,
            'token_grid_hw': list(self.token_grid_hw),
            'shift_yx': [self.shift_y, self.shift_x],
            'normalized_shift_yx': [self.normalized_shift_y, self.normalized_shift_x],
            'send_matrix': [list(row) for row in self.send_matrix],
            'remote_tokens': self.remote_tokens,
            'remote_send_by_rank': list(self.remote_send_by_rank),
            'remote_receive_by_rank': list(self.remote_receive_by_rank),
            'local_tokens_by_rank': list(self.local_tokens_by_rank),
            'max_rank_send': self.max_rank_send,
            'max_rank_receive': self.max_rank_receive,
            'peer_pairs': [list(pair) for pair in self.peer_pairs],
            'peer_pair_count': len(self.peer_pairs),
            'total_token_rows': sum(sum(row) for row in self.send_matrix),
            'matrix_convention': 'send_matrix[source_rank][destination_rank]; diagonal stays local',
            'token_count_convention': 'all ranks, one shift, each remote token row counted once',
            'wire_dtype_observed': False,
            'wire_bytes': None,
            'physical_bandwidth': None,
        }


def build_layout(rows, columns, ranks, order='row', cut_policy='paper_floor'):
    """Build balanced complete-window ownership, including possible empty ranks.

    paper_floor uses b[p] = floor(p*N/P), matching the paper's cut convention.
    prefix_remainder uses b[p] = p*floor(N/P) + min(p,N%P), the v8 convention.
    Both return the original row-major global window IDs in the chosen order.
    """
    _positive_integer(rows, 'rows')
    _positive_integer(columns, 'columns')
    _positive_integer(ranks, 'ranks')
    if order not in ORDERS:
        raise ValueError('order must be row or column')
    if cut_policy not in POLICIES:
        raise ValueError('Unsupported cut_policy')
    windows = rows * columns
    q, remainder = divmod(windows, ranks)
    if cut_policy == 'paper_floor':
        cuts = tuple(p * windows // ranks for p in range(ranks + 1))
    else:
        cuts = tuple(p * q + min(p, remainder) for p in range(ranks + 1))
    owners = [[-1] * columns for _ in range(rows)]
    rank_windows = []
    for rank in range(ranks):
        ids = []
        for index in range(cuts[rank], cuts[rank + 1]):
            if order == 'row':
                i, j = divmod(index, columns)
            else:
                j, i = divmod(index, rows)
            owners[i][j] = rank
            ids.append(i * columns + j)
        rank_windows.append(tuple(ids))
    return WindowLayout(rows, columns, ranks, order, cut_policy, cuts,
                        tuple(tuple(row) for row in owners), tuple(rank_windows))


def shift_plan(layout, window_size, shift_y, shift_x):
    """Return the exact ownership permutation plan for any integer cyclic shift.

    A source token (y,x) uses the target owner at
    ((y-shift_y) % H, (x-shift_x) % W), H=rows*w and W=columns*w.
    Quotient/remainder decomposition counts at most four destination windows
    per source window, so large w does not require a per-token iteration.
    The matrix includes local values on its diagonal; only off-diagonal
    entries count as remote communication. This function sends no data.
    """
    if not isinstance(layout, WindowLayout):
        raise TypeError('layout must be a WindowLayout returned by build_layout')
    _positive_integer(window_size, 'window_size')
    _integer(shift_y, 'shift_y')
    _integer(shift_x, 'shift_x')
    height, width = layout.rows * window_size, layout.columns * window_size
    dy, dx = shift_y % height, shift_x % width
    whole_y, remainder_y = divmod(dy, window_size)
    whole_x, remainder_x = divmod(dx, window_size)
    y_parts = ((whole_y, window_size - remainder_y), (whole_y + 1, remainder_y))
    x_parts = ((whole_x, window_size - remainder_x), (whole_x + 1, remainder_x))
    matrix = [[0] * layout.ranks for _ in range(layout.ranks)]
    for i, row in enumerate(layout.owner_grid):
        for j, source_rank in enumerate(row):
            for offset_y, count_y in y_parts:
                for offset_x, count_x in x_parts:
                    if count_y and count_x:
                        target_rank = layout.owner_grid[(i - offset_y) % layout.rows][
                            (j - offset_x) % layout.columns]
                        matrix[source_rank][target_rank] += count_y * count_x
    return ShiftPlan(window_size, (height, width), shift_y, shift_x, dy, dx,
                     tuple(tuple(row) for row in matrix))


def token_owner_grid(layout, window_size):
    """Materialize a small token-level owner grid for illustrations or checking."""
    _positive_integer(window_size, 'window_size')
    return tuple(tuple(layout.owner_grid[y // window_size][x // window_size]
                       for x in range(layout.columns * window_size))
                 for y in range(layout.rows * window_size))


def remote_token_mask(layout, window_size, shift_y, shift_x):
    """True at source token coordinates that require another rank after shift."""
    _integer(shift_y, 'shift_y')
    _integer(shift_x, 'shift_x')
    owners = token_owner_grid(layout, window_size)
    height, width = len(owners), len(owners[0])
    return tuple(tuple(owners[y][x] != owners[(y - shift_y) % height][(x - shift_x) % width]
                       for x in range(width)) for y in range(height))


def demo_case(rows=4, columns=8, ranks=4, window_size=4,
              shift_y=None, shift_x=None, cut_policy='paper_floor', include_tokens=False):
    """Build the same-geometry row/column comparison used by the visual demo."""
    _positive_integer(window_size, 'window_size')
    if shift_y is None:
        shift_y = window_size // 2
    if shift_x is None:
        shift_x = window_size // 2
    result = {
        'format': 'terra_window_layout_comparison_demo_v1',
        'kind': 'mathematical layout and route plan; no hardware measurements',
        'geometry': {'window_grid_hw': [rows, columns], 'ranks': ranks,
                     'window_size': window_size, 'shift_yx': [shift_y, shift_x]},
        'cut_policy': cut_policy,
        'scope': 'pure WP, complete periodic model domain, entry/exit excluded',
    }
    for order in ORDERS:
        layout = build_layout(rows, columns, ranks, order, cut_policy)
        plan = shift_plan(layout, window_size, shift_y, shift_x)
        entry = {'layout': layout.to_dict(), 'shift_plan': plan.to_dict()}
        if include_tokens:
            entry['token_owner_grid'] = [list(row) for row in token_owner_grid(layout, window_size)]
            entry['remote_token_mask'] = [list(row) for row in remote_token_mask(
                layout, window_size, shift_y, shift_x)]
        result[order] = entry
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--columns', type=int, default=8)
    parser.add_argument('--ranks', type=int, default=4)
    parser.add_argument('--window-size', type=int, default=4)
    parser.add_argument('--shift-y', type=int)
    parser.add_argument('--shift-x', type=int)
    parser.add_argument('--cut-policy', choices=POLICIES, default='paper_floor')
    parser.add_argument('--include-tokens', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        result = demo_case(args.rows, args.columns, args.ranks, args.window_size,
                           args.shift_y, args.shift_x, args.cut_policy, args.include_tokens)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    output = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
        print(json.dumps({'output': str(args.output),
                          'row_remote_tokens': result['row']['shift_plan']['remote_tokens'],
                          'column_remote_tokens': result['column']['shift_plan']['remote_tokens']}))
    else:
        print(output, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
