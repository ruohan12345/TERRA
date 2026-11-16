"""Independent coordinate-oracle checks for the public layout-plan demo."""
from dataclasses import FrozenInstanceError
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from window_layout_demo import (build_layout, demo_case, main, remote_token_mask,
                                shift_plan, token_owner_grid)


def pixel_oracle(rows, columns, ranks, window, order, policy, dy, dx):
    """Direct token loop with arithmetic rank lookup, independent of grid fill."""
    n = rows * columns
    q, remainder = divmod(n, ranks)
    def owner(y, x):
        i, j = y // window, x // window
        index = i * columns + j if order == 'row' else j * rows + i
        if policy == 'paper_floor':
            return ((index + 1) * ranks - 1) // n
        prefix = (q + 1) * remainder
        return index // (q + 1) if index < prefix else remainder + (index - prefix) // q
    h, w = rows * window, columns * window
    matrix = [[0] * ranks for _ in range(ranks)]
    for y in range(h):
        for x in range(w):
            matrix[owner(y, x)][owner((y - dy) % h, (x - dx) % w)] += 1
    return tuple(tuple(row) for row in matrix)


class LayoutDemoTests(unittest.TestCase):
    def test_hand_layout_ids_and_immutability(self):
        row = build_layout(4, 8, 4, 'row')
        column = build_layout(4, 8, 4, 'column')
        self.assertEqual(row.owner_grid, tuple((i,) * 8 for i in range(4)))
        self.assertEqual(column.owner_grid, ((0, 0, 1, 1, 2, 2, 3, 3),) * 4)
        self.assertEqual(row.counts, (8, 8, 8, 8))
        self.assertEqual(column.window_ids_by_rank[0], (0, 8, 16, 24, 1, 9, 17, 25))
        self.assertEqual(sorted(i for ids in column.window_ids_by_rank for i in ids), list(range(32)))
        self.assertEqual(column.owner_of_window(8), 0)
        self.assertEqual(column.owner_of_window(2), 1)
        with self.assertRaises(FrozenInstanceError):
            column.rows = 8
        snapshot = column.to_dict()
        snapshot['owner_grid'][0][0] = 99
        self.assertEqual(column.owner_grid[0][0], 0)

    def test_hand_shift_counts_and_rank_loads(self):
        row = shift_plan(build_layout(4, 8, 4, 'row'), 4, 2, 2)
        column = shift_plan(build_layout(4, 8, 4, 'column'), 4, 2, 2)
        self.assertEqual((row.remote_tokens, column.remote_tokens), (256, 128))
        self.assertEqual(row.remote_send_by_rank, (64, 64, 64, 64))
        self.assertEqual(column.remote_send_by_rank, (32, 32, 32, 32))
        self.assertEqual((row.max_rank_send, column.max_rank_send), (64, 32))
        self.assertEqual(len(row.peer_pairs), 4)
        self.assertEqual(sum(map(sum, row.send_matrix)), 512)
        self.assertEqual(sum(map(sum, column.send_matrix)), 512)
        self.assertIsNone(column.to_dict()['wire_bytes'])

    def test_current_geometries_keep_distinct_cut_policies(self):
        for policy, col_w6 in (('prefix_remainder', 3150), ('paper_floor', 3132)):
            for scan, expected in (('row', 6534), ('column', col_w6)):
                layout = build_layout(43, 90, 4, scan, policy)
                plan = shift_plan(layout, 6, 3, 3)
                self.assertEqual(plan.remote_tokens, expected)
                self.assertEqual(tuple(sum(x) for x in plan.send_matrix),
                                 tuple(count * 36 for count in layout.counts))
            for scan, expected in (('row', 34816), ('column', 17920)):
                self.assertEqual(shift_plan(build_layout(8, 17, 4, scan, policy), 32, 16, 16)
                                 .remote_tokens, expected)

    def test_generic_shifts_match_independent_pixel_oracle(self):
        checked = 0
        for rows, columns in ((1, 1), (1, 3), (2, 3), (3, 2), (3, 4)):
            for ranks in (1, 2, rows * columns + 1):
                for window in (1, 2, 3, 4):
                    shifts = ((0, 0), (window // 2, window // 2), (0, 1), (1, 0),
                              (-1, 2), (window + 1, -window - 2))
                    for policy in ('paper_floor', 'prefix_remainder'):
                        for scan in ('row', 'column'):
                            layout = build_layout(rows, columns, ranks, scan, policy)
                            for dy, dx in shifts:
                                plan = shift_plan(layout, window, dy, dx)
                                expected = pixel_oracle(rows, columns, ranks, window, scan, policy, dy, dx)
                                self.assertEqual(plan.send_matrix, expected,
                                                 (rows, columns, ranks, window, scan, policy, dy, dx))
                                inverse = shift_plan(layout, window, -dy, -dx)
                                self.assertEqual(inverse.send_matrix, tuple(zip(*plan.send_matrix)))
                                self.assertEqual(sum(map(sum, plan.send_matrix)), rows * columns * window ** 2)
                                checked += 1
        self.assertEqual(checked, 1440)

    def test_tokens_masks_empty_owners_and_odd_counterexample(self):
        for policy in ('paper_floor', 'prefix_remainder'):
            layout = build_layout(2, 3, 7, 'column', policy)
            self.assertEqual(sum(count == 0 for count in layout.counts), 1)
            owners = token_owner_grid(layout, 3)
            self.assertEqual((len(owners), len(owners[0])), (6, 9))
            mask = remote_token_mask(layout, 3, -1, 2)
            self.assertEqual(sum(map(sum, mask)), shift_plan(layout, 3, -1, 2).remote_tokens)
            row = shift_plan(build_layout(5, 6, 5, 'row', policy), 3, 1, 1)
            column = shift_plan(build_layout(5, 6, 5, 'column', policy), 3, 1, 1)
            self.assertEqual((row.remote_tokens, column.remote_tokens), (90, 91))

    def test_validation_and_import_has_no_execution(self):
        for kwargs in ({'rows': True}, {'columns': 0}, {'ranks': -1},
                       {'order': 'bad'}, {'cut_policy': 'bad'}):
            defaults = dict(rows=4, columns=8, ranks=4)
            defaults.update(kwargs)
            with self.assertRaises(ValueError):
                build_layout(**defaults)
        layout = build_layout(4, 8, 4)
        for args in ((layout, 0, 2, 2), (layout, True, 2, 2), (layout, 4, .5, 2)):
            with self.assertRaises(ValueError):
                shift_plan(*args)
        with self.assertRaises(TypeError):
            shift_plan(object(), 4, 2, 2)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            spec = importlib.util.spec_from_file_location('layout_import_check', Path(__file__).with_name('window_layout_demo.py'))
            module = importlib.util.module_from_spec(spec)
            # dataclasses consult sys.modules on some supported Python versions.
            import sys
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
            finally:
                del sys.modules[spec.name]
        self.assertEqual(buffer.getvalue(), '')

    def test_direct_demo_and_cli_json(self):
        value = demo_case(include_tokens=True)
        self.assertEqual(value['row']['shift_plan']['remote_tokens'], 256)
        self.assertEqual(value['column']['shift_plan']['remote_tokens'], 128)
        self.assertEqual(sum(map(sum, value['column']['remote_token_mask'])), 128)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(['--rows', '4', '--columns', '8', '--ranks', '4', '--window-size', '4'])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(buffer.getvalue())['column']['shift_plan']['remote_tokens'], 128)


if __name__ == '__main__':
    unittest.main(verbosity=2)
