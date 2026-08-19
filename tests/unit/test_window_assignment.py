import pytest

from core.parallel.window_assignment import get_window_indices


def _flat_ids(indices, width):
    return [row * width + col for row, col in indices]


def test_terra_m1_auto_uses_contiguous_balanced_chunks():
    rank_to_indices, global_to_local = get_window_indices(
        num_windows_h=2,
        num_windows_w=5,
        wp_group_h=4,
        wp_group_w=1,
        mode="terra_m1_ragged_auto",
    )

    per_rank_ids = [_flat_ids(rank_to_indices[rank], 5) for rank in range(4)]
    assert per_rank_ids == [[0, 1], [2, 3, 4], [5, 6], [7, 8, 9]]
    assert sorted(global_to_local) == [(row, col) for row in range(2) for col in range(5)]
    assert max(map(len, per_rank_ids)) - min(map(len, per_rank_ids)) <= 1
    assert all(ids == list(range(ids[0], ids[-1] + 1)) for ids in per_rank_ids)


def test_terra_m1_mode_rejects_two_dimensional_wp_topology():
    with pytest.raises(ValueError, match=r"requires xfmr_wp_topo=\(m, 1\)"):
        get_window_indices(
            num_windows_h=4,
            num_windows_w=4,
            wp_group_h=2,
            wp_group_w=2,
            mode="terra_m1_ragged_auto",
        )
