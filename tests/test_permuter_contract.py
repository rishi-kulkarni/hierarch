"""Contract of cluster-aware permutation, independent of implementation.

A permutation preserves the multiset of labels (within each stratum when nested),
moves whole clusters when the permuted column has repeated rows beneath it, is
uniform over the admissible orderings, and in exact mode enumerates every distinct
ordering once before cycling.
"""

from collections import Counter
from itertools import cycle

import numpy as np
import pytest
import scipy.stats as stats

from hierarch.resampling import (
    draw_permuted_labels,
    exact_label_matrix,
    permutation_plan,
)
from tests._reference import groupby_mean, make_design, n_distinct_permutations


def _cluster_ids(data, ncols):
    """Integer id of each row's cluster defined by the first `ncols` columns."""
    if ncols == 0:
        return np.zeros(len(data), dtype=int)
    _, inv = np.unique(data[:, :ncols], axis=0, return_inverse=True)
    return inv.ravel()


def _is_balanced(hier):
    return all(isinstance(h, int) for h in hier)


def _permute(plan, rng, target, col):
    """Draw one permutation and assign it into ``target`` in place, mirroring
    the old Permuter.transform contract (returns the same, mutated array)."""
    target[:, col] = draw_permuted_labels(plan, rng, 1)[0]
    return target


class TestPermuterContract:
    def test_col0_permutation_preserves_multiset_and_other_columns(self, design):
        _, hier, data = design
        agg = groupby_mean(
            data, iterations=data.shape[1] - 3
        )  # rows unique in cols 0,1
        rng = np.random.default_rng(1)
        plan = permutation_plan(agg, 0)
        for _ in range(20):
            work = agg.copy()
            out = _permute(plan, rng, work, 0)
            assert out is work  # in-place contract
            assert Counter(out[:, 0]) == Counter(agg[:, 0])
            np.testing.assert_array_equal(out[:, 1:], agg[:, 1:])

    def test_stratified_permutation_stays_within_strata(self, design):
        _, hier, data = design
        if len(hier) < 4:
            pytest.skip(
                "need >= 4 design columns to permute column 1 after aggregation"
            )
        agg = groupby_mean(
            data, iterations=data.shape[1] - 4
        )  # rows unique in cols 0..2
        strata = _cluster_ids(agg, 1)
        rng = np.random.default_rng(1)
        plan = permutation_plan(agg, 1)
        for _ in range(20):
            out = _permute(plan, rng, agg.copy(), 1)
            for s in np.unique(strata):
                assert Counter(out[strata == s, 1]) == Counter(agg[strata == s, 1])
            np.testing.assert_array_equal(
                out[:, [0] + list(range(2, agg.shape[1]))],
                agg[:, [0] + list(range(2, agg.shape[1]))],
            )

    def test_permutation_of_clustered_column_moves_whole_clusters(self):
        """When the permuted column has repeated rows beneath it (no aggregation),
        a label must move with all rows of its cluster, not row by row."""
        data = make_design([2, 3, 3], rng=1)
        rng = np.random.default_rng(1)
        plan = permutation_plan(data, 0)
        ids = _cluster_ids(data, 2)  # (col0, col1) clusters
        for _ in range(20):
            out = _permute(plan, rng, data.copy(), 0)
            for c in np.unique(ids):
                assert np.ptp(out[ids == c, 0]) == 0
            assert Counter(out[:, 0]) == Counter(data[:, 0])

    def test_shuffle_is_uniform_over_permutations(self):
        """Every one of the 4! orderings of a 4-row column must be equally likely
        (catches biased Fisher-Yates variants such as Sattolo's cycle)."""
        data = np.column_stack([np.arange(4.0), np.zeros(4)])
        rng = np.random.default_rng(3)
        plan = permutation_plan(data, 0)
        reps = 12000
        counts = Counter(
            tuple(_permute(plan, rng, data.copy(), 0)[:, 0]) for _ in range(reps)
        )
        assert len(counts) == 24
        chi = stats.chisquare(list(counts.values()))
        assert chi.pvalue > 1e-4, counts

    def test_stratified_shuffle_is_uniform_within_strata(self):
        data = np.column_stack(
            [np.repeat([0.0, 1.0], 3), np.tile(np.arange(3.0), 2), np.zeros(6)]
        )
        rng = np.random.default_rng(3)
        plan = permutation_plan(data, 1)
        reps = 6000
        counts = Counter(
            tuple(_permute(plan, rng, data.copy(), 1)[:, 1]) for _ in range(reps)
        )
        assert len(counts) == 36  # 3! * 3!
        chi = stats.chisquare(list(counts.values()))
        assert chi.pvalue > 1e-4

    def test_exact_enumerates_every_distinct_permutation_once_then_cycles(self):
        for labels in ([0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [1, 2, 3, 4]):
            data = np.column_stack(
                [np.array(labels, dtype=float), np.arange(len(labels), dtype=float)]
            )
            n_perm = n_distinct_permutations(labels)
            it = cycle(exact_label_matrix(data, 0))
            seen = [tuple(next(it)) for _ in range(n_perm)]
            assert len(set(seen)) == n_perm
            assert all(Counter(s) == Counter(labels) for s in seen)
            # after exhausting, it must cycle back
            assert tuple(next(it)) == seen[0]

    def test_exact_with_repeated_rows_moves_whole_clusters(self):
        data = make_design([2, 2, 3], rng=1)  # col-0 labels each cover 6 rows
        ids = _cluster_ids(data, 2)
        it = cycle(exact_label_matrix(data, 0))
        seen = set()
        for _ in range(6):  # C(4,2) = 6 cluster-level orderings
            out = next(it)
            for c in np.unique(ids):
                assert np.ptp(out[ids == c]) == 0
            seen.add(tuple(out))
        assert len(seen) == 6

    def test_seed_reproducibility(self):
        data = make_design([2, 4, 1], rng=1)

        def draws(seed):
            rng = np.random.default_rng(seed)
            plan = permutation_plan(data, 0)
            return [_permute(plan, rng, data.copy(), 0) for _ in range(5)]

        for x, y in zip(draws(9), draws(9)):
            np.testing.assert_array_equal(x, y)
