"""Contract of the cluster-aware Permuter, independent of implementation.

A permutation preserves the multiset of labels (within each stratum when nested),
moves whole clusters when the permuted column has repeated rows beneath it, is
uniform over the admissible orderings, and in exact mode enumerates every distinct
ordering once before cycling.
"""

from collections import Counter

import numpy as np
import pytest
import scipy.stats as stats

from hierarch.resampling import Permuter
from tests._reference import groupby_mean, make_design, n_distinct_permutations


def _cluster_ids(data, ncols):
    """Integer id of each row's cluster defined by the first `ncols` columns."""
    if ncols == 0:
        return np.zeros(len(data), dtype=int)
    _, inv = np.unique(data[:, :ncols], axis=0, return_inverse=True)
    return inv.ravel()


def _is_balanced(hier):
    return all(isinstance(h, int) for h in hier)


class TestPermuterContract:
    def test_col0_permutation_preserves_multiset_and_other_columns(self, design):
        _, hier, data = design
        agg = groupby_mean(
            data, iterations=data.shape[1] - 3
        )  # rows unique in cols 0,1
        p = Permuter(random_state=1)
        p.fit(agg, col_to_permute=0)
        for _ in range(20):
            work = agg.copy()
            out = p.transform(work)
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
        p = Permuter(random_state=1)
        p.fit(agg, col_to_permute=1)
        for _ in range(20):
            out = p.transform(agg.copy())
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
        p = Permuter(random_state=1)
        p.fit(data, col_to_permute=0)
        ids = _cluster_ids(data, 2)  # (col0, col1) clusters
        for _ in range(20):
            out = p.transform(data.copy())
            for c in np.unique(ids):
                assert np.ptp(out[ids == c, 0]) == 0
            assert Counter(out[:, 0]) == Counter(data[:, 0])

    def test_shuffle_is_uniform_over_permutations(self):
        """Every one of the 4! orderings of a 4-row column must be equally likely
        (catches biased Fisher-Yates variants such as Sattolo's cycle)."""
        data = np.column_stack([np.arange(4.0), np.zeros(4)])
        p = Permuter(random_state=3)
        p.fit(data, col_to_permute=0)
        reps = 12000
        counts = Counter(tuple(p.transform(data.copy())[:, 0]) for _ in range(reps))
        assert len(counts) == 24
        chi = stats.chisquare(list(counts.values()))
        assert chi.pvalue > 1e-4, counts

    def test_stratified_shuffle_is_uniform_within_strata(self):
        data = np.column_stack(
            [np.repeat([0.0, 1.0], 3), np.tile(np.arange(3.0), 2), np.zeros(6)]
        )
        p = Permuter(random_state=3)
        p.fit(data, col_to_permute=1)
        reps = 6000
        counts = Counter(tuple(p.transform(data.copy())[:, 1]) for _ in range(reps))
        assert len(counts) == 36  # 3! * 3!
        chi = stats.chisquare(list(counts.values()))
        assert chi.pvalue > 1e-4

    def test_exact_enumerates_every_distinct_permutation_once_then_cycles(self):
        for labels in ([0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [1, 2, 3, 4]):
            data = np.column_stack(
                [np.array(labels, dtype=float), np.arange(len(labels), dtype=float)]
            )
            n_perm = n_distinct_permutations(labels)
            p = Permuter()
            p.fit(data, col_to_permute=0, exact=True)
            seen = [tuple(p.transform(data.copy())[:, 0]) for _ in range(n_perm)]
            assert len(set(seen)) == n_perm
            assert all(Counter(s) == Counter(labels) for s in seen)
            # after exhausting, it must cycle back
            assert tuple(p.transform(data.copy())[:, 0]) == seen[0]

    def test_exact_with_repeated_rows_moves_whole_clusters(self):
        data = make_design([2, 2, 3], rng=1)  # col-0 labels each cover 6 rows
        p = Permuter()
        p.fit(data, col_to_permute=0, exact=True)
        ids = _cluster_ids(data, 2)
        seen = set()
        for _ in range(6):  # C(4,2) = 6 cluster-level orderings
            out = p.transform(data.copy())
            for c in np.unique(ids):
                assert np.ptp(out[ids == c, 0]) == 0
            seen.add(tuple(out[:, 0]))
        assert len(seen) == 6

    def test_exact_not_supported_for_nested_columns(self):
        data = make_design([2, 3, 3], rng=1)
        p = Permuter()
        with pytest.raises(NotImplementedError):
            p.fit(data, col_to_permute=1, exact=True)

    def test_seed_reproducibility(self):
        data = make_design([2, 4, 1], rng=1)

        def draws(seed):
            p = Permuter(random_state=seed)
            p.fit(data, 0)
            return [p.transform(data.copy()) for _ in range(5)]

        for x, y in zip(draws(9), draws(9)):
            np.testing.assert_array_equal(x, y)
