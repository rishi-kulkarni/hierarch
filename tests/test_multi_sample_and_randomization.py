"""multi_sample_test and hierarchical_randomization contracts.

multi_sample_test tests every requested pair, sorts by p, applies a monotone
correction, and each pairwise p is a plain hypothesis_test on the pair.
hierarchical_randomization yields bootstraps*permutations valid resamples.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import hierarch.stats as hs
from tests._reference import make_design

ALTERNATIVES = ["two-sided", "less", "greater"]


def _quiet(**kwargs):
    """Call hypothesis_test with the 'no levels to bootstrap' warning silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return hs.hypothesis_test(**kwargs)


def _cluster_ids(data, ncols):
    if ncols == 0:
        return np.zeros(len(data), dtype=int)
    _, inv = np.unique(data[:, :ncols], axis=0, return_inverse=True)
    return inv.ravel()


def _slope(x, y):
    return np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)


class TestMultiSampleTest:
    @pytest.fixture(scope="class")
    def data(self):
        return make_design([4, 3, 3], rng=21, treatment_effect=2.0)

    def test_all_pairs_are_tested_and_sorted(self, data):
        out = hs.multi_sample_test(
            data, 0, bootstraps=10, permutations=50, random_state=1
        )
        assert list(out.columns) == [
            "Condition 1",
            "Condition 2",
            "p-value",
            "Corrected p-value",
        ]
        assert len(out) == 6  # C(4, 2)
        p = out["p-value"].to_numpy(dtype=float)
        q = out["Corrected p-value"].to_numpy(dtype=float)
        assert np.all(np.diff(p) >= 0)
        assert np.all((0 < p) & (p <= 1))
        assert np.all(q >= p - 1e-12) and np.all(q <= 1 + 1e-12)
        assert np.all(np.diff(q) >= -1e-12)  # BH-adjusted values are monotone

    def test_no_correction(self, data):
        out = hs.multi_sample_test(
            data, 0, correction=None, bootstraps=10, permutations=50
        )
        assert list(out.columns) == ["Condition 1", "Condition 2", "p-value"]

    def test_explicit_hypotheses(self, data):
        hyp = [[1.0, 2.0], [1.0, 4.0]]
        out = hs.multi_sample_test(
            data, 0, hypotheses=hyp, bootstraps=10, permutations=50
        )
        pairs = {tuple(r) for r in out[["Condition 1", "Condition 2"]].to_numpy()}
        assert pairs == {(1.0, 2.0), (1.0, 4.0)}
        with pytest.raises(KeyError):
            hs.multi_sample_test(
                data, 0, hypotheses=[["x", 2.0]], bootstraps=2, permutations=5
            )

    def test_first_comparison_matches_hypothesis_test(self, data):
        seed = 77
        out = hs.multi_sample_test(
            data, 0, bootstraps=10, permutations=50, random_state=seed
        )
        row = out[(out["Condition 1"] == 1.0) & (out["Condition 2"] == 2.0)].iloc[0]
        subset = data[np.isin(data[:, 0], [1.0, 2.0])]
        p = hs.hypothesis_test(
            subset,
            0,
            compare="means",
            bootstraps=10,
            permutations=50,
            random_state=np.random.default_rng(seed),
        )
        assert row["p-value"] == p

    def test_dataframe_input(self, data):
        df = pd.DataFrame(data, columns=["treat", "a", "b", "y"])
        out = hs.multi_sample_test(
            df, "treat", bootstraps=5, permutations=20, random_state=1
        )
        assert len(out) == 6

    def test_bad_correction(self, data):
        with pytest.raises(KeyError):
            hs.multi_sample_test(
                data, 0, correction="bonferroni?", bootstraps=2, permutations=5
            )


class TestHierarchicalRandomization:
    def test_yields_valid_resamples(self):
        data = make_design([2, 3, 3], rng=4)
        n = 0
        for sample in hs.hierarchical_randomization(
            data, 0, bootstraps=4, permutations=5, random_state=1
        ):
            n += 1
            assert sample.shape == data.shape
            # everything but the (permuted) treatment column is an original row
            tail_rows = {tuple(r[1:]) for r in data}
            assert all(tuple(r[1:]) in tail_rows for r in sample)
            # treatment labels: multiset preserved and constant within units
            assert sorted(sample[:, 0]) == sorted(data[:, 0])
            unit = _cluster_ids(sample, 2)
            for u in np.unique(unit):
                assert np.ptp(sample[unit == u, 0]) == 0
        assert n == 20

    def test_reproducible(self):
        data = make_design([2, 3, 3], rng=4)
        a = [
            s.copy()
            for s in hs.hierarchical_randomization(
                data, 0, bootstraps=2, permutations=3, random_state=9
            )
        ]
        b = [
            s.copy()
            for s in hs.hierarchical_randomization(
                data, 0, bootstraps=2, permutations=3, random_state=9
            )
        ]
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
