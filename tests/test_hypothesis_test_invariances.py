"""hypothesis_test p-value conventions and invariances.

Conventions: `return_null` reproduces p; one-sided p-values are complementary and
the two-sided p is twice the smaller tail; p is never 0. Invariances (same seed =>
bit-identical p): affine transforms of y, monotone relabelling of the treatment,
negating y swaps tails, input row order, DataFrame vs ndarray, string labels,
skipping every bootstrap level == bootstraps=1, 'indexes' == 'weights'.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import hierarch.stats as hs
from tests._reference import (
    make_design,
    pvalue_from_null,
)
from tests.conftest import design_names

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


class TestPValueConventions:
    def test_return_null_is_consistent_with_p(self, design):
        name, hier, data = design
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        # The observed statistic must come from the library's own
        # aggregation+statistic path: the first permutation round acts on the
        # unbootstrapped data, so the null contains entries *bit-identical* to
        # the library's observed value, and the tie-inclusive counting is
        # discontinuous there. An independently recomputed observed lands a few
        # ulps away and miscounts the ties. (The statistic's value itself is
        # cross-checked against the reference in test_exact_matches_brute_force.)
        from hierarch.internal_functions import GroupbyMean

        encoded = hs._preprocess_data(data)
        grouper = GroupbyMean()
        grouper.fit(encoded)
        agg = grouper.transform(encoded, iterations=encoded.shape[1] - 3)
        batched = hs._batched_stat_factory(tuple(encoded[:, 0].tolist()), "corr")
        observed = batched(
            np.ascontiguousarray(agg[:, 0])[None, :],
            np.ascontiguousarray(agg[:, -1]),
        )[0]
        for alternative in ALTERNATIVES:
            p, null = hs.hypothesis_test(
                data,
                0,
                alternative=alternative,
                bootstraps=10,
                permutations=200,
                return_null=True,
                random_state=4,
            )
            assert isinstance(null, list) and len(null) == 10 * 200
            # exact equality: the p-value convention (ties counted as extreme
            # on both tails, 2*min two-sided, floor at 1/total) must reproduce
            # the returned p bit-for-bit from the returned null
            assert p == pvalue_from_null(observed, null, alternative)

    def test_one_sided_p_values_are_complementary(self, design):
        name, hier, data = design
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        kw = dict(bootstraps=5, permutations=100, random_state=11)
        p_less, null_l = hs.hypothesis_test(
            data, 0, alternative="less", return_null=True, **kw
        )
        p_greater, null_g = hs.hypothesis_test(
            data, 0, alternative="greater", return_null=True, **kw
        )
        p_two = hs.hypothesis_test(data, 0, alternative="two-sided", **kw)
        assert null_l == null_g  # same seed => same null
        assert p_less + p_greater >= 1  # ties counted on both sides
        assert p_two == pytest.approx(2 * min(p_less, p_greater))

    def test_p_is_never_zero(self):
        data = make_design([2, 4, 3], rng=1, treatment_effect=50.0)
        for seed in range(30):
            for alternative in ALTERNATIVES:
                p = hs.hypothesis_test(
                    data,
                    0,
                    alternative=alternative,
                    bootstraps=1,
                    permutations=7,
                    random_state=seed,
                )
                assert p > 0
                assert p >= 1 / 7 - 1e-12

    def test_p_in_unit_interval_and_two_sided_at_most_one_after_floor(self, design):
        name, hier, data = design
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        for kind in ("weights", "bayesian"):
            p = hs.hypothesis_test(
                data, 0, bootstraps=5, permutations=50, kind=kind, random_state=0
            )
            assert 0 < p <= 1

    def test_invalid_alternative_raises(self):
        data = make_design([2, 3, 2], rng=0)
        with pytest.raises(Exception):
            hs.hypothesis_test(
                data, 0, alternative="sideways", bootstraps=1, permutations=10
            )

    def test_argument_validation(self):
        data = make_design([2, 3, 2], rng=0)
        with pytest.raises(TypeError):
            hs.hypothesis_test([[1, 2, 3]], 0)
        with pytest.raises(TypeError):
            hs.hypothesis_test(data, 0, bootstraps=0)
        with pytest.raises(TypeError):
            hs.hypothesis_test(data, 0, permutations=0)
        with pytest.raises(TypeError):
            hs.hypothesis_test(data, 0, permutations="some")
        with pytest.raises(AttributeError):
            hs.hypothesis_test(data, 0, compare=3)
        with pytest.raises(KeyError):
            hs.hypothesis_test(data, 0, compare="nope")
        with pytest.raises(KeyError):
            hs.hypothesis_test(data, 0, kind="nope")


class TestInvariances:
    KW = dict(bootstraps=20, permutations=100, random_state=7)

    def _p(self, data, tc=0, **over):
        kw = {**self.KW, **over}
        return _quiet(data_array=data, treatment_col=tc, **kw)

    @pytest.mark.parametrize("compare", ["corr", "jackknife_corr", "means"])
    def test_affine_transform_of_y(self, design, compare):
        name, hier, data = design
        if compare == "means" and len(np.unique(data[:, 0])) != 2:
            pytest.skip()
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        base = self._p(data, compare=compare)
        scaled = data.copy()
        scaled[:, -1] = 3.7 * scaled[:, -1] - 12.0
        assert self._p(scaled, compare=compare) == base
        for alternative in ("less", "greater"):
            assert self._p(scaled, compare=compare, alternative=alternative) == self._p(
                data, compare=compare, alternative=alternative
            )

    def test_negating_y_swaps_tails(self, design):
        name, hier, data = design
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        neg = data.copy()
        neg[:, -1] *= -1
        assert self._p(neg, alternative="less") == self._p(data, alternative="greater")
        assert self._p(neg, alternative="greater") == self._p(data, alternative="less")
        assert self._p(neg) == self._p(data)

    def test_relabelling_treatment_monotonically(self, design):
        name, hier, data = design
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        relab = data.copy()
        relab[:, 0] = 10 * relab[:, 0] + 100
        for compare in ("corr", "jackknife_corr"):
            for alternative in ALTERNATIVES:
                assert self._p(
                    relab, compare=compare, alternative=alternative
                ) == self._p(data, compare=compare, alternative=alternative)

    def test_swapping_two_group_labels_flips_tails(self):
        data = make_design([2, 4, 3], rng=8, treatment_effect=1.0)
        swapped = data.copy()
        swapped[:, 0] = 3 - swapped[:, 0]  # 1<->2 ; the library re-sorts rows
        # bootstraps=1: with resampling, the (re-sorted) units would receive
        # different weights, so only the pure permutation test is invariant.
        kw = dict(bootstraps=1, permutations=300, random_state=7)
        for compare in ("corr", "means"):
            assert self._p(swapped, compare=compare, **kw) == self._p(
                data, compare=compare, **kw
            )
            assert self._p(
                swapped, compare=compare, alternative="less", **kw
            ) == self._p(data, compare=compare, alternative="greater", **kw)

    def test_input_row_order_does_not_matter(self, design, rng):
        name, hier, data = design
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        shuffled = data[rng.permutation(len(data))]
        assert self._p(shuffled) == self._p(data)

    def test_dataframe_and_ndarray_agree(self, design):
        name, hier, data = design
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        cols = [f"c{i}" for i in range(data.shape[1] - 1)] + ["y"]
        df = pd.DataFrame(data, columns=cols)
        assert self._p(df) == self._p(data)
        assert self._p(df, tc="c0") == self._p(data)

    def test_string_labels_are_encoded(self):
        data = make_design([2, 4, 3], rng=8, treatment_effect=1.0)
        df = pd.DataFrame(data, columns=["treat", "mouse", "cell", "y"])
        df["treat"] = df["treat"].map({1.0: "control", 2.0: "drug"})
        df["mouse"] = df["mouse"].map(lambda v: f"m{int(v)}")
        assert self._p(df, tc="treat") == self._p(data)

    @pytest.mark.parametrize("name", design_names())
    def test_indexes_and_weights_kinds_agree(self, name, design_pool):
        hier, data = design_pool[name]
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        for compare in ("corr", "means"):
            if compare == "means" and len(np.unique(data[:, 0])) != 2:
                continue
            _, null_w = _quiet(
                data_array=data,
                treatment_col=0,
                compare=compare,
                kind="weights",
                return_null=True,
                **self.KW,
            )
            _, null_i = _quiet(
                data_array=data,
                treatment_col=0,
                compare=compare,
                kind="indexes",
                return_null=True,
                **self.KW,
            )
            np.testing.assert_allclose(null_w, null_i)

    def test_skipping_every_bootstrap_level_equals_bootstraps_1(self):
        for hier, skip in ([2, 3, 3], [2]), ([2, 3, 2, 3], [2, 3]):
            data = make_design(hier, rng=2)
            with pytest.warns(UserWarning, match="No levels to bootstrap"):
                p_skip, null_skip = hs.hypothesis_test(
                    data,
                    0,
                    skip=skip,
                    bootstraps=20,
                    permutations=100,
                    return_null=True,
                    random_state=7,
                )
            p_one, null_one = hs.hypothesis_test(
                data,
                0,
                bootstraps=1,
                permutations=100,
                return_null=True,
                random_state=7,
            )
            assert p_skip == p_one
            assert null_skip == null_one

    def test_skip_below_treatment_level_is_ignored_with_a_warning(self):
        data = make_design([2, 3, 2, 3], rng=2)
        with pytest.warns(UserWarning, match="No need to include"):
            p_a = hs.hypothesis_test(data, 0, skip=[0, 1], **self.KW)
        p_b = hs.hypothesis_test(data, 0, **self.KW)
        assert p_a == p_b

    def test_skipping_one_level_changes_the_null(self):
        data = make_design([2, 3, 2, 3], rng=2, icc=1.0)
        _, null_all = hs.hypothesis_test(
            data, 0, bootstraps=50, permutations=50, return_null=True, random_state=1
        )
        _, null_skip = hs.hypothesis_test(
            data,
            0,
            skip=[3],
            bootstraps=50,
            permutations=50,
            return_null=True,
            random_state=1,
        )
        assert null_all != null_skip

    def test_seed_reproducibility_and_variation(self, design):
        name, hier, data = design
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        kw = dict(bootstraps=10, permutations=50, return_null=True)
        p1, n1 = hs.hypothesis_test(data, 0, random_state=123, **kw)
        p2, n2 = hs.hypothesis_test(data, 0, random_state=123, **kw)
        p3, n3 = hs.hypothesis_test(data, 0, random_state=124, **kw)
        assert (p1, n1) == (p2, n2)
        assert n1 != n3
        g1, _ = hs.hypothesis_test(data, 0, random_state=np.random.default_rng(5), **kw)
        g2, _ = hs.hypothesis_test(data, 0, random_state=np.random.default_rng(5), **kw)
        assert g1 == g2

    def test_permutation_only_test_ignores_kind(self, design):
        name, hier, data = design
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        kw = dict(bootstraps=1, permutations="all")
        ps = {
            kind: _quiet(data_array=data, treatment_col=0, kind=kind, **kw)
            for kind in ("weights", "indexes", "bayesian")
        }
        assert len(set(ps.values())) == 1


class TestStatisticalSanity:
    def test_large_effect_is_detected(self):
        data = make_design([2, 4, 3], rng=1, treatment_effect=20.0)
        for compare in ("corr", "means", "jackknife_corr"):
            p = hs.hypothesis_test(
                data,
                0,
                compare=compare,
                bootstraps=100,
                permutations="all",
                random_state=1,
            )
            assert p < 0.05, (compare, p)
