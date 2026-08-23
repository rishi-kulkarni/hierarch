"""Exact tests equal brute-force enumeration.

With bootstraps=1 the p-value of hypothesis_test is a deterministic function of the
data: it must equal an itertools enumeration of the distinct (stratified) label
orderings scored with an independent transcription of each statistic, for every
`compare` and `alternative`. Where exact stratified tests are not implemented, a
large randomized test must agree within Monte Carlo error.
"""

import warnings

import numpy as np
import pytest

import hierarch.stats as hs
from tests._reference import (
    STATISTICS,
    aggregate_to_treatment_level,
    exact_permutation_pvalue,
    make_design,
    n_distinct_permutations,
    pvalue_from_null,
)
from tests.conftest import DESIGN_POOL

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


class TestExactTestMatchesBruteForce:
    """With bootstraps=1 and permutations='all' the p-value is a deterministic
    function of the data and must equal an itertools enumeration."""

    @pytest.mark.parametrize("compare", ["corr", "jackknife_corr", "means"])
    @pytest.mark.parametrize("name", list(DESIGN_POOL))
    def test_treatment_col_0(self, name, compare, design_pool):
        hier, data = design_pool[name]
        n_groups = len(np.unique(data[:, 0]))
        if compare == "means" and n_groups != 2:
            pytest.skip("means needs exactly two groups")
        if n_groups == 1:
            pytest.skip("nothing to permute")
        agg = aggregate_to_treatment_level(data, 0)
        if n_distinct_permutations(agg[:, 0]) > 5000:
            pytest.skip("too many permutations for brute force")
        # enumerate the null once; the three alternatives share it
        observed = STATISTICS[compare](agg[:, 0], agg[:, -1])
        _, null = exact_permutation_pvalue(
            agg[:, 0], agg[:, -1], STATISTICS[compare], "two-sided"
        )
        for alternative in ALTERNATIVES:
            p_ref = pvalue_from_null(observed, null, alternative)
            p_lib = _quiet(
                data_array=data,
                treatment_col=0,
                compare=compare,
                alternative=alternative,
                bootstraps=1,
                permutations="all",
            )
            if n_groups > 2:
                # BUG (documented in the docstring as unsupported, but it does not
                # raise): with >2 groups the library only scores C(n, n_0) of the
                # multiset permutations. Pin the CORRECT value and expect failure.
                if abs(p_lib - p_ref) > 1e-12:
                    pytest.xfail(
                        "known bug: >2 groups with permutations='all' "
                        "scores only the first C(n, n_0) permutations"
                    )
                pytest.fail("known >2-group 'all' bug appears fixed: drop the xfail")
            assert p_lib == pytest.approx(p_ref, abs=1e-12), (
                name,
                compare,
                alternative,
            )

    @pytest.mark.parametrize("compare", ["corr", "jackknife_corr"])
    def test_treatment_col_1_stratified_random_test_matches_brute_force(self, compare):
        """Exact stratified tests are not implemented, so compare a large
        randomized test to the brute-force stratified enumeration."""
        for hier, tc in ([2, 3, 2, 3], 1), ([2, 2, 3], 1):
            data = make_design(hier, rng=5)
            agg = aggregate_to_treatment_level(data, tc)
            strata = _cluster_ids(agg, tc)
            p_ref, null = exact_permutation_pvalue(
                agg[:, tc], agg[:, -1], STATISTICS[compare], "two-sided", strata=strata
            )
            p_lib = _quiet(
                data_array=data,
                treatment_col=tc,
                compare=compare,
                bootstraps=1,
                permutations=6000,
                random_state=1,
            )
            # MC standard error at 6000 permutations is < 0.0065
            assert abs(p_lib - p_ref) < 0.035, (hier, compare, p_lib, p_ref)

    def test_custom_callable_statistic(self, design_pool):
        hier, data = design_pool["3lvl_balanced_2x4x3"]

        def stat(x, y):
            return np.corrcoef(x, y)[0, 1]

        agg = aggregate_to_treatment_level(data, 0)
        p_ref, _ = exact_permutation_pvalue(agg[:, 0], agg[:, -1], stat, "two-sided")
        p_lib = hs.hypothesis_test(
            data, 0, compare=stat, bootstraps=1, permutations="all"
        )
        assert p_lib == pytest.approx(p_ref, abs=1e-12)

    def test_no_levels_to_bootstrap_falls_back_to_pure_permutation(self):
        data = make_design([2, 3, 2], rng=3)
        agg = aggregate_to_treatment_level(data, 1)  # zero iterations
        strata = _cluster_ids(agg, 1)
        p_ref, _ = exact_permutation_pvalue(
            agg[:, 1], agg[:, -1], STATISTICS["corr"], "two-sided", strata=strata
        )
        with pytest.warns(UserWarning, match="No levels to bootstrap"):
            p_lib = hs.hypothesis_test(
                data, 1, bootstraps=50, permutations=6000, random_state=2
            )
        assert abs(p_lib - p_ref) < 0.035
