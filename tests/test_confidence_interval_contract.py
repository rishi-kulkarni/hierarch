"""confidence_interval contract.

The interval contains the slope point estimate, widens with the confidence level,
is equivariant under affine transforms of y and scaling of x, shifts with an added
slope, agrees between compare='means' and 'corr' for two groups, and warns when the
refinement fails to converge.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import hierarch.stats as hs
from tests._reference import aggregate_to_treatment_level, make_design

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


class TestConfidenceInterval:
    KW = dict(bootstraps=30, permutations=100, random_state=3)

    def _ci(self, data, tc=0, **over):
        kw = {**self.KW, **over}
        return hs.confidence_interval(data, tc, **kw)

    def test_shape_ordering_and_contains_point_estimate(self, design):
        name, hier, data = design
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        lo, hi = self._ci(data)
        assert isinstance(lo, float) and isinstance(hi, float)
        assert lo < hi
        agg = aggregate_to_treatment_level(data, 0)
        assert lo < _slope(agg[:, 0], agg[:, -1]) < hi

    def test_wider_interval_for_higher_confidence(self, design):
        name, hier, data = design
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        lo95, hi95 = self._ci(data, interval=95)
        lo68, hi68 = self._ci(data, interval=68)
        assert lo95 < lo68 < hi68 < hi95

    def test_affine_equivariance_in_y(self, design):
        name, hier, data = design
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        lo, hi = self._ci(data)
        t = data.copy()
        t[:, -1] = 2.5 * t[:, -1] + 4.0
        lo2, hi2 = self._ci(t)
        assert lo2 == pytest.approx(2.5 * lo, rel=1e-9)
        assert hi2 == pytest.approx(2.5 * hi, rel=1e-9)

    def test_scaling_x_scales_slope_interval(self, design):
        name, hier, data = design
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        lo, hi = self._ci(data)
        t = data.copy()
        t[:, 0] = 4.0 * t[:, 0]
        lo2, hi2 = self._ci(t)
        assert lo2 == pytest.approx(lo / 4, rel=1e-9)
        assert hi2 == pytest.approx(hi / 4, rel=1e-9)

    def test_adding_a_slope_shifts_the_interval(self, design):
        name, hier, data = design
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        lo, hi = self._ci(data)
        t = data.copy()
        t[:, -1] = t[:, -1] + 1.5 * t[:, 0]
        lo2, hi2 = self._ci(t)
        assert lo2 == pytest.approx(lo + 1.5, rel=1e-9, abs=1e-9)
        assert hi2 == pytest.approx(hi + 1.5, rel=1e-9, abs=1e-9)

    def test_means_and_corr_agree_for_two_groups(self):
        data = make_design([2, 4, 3], rng=9, treatment_effect=1.0)
        assert self._ci(data, compare="means") == pytest.approx(
            self._ci(data, compare="corr")
        )

    def test_jackknife_is_close_to_corr(self):
        data = make_design([2, 6, 3], rng=9, treatment_effect=1.0)
        a = np.array(self._ci(data, compare="jackknife_corr"))
        b = np.array(self._ci(data, compare="corr"))
        # tolerance must be scaled by the interval width, not the endpoint
        # values: an endpoint near zero makes any rtol comparison meaningless
        # (observed: endpoints differing by ~3% of the width but ~70% of a
        # near-zero lower bound)
        width = b[1] - b[0]
        assert np.allclose(a, b, atol=0.25 * width, rtol=0)

    def test_dataframe_agrees_with_ndarray(self, design):
        name, hier, data = design
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        cols = [f"c{i}" for i in range(data.shape[1] - 1)] + ["y"]
        df = pd.DataFrame(data, columns=cols)
        assert self._ci(df) == self._ci(data)
        assert self._ci(df, tc="c0") == self._ci(data)

    def test_seed_reproducibility(self, design):
        name, hier, data = design
        if len(np.unique(data[:, 0])) < 2:
            pytest.skip()
        assert self._ci(data, random_state=5) == self._ci(data, random_state=5)
        assert self._ci(data, random_state=5) != self._ci(data, random_state=6)

    def test_convergence_warning_when_iterations_exhausted(self):
        data = make_design([2, 4, 3], rng=9)
        with pytest.warns(hs.ConvergenceWarning):
            self._ci(data, iterations=1, tolerance=0.0)

    def test_treatment_col_1(self):
        data = make_design([2, 3, 2, 3], rng=9)
        lo, hi = self._ci(data, tc=1)
        agg = aggregate_to_treatment_level(data, 1)
        assert lo < _slope(agg[:, 1], agg[:, -1]) < hi

    def test_bad_input_raises(self):
        with pytest.raises(TypeError):
            hs.confidence_interval("nope", 0)
