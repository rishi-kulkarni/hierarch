"""Contract of GroupbyMean and of aggregating a bootstrapped sample.

On raw data GroupbyMean is a plain successive groupby-mean (== pandas). On a
weight- or index-bootstrapped sample, aggregating up to the level the bootstrap
started from must yield the nested *multiplicity*-weighted mean of the resample
("mean of cluster means"; clusters are weighted by how many times they were drawn,
not by their row count -- this matters for unbalanced designs). 'indexes' and
'weights' are two encodings of the same resample and must agree after aggregation.
"""

import numpy as np
import pytest

from hierarch.internal_functions import GroupbyMean
from hierarch.resampling import bootstrap_plan, draw_bootstrap_weights
from tests.conftest import design_names
from tests._reference import groupby_mean, nested_weighted_mean


def _cluster_ids(data, ncols):
    """Integer id of each row's cluster defined by the first `ncols` columns."""
    if ncols == 0:
        return np.zeros(len(data), dtype=int)
    _, inv = np.unique(data[:, :ncols], axis=0, return_inverse=True)
    return inv.ravel()


def _is_balanced(hier):
    return all(isinstance(h, int) for h in hier)


def _transform(plan, rng, start, kind, data):
    """Draw one set of bootstrap weights and apply them to data."""
    weights = draw_bootstrap_weights(plan, rng, start, kind)
    if kind == "indexes":
        return data.astype(np.float64)[np.repeat(np.arange(data.shape[0]), weights)]
    out = data.astype(np.float64).copy()
    out[:, -1] = out[:, -1] * weights
    return out


class TestBootstrapKindsAgree:
    @pytest.mark.parametrize("name", design_names())
    def test_indexes_and_weights_agree_after_aggregation(self, name, design_pool):
        """Documented contract: 'indexes' and 'weights' describe the same
        resample; once aggregated up to the level the bootstrap started from,
        the two are indistinguishable."""
        hier, data = design_pool[name]
        ncols = data.shape[1]
        plan = bootstrap_plan(data[:, :-1])
        for start in range(1, ncols - 1):
            iterations = ncols - 1 - start  # aggregate to level start-1
            for seed in range(3):
                # same seed => same underlying draw for "weights" and "indexes",
                # which only differ in how the draw is applied to the data
                w_out = _transform(
                    plan, np.random.default_rng(seed), start, "weights", data
                )
                i_out = _transform(
                    plan, np.random.default_rng(seed), start, "indexes", data
                )
                g = GroupbyMean()
                g.fit(data)
                np.testing.assert_allclose(
                    g.transform(w_out, iterations=iterations),
                    g.transform(i_out, iterations=iterations, resampled=True),
                )


class TestGroupbyMeanContract:
    def test_matches_pandas_for_every_depth(self, design):
        _, hier, data = design
        g = GroupbyMean()
        g.fit(data)
        for iters in range(1, data.shape[1] - 1):
            np.testing.assert_allclose(
                g.transform(data, iterations=iters),
                groupby_mean(data, iterations=iters),
            )

    @pytest.mark.parametrize(
        "name, kind",
        design_names(kinds=["weights", "indexes", "bayesian"]),
    )
    def test_aggregating_a_bootstrap_sample_gives_the_nested_weighted_mean(
        self, name, kind, design_pool
    ):
        """hypothesis_test aggregates a bootstrapped sample with a GroupbyMean
        fitted on the ORIGINAL data. Whatever the intermediate levels look like,
        the result at the level the bootstrap started from must be the nested
        weighted mean of the resample."""
        hier, data = design_pool[name]
        ncols = data.shape[1]
        unit = data.copy()
        unit[:, -1] = 1.0
        plan = bootstrap_plan(data[:, :-1])
        for start in range(1, ncols - 1):
            iterations = ncols - 1 - start
            for seed in range(3):
                wkind = "weights" if kind == "indexes" else kind
                w = _transform(plan, np.random.default_rng(seed), start, wkind, unit)[
                    :, -1
                ]  # per-row weights
                if kind == "indexes":
                    # same seed => same resample as 'weights'; the transform of
                    # the real data is the row-expanded sample
                    sample = _transform(
                        plan, np.random.default_rng(seed), start, kind, data
                    )
                    ref_weights = np.bincount(
                        [
                            int(np.flatnonzero((data[:, :-1] == r[:-1]).all(1))[0])
                            for r in sample
                        ],
                        minlength=len(data),
                    )
                    np.testing.assert_array_equal(ref_weights, w)  # multiplicities
                else:
                    sample = data.copy()
                    sample[:, -1] *= w
                g = GroupbyMean()
                g.fit(data)
                np.testing.assert_allclose(
                    g.transform(
                        sample, iterations=iterations, resampled=(kind == "indexes")
                    ),
                    nested_weighted_mean(data, w, iterations),
                    atol=1e-12,
                )

    def test_transform_does_not_mutate_input(self, design):
        _, hier, data = design
        g = GroupbyMean()
        g.fit(data)
        before = data.copy()
        g.transform(data, iterations=1)
        np.testing.assert_array_equal(data, before)
