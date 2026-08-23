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
from hierarch.resampling import Bootstrapper
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


class TestBootstrapKindsAgree:
    @pytest.mark.parametrize("name", design_names(xfail_indexes_unbalanced=True))
    def test_indexes_and_weights_agree_after_aggregation(self, name, design_pool):
        """Documented contract: 'indexes' and 'weights' describe the same
        resample; once aggregated up to the level the bootstrap started from,
        the two are indistinguishable."""
        hier, data = design_pool[name]
        ncols = data.shape[1]
        # NB: seeding is (currently) a global side effect, so build and use each
        # bootstrapper in sequence rather than constructing both up front.
        for start in range(1, ncols - 1):
            iterations = ncols - 1 - start  # aggregate to level start-1
            for seed in range(3):
                bw = Bootstrapper(random_state=seed, kind="weights")
                bw.fit(data)
                w_out = bw.transform(data, start=start)
                bi = Bootstrapper(random_state=seed, kind="indexes")
                bi.fit(data)
                i_out = bi.transform(data, start=start)
                g = GroupbyMean()
                g.fit(data)
                np.testing.assert_allclose(
                    g.transform(w_out, iterations=iterations),
                    g.transform(i_out, iterations=iterations),
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
        design_names(
            xfail_indexes_unbalanced=True, kinds=["weights", "indexes", "bayesian"]
        ),
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
        for start in range(1, ncols - 1):
            iterations = ncols - 1 - start
            for seed in range(3):
                wkind = "weights" if kind == "indexes" else kind
                boot = Bootstrapper(random_state=seed, kind=wkind)
                boot.fit(unit)
                w = boot.transform(unit, start=start)[:, -1]  # per-row weights
                if kind == "indexes":
                    # same seed => same resample as 'weights'; the transform of
                    # the real data is the row-expanded sample
                    boot = Bootstrapper(random_state=seed, kind=kind)
                    boot.fit(data)
                    sample = boot.transform(data, start=start)
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
                    g.transform(sample, iterations=iterations),
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
