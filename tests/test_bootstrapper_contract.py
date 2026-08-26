"""Contract of the nested bootstrap, independent of implementation.

Weights are non-negative (integer for Efron, continuous for Bayesian), mass is
redistributed only *within* the clusters at the level the bootstrap starts from,
resampling is nested (a zero-weight parent zeroes its descendants), skipped levels
inherit their parent's weight, and the marginal weight distributions at the last
level are Multinomial(k, 1/k) / k*Dirichlet(1,...,1).
"""

import numpy as np
import pytest
import scipy.stats as stats

from hierarch.resampling import bootstrap_plan, draw_bootstrap_weights
from tests._reference import make_design


def _cluster_ids(data, ncols):
    """Integer id of each row's cluster defined by the first `ncols` columns."""
    if ncols == 0:
        return np.zeros(len(data), dtype=int)
    _, inv = np.unique(data[:, :ncols], axis=0, return_inverse=True)
    return inv.ravel()


def _is_balanced(hier):
    return all(isinstance(h, int) for h in hier)


def _transform(plan, rng, start, kind, data, skip=()):
    """Draw one set of bootstrap weights and apply them, mirroring the old
    Bootstrapper.transform contract."""
    weights = draw_bootstrap_weights(plan, rng, start, kind)
    if kind == "indexes":
        return data.astype(np.float64)[np.repeat(np.arange(data.shape[0]), weights)]
    out = data.astype(np.float64).copy()
    out[:, -1] = out[:, -1] * weights
    return out


class TestBootstrapperContract:
    @pytest.mark.parametrize("kind", ["weights", "bayesian"])
    def test_weights_only_touch_y(self, design, kind):
        _, hier, data = design
        plan = bootstrap_plan(data[:, :-1])
        rng = np.random.default_rng(0)
        for start in range(1, data.shape[1] - 1):
            out = _transform(plan, rng, start, kind, data)
            assert out.shape == data.shape
            np.testing.assert_array_equal(out[:, :-1], data[:, :-1])

    def test_efron_weights_are_nonnegative_integers(self, design):
        _, hier, data = design
        data = data.copy()
        data[:, -1] = 1.0
        plan = bootstrap_plan(data[:, :-1])
        rng = np.random.default_rng(1)
        for start in range(1, data.shape[1] - 1):
            for _ in range(20):
                w = _transform(plan, rng, start, "weights", data)[:, -1]
                assert np.all(w >= 0)
                assert np.all(w == np.round(w))

    def test_bayesian_weights_are_nonnegative_and_continuous(self, design):
        _, hier, data = design
        data = data.copy()
        data[:, -1] = 1.0
        plan = bootstrap_plan(data[:, :-1])
        rng = np.random.default_rng(1)
        w = _transform(plan, rng, 1, "bayesian", data)[:, -1]
        assert np.all(w >= 0)
        # a Dirichlet draw is (a.s.) never integer valued
        assert not np.all(w == np.round(w))

    @pytest.mark.parametrize("kind", ["weights", "bayesian"])
    def test_weight_mass_is_conserved_within_each_unresampled_cluster(
        self, design, kind
    ):
        """The nested bootstrap redistributes weight *within* the clusters at
        level start-1; each of those clusters keeps exactly its original mass
        (= its number of rows) when the design below it is balanced."""
        _, hier, data = design
        if not _is_balanced(hier):
            pytest.skip(
                "exact conservation only holds per-cluster for balanced designs"
            )
        data = data.copy()
        data[:, -1] = 1.0
        plan = bootstrap_plan(data[:, :-1])
        rng = np.random.default_rng(3)
        for start in range(1, data.shape[1] - 1):
            ids = _cluster_ids(data, start)
            for _ in range(10):
                w = _transform(plan, rng, start, kind, data)[:, -1]
                got = np.bincount(ids, weights=w)
                want = np.bincount(ids)
                np.testing.assert_allclose(got, want, rtol=1e-9)

    @pytest.mark.parametrize("kind", ["weights", "bayesian"])
    def test_weight_mass_conserved_in_expectation_when_unbalanced(self, kind):
        """For unbalanced designs per-cluster mass is conserved only in
        expectation: E[sum of weights in cluster] == cluster size."""
        data = make_design([2, (2, 4), (1, 5)], rng=7)
        data[:, -1] = 1.0
        plan = bootstrap_plan(data[:, :-1])
        rng = np.random.default_rng(3)
        ids = _cluster_ids(data, 1)
        reps = 3000
        acc = np.zeros(ids.max() + 1)
        for _ in range(reps):
            acc += np.bincount(ids, weights=_transform(plan, rng, 1, kind, data)[:, -1])
        np.testing.assert_allclose(acc / reps, np.bincount(ids), rtol=0.05)

    def test_each_row_has_unit_expected_weight(self, design):
        _, hier, data = design
        data = data.copy()
        data[:, -1] = 1.0
        plan = bootstrap_plan(data[:, :-1])
        rng = np.random.default_rng(5)
        reps = 3000
        acc = np.zeros(len(data))
        for _ in range(reps):
            acc += _transform(plan, rng, 1, "weights", data)[:, -1]
        mean_w = acc / reps
        # E[w] = 1 for every row; SE of the mean is ~ sqrt(var/reps) <= ~0.03
        assert np.all(np.abs(mean_w - 1) < 0.12), mean_w

    def test_last_level_efron_weights_are_multinomial(self):
        """Resampling only the deepest level of a balanced design: each parent's
        k children get Multinomial(k, 1/k) weights, so Var(w) = 1 - 1/k."""
        k = 4
        data = make_design([2, 3, k], rng=11)
        data[:, -1] = 1.0
        plan = bootstrap_plan(data[:, :-1])
        rng = np.random.default_rng(8)
        start = data.shape[1] - 2  # last design column
        reps = 4000
        draws = np.array(
            [_transform(plan, rng, start, "weights", data)[:, -1] for _ in range(reps)]
        )
        # every parent's children sum to k on every draw
        parent = _cluster_ids(data, start)
        sums = np.array([np.bincount(parent, weights=d) for d in draws])
        assert np.all(sums == k)
        # marginal variance of each weight
        var = draws.var(axis=0)
        np.testing.assert_allclose(var, 1 - 1 / k, atol=0.08)
        # and P(w == 0) = (1 - 1/k)^k
        p0 = (draws == 0).mean(axis=0)
        np.testing.assert_allclose(p0, (1 - 1 / k) ** k, atol=0.05)

    def test_bayesian_last_level_weights_are_dirichlet(self):
        """Deepest-level Bayesian weights are k * Dirichlet(1,...,1): mean 1,
        variance (k-1)/(k+1) and marginally Beta(1, k-1) scaled by k."""
        k = 4
        data = make_design([2, 3, k], rng=11)
        data[:, -1] = 1.0
        plan = bootstrap_plan(data[:, :-1])
        rng = np.random.default_rng(8)
        start = data.shape[1] - 2
        reps = 4000
        draws = np.array(
            [_transform(plan, rng, start, "bayesian", data)[:, -1] for _ in range(reps)]
        )
        np.testing.assert_allclose(draws.mean(axis=0), 1, atol=0.08)
        np.testing.assert_allclose(draws.var(axis=0), (k - 1) / (k + 1), atol=0.08)
        ks = stats.kstest(draws[:, 0] / k, stats.beta(1, k - 1).cdf)
        assert ks.pvalue > 1e-3, ks

    def test_nesting_zero_parent_zeroes_all_descendants(self):
        """If a level-1 cluster draws weight zero, every row beneath it is zero
        (and, conversely, weight lands only inside resampled parents)."""
        data = make_design([1, 3, 2, 2], rng=2)  # single top cluster, 3 children
        data[:, -1] = 1.0
        plan = bootstrap_plan(data[:, :-1])
        rng = np.random.default_rng(4)
        ids1 = _cluster_ids(data, 2)  # level-1 clusters (cols 0,1)
        seen_zero_parent = False
        for _ in range(200):
            w = _transform(plan, rng, 1, "weights", data)[:, -1]
            per_parent = np.bincount(ids1, weights=w)
            for p, mass in enumerate(per_parent):
                if mass == 0:
                    seen_zero_parent = True
                    assert np.all(w[ids1 == p] == 0)
        assert (
            seen_zero_parent
        ), "expected at least one zero-weighted parent in 200 draws"

    def test_skipping_a_level_gives_children_their_parents_weight(self):
        data = make_design([2, 3, 4], rng=9)
        data[:, -1] = 1.0
        plan = bootstrap_plan(data[:, :-1], skip=[2])  # last level not resampled
        rng = np.random.default_rng(6)
        parent = _cluster_ids(data, 2)
        for _ in range(20):
            w = _transform(plan, rng, 1, "weights", data)[:, -1]
            # constant within each level-1 cluster ...
            for p in np.unique(parent):
                assert np.ptp(w[parent == p]) == 0
            # ... and not degenerate overall (level 1 was resampled)
        # over many draws some parent must have weight != 1
        assert any(
            np.any(_transform(plan, rng, 1, "weights", data)[:, -1] != 1)
            for _ in range(20)
        )

    def test_skipping_every_level_is_the_identity(self):
        data = make_design([2, 3, 4], rng=9)
        plan = bootstrap_plan(data[:, :-1], skip=[1, 2])
        rng = np.random.default_rng(6)
        for _ in range(5):
            np.testing.assert_array_equal(
                _transform(plan, rng, 1, "weights", data), data
            )

    def test_indexes_kind_returns_original_rows(self, design):
        _, hier, data = design
        plan = bootstrap_plan(data[:, :-1])
        rng = np.random.default_rng(1)
        rows = {tuple(r) for r in data}
        for start in range(1, data.shape[1] - 1):
            out = _transform(plan, rng, start, "indexes", data)
            assert all(tuple(r) in rows for r in out)
            if _is_balanced(hier):
                assert len(out) == len(data)

    def test_seed_reproducibility_and_variation(self, design):
        _, hier, data = design

        def draws(seed):
            plan = bootstrap_plan(data[:, :-1])
            rng = np.random.default_rng(seed)
            return [_transform(plan, rng, 1, "weights", data) for _ in range(3)]

        ra, rb, rc = draws(42), draws(42), draws(43)
        for x, y in zip(ra, rb):
            np.testing.assert_array_equal(x, y)
        assert not all(np.array_equal(x, y) for x, y in zip(ra, rc))
        # generator objects are honoured too
        np.testing.assert_array_equal(
            draws(np.random.default_rng(7))[0], draws(np.random.default_rng(7))[0]
        )

    def test_successive_transforms_differ(self, design):
        _, hier, data = design
        plan = bootstrap_plan(data[:, :-1])
        rng = np.random.default_rng(0)
        outs = [
            _transform(plan, rng, 1, "weights", data)[:, -1].tobytes()
            for _ in range(10)
        ]
        assert len(set(outs)) > 1
