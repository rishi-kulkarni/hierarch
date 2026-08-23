"""Independent, deliberately naive reference implementations used by the test suite.

Nothing in here imports from ``hierarch``. Everything is written the slow, obvious
way (itertools enumeration, pandas groupby, straight numpy formulas) so that it does
not share code -- or bugs -- with the library it is checking.
"""

import itertools
import math

import numpy as np
import pandas as pd
import scipy.stats as stats

# --------------------------------------------------------------------------- #
# design matrices
# --------------------------------------------------------------------------- #


def make_design(hierarchy, rng=None, y="normal", treatment_effect=0.0, icc=0.0):
    """Build a lexicographically sorted design matrix with a y column.

    Parameters
    ----------
    hierarchy : list
        One entry per design column. Each entry is either an int (every parent
        gets that many children) or a (lo, hi) tuple (each parent gets a random
        integer number of children in [lo, hi] -- an unbalanced design).
        Level 0 is the number of top-level clusters.
    rng : numpy Generator
    y : {"normal", "lognormal", "integers"} or callable(rng, n) -> array
    treatment_effect : float
        Added to y for rows whose column-0 label is the largest label.
    icc : float
        Standard deviation of a random effect added at every non-terminal level,
        to induce intraclass correlation.
    """
    rng = np.random.default_rng(rng)
    rows = [[]]
    for level in hierarchy:
        new_rows = []
        for parent in rows:
            k = (
                level
                if isinstance(level, int)
                else int(rng.integers(level[0], level[1] + 1))
            )
            for c in range(1, k + 1):
                new_rows.append(parent + [c])
        rows = new_rows
    design = np.array(rows, dtype=np.float64)
    n = design.shape[0]

    if callable(y):
        yvals = np.asarray(y(rng, n), dtype=np.float64)
    elif y == "normal":
        yvals = rng.standard_normal(n)
    elif y == "lognormal":
        yvals = rng.lognormal(size=n)
    elif y == "integers":
        yvals = rng.integers(0, 4, size=n).astype(np.float64)
    else:
        raise ValueError(y)

    if icc:
        for j in range(design.shape[1] - 1):
            keys = [tuple(r) for r in design[:, : j + 1]]
            uniq = {k: rng.normal(scale=icc) for k in set(keys)}
            yvals = yvals + np.array([uniq[k] for k in keys])

    if treatment_effect:
        top = design[:, 0] == design[:, 0].max()
        yvals = yvals + treatment_effect * top

    return np.column_stack([design, yvals])


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def studentized_covariance(x, y):
    """Straight transcription of the formula in the hierarch paper/docstring."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    xc, yc = x - x.mean(), y - y.mean()

    def moment(a, b, pow_, ddof):
        return np.sum((a**pow_) * (b**pow_)) / (n - ddof)

    numer = moment(xc, yc, 1, 1)
    d1 = moment(xc, yc, 2, math.sqrt(2))
    d2 = moment(xc, xc, 1, 1) * moment(yc, yc, 1, 1) / (n - 1)
    d3 = (n - 2) * moment(xc, yc, 1, 1.75) ** 2 / (n - 1)
    return numer / math.sqrt((1 / (n - 1.5)) * (d1 + d2 - d3))


def jackknife_studentized_covariance(x, y):
    """Sample covariance over its leave-one-out jackknife standard error."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    full = np.cov(x, y, ddof=1)[0, 1]
    loo = np.array(
        [np.cov(np.delete(x, i), np.delete(y, i), ddof=1)[0, 1] for i in range(n)]
    )
    se = math.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2))
    return full / se


def welch(x, y):
    """Welch t statistic; the label with the smallest value is group A."""
    x = np.asarray(x)
    y = np.asarray(y, dtype=np.float64)
    labels = np.unique(x)
    assert labels.size == 2
    a, b = y[x == labels[0]], y[x == labels[1]]
    return stats.ttest_ind(a, b, equal_var=False).statistic


STATISTICS = {
    "corr": studentized_covariance,
    "jackknife_corr": jackknife_studentized_covariance,
    "means": welch,
}


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


def groupby_mean(data, iterations=1):
    """Successive groupby-mean of the last column over the leading columns.

    Each iteration drops the last design column and averages y within the
    remaining columns. Returns a lexsorted float array like the library does.
    """
    out = np.asarray(data, dtype=np.float64)
    for _ in range(iterations):
        ncol = out.shape[1]
        df = pd.DataFrame(out)
        keys = list(range(ncol - 2))
        out = df.groupby(keys, sort=True)[ncol - 1].mean().reset_index().to_numpy()
    return out


def aggregate_to_treatment_level(data, treatment_col):
    """Aggregate a raw dataset the way hypothesis_test does before permuting."""
    levels_to_agg = data.shape[1] - treatment_col - 3
    return groupby_mean(data, iterations=levels_to_agg)


# --------------------------------------------------------------------------- #
# p-values
# --------------------------------------------------------------------------- #


def pvalue_from_null(observed, null, alternative):
    null = np.asarray(null)
    total = null.size
    p_less = np.sum(observed >= null) / total
    p_greater = np.sum(observed <= null) / total
    p = {"two-sided": 2 * min(p_less, p_greater), "less": p_less, "greater": p_greater}[
        alternative
    ]
    if p == 0:
        p += 1 / total
    return p


def distinct_permutations(x):
    """All distinct orderings of the multiset x.

    Textbook recursion: at each position, place each *unique* remaining value
    (the input is sorted, so duplicates are adjacent and skipped). This visits
    only the distinct orderings rather than filtering all n! of them, and shares
    no structure with hierarch's Takaoka-linked-list ``msp``.
    """

    def rec(remaining):
        if not remaining:
            yield ()
            return
        for i, v in enumerate(remaining):
            if i and v == remaining[i - 1]:
                continue
            for rest in rec(remaining[:i] + remaining[i + 1 :]):
                yield (v,) + rest

    for perm in rec(sorted(x)):
        yield np.array(perm, dtype=np.float64)


def distinct_stratified_permutations(x, strata):
    """All distinct orderings of x that keep every value inside its stratum."""
    x = np.asarray(x)
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    per_group = [list(distinct_permutations(x[g])) for g in groups]
    for combo in itertools.product(*per_group):
        out = np.empty_like(x, dtype=np.float64)
        for g, vals in zip(groups, combo):
            out[g] = vals
        yield out


BRUTE_FORCE_LIMIT = 50_000


def n_distinct_stratified_permutations(x, strata):
    x = np.asarray(x)
    strata = np.asarray(strata)
    out = 1
    for s in np.unique(strata):
        out *= n_distinct_permutations(x[strata == s])
    return out


def exact_permutation_pvalue(x, y, statistic, alternative="two-sided", strata=None):
    """Brute-force exact permutation p-value with hierarch's p-value convention.

    Refuses (rather than hangs) when the enumeration exceeds BRUTE_FORCE_LIMIT.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    size = (
        n_distinct_permutations(x)
        if strata is None
        else n_distinct_stratified_permutations(x, strata)
    )
    if size > BRUTE_FORCE_LIMIT:
        raise ValueError(f"brute force would enumerate {size} permutations; shrink the design")
    observed = statistic(x, y)
    perms = (
        distinct_permutations(x)
        if strata is None
        else distinct_stratified_permutations(x, strata)
    )
    null = np.array([statistic(p, y) for p in perms])
    return pvalue_from_null(observed, null, alternative), null


def n_distinct_permutations(x):
    """Multinomial coefficient: number of distinct orderings of multiset x."""
    _, counts = np.unique(x, return_counts=True)
    out = math.factorial(int(counts.sum()))
    for c in counts:
        out //= math.factorial(int(c))
    return out


def nested_weighted_mean(data, row_weights, iterations):
    """The statistic a nested bootstrap should produce after aggregation.

    Going up one level at a time, a cluster's value is the mean of its children's
    values weighted by the children's bootstrap *multiplicities*, and the cluster's
    own multiplicity is the mean multiplicity of its children (a nested Efron
    bootstrap draws ``k * w_parent`` children, so ``sum(child weights) / k`` is how
    many times the parent was drawn). Row weights are the deepest multiplicities.
    A cluster with zero total weight contributes nothing and gets value 0.

    NB: this weights clusters by multiplicity, NOT by their number of rows -- in an
    unbalanced design "mean of cluster means" is what hierarch computes.
    """
    df = pd.DataFrame(np.asarray(data, dtype=np.float64))
    ncol = df.shape[1]
    df["_w"] = np.asarray(row_weights, dtype=np.float64)
    for it in range(iterations):
        keys = list(range(ncol - 2 - it))
        g = df.groupby(keys, sort=True)
        wsum = g["_w"].sum()
        wy = (df["_w"] * df[ncol - 1]).groupby([df[k] for k in keys], sort=True).sum()
        value = np.where(wsum > 0, wy / wsum.where(wsum > 0, 1), 0.0)
        multiplicity = wsum / g.size()
        df = wsum.reset_index()[keys]
        df[ncol - 1] = value
        df["_w"] = multiplicity.to_numpy()
    return df[[*range(ncol - 2 - (iterations - 1)), ncol - 1]].to_numpy()
