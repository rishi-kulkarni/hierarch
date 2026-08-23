from dataclasses import dataclass
from itertools import cycle
from typing import Iterable, Tuple, Union

import numpy as np

from hierarch.internal_functions import (
    _repeat,
    id_cluster_counts,
    msp,
)


@dataclass(frozen=True)
class BootstrapPlan:
    """Precomputed cluster geometry for nested bootstrapping.

    ``children_per_level[k]`` holds the number of level-(k+1) subclusters
    within each level-k cluster (rows are the deepest level's children), in
    lexicographic order. ``resampled_levels[k]`` is False for levels listed
    in ``skip`` at plan time. ``block_starts_per_level`` and
    ``multinomial_groups_per_level`` are shape-derived lookup tables so that
    drawing weights spends no time re-deriving geometry: each multinomial
    group is one ``(size, cluster_indices, scatter_indices, pvals)`` batch of
    same-sized clusters that can be drawn in a single Generator call.
    """

    children_per_level: Tuple[np.ndarray, ...]
    resampled_levels: Tuple[bool, ...]
    block_starts_per_level: Tuple[np.ndarray, ...]
    multinomial_groups_per_level: Tuple[tuple, ...]


def bootstrap_plan(design: np.ndarray, skip: Iterable[int] = ()) -> BootstrapPlan:
    """Analyze the hierarchy of a lexsorted design matrix.

    Parameters
    ----------
    design : 2D numeric ndarray
        Design columns only (no dependent-variable column). Must be
        lexicographically sorted.
    skip : iterable of ints, optional
        Levels that should not be resampled when drawing weights.

    Returns
    -------
    BootstrapPlan
    """
    cluster_dict = id_cluster_counts(design)
    children = tuple(reversed(list(cluster_dict.values())))
    skip = frozenset(skip)
    resampled = tuple(level not in skip for level in range(len(children)))

    starts, groups = [], []
    for level_children in children:
        block_starts = np.concatenate(([0], np.cumsum(level_children[:-1])))
        starts.append(block_starts)
        level_groups = []
        for size in np.unique(level_children):
            clusters = np.flatnonzero(level_children == size)
            scatter = block_starts[clusters][:, None] + np.arange(size)
            level_groups.append((int(size), clusters, scatter, [1 / size] * size))
        groups.append(tuple(level_groups))
    return BootstrapPlan(children, resampled, tuple(starts), tuple(groups))


def draw_bootstrap_weights(
    plan: BootstrapPlan,
    rng: np.random.Generator,
    start: int,
    kind: str,
) -> np.ndarray:
    """Draw one set of per-row bootstrap weights for a fitted plan.

    Parameters
    ----------
    plan : BootstrapPlan
    rng : numpy.random.Generator
        Source of randomness; advanced by this call.
    start : int
        First level to resample.
    kind : { "weights", "indexes", "bayesian" }
        "weights" and "indexes" draw integer (Efron) weights; "bayesian"
        draws continuous Dirichlet weights.

    Returns
    -------
    1D array of per-row weights
        Integer dtype for the Efron bootstrap, float64 for the Bayesian.
    """
    if kind == "bayesian":
        weights = np.ones(len(plan.children_per_level[start]))
    else:
        weights = np.ones(len(plan.children_per_level[start]), dtype=np.int64)
    for level in range(start, len(plan.children_per_level)):
        children = plan.children_per_level[level]
        if not plan.resampled_levels[level]:
            weights = np.repeat(weights, children)
        elif kind == "bayesian":
            weights = _draw_dirichlet_level(
                weights, children, plan.block_starts_per_level[level], rng
            )
        else:
            weights = _draw_multinomial_level(
                weights,
                plan.multinomial_groups_per_level[level],
                int(children.sum()),
                rng,
            )
    return weights


def _draw_multinomial_level(weights, groups, total, rng):
    """One nested Efron resampling step: each cluster's weight is split
    among its children by a uniform multinomial draw."""
    out = np.empty(total, dtype=np.int64)
    for size, clusters, scatter, pvals in groups:
        out[scatter] = rng.multinomial(size * weights[clusters], pvals)
    return out


def _draw_dirichlet_level(weights, children, block_starts, rng):
    """One nested Bayesian resampling step: each cluster's weight is split
    among its children by a flat Dirichlet draw."""
    gammas = rng.standard_gamma(1.0, size=int(children.sum()))
    sums = np.add.reduceat(gammas, block_starts)
    scale = weights * children
    return gammas / np.repeat(sums, children) * np.repeat(scale, children)


class Bootstrapper:
    """Bootstrapper(random_state=None, kind="weights")

    This transformer performs a nested bootstrap on the target data.
    Undefined behavior if the target data is not lexicographically
    sorted.

    Parameters
    ----------
    random_state : int or numpy.random.Generator instance, optional
        Seeds the Bootstrapper for reproducibility, by default None
    kind : { "weights", "bayesian", "indexes" }
        Specifies the bootstrapping algorithm.

        "weights" generates a set of new integer weights for
        each datapoint.

        "bayesian" generates a set of new real weights for
        each datapoint.

        "indexes" generates a set of new indexes for the dataset.
        Mathematically, this is equivalent to demanding integer weights.

    Notes
    -----
    These approaches have different outputs - "weights" and "bayesian"
    output arrays the same size of the original array, but with
    every y-value multiplied by generated weight. "indexes" will
    output an array that is not necessarily the same size as the
    original array, but the weight of each y-value is 1, so certain
    metrics are easier to compute. Assuming both algorithms generated
    the "same" sample in terms of reweights, the arrays will be
    equivalent after the groupby and aggregate step.

    "bayesian" has no reindexing equivalent.

    Examples
    --------
    Generate a simple design matrix with dependent variable always equal to 1.

    >>> from hierarch.power import DataSimulator
    >>> paramlist = [[1]*2, [0]*6, [0]*18]
    >>> hierarchy = [2, 3, 3]
    >>> datagen = DataSimulator(paramlist)
    >>> datagen.fit(hierarchy)
    >>> data = datagen.generate()
    >>> data
    array([[1., 1., 1., 1.],
           [1., 1., 2., 1.],
           [1., 1., 3., 1.],
           [1., 2., 1., 1.],
           [1., 2., 2., 1.],
           [1., 2., 3., 1.],
           [1., 3., 1., 1.],
           [1., 3., 2., 1.],
           [1., 3., 3., 1.],
           [2., 1., 1., 1.],
           [2., 1., 2., 1.],
           [2., 1., 3., 1.],
           [2., 2., 1., 1.],
           [2., 2., 2., 1.],
           [2., 2., 3., 1.],
           [2., 3., 1., 1.],
           [2., 3., 2., 1.],
           [2., 3., 3., 1.]])

    Generate a bootstrapped sample by resampling column 1, then column 2. The "weights"
    algorithm multiplies all of the dependent variable values by the resampled weights.
    Starting at column 1 means that some column 2 clusters might be zero-weighted.

    >>> boot = Bootstrapper(random_state=1, kind="weights")
    >>> boot.fit(data, skip=None)
    >>> boot.transform(data, start=1)
    array([[1., 1., 1., 1.],
           [1., 1., 2., 1.],
           [1., 1., 3., 1.],
           [1., 2., 1., 3.],
           [1., 2., 2., 1.],
           [1., 2., 3., 2.],
           [1., 3., 1., 0.],
           [1., 3., 2., 0.],
           [1., 3., 3., 0.],
           [2., 1., 1., 0.],
           [2., 1., 2., 0.],
           [2., 1., 3., 0.],
           [2., 2., 1., 3.],
           [2., 2., 2., 1.],
           [2., 2., 3., 5.],
           [2., 3., 1., 0.],
           [2., 3., 2., 0.],
           [2., 3., 3., 0.]])

    Starting at column 2 means that every column 1 cluster has equal weight.

    >>> boot = Bootstrapper(random_state=1, kind="weights")
    >>> boot.fit(data, skip=None)
    >>> boot.transform(data, start=2)
    array([[1., 1., 1., 1.],
           [1., 1., 2., 2.],
           [1., 1., 3., 0.],
           [1., 2., 1., 0.],
           [1., 2., 2., 3.],
           [1., 2., 3., 0.],
           [1., 3., 1., 1.],
           [1., 3., 2., 1.],
           [1., 3., 3., 1.],
           [2., 1., 1., 2.],
           [2., 1., 2., 0.],
           [2., 1., 3., 1.],
           [2., 2., 1., 1.],
           [2., 2., 2., 0.],
           [2., 2., 3., 2.],
           [2., 3., 1., 2.],
           [2., 3., 2., 1.],
           [2., 3., 3., 0.]])

    Skipping column 2 results in only column 1 clusters being resampled.

    >>> boot = Bootstrapper(random_state=1, kind="weights")
    >>> boot.fit(data, skip=[2])
    >>> boot.transform(data, start=1)
    array([[1., 1., 1., 1.],
           [1., 1., 2., 1.],
           [1., 1., 3., 1.],
           [1., 2., 1., 2.],
           [1., 2., 2., 2.],
           [1., 2., 3., 2.],
           [1., 3., 1., 0.],
           [1., 3., 2., 0.],
           [1., 3., 3., 0.],
           [2., 1., 1., 0.],
           [2., 1., 2., 0.],
           [2., 1., 3., 0.],
           [2., 2., 1., 3.],
           [2., 2., 2., 3.],
           [2., 2., 3., 3.],
           [2., 3., 1., 0.],
           [2., 3., 2., 0.],
           [2., 3., 3., 0.]])

    Changing the algorithm to "indexes" gives a more familiar result.

    >>> boot = Bootstrapper(random_state=1, kind="indexes")
    >>> boot.fit(data, skip=None)
    >>> boot.transform(data, start=1)
    array([[1., 1., 1., 1.],
           [1., 1., 2., 1.],
           [1., 1., 3., 1.],
           [1., 2., 1., 1.],
           [1., 2., 1., 1.],
           [1., 2., 1., 1.],
           [1., 2., 2., 1.],
           [1., 2., 3., 1.],
           [1., 2., 3., 1.],
           [2., 2., 1., 1.],
           [2., 2., 1., 1.],
           [2., 2., 1., 1.],
           [2., 2., 2., 1.],
           [2., 2., 3., 1.],
           [2., 2., 3., 1.],
           [2., 2., 3., 1.],
           [2., 2., 3., 1.],
           [2., 2., 3., 1.]])

    The Bayesian bootstrap is the same as the Efron bootstrap, but allows
    the resampled weights to take any real value up to the sum of the original
    weights in that cluster.

    >>> boot = Bootstrapper(random_state=2, kind="bayesian")
    >>> boot.fit(data, skip=None)
    >>> boot.transform(data, start=1)
    array([[1.        , 1.        , 1.        , 0.36182397],
           [1.        , 1.        , 2.        , 0.25496673],
           [1.        , 1.        , 3.        , 0.73887459],
           [1.        , 2.        , 1.        , 0.47257995],
           [1.        , 2.        , 2.        , 1.51837286],
           [1.        , 2.        , 3.        , 0.29299717],
           [1.        , 3.        , 1.        , 2.03854886],
           [1.        , 3.        , 2.        , 2.34934884],
           [1.        , 3.        , 3.        , 0.97248704],
           [2.        , 1.        , 1.        , 0.74915905],
           [2.        , 1.        , 2.        , 0.44384276],
           [2.        , 1.        , 3.        , 0.75993649],
           [2.        , 2.        , 1.        , 0.7887371 ],
           [2.        , 2.        , 2.        , 1.62961596],
           [2.        , 2.        , 3.        , 0.93041948],
           [2.        , 3.        , 1.        , 0.57605409],
           [2.        , 3.        , 2.        , 1.70326713],
           [2.        , 3.        , 3.        , 1.41896793]])

    """

    #: ("weights", "indexes", "bayesian) The three possible arguments that
    # can be provided to the "kind" keyword argument.
    _BOOTSTRAP_ALGORITHMS = tuple(["weights", "indexes", "bayesian"])

    def __init__(
        self,
        random_state: Union[np.random.Generator, int, None] = None,
        kind: str = "weights",
    ) -> None:

        self.random_generator = np.random.default_rng(random_state)
        if kind in self._BOOTSTRAP_ALGORITHMS:
            self.kind = kind
        else:
            raise KeyError("Invalid 'kind' argument.")
        self._plan = None

    def fit(self, data: np.ndarray, skip=None, y=-1) -> None:
        """Fit the bootstrapper to the target data.

        Parameters
        ----------
        data : 2D array
            Target data. Must be lexicographically sorted.
        sort : bool
            Set to false is data is already sorted by row, by default True.
        skip : list of integers, optional
            Columns to skip in the bootstrap. Skip columns that were sampled
            without replacement from the prior column, by default [].
        y : int, optional
            column index of the dependent variable, by default -1

        Raises
        ------
        ValueError
            Raises error if the input data is not a numpy numeric array.
        AttributeError
            Raises error if the input data is not a numpy array.

        """
        try:
            if not np.issubdtype(data.dtype, np.number):
                raise ValueError(
                    "Bootstrapper can only handle numeric datatypes. Please pre-process your data."
                )
        except AttributeError:
            raise AttributeError(
                "Bootstrapper can only handle numpy arrays. Please pre-process your data."
            )

        if skip is not None:
            skip = list(skip)
            for v in iter(skip):
                if not isinstance(v, int):
                    raise IndexError(
                        "skip values must be integers corresponding to column indices."
                    )
                if v >= data.shape[1] - 1:
                    raise IndexError("skip index out of bounds for this array.")
        else:
            skip = []

        y %= data.shape[1]
        self._plan = bootstrap_plan(data[:, :y], skip=skip)

    def transform(self, data: np.ndarray, start: int) -> np.ndarray:
        """Generate a bootstrapped sample from target data.

        Parameters
        ----------
        data : 2D array
            Target data. Must be sorted by row.
        start : int
            Column index of the first column to be bootstrapped.

        Returns
        -------
        2D array
            Array matching target data, but resampled with replacement
            according to "kind" argument.

        """
        if self._plan is None:
            raise Exception("Use fit() before using transform().")
        weights = draw_bootstrap_weights(
            self._plan, self.random_generator, start, self.kind
        )
        out = data.astype(np.float64)
        if self.kind == "indexes":
            return out[np.repeat(np.arange(out.shape[0]), weights)]
        out[:, -1] = out[:, -1] * weights
        return out


class Permuter:
    """Class for performing cluster-aware permutation on a target column.

    Parameters
    ----------
    random_state : int or numpy.random.Generator instance, optional
        Seedable for reproducibility, by default None

    Examples
    --------
    When the column to resample is the first column, Permuter performs an
    ordinary shuffle.

    >>> from hierarch.power import DataSimulator
    >>> from hierarch.internal_functions import GroupbyMean
    >>> paramlist = [[1]*2, [0]*6, [0]*18]
    >>> hierarchy = [2, 3, 3]
    >>> datagen = DataSimulator(paramlist)
    >>> datagen.fit(hierarchy)
    >>> data = datagen.generate()
    >>> agg = GroupbyMean()
    >>> test = agg.fit_transform(data)
    >>> test
    array([[1., 1., 1.],
           [1., 2., 1.],
           [1., 3., 1.],
           [2., 1., 1.],
           [2., 2., 1.],
           [2., 3., 1.]])

    Permuter performs an in-place shuffle on the fitted data.

    >>> permute = Permuter(random_state=1)
    >>> permute.fit(test, col_to_permute=0, exact=False)
    >>> permute.transform(test)
    array([[1., 1., 1.],
           [2., 2., 1.],
           [2., 3., 1.],
           [1., 1., 1.],
           [2., 2., 1.],
           [1., 3., 1.]])

    If exact=True, Permuter will not repeat a permutation until all possible
    permutations have been exhausted.

    >>> test = agg.fit_transform(data)
    >>> permute = Permuter(random_state=1)
    >>> permute.fit(test, col_to_permute=0, exact=True)
    >>> permute.transform(test)
    array([[2., 1., 1.],
           [2., 2., 1.],
           [2., 3., 1.],
           [1., 1., 1.],
           [1., 2., 1.],
           [1., 3., 1.]])
    >>> next(permute.iterator)
    [1.0, 2.0, 2.0, 2.0, 1.0, 1.0]
    >>> next(permute.iterator)
    [2.0, 1.0, 2.0, 2.0, 1.0, 1.0]

    If the column to permute is not 0, Permuter performs a within-cluster shuffle.
    Note that values of column 1 were shuffled within their column 0 cluster.

    >>> test = agg.fit_transform(data)
    >>> permute = Permuter(random_state=2)
    >>> permute.fit(test, col_to_permute=1, exact=False)
    >>> permute.transform(test)
    array([[1., 1., 1.],
           [1., 2., 1.],
           [1., 3., 1.],
           [2., 1., 1.],
           [2., 2., 1.],
           [2., 3., 1.]])

    Exact within-cluster permutations are not implemented, but there are typically
    too many to be worth attempting.

    >>> permute = Permuter(random_state=2)
    >>> permute.fit(test, col_to_permute=1, exact=True)
    Traceback (most recent call last):
        ...
    NotImplementedError: Exact permutation only available for col_to_permute = 0.
    """

    def __init__(
        self, random_state: Union[np.random.Generator, int, None] = None
    ) -> None:
        self.random_generator = np.random.default_rng(random_state)
        self._plan = None
        self._exact = False

    def fit(self, data: np.ndarray, col_to_permute: int, exact: bool = False) -> None:
        """Fit the permuter to the target data.

        Parameters
        ----------
        data : 2D numeric ndarray
            Target data.
        col_to_permute : int
            Index of target column.
        exact : bool, optional
            If True, will enumerate all possible permutations and
            iterate through them one by one, by default False. Only
            works if target column has index 0.
        """
        if col_to_permute != 0 and exact is True:
            raise NotImplementedError(
                "Exact permutation only available for col_to_permute = 0."
            )

        self._exact = exact
        self._col = col_to_permute

        if exact is True:
            values, counts = np.unique(
                data[:, : col_to_permute + 2], return_counts=True, axis=0
            )
            col_values = values[:, -2].tolist()
            self.iterator = cycle(msp(col_values))
            self._counts = None if len(col_values) == len(data) else counts
        else:
            self._plan = permutation_plan(data, col_to_permute)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Permute target column in-place.

        Parameters
        ----------
        data : 2D numeric ndarray
            Target data.

        Returns
        -------
        data : 2D numeric ndarray
            Original data with target column shuffled, in a stratified fashion if necessary.
        """
        if self._exact:
            labels = next(self.iterator)
            if self._counts is not None:
                labels = _repeat(tuple(labels), self._counts)
            data[:, self._col] = labels
        elif self._plan is not None:
            data[:, self._col] = draw_permuted_labels(
                self._plan, self.random_generator, 1
            )[0]
        else:
            raise Exception("Use fit() before using transform().")
        return data


@dataclass(frozen=True)
class PermutationPlan:
    """Precomputed structure for cluster-aware permutation of one column.

    A permutable *unit* is a distinct row of the design columns up to and
    including the column after the target column: whole clusters move
    together when the target column is above the row level. ``unit_values``
    holds each unit's target-column value; ``stratum_ids`` groups units that
    may exchange values (all zeros when the target column is column 0);
    ``row_repeats`` expands unit labels back to data rows, or None when
    units and rows coincide.
    """

    col: int
    unit_values: np.ndarray
    stratum_ids: np.ndarray
    row_repeats: Union[np.ndarray, None]


def permutation_plan(data: np.ndarray, col_to_permute: int) -> PermutationPlan:
    """Analyze the target data for cluster-aware permutation.

    Parameters
    ----------
    data : 2D numeric ndarray
        Target data. Must be lexicographically sorted.
    col_to_permute : int
        Index of the column to be permuted.

    Returns
    -------
    PermutationPlan
    """
    values, indexes, counts = np.unique(
        data[:, : col_to_permute + 2], return_index=True, return_counts=True, axis=0
    )
    if col_to_permute == 0:
        stratum_ids = np.zeros(len(values), dtype=np.int64)
    else:
        stratum_ids = np.unique(values[:, :-2], axis=0, return_inverse=True)[1].ravel()
    unit_values = values[:, -2].copy()
    row_repeats = None if indexes.size == len(data) else counts
    return PermutationPlan(col_to_permute, unit_values, stratum_ids, row_repeats)


def draw_permuted_labels(
    plan: PermutationPlan, rng: np.random.Generator, size: int
) -> np.ndarray:
    """Draw a batch of cluster-aware permutations of the target column.

    Each output row is one independent uniform (stratified) permutation of
    the unit labels, expanded to data rows if units span multiple rows.

    Parameters
    ----------
    plan : PermutationPlan
    rng : numpy.random.Generator
        Source of randomness; advanced by this call.
    size : int
        Number of permutations to draw.

    Returns
    -------
    2D array of shape (size, number of data rows)
    """
    n_units = len(plan.unit_values)
    # sorting random keys offset by stratum id keeps every unit inside its
    # stratum while shuffling uniformly within it
    keys = plan.stratum_ids + rng.random((size, n_units))
    order = np.argsort(keys, axis=1)
    labels = plan.unit_values[order]
    if plan.row_repeats is not None:
        labels = np.repeat(labels, plan.row_repeats, axis=1)
    return labels


def exact_label_matrix(data: np.ndarray, col_to_permute: int) -> np.ndarray:
    """Enumerate every distinct permutation of the target column's unit
    labels, in multiset-permutation order, expanded to data rows.

    Parameters
    ----------
    data : 2D numeric ndarray
        Target data. Must be lexicographically sorted.
    col_to_permute : int
        Index of the column to be permuted. Must be 0.

    Returns
    -------
    2D array of shape (number of distinct permutations, number of data rows)
    """
    values, counts = np.unique(
        data[:, : col_to_permute + 2], return_counts=True, axis=0
    )
    col_values = values[:, -2].tolist()
    labels = np.array(list(msp(col_values)))
    if len(col_values) != len(data):
        labels = np.repeat(labels, counts, axis=1)
    return labels
