from collections import deque
from functools import lru_cache
from np_cache import np_lru_cache
from typing import Callable, Generator, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
from numba import guvectorize

from .internal_functions import (
    _repeat,
    msp,
    bounded_uint,
    set_numba_random_state,
)
from .pipeline import Pipeline


def bootstrap(
    *arrays: np.ndarray,
    start_col: int = 0,
    skip: Optional[List[int]] = None,
    n_resamples: int = 1000,
    kind: str = "weights",
    batch_size: int = 50,
    random_state: Union[np.random.Generator, int, None] = None,
) -> Iterator[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """

    Performs a nested bootstrap on the target data.
    Undefined behavior if the target data is not lexicographically
    sorted.

    Parameters
    ----------
    *arrays : np.ndarray(s)
        Arrays to bootstrap. Presumes that the first passed array is
        the design matrix and the second is the regressand.
    start_col : int, optional
        The first column of the design matrix to bootstrap, by default 0
    skip : _type_, optional
        Columns to skip in the bootstrap. Skip columns that were sampled
        without replacement from the prior column, by default None.
    n_resamples : int
        Number of resamples for Bootstrapper's resample method to generate,
        by default 1000.
    kind : { "weights", "bayesian", "indexes" }
        Specifies the bootstrapping algorithm.

        "weights" generates a set of new integer weights for
        each datapoint.

        "bayesian" generates a set of new real weights for
        each datapoint.

        "indexes" generates a set of new indexes for the dataset.
        Mathematically, this is equivalent to demanding integer weights.
    batch_size : int
        Number of bootstrap resamples to generate at once, by default 50.
        The resamples are yielded one at a time, but can be generated in
        a vectorized manner at the expense of memory.
    random_state : int or numpy.random.Generator instance, optional
        Seeds the Bootstrapper for reproducibility, by default None

    Yields
    ------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        If y is not provided, each resample will be a 1D np.ndarray containing
        resampled weights (or indexes, if kind="indexes"). If y is provided,
        each resample will be a tuple of X (or reindexed X), reweighted y
        (or reindexed y) values.

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
    Consider a simple two-level design matrix, with first level units in column 0.
    Second-level units are nested within first-level units, and observations are
    nested within second-level units.

    >>> design = np.array([[1, 1],
    ...                    [1, 2],
    ...                    [1, 3],
    ...                    [2, 4],
    ...                    [2, 5],
    ...                    [2, 6]]).repeat(3, axis=0)
    >>> design
    array([[1, 1],
           [1, 1],
           [1, 1],
           [1, 2],
           [1, 2],
           [1, 2],
           [1, 3],
           [1, 3],
           [1, 3],
           [2, 4],
           [2, 4],
           [2, 4],
           [2, 5],
           [2, 5],
           [2, 5],
           [2, 6],
           [2, 6],
           [2, 6]])

    Specifying kind="weights" to the Bootstrapper constructor makes the resample method yield
    the weights of the resampled data.

    One potential resampling plan is generating a bootstrapped sample by resampling second-level
    units from first-level units, then resampling observations from second-level units. This is
    done by specifying start_col=1 to the resample method.

    >>> resamples = list(bootstrap(design,
    ...                            start_col=1,
    ...                            n_resamples=10,
    ...                            kind="weights",
    ...                            random_state=1))
    >>> resamples[0]
    array([1, 2, 0, 1, 2, 0, 1, 1, 1, 1, 1, 4, 2, 0, 1, 0, 0, 0])

    These weights can be consumed by a function that computes a weighted statistical quantity.

    >>> y = np.array([1 for row in design])
    >>> np.average(y, weights=resamples[0])
    1.0

    Users may be more familiar with bootstrap resamples that return resampled indexes rather than
    weights. Setting kind="indexes" causes the resample method to yield indexes. Note that the
    if the random_state is the same, "indexes" and "weights" return the same resampled dataset.

    >>> resamples = list(bootstrap(design,
    ...                            start_col=1,
    ...                            n_resamples=10,
    ...                            kind="indexes",
    ...                            random_state=1))
    >>> resamples[0]
    array([ 0,  1,  1,  3,  4,  4,  6,  7,  8,  9, 10, 11, 11, 11, 11, 12, 12,
           14])

    Bootstrapper also implements the Bayesian bootstrap, which resamples weights from a dirichlet
    distribution rather than a multinomial distribution.

    >>> resamples = list(bootstrap(design,
    ...                            start_col=1,
    ...                            n_resamples=10,
    ...                            kind="bayesian",
    ...                            random_state=1))
    >>> resamples[0]
    array([0.88309416, 0.25491275, 0.17914842, 0.40079262, 1.88974779,
           0.7963255 , 2.41925443, 2.15130489, 0.02541944, 1.69127994,
           1.52348525, 0.42456709, 0.1854059 , 0.70222986, 2.28517886,
           0.48974997, 0.35698548, 1.34111767])

    For convenience, the resample method can be called with both X and y, causing the resampled
    weights/indexes to automatically be applied to both matrices.

    >>> resamples = list(bootstrap(design, y,
    ...                            start_col=1,
    ...                            n_resamples=10,
    ...                            kind="bayesian",
    ...                            random_state=1))
    >>> resampled_X, resampled_y = resamples[0]
    >>> resampled_y
    array([0.88309416, 0.25491275, 0.17914842, 0.40079262, 1.88974779,
           0.7963255 , 2.41925443, 2.15130489, 0.02541944, 1.69127994,
           1.52348525, 0.42456709, 0.1854059 , 0.70222986, 2.28517886,
           0.48974997, 0.35698548, 1.34111767])
    """

    if random_state is not None:
        # this is hacky, but we can use the numpy rng to seed the
        # numba rng. using seedsequence with this makes it process-safe
        random_generator = np.random.default_rng(random_state)
        nb_seed = random_generator.integers(low=2**32)
        set_numba_random_state(nb_seed)

    try:
        bootstrap_sampler = _bootstrapper_factory(kind)
    except KeyError:
        raise KeyError(f"Invalid 'kind' argument: {kind}")

    X, y = _validate(*arrays)

    if skip is None:
        skip = []
    else:
        if not set(skip).issubset(range(X.shape[1] + 1)):
            raise IndexError(f"skip contains invalid column indexes for X: {skip}")

    resampling_plan = tuple(
        (cluster, False) if idx in skip else (cluster, True)
        for idx, cluster in enumerate(id_cluster_counts(X))
    )

    bootstrap_pipeline = Pipeline(
        components=[
            (
                bootstrap_sampler,
                {
                    "n_resamples": n_resamples,
                    "batch_size": batch_size,
                },
            ),
        ]
    )

    if y is not None:
        if kind == "indexes":
            bootstrap_pipeline.add_component((_reindex, {"X": X, "y": y}))
        else:
            bootstrap_pipeline.add_component((_reweight, {"X": X, "y": y}))

    yield from bootstrap_pipeline.process(resampling_plan[start_col:])


def permute(
    X: np.ndarray,
    col_to_permute: int,
    *,
    n_resamples: int = 1000,
    exact: bool = False,
    batch_size: int = 100,
    random_state: Union[np.random.Generator, int, None] = None,
) -> Iterator[np.ndarray]:

    """Permuter(n_resamples, exact=False, random_state=None)

    Class for performing cluster-aware permutation on a target column.

    Parameters
    ----------
    X : np.ndarray
        Design matrix
    col_to_permute : int
        Target level to permute
    n_resamples: int
        Number of resamples for Permuter's resample method to generate,
        by default 1000. Overridden by exact, if true.
    exact : bool, optional
        If True, overrides n_resamples and instead causes resample to
        enumerate all possible permutations and iterate through them
        one by one, by default False.
        Warning: there can be a very large number of permutations for
        large datasets.
    random_state : int or numpy.random.Generator instance, optional
        Seedable for reproducibility, by default None

    Yields
    ------
    np.ndarray
        X with col_to_permute randomly shuffled

    Raises
    ------
    NotImplementedError
        Exact permutation when col_to_permute != 0 has not been implemented.

    Examples
    --------

    >>> design = np.array([[1, 1],
    ...                    [1, 2],
    ...                    [1, 3],
    ...                    [2, 4],
    ...                    [2, 5],
    ...                    [2, 6]])
    >>> design
    array([[1, 1],
           [1, 2],
           [1, 3],
           [2, 4],
           [2, 5],
           [2, 6]])

    If the first column is chosen as the target, Permuter will perform an ordinary shuffle
    and return the permuted column.

    >>> resamples = list(permute(design, col_to_permute=0, n_resamples=10, random_state=1))
    >>> resamples[0]
    array([1, 2, 2, 1, 2, 1])

    If exact=True, Permuter will not repeat a permutation until all possible
    permutations have been exhausted. For this design matrix, there are only
    20 possible permutations and we can see that Permuter enumerates all of them.
    Note that the n_resamples value is overriden when exact=True.

    >>> resamples = list(permute(design, col_to_permute=0, exact=True))
    >>> len(resamples)
    20

    Asking Permuter to permute the second column will result in a cluster-aware permutation -
    that is, second-level units will be shuffled within first-level units. In the following
    example, note that 1, 2, 3 remain nested in 1, while 4, 5, 6 remain nested in 2.

    >>> resamples = list(permute(design, col_to_permute=1, n_resamples=10, random_state=1))
    >>> resamples[0]
    array([1, 3, 2, 5, 4, 6])

    If any values are repeated, Permuter ensures that they are shuffled together.
    Note that there's rarely a reason to do this in a hypothesis-testing context.

    >>> repeated_design = design.repeat(2, axis=0)
    >>> resamples = list(permute(repeated_design,
    ...                          col_to_permute=0,
    ...                          n_resamples=10,
    ...                          random_state=1))
    >>> resamples[0]
    array([1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1])

    """

    if random_state is not None:
        # this is hacky, but we can use the numpy rng to seed the
        # numba rng. using seedsequence with this makes it process-safe
        random_generator = np.random.default_rng(random_state)
        nb_seed = random_generator.integers(low=2**32)
        set_numba_random_state(nb_seed)

    if col_to_permute != 0 and exact is True:
        raise NotImplementedError(
            "Exact permutation only available for col_to_permute = 0."
        )

    col_values, permutation_pipeline = make_permutation_pipeline(
        X, col_to_permute, exact, n_resamples, batch_size
    )

    yield from permutation_pipeline.process(np.array(col_values))


@np_lru_cache
def id_cluster_counts(design: np.ndarray) -> Tuple[np.ndarray]:
    """Identifies the hierarchy in a design matrix.

    Constructs a tuple of arrays corresponding describing the hierarchy
    in a design matrix. This presumes that the design matrix is organized
    hierarchically from left to right.

    Parameters
    ----------
    design : 2D numeric ndarray

    Returns
    -------
    Tuple
        Each index corresponds to a column index in the design matrix
        and each value is the number cluster in that column. Subclusters
        are expressed separately.

    Examples
    --------
    Consider a simple design matrix that has one y-value per
    x-value and no nesting:

    >>> design = np.array([1, 2, 3, 4, 5])[:, None]
    >>> design.shape
    (5, 1)

    >>> id_cluster_counts(design)
    (array([5]), array([1, 1, 1, 1, 1]))

    This reflects the words we used to describe the matrix - there are 5 x-values,
    each of which corresponds to a single y-value. This approach can describe
    design matrices of any level of hierarchy:

    >>> design = np.array([[1, 1],
    ...                    [1, 2],
    ...                    [1, 3],
    ...                    [2, 4],
    ...                    [2, 5],
    ...                    [2, 5]])
    >>> design.shape
    (6, 2)

    This matrix has two first-level x-values, the first of which contains three
    second-level x-values and the second of which contains two second-level x-values.
    Finally, each second-level x-value corresponds to a single y-value, except for 5,
    which corresponds to two y-values.

    id_cluster_counts returns this description:

    >>> id_cluster_counts(design)
    (array([2]), array([3, 2]), array([1, 1, 1, 1, 2]))

    """

    # deque is nice because we're traversing the hierarchy backwards
    cluster_desc = deque()

    # the number of clusters in the y-values is just the counts
    # from np.uniques of the design matrix
    prior_uniques, final_counts = np.unique(design, axis=0, return_counts=True)
    cluster_desc.appendleft(final_counts)

    # get the number of clusters in each column of the design matrix
    for i in range(design.shape[1]):
        prior_uniques, counts = np.unique(
            prior_uniques[:, :-1], axis=0, return_counts=True
        )
        cluster_desc.appendleft(counts)
    # return a tuple because it plays nicer with numba
    return tuple(cluster_desc)


@np_lru_cache
def make_permutation_pipeline(
    design: Tuple[Tuple],
    col_to_permute: int,
    exact: bool,
    n_resamples: int,
    batch_size: int,
) -> Pipeline:
    """This function produces the column to permute, any subclusters
    it contains, and any superclusters that contain it. This information
    is necessary for cluster-aware permutation.

    Delegates computation to a function with an lru_cache, which allows
    the permutation generator to be quickly reconstructed for nested
    resampling plans (say, performing a permutation test on each of many
    bootstrap samples).

    Parameters
    ----------
    design_matrix : Tuple[Tuple]
    col_to_permute : int
    exact: bool
    n_resamples: int

    Returns
    -------
    Pipeline
        Pipeline that yields permutations of design_matrix
    """
    # make a copy so the original is not shuffled
    cached_design = design.copy()

    if cached_design.ndim != 2:
        raise ValueError(
            f"design_matrix should be a 2D design matrix, got {cached_design.ndim} dimensions"
        )

    # we're not actually looking at the y-values, so if the column
    # to permute is the last column (which is generally should be),
    # we assume that there are no subclusters
    if col_to_permute % cached_design.shape[1] == cached_design.shape[1] - 1:
        permutation_matrix = cached_design
        subclusters = np.array([1 for row in cached_design])

    else:
        # we will actually be permuting the unique rows in this submatrix,
        # then duplicating any rows that contain subclusters
        permutation_matrix, subclusters = np.unique(
            cached_design[:, : col_to_permute + 2], axis=0, return_counts=True
        )

    # need to make this immutable
    col_values = permutation_matrix[:, col_to_permute]

    # if the target column is nested within another level, we have
    # to stratify the fisher-yates shuffle
    _, supercluster_idxs = np.unique(
        permutation_matrix[:, :col_to_permute], axis=0, return_index=True
    )
    supercluster_idxs = np.append(supercluster_idxs, len(permutation_matrix))

    permutation_pipeline = _shuffle_generator_factory(
        supercluster_idxs, subclusters, exact, n_resamples, batch_size
    )

    return col_values, permutation_pipeline


def _shuffle_generator_factory(
    supercluster_idxs: Tuple,
    subclusters: Tuple,
    exact: bool,
    n_resamples: int,
    batch_size: int,
) -> Pipeline:
    """This factory function generates a permutation algorithm per our needs.

    Parameters
    ----------
    col_values : Tuple
    supercluster_idxs : Tuple
    subclusters : Tuple
    exact : bool
    """
    permutation_pipeline = Pipeline()
    if exact is True:
        permutation_pipeline.add_component(msp)
    else:
        permutation_pipeline.add_component(
            (
                permutation_resampler,
                {
                    "n_resamples": n_resamples,
                    "stratification": supercluster_idxs,
                    "batch_size": batch_size,
                },
            )
        )

    if not np.all(subclusters == 1):
        permutation_pipeline.add_component(
            (
                _repeat_array,
                {"counts": np.array(subclusters)},
            )
        )
    return permutation_pipeline


@guvectorize(
    ["(int64[:], int64[:], int64[:])"], "(n),(n),(m)", target="cpu", cache=True
)
def efron_bootstrap(clusters, weights, res):
    """Vectorized Efron bootstrap resampling.

    Note: because the length of res cannot be inferred from the lengths of
    clusters and weights, it has to be preallocated and given as an argument
    to this function.

    Parameters
    ----------
    clusters : np.ndarray
        Number of records in each cluster on this level.
    weights : np.ndarray
        Pre-existing weights.
    res : np.ndarray
        Array to put resampled weights in.
    """
    spot = 0
    for cluster, weight in zip(clusters, weights):
        res[spot : spot + cluster] = np.random.multinomial(
            cluster * weight, np.full(cluster, 1 / cluster)
        )
        spot += cluster


@guvectorize(
    ["(int64[:], float64[:], float64[:])"], "(n),(n),(m)", target="cpu", cache=True
)
def bayesian_bootstrap(clusters: np.ndarray, weights: np.ndarray, res: np.ndarray):
    """Vectorized bayesian bootstrap resampling.

    Note: because the length of res cannot be inferred from the lengths of
    clusters and weights, it has to be preallocated and given as an argument
    to this function.

    Parameters
    ----------
    clusters : np.ndarray
        Number of records in each cluster on this level.
    weights : np.ndarray
        Pre-existing weights.
    res : np.ndarray
        Array to put resampled weights in.
    """

    spot = 0
    for cluster, weight in zip(clusters, weights):
        res[spot : spot + cluster] = (
            np.random.dirichlet(np.ones(cluster)) * weight * cluster
        )
        spot += cluster


@guvectorize(["(int64[:], int64[:])"], "(n)->(n)", cache=True)
def _weights_to_index(weights, indexes):
    """Helper function to convert multinomial bootstrap weights back
    to indexes if the traditional bootstrap sample presentation is
    desired."""
    spot = 0
    for i, v in enumerate(weights):
        for j in range(v):
            indexes[spot] = i
            spot += 1


@lru_cache
def _bootstrapper_factory(kind: str) -> Callable:
    """Factory function that returns the appropriate transform()."""

    def _resample_weights(resampling_plan, batch_size):
        # at the start, all samples are weighted equally
        weights = np.ones(
            (batch_size, resampling_plan[0][0].shape[0]), dtype=_weight_dtype
        )
        for clusters, to_resample in resampling_plan:
            if not to_resample:
                # expand the old weights to fit into the column
                weights = np.repeat(weights, clusters, axis=1)
            else:
                out = np.zeros((batch_size, clusters.sum()), dtype=_weight_dtype)
                _bootstrap_gufunc(clusters, weights, out)
                weights = out
        return weights

    _KIND_DISPATCHER = {
        "weights": (np.int64, efron_bootstrap),
        "indexes": (np.int64, efron_bootstrap),
        "bayesian": (np.float64, bayesian_bootstrap),
    }
    _weight_dtype, _bootstrap_gufunc = _KIND_DISPATCHER[kind]

    if kind == "indexes":

        def _bootstrapper_impl(resampling_plan, n_resamples, batch_size):
            repeats, extra_batch = divmod(n_resamples, batch_size)
            for i in range(repeats):
                yield from _weights_to_index(
                    _resample_weights(resampling_plan, batch_size)
                )
            yield from _weights_to_index(
                _resample_weights(resampling_plan, extra_batch)
            )

    else:

        def _bootstrapper_impl(resampling_plan, n_resamples, batch_size):
            repeats, extra_batch = divmod(n_resamples, batch_size)
            for i in range(repeats):
                yield from _resample_weights(resampling_plan, batch_size)
            yield from _resample_weights(resampling_plan, extra_batch)

    return _bootstrapper_impl


def _validate(*arrs):
    """Helper function to validate that incoming arrays are numeric numpy arrays.

    Raises
    ------
    ValueError
        Raised if any array contains non-numeric data.
    AttributeError
        Raised if any argument is not a numpy array.
    """
    for arr in arrs:

        try:
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError(
                    "Bootstrapper can only handle numeric datatypes. Please pre-process your data."
                )
        except AttributeError:
            raise AttributeError(
                "Bootstrapper can only handle numpy arrays. Please pre-process your data."
            )
    if len(arrs) == 1:
        (X,) = arrs
        y = None
    elif len(arrs) == 2:
        X, y = arrs
    else:
        raise ValueError(f"arrays can be one or two arrays, got {len(arrs)}")

    return X, y


@guvectorize(["(int64[:], int64[:], int64[:])"], "(n),(m)->(n)", cache=True)
def stratified_permuted(arr: np.ndarray, stratification: np.ndarray, res: np.ndarray):
    """Vectorized stratified permutation. Similar to np.random.Generator.permuted,
    but makes a copy.

    Parameters
    ----------
    arr : np.ndarray
        Array to be permuted
    stratification : np.ndarray
        Permutation will occur between indexes. Must contain 0 and n for
        arr of length n.
    res : np.ndarray
        Permuted copy of arr
    """
    res[:] = arr[:]
    for low, high in zip(stratification[:-1], stratification[1:]):
        for i in range(low, high - 1):
            j = bounded_uint(high - i) + i
            res[i], res[j] = res[j], res[i]


def permutation_resampler(
    col_to_resample: np.ndarray,
    stratification: np.ndarray,
    n_resamples: int,
    batch_size: int,
) -> Iterator[np.ndarray]:
    """Wraps stratified_permuted to yield permutations one by one.

    Parameters
    ----------
    col_to_resample : np.ndarray
    stratification : np.ndarray
        Indexes that shuffling should occur between. Must
        contain 0 and len(col_to_resample)
    n_resamples : int
    batch_size : int
        Number of permutations to generate at once

    Yields
    ------
    Iterator[np.ndarray]
        row of permuted indexes from batch
    """
    repeats, extra = divmod(n_resamples, batch_size)

    raw_indexes = np.arange(len(col_to_resample))[None, :]
    index_batch = raw_indexes.repeat(batch_size, axis=0)

    for i in range(repeats):
        yield from np.take(
            col_to_resample, stratified_permuted(index_batch, stratification)
        )

    extra_batch = raw_indexes.repeat(extra, axis=0)
    yield from np.take(
        col_to_resample, stratified_permuted(extra_batch, stratification)
    )


def _place_permutation(
    permutation_generator: Generator[Iterable, None, None],
    target_array: np.ndarray,
    col_idx: int,
) -> np.ndarray:
    """Place permuted column in target array and yields a copy

    Parameters
    ----------
    permutation_generator : Generator[Iterable, None, None]
    target_array : np.ndarray
    col_idx : int

    Yields
    ------
    np.ndarray
    """
    for permutation in permutation_generator:
        target_array[:, col_idx] = permutation
        yield np.array(target_array)


def _repeat_array(
    array_generator: Iterator[np.ndarray], counts: np.ndarray
) -> Iterator[np.ndarray]:
    """Pipeline component that calls np.repeat(x, counts) on the output of a generator."""
    yield from (_repeat(x, counts) for x in array_generator)


def _reindex(
    index_generator: Iterator[np.ndarray], X: np.ndarray, y: np.ndarray
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Pipeline component that reindexes X and y using indexes pulled from index_generator."""
    yield from ((X[indexes], y[indexes]) for indexes in index_generator)


def _reweight(
    weight_generator: Iterator[np.ndarray], X: np.ndarray, y: np.ndarray
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Pipeline component that reweights y using weights pulled from weight_generator."""
    yield from ((X, y * weights) for weights in weight_generator)
