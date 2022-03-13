from collections import deque
from functools import lru_cache
from itertools import islice
from typing import Callable, Generator, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
from numba import jit

from .internal_functions import (
    _repeat,
    msp,
    nb_strat_shuffle,
    set_numba_random_state,
    nb_chain_from_iterable,
)
from .pipeline import Pipeline


class Bootstrapper:
    """Bootstrapper(n_resamples, kind="weights", random_state=None)

    This transformer performs a nested bootstrap on the target data.
    Undefined behavior if the target data is not lexicographically
    sorted.

    Parameters
    ----------
    n_resamples: int
        Number of resamples for Bootstrapper's resample method to generate.
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

    >>> boot = Bootstrapper(n_resamples=10, kind="weights", random_state=1)

    One potential resampling plan is generating a bootstrapped sample by resampling second-level
    units from first-level units, then resampling observations from second-level units. This is
    done by specifying start_col=1 to the resample method.

    >>> resamples = list(boot.resample(design, start_col=1))
    >>> resamples[0]
    array([3, 0, 3, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 3, 1])

    These weights can be consumed by a function that computes a weighted statistical quantity.

    >>> y = np.array([1 for row in design])
    >>> np.average(y, weights=resamples[0])
    1.0

    Users may be more familiar with bootstrap resamples that return resampled indexes rather than
    weights. Setting kind="indexes" causes the resample method to yield indexes. Note that the
    if the random_state is the same, "indexes" and "weights" return the same resampled dataset.

    >>> boot = Bootstrapper(n_resamples=10, kind="indexes", random_state=1)
    >>> resamples = list(boot.resample(design, start_col=1))
    >>> resamples[0]
    array([ 0,  0,  0,  2,  2,  2,  6,  7,  8, 12, 13, 14, 15, 15, 16, 16, 16,
           17])

    Bootstrapper also implements the Bayesian bootstrap, which resamples weights from a dirichlet
    distribution rather than a multinomial distribution.

    >>> boot = Bootstrapper(n_resamples=10, kind="bayesian", random_state=1)
    >>> resamples = list(boot.resample(design, start_col=1))
    >>> resamples[0]
    array([2.30576453e+00, 2.61738579e+00, 2.85140165e+00, 4.78956088e-02,
           5.69594722e-01, 4.85766407e-01, 3.58185441e-02, 7.73710834e-02,
           9.00167099e-03, 2.23018979e-01, 1.84193794e-01, 5.87413492e-01,
           7.80809178e-02, 1.49869223e+00, 6.27871188e+00, 4.41052278e-03,
           1.08636700e-01, 3.68414832e-02])

    For convenience, the resample method can be called with both X and y, causing the resampled
    weights/indexes to automatically be applied to both matrices.

    >>> resamples = list(boot.resample(design, y, start_col=1))
    >>> resampled_X, resampled_y = resamples[0]
    >>> resampled_y
    array([1.34926314, 0.59942441, 0.4360995 , 2.75019554, 0.03237446,
           3.05337544, 0.43723699, 0.05818418, 0.28384635, 0.2745058 ,
           0.20724543, 0.07763387, 0.01048208, 0.9074077 , 1.06618557,
           1.99243332, 1.10400974, 3.36009649])
    """

    #: ("weights", "indexes", "bayesian) The three possible arguments that
    # can be provided to the "kind" keyword argument.
    _BOOTSTRAP_ALGORITHMS = tuple(["weights", "indexes", "bayesian"])

    def __init__(
        self,
        n_resamples,
        *,
        kind: str = "weights",
        random_state: Union[np.random.Generator, int, None] = None,
    ) -> None:

        self._n_resamples = n_resamples

        self._random_generator = np.random.default_rng(random_state)
        # this is a bit hacky, but we use the numpy generator to seed Numba
        # this makes it both reproducible and thread-safe enough
        nb_seed = self._random_generator.integers(low=2**32 - 1)
        set_numba_random_state(nb_seed)

        if kind in self._BOOTSTRAP_ALGORITHMS:
            self._kind = kind
            self.bootstrap_sampler = _bootstrapper_factory(kind)
        else:
            raise KeyError("Invalid 'kind' argument.")

    def __repr__(self):
        return (
            f"<Bootstrapper(n_resamples={self._n_resamples}, kind={self._kind}, "
            f"random_state={self._random_generator}>"
        )

    def resample(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        start_col: int = 0,
        skip: Optional[List[int]] = None,
    ) -> Generator[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], None, None]:
        """Generate bootstrapped samples from design matrix X.

        Parameters
        ----------
        X : np.ndarray
            2D Design matrix
        y : np.ndarray, optional
            1D Regressand values, by default None
        start_col : int, optional
            The first column of the design matrix to bootstrap, by default 0
        skip : _type_, optional
            Columns to skip in the bootstrap. Skip columns that were sampled
            without replacement from the prior column, by default None.

        Yields
        ------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            If y is not provided, each resample will be a 1D np.ndarray containing
            resampled weights (or indexes, if kind="indexes"). If y is provided,
            each resample will be a tuple of X (or reindexed X), reweighted y
            (or reindexed y) values.

        Raises
        ------
        IndexError
            Raised if skip does not contain valid indexes for X.
        """
        _validate(X, y)

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
                (_repeat_func, {"func": self.bootstrap_sampler}),
                (_islice_wrapper, {"stop": self._n_resamples}),
            ]
        )

        if y is not None:
            if self._kind == "indexes":
                bootstrap_pipeline.add_component((_reindex, {"X": X, "y": y}))
            else:
                bootstrap_pipeline.add_component((_reweight, {"X": X, "y": y}))

        yield from bootstrap_pipeline.process(resampling_plan[start_col:])


class Permuter:

    """Class for performing cluster-aware permutation on a target column.

    Parameters
    ----------
    n_resamples: int
        Number of resamples for Permuter's resample method to generate.
        Overridden by exact, if true.
    exact : bool, optional
        If True, overrides n_resamples and instead causes resample to
        enumerate all possible permutations and iterate through them
        one by one, by default False.
        Warning: there can be a very large number of permutations for
        large datasets.
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
    array([[2., 1., 1.],
           [2., 2., 1.],
           [1., 3., 1.],
           [2., 1., 1.],
           [1., 2., 1.],
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
           [2., 2., 1.],
           [2., 1., 1.],
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
        self,
        n_resamples: int,
        *,
        exact: bool = False,
        random_state: Union[np.random.Generator, int, None] = None,
    ) -> None:

        self._n_resamples = n_resamples
        self._exact = exact

        self.random_generator = np.random.default_rng(random_state)
        if random_state is not None:
            nb_seed = self.random_generator.integers(low=2**32)
            set_numba_random_state(nb_seed)

    def __repr__(self):
        return (
            f"<Permuter(n_resamples={self._n_resamples}, exact={self._exact}, "
            f"random_state={self.random_generator}>"
        )

    def resample(
        self, X: np.ndarray, col_to_permute: int
    ) -> Generator[np.ndarray, None, None]:
        """Yield copies of X with the target column randomly shuffled.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        col_to_permute : int
            Target level to permute

        Yields
        ------
        Generator[np.ndarray, None, None]
            X with col_to_permute randomly shuffled

        Raises
        ------
        NotImplementedError
            Exact permutation when col_to_permute != 0 has not been implemented.
        """

        if col_to_permute != 0 and self._exact is True:
            raise NotImplementedError(
                "Exact permutation only available for col_to_permute = 0."
            )

        # permute values in this copy so that original array
        # is untouched
        cached_X = X.copy()

        col_values, subclusters, supercluster_idxs = permutation_design_info(
            cached_X, col_to_permute
        )

        permutation_pipeline = _shuffle_generator_factory(
            supercluster_idxs, subclusters, self._exact, self._n_resamples
        )

        permutation_pipeline.add_component(
            (_place_permutation, {"target_array": cached_X, "col_idx": col_to_permute})
        )

        yield from permutation_pipeline.process(np.array(col_values))


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

    # turn the design matrix into a tuple so lru_cache can be used
    # this presumes a 2D matrix, but all design matrices are
    hashable = tuple(map(tuple, design.tolist()))

    return _id_cluster_impl(hashable)


@lru_cache
def _id_cluster_impl(design: Tuple[Tuple]) -> Tuple[Tuple]:
    # convert tuple back to array
    design = np.array(design)

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


def permutation_design_info(
    design_matrix: Tuple[Tuple], col_to_permute: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        column to permute, subclusters, supercluster indexes
    """
    # makes array hashable, assumes its 2D
    hashable_design = tuple(map(tuple, design_matrix))

    return _permutation_design_info_impl(hashable_design, col_to_permute)


@lru_cache
def _permutation_design_info_impl(
    design: Tuple[Tuple], col_to_permute: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    design = np.array(design)

    # we will actually be permuting the unique rows in this submatrix,
    # then duplicating any rows that contain subclusters
    permutation_matrix, subclusters = np.unique(
        design[:, : col_to_permute + 2], axis=0, return_counts=True
    )
    # if the target column is nested within another level, we have
    # to stratify the fisher-yates shuffle
    _, supercluster_idxs = np.unique(
        permutation_matrix[:, :col_to_permute], axis=0, return_index=True
    )
    supercluster_idxs = np.append(supercluster_idxs, len(permutation_matrix))
    supercluster_idxs = tuple(
        (low, high) for low, high in zip(supercluster_idxs[:-1], supercluster_idxs[1:])
    )

    # need to make this immutable
    col_values = tuple(permutation_matrix[:, col_to_permute].tolist())

    return col_values, subclusters, supercluster_idxs


def _shuffle_generator_factory(
    supercluster_idxs: Tuple,
    subclusters: Tuple,
    exact: bool,
    n_resamples: int,
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
                _repeat_func,
                {
                    "func": nb_strat_shuffle,
                    "stratification": supercluster_idxs,
                },
            )
        )
        permutation_pipeline.add_component((_islice_wrapper, {"stop": n_resamples}))
    if not np.all(subclusters == 1):
        permutation_pipeline.add_component(
            (
                lambda generator, counts: (_repeat(x, counts) for x in generator),
                {"counts": subclusters},
            )
        )

    return permutation_pipeline


@lru_cache
def _bootstrapper_factory(kind: str) -> Callable:
    """Factory function that returns the appropriate transform()."""

    # these helper functions wrap the distributions so that they take the same arguments
    @jit(nopython=True)
    def _multinomial_distribution(weight, v):
        return np.random.multinomial(v * weight, np.full(v, 1 / v))

    @jit(nopython=True)
    def _dirichlet_distribution(weight, v):
        return np.random.dirichlet(np.ones(v), size=None) * weight * v

    @jit(nopython=True)
    def _resample_weights(resampling_plan):
        # at the start, all samples are weighted equally
        weights = np.array([1 for i in resampling_plan[0][0]], dtype=_weight_dtype)

        for subclusters, to_resample in resampling_plan:

            if not to_resample:
                # expand the old weights to fit into the column
                weights = np.repeat(weights, subclusters)
            else:
                # generate new weights from the distribution we're using
                weights = nb_chain_from_iterable(
                    [_dist(weights[idx], v) for idx, v in enumerate(subclusters)]
                )
        return weights

    _KIND_DISPATCHER = {
        "weights": (np.int64, _multinomial_distribution),
        "indexes": (np.int64, _multinomial_distribution),
        "bayesian": (np.float64, _dirichlet_distribution),
    }
    _weight_dtype, _dist = _KIND_DISPATCHER[kind]

    if kind == "indexes":

        @jit(nopython=True)
        def _bootstrapper_impl(resampling_plan):
            weights = _resample_weights(resampling_plan)
            return _weights_to_index(weights)

    else:

        @jit(nopython=True)
        def _bootstrapper_impl(resampling_plan):
            return _resample_weights(resampling_plan)

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
        if arr is None:
            continue
        try:
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError(
                    "Bootstrapper can only handle numeric datatypes. Please pre-process your data."
                )
        except AttributeError:
            raise AttributeError(
                "Bootstrapper can only handle numpy arrays. Please pre-process your data."
            )


def _repeat_func(first_argument, func, **kwargs):
    """Utility function that repeats a function on a set
    of arguments infinitely."""
    while True:
        yield func(first_argument, **kwargs)


def _islice_wrapper(iterable, stop):
    """Wrapper to make islice take a keyword argument."""
    return islice(iterable, stop)


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
        yield target_array.copy()


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


@jit(nopython=True)
def _weights_to_index(weights):
    """Converts a 1D array of integer weights to indices.

    Equivalent to np.array(list(range(n))).repeat(weights).

    Parameters
    ----------
    weights : array-like of ints

    Returns
    -------
    indexes: array-like of ints
    """

    indexes = np.empty(weights.sum(), dtype=np.int64)
    spot = 0
    for i, v in enumerate(weights):
        for j in range(v):
            indexes[spot] = i
            spot += 1
    return indexes
