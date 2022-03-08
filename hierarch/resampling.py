from collections import deque
from functools import lru_cache
from itertools import repeat
from typing import Callable, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
from numba import jit, types
from numba.extending import overload, register_jitable

from .internal_functions import (
    _repeat,
    msp,
    nb_strat_shuffle,
    set_numba_random_state,
)
from .pipeline import Pipeline


def _id_cluster_counts(design: np.ndarray) -> Tuple[np.ndarray]:
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
    ((5,), (1, 1, 1, 1, 1))

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
    ((2,), (3, 2), (1, 1, 1, 1, 2))

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


class Bootstrapper:
    """Bootstrapper(n_resamples, random_state=None, kind="weights")

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
    array([[1., 1., 1., 3.],
           [1., 1., 2., 0.],
           [1., 1., 3., 3.],
           [1., 2., 1., 0.],
           [1., 2., 2., 0.],
           [1., 2., 3., 0.],
           [1., 3., 1., 1.],
           [1., 3., 2., 1.],
           [1., 3., 3., 1.],
           [2., 1., 1., 0.],
           [2., 1., 2., 0.],
           [2., 1., 3., 0.],
           [2., 2., 1., 1.],
           [2., 2., 2., 1.],
           [2., 2., 3., 1.],
           [2., 3., 1., 2.],
           [2., 3., 2., 3.],
           [2., 3., 3., 1.]])

    Starting at column 2 means that every column 1 cluster has equal weight.

    >>> boot = Bootstrapper(random_state=1, kind="weights")
    >>> boot.fit(data, skip=None)
    >>> boot.transform(data, start=2)
    array([[1., 1., 1., 2.],
           [1., 1., 2., 0.],
           [1., 1., 3., 1.],
           [1., 2., 1., 0.],
           [1., 2., 2., 1.],
           [1., 2., 3., 2.],
           [1., 3., 1., 2.],
           [1., 3., 2., 0.],
           [1., 3., 3., 1.],
           [2., 1., 1., 1.],
           [2., 1., 2., 1.],
           [2., 1., 3., 1.],
           [2., 2., 1., 1.],
           [2., 2., 2., 0.],
           [2., 2., 3., 2.],
           [2., 3., 1., 1.],
           [2., 3., 2., 1.],
           [2., 3., 3., 1.]])

    Skipping column 2 results in only column 1 clusters being resampled.

    >>> boot = Bootstrapper(random_state=1, kind="weights")
    >>> boot.fit(data, skip=[2])
    >>> boot.transform(data, start=1)
    array([[1., 1., 1., 2.],
           [1., 1., 2., 2.],
           [1., 1., 3., 2.],
           [1., 2., 1., 0.],
           [1., 2., 2., 0.],
           [1., 2., 3., 0.],
           [1., 3., 1., 1.],
           [1., 3., 2., 1.],
           [1., 3., 3., 1.],
           [2., 1., 1., 0.],
           [2., 1., 2., 0.],
           [2., 1., 3., 0.],
           [2., 2., 1., 1.],
           [2., 2., 2., 1.],
           [2., 2., 3., 1.],
           [2., 3., 1., 2.],
           [2., 3., 2., 2.],
           [2., 3., 3., 2.]])

    Changing the algorithm to "indexes" gives a more familiar result.

    >>> boot = Bootstrapper(random_state=1, kind="indexes")
    >>> boot.fit(data, skip=None)
    >>> boot.transform(data, start=1)
    array([[1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 3., 1.],
           [1., 1., 3., 1.],
           [1., 1., 3., 1.],
           [1., 3., 1., 1.],
           [1., 3., 2., 1.],
           [1., 3., 3., 1.],
           [2., 2., 1., 1.],
           [2., 2., 2., 1.],
           [2., 2., 3., 1.],
           [2., 3., 1., 1.],
           [2., 3., 1., 1.],
           [2., 3., 2., 1.],
           [2., 3., 2., 1.],
           [2., 3., 2., 1.],
           [2., 3., 3., 1.]])

    The Bayesian bootstrap is the same as the Efron bootstrap, but allows
    the resampled weights to take any real value up to the sum of the original
    weights in that cluster.

    >>> boot = Bootstrapper(random_state=2, kind="bayesian")
    >>> boot.fit(data, skip=None)
    >>> boot.transform(data, start=1)
    array([[1.        , 1.        , 1.        , 0.92438197],
           [1.        , 1.        , 2.        , 1.65820553],
           [1.        , 1.        , 3.        , 1.31019207],
           [1.        , 2.        , 1.        , 3.68556477],
           [1.        , 2.        , 2.        , 0.782951  ],
           [1.        , 2.        , 3.        , 0.01428243],
           [1.        , 3.        , 1.        , 0.03969449],
           [1.        , 3.        , 2.        , 0.04616013],
           [1.        , 3.        , 3.        , 0.53856761],
           [2.        , 1.        , 1.        , 4.4725425 ],
           [2.        , 1.        , 2.        , 1.83458204],
           [2.        , 1.        , 3.        , 0.16269176],
           [2.        , 2.        , 1.        , 0.53223701],
           [2.        , 2.        , 2.        , 0.37478853],
           [2.        , 2.        , 3.        , 0.07456895],
           [2.        , 3.        , 1.        , 0.27616575],
           [2.        , 3.        , 2.        , 0.11271856],
           [2.        , 3.        , 3.        , 1.15970489]])

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
            for idx, cluster in enumerate(_id_cluster_counts(X))
        )

        sampler = self.bootstrap_sampler

        bootstrap_pipeline = Pipeline()
        bootstrap_pipeline.add_component(
            (
                _repeat_func,
                {"func": sampler, "start": start_col, "repetitions": self._n_resamples},
            )
        )

        if y is None:
            pass

        elif self._kind == "indexes":
            bootstrap_pipeline.add_component(
                lambda generator: ((X[indexes], y[indexes]) for indexes in generator)
            )
        else:
            bootstrap_pipeline.add_component(
                lambda generator: ((X, y[weights]) for weights in generator)
            )

        yield from bootstrap_pipeline.process(resampling_plan)


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


def _concatenate_1D_from_list(arr_list: np.ndarray) -> np.ndarray:
    """Helper function to do np.concatenate from a list of arguments.

    Numba doesn't support this because it doesn't really work well,
    but if we are only passing in lists that are made inside jitted
    functions, it's okay."""
    out = np.empty(shape=sum([arr.size for arr in arr_list]))
    put_idx = 0
    for arr in arr_list:
        out[put_idx : put_idx + arr.size] = arr
        put_idx += arr.size

    return out


@register_jitable
def _do_conc_impl(arr_list, dtype):
    out = np.empty(shape=sum([arr.size for arr in arr_list]), dtype=dtype)
    put_idx = 0
    for arr in arr_list:
        out[put_idx : put_idx + arr.size] = arr
        put_idx += arr.size
    return out


@overload(_concatenate_1D_from_list)
def ol_concatenate_1D_from_list(arr_list):
    if isinstance(arr_list[0].dtype, types.Integer):

        def conc_1D_impl(arr_list):
            return _do_conc_impl(arr_list, dtype=np.int64)

    else:

        def conc_1D_impl(arr_list):
            return _do_conc_impl(arr_list, dtype=np.float64)

    return conc_1D_impl


@lru_cache
def _bootstrapper_factory(kind: str) -> Callable:
    """Factory function that returns the appropriate transform()."""

    # these helper functions wrap the distributions so that they take the same arguments
    @jit(nopython=True)
    def _multinomial_distribution(weights, v):
        return np.random.multinomial(v * weights, [1 / v] * v)

    @jit(nopython=True)
    def _dirichlet_distribution(weights, v):
        return np.random.dirichlet([1] * v, size=None) * weights * v

    @jit(nopython=True)
    def _resample_weights(resampling_plan, start):
        # at the start, all samples are weighted equally
        weights = np.array([1 for i in resampling_plan[start][0]], dtype=_weight_dtype)

        for idx, (subclusters, to_resample) in enumerate(resampling_plan):
            if idx < start:
                continue
            elif not to_resample:
                # expand the old weights to fit into the column
                weights = np.repeat(weights, subclusters)
            else:
                # generate new weights from the distribution we're using
                weights = _concatenate_1D_from_list(
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
        def _bootstrapper_impl(resampling_plan, start):
            weights = _resample_weights(resampling_plan, start)
            return _weights_to_index(weights)

    else:

        @jit(nopython=True)
        def _bootstrapper_impl(resampling_plan, start):
            return _resample_weights(resampling_plan, start)

    return _bootstrapper_impl


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

        # we will actually be permuting the unique rows in this submatrix,
        # then duplicating any rows that contain subclusters
        permutation_matrix, subclusters = np.unique(
            X[:, : col_to_permute + 2], axis=0, return_counts=True
        )
        # if the target column is nested within another level, we have
        # to stratify the fisher-yates shuffle
        _, supercluster_idxs = np.unique(
            permutation_matrix[:, :col_to_permute], axis=0, return_index=True
        )
        supercluster_idxs = np.append(supercluster_idxs, len(permutation_matrix))
        supercluster_idxs = tuple(
            (low, high)
            for low, high in zip(supercluster_idxs[:-1], supercluster_idxs[1:])
        )

        col_values = permutation_matrix[:, col_to_permute]

        permutation_pipeline = _shuffle_generator_factory(
            supercluster_idxs, subclusters, self._exact, self._n_resamples
        )

        permutation_pipeline.add_component(
            (_place_permutation, {"target_array": cached_X, "col_idx": col_to_permute})
        )

        yield from permutation_pipeline.process(col_values)


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
                    "repetitions": n_resamples,
                },
            )
        )
    if not np.all(subclusters == 1):
        permutation_pipeline.add_component(
            (
                lambda generator, counts: (_repeat(x, counts) for x in generator),
                {"counts": subclusters},
            )
        )

    return permutation_pipeline


def _repeat_func(first_argument, func, repetitions=None, **kwargs):
    """Utility function that repeats a function on a set
    of arguments infinitely."""
    return (func(first_argument, **kwargs) for _ in repeat(None, times=repetitions))


def _place_permutation(
    permutation_generator: Generator[Iterable, None, None],
    target_array: np.ndarray,
    col_idx: int,
):
    """_summary_

    Parameters
    ----------
    permutation_generator : Generator[Iterable, None, None]
        _description_
    target_array : np.ndarray
        _description_
    col_idx : int
        _description_

    Yields
    ------
    _type_
        _description_
    """
    for permutation in permutation_generator:
        target_array[:, col_idx] = permutation
        yield target_array.copy()
