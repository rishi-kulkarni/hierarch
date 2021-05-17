import numpy as np
from itertools import cycle
from hierarch.internal_functions import (
    nb_reweighter,
    nb_reweighter_real,
    nb_unique,
    id_cluster_counts,
    msp,
    set_numba_random_state,
    iter_return,
    randomized_return,
)


BOOTSTRAP_ALGORITHMS = ["weights", "indexes", "bayesian"]


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

    """

    def __init__(self, random_state=None, kind="weights"):

        self.random_generator = np.random.default_rng(random_state)
        # this is a bit hacky, but we use the numpy generator to seed Numba
        # this makes it both reproducible and thread-safe enough
        nb_seed = self.random_generator.integers(low=2 ** 32 - 1)
        set_numba_random_state(nb_seed)
        if kind in BOOTSTRAP_ALGORITHMS:
            self.kind = kind
        else:
            raise KeyError("Invalid 'kind' argument.")

    def fit(self, data, skip=None, y=-1):
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
            print(
                "Bootstrapper can only handle numpy arrays. Please pre-process your data."
            )
            raise

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

        self.cluster_dict = id_cluster_counts(data[:, :y])
        y %= data.shape[1]
        self.shape = y

        self.columns_to_resample = np.array([True for k in range(self.shape)])
        for key in skip:
            self.columns_to_resample[key] = False

    def transform(self, data, start: int):
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
        if self.kind == "weights":
            resampled = nb_reweighter(
                data,
                self.columns_to_resample,
                self.cluster_dict,
                start,
                self.shape,
                indexes=False,
            )
        elif self.kind == "indexes":
            resampled = nb_reweighter(
                data,
                self.columns_to_resample,
                self.cluster_dict,
                start,
                self.shape,
                indexes=True,
            )

        elif self.kind == "bayesian":
            resampled = nb_reweighter_real(
                data, self.columns_to_resample, self.cluster_dict, start, self.shape
            )
        return resampled


class Permuter:

    """Class for performing cluster-aware permutation on a target column.

    Parameters
    ----------
    random_state : int or numpy.random.Generator instance, optional
        Seedable for reproducibility, by default None
    """

    def __init__(self, random_state=None):
        self.random_generator = np.random.default_rng(random_state)
        if random_state is not None:
            nb_seed = self.random_generator.integers(low=2 ** 32)
            set_numba_random_state(nb_seed)

    def fit(self, data, col_to_permute: int, exact=False):
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
        self.values, self.indexes, self.counts = np.unique(
            data[:, : col_to_permute + 2], return_index=True, return_counts=True, axis=0
        )

        if col_to_permute != 0 and exact is True:
            raise NotImplementedError(
                "Exact permutation only available for col_to_permute = 0."
            )
        self.col_to_permute = col_to_permute
        self.exact = exact

        try:
            self.values[:, -3]
            self.keys = nb_unique(self.values[:, :-2])[1]
            self.keys = np.append(self.keys, data.shape[0])
        except IndexError:
            self.keys = np.empty(0)

        if self.exact is True:
            col_values = self.values[:, -2].copy()
            self._len = len(col_values)
            self.iterator = cycle(msp(col_values))

        else:
            self.shuffled_col_values = data[:, self.col_to_permute][self.indexes]

    def transform(self, data, col_to_permute):
        """Permute target column in-place.

        Parameters
        ----------
        data : 2D numeric ndarray
            Target data.

        Returns
        -------
        data : 2D numeric ndarray
            Original data with target column shuffled, in a cluster-aware fashion if necessary.
        """
        if self.exact is False:
            randomized_return(
                data, col_to_permute, self.shuffled_col_values, self.keys, self.counts,
            )
        else:
            if self._len == len(data):
                data[:, col_to_permute] = next(self.iterator)
            else:
                iter_return(
                    data, col_to_permute, tuple(next(self.iterator)), self.counts
                )
        return data
