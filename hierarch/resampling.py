import numpy as np
from itertools import cycle
from hierarch.internal_functions import nb_reweighter, nb_reweighter_real
from hierarch.internal_functions import unique_idx_w_cache, id_cluster_counts
from hierarch.internal_functions import msp, set_numba_random_state
from hierarch.internal_functions import iter_return, randomized_return


class Bootstrapper:
    """
    Nested bootstrapping class.

    Methods
    -------

    __init__(random_state, kind="weights"):
        random_state: int or numpy.random Generator
            Seeds the Bootstrapper for reproducibility.

        kind: str = "weights" or "bayesian" or "indexes"
            Specifies the bootstrapping algorithm.

            "weights" generates a set of new integer weights for
            each datapoint.

            "bayesian" generates a set of new real weights for
            each datapoint.

            "indexes" generates a set of new indexes for the dataset.
            Mathematically, this is equivalent to demanding integer weights.

            These approaches have different outputs - "weights" and "bayesian"
            output arrays the same size of the original array, but with
            every y-value multiplied by generated weight. "indexes" will
            output an array that is not necessarily the same size as the
            original array, but the weight of each y-value is 1, so certain
            metrics are easier to compute. Assuming both algorithms generated
            the "same" sample in terms of reweights, the arrays will be
            equivalent after the groupby and aggregate step.

            There is no reindexing equivalent of "bayesian"


    fit(data, skip=[], y=-1)
        Fit the bootstrapper to data.

        skip: List of ints. Columns to skip in the bootstrap. Skip columns
        that were sampled without replacement from the prior column.

        y indicates the column containing the dependent variable. This
        variable is exposed just to make the user aware of it - many functions
        break if the dependent variable column is in the middle of the array.

    transform(data, start=0)
        Generate a bootstrapped sample from data. start indicates the last
        column in the array that should not be resampled.

    """

    def __init__(self, random_state=None, kind="weights"):
        self.random_generator = np.random.default_rng(random_state)
        # this is a bit hacky, but we use the numpy generator to seed Numba
        # this makes it both reproducible and thread-safe enough
        nb_seed = self.random_generator.integers(low=2 ** 32 - 1)
        set_numba_random_state(nb_seed)
        self.kind = kind

    def fit(self, data, skip=[], y=-1):
        self.cluster_dict = id_cluster_counts(data[:, :y])
        y %= data.shape[1]
        self.shape = y

        self.columns_to_resample = np.array([True for k in range(self.shape)])
        for key in skip:
            self.columns_to_resample[key] = False

    def transform(self, data, start=1):
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
        else:
            raise ValueError('invalid "kind" argument!')
        return resampled


class Permuter:

    """
    Cluster permutation class.

    Methods
    -------
    fit(data, col_to_permute, exact=False)
        Fit the permuter to data.

        col_to_permute indicates the column immediately to the right of the
        shuffled column - i.e. the column that indicates the treated samples.

        exact, if True, will construct a generator that will iterate
        deterministically through every possible permutation.
        Note: this functions properly only if there are no clusters
        in the column, though that might be implemented at a later date.
        It is typically not that useful.

    transform(data)
        Generates a permuted sample. If exact is True, permutations will
        always be generated in a deterministic order, otherwise
        they will be random.

    """

    def __init__(self, random_state=None):
        self.random_generator = np.random.default_rng(random_state)
        if random_state is not None:
            nb_seed = self.random_generator.integers(low=2 ** 32)
            set_numba_random_state(nb_seed)

    def fit(self, data, col_to_permute, exact=False):
        self.values, self.indexes, self.counts = np.unique(
            data[:, : col_to_permute + 1], return_index=True, return_counts=True, axis=0
        )
        self.col_to_permute = col_to_permute
        self.exact = exact

        try:
            self.keys = unique_idx_w_cache(self.values)[-2]
            self.keys = np.append(self.keys, data.shape[0])
        except IndexError:
            self.keys = unique_idx_w_cache(self.values)[-1]

        if self.exact is True:
            col_values = self.values[:, -2].copy()
            self.iterator = cycle(msp(col_values))

        else:
            self.shuffled_col_values = data[:, self.col_to_permute - 1][self.indexes]

    def transform(self, data):
        if self.exact is False:
            randomized_return(
                data,
                self.col_to_permute,
                self.shuffled_col_values,
                self.keys,
                self.counts,
            )
        else:
            next_iter = next(self.iterator)
            iter_return(data, self.col_to_permute, tuple(next_iter), self.counts)
        return data
