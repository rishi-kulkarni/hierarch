import numpy as np
import hierarch.internal_functions as internal_functions
from itertools import cycle


class Bootstrapper:
    '''
    Nested bootstrapping class.

    Methods
    -------
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

    '''
    def __init__(self, random_state=None):
        self.random_generator = np.random.default_rng(random_state)

    def fit(self, data, skip=[], y=-1):
        self.unique_idx_list = internal_functions.unique_idx_w_cache(data)
        self.cluster_dict = internal_functions.id_clusters(
                            tuple(self.unique_idx_list))
        if y < 0:
            self.shape = data.shape[1] + y
        else:
            self.shape = y - 1
        self.columns_to_resample = np.array([True for k in range(self.shape)])
        for key in skip:
            self.columns_to_resample[key] = False

    def transform(self, data, start=0):
        resampled_idx = self.unique_idx_list[start]
        randnos = self.random_generator.integers(low=2**32, size=data.size)
        resampled = internal_functions.nb_reindexer(resampled_idx, data,
                                                    self.columns_to_resample,
                                                    self.cluster_dict, randnos,
                                                    start, self.shape)
        return resampled


class Permuter:

    '''
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

    '''

    def __init__(self):
        pass

    def fit(self, data, col_to_permute, exact=False):
        self.values, self.indexes, self.counts = np.unique(
            data[:, :col_to_permute+1], return_index=True,
            return_counts=True, axis=0)
        self.col_to_permute = col_to_permute
        self.exact = exact

        try:
            self.keys = internal_functions.unique_idx_w_cache(self.values)[-2]
        except:
            self.keys = internal_functions.unique_idx_w_cache(self.values)[-1]

        if self.exact is True:
            col_values = self.values[:, -2].copy()
            self.iterator = cycle(internal_functions.msp(col_values))

        else:
            self.shuffled_col_values = (data[:, self.col_to_permute-1]
                                        [self.indexes])

    def transform(self, data):
        if self.exact is True:
            next_iter = next(self.iterator)
            return internal_functions.iter_return(data, self.col_to_permute,
                                                  tuple(next_iter),
                                                  self.counts)
        else:
            return internal_functions.randomized_return(
                data, self.col_to_permute,
                self.shuffled_col_values, self.keys, self.counts)
