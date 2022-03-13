import numpy as np
from numba import jit

from .internal_functions import nb_unique


class GroupbyMean:
    """Class for performing groupby reductions on numpy arrays.

    Currently only supports mean reduction.
    """

    def __init__(self):

        self.cache_dict = {}

    def fit(self, reference_data):
        """Fits the class to reference data.

        Parameters
        ----------
        reference_data : 2D numeric numpy array
            Reference data to use for the reduction.

        """
        self.reference_dict = {}

        reference = reference_data[:, :-1]

        for i in reversed(range(1, reference.shape[1])):

            reference, counts = nb_unique(reference[:, :-1])[0::2]

            self.reference_dict[i] = (reference, counts.astype(np.int64))

    def transform(self, target, iterations=1):
        """Performs iterative groupby reductions.

        Parameters
        ----------
        target : 2D numeric array
            Array to be reduced.
        iterations : int, optional
            Number of reductions to perform, by default 1

        Returns
        -------
        2D numeric array
            Array with the same number of rows as target data, but one fewer column
            for each iteration. Final column values are combined by taking the mean.
        """
        for i in range(iterations):

            key = hash((target[:, :-2]).tobytes())

            try:

                reduce_at_list, reduce_at_counts = self.cache_dict[key]

            except KeyError:

                column = target.shape[1] - 2

                reference, counts = self.reference_dict[column]

                reduce_at_list = class_make_ufunc_list(
                    target[:, :-2], reference, counts
                )

                reduce_at_counts = (
                    np.append(reduce_at_list[1:], target[:, -2].size) - reduce_at_list
                )

                self.cache_dict[key] = reduce_at_list, reduce_at_counts

                if len(self.cache_dict.keys()) > 50:

                    self.cache_dict.pop(list(self.cache_dict)[0])

            agg_col = np.add.reduceat(target[:, -1], reduce_at_list) / reduce_at_counts

            target = target[reduce_at_list][:, :-1]

            target[:, -1] = agg_col

        return target

    def fit_transform(self, target, reference_data=None, iterations=1):
        """Combines fit() and transform() for convenience. See those methods for details."""
        if reference_data is None:
            reference_data = target
        self.fit(reference_data)
        return self.transform(target, iterations=iterations)


jit(nopython=True, cache=True)


def class_make_ufunc_list(target, reference, counts):
    """Makes a list of indices to perform a ufunc.reduceat operation along.

    This is necessary when an aggregation operation is performed
    while grouping by a column that was resampled. The target array
    must be lexsorted.

    Parameters
    ----------
    target : 2D numeric array
        Array that the groupby-aggregate operation will be performed on.
    reference : 2D numeric array
        Unique rows in target for the column that will be aggregated to.
    counts : 1D array of ints
        Number of times each row in reference should appear in target.

    Returns
    -------
    1D array of ints
        Indices to reduceat along.
    """

    ufunc_list = np.empty(len(reference), dtype=np.int64)
    i = 0

    if reference.shape[1] > 1:
        for idx, _ in enumerate(ufunc_list):
            ufunc_list[idx] = i
            i += counts[np.all((reference == target[i]), axis=1)][0]

    else:
        for idx, _ in enumerate(ufunc_list):
            ufunc_list[idx] = i
            i += counts[(reference == target[i]).flatten()].item()

    return ufunc_list
