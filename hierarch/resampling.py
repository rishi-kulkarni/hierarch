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
    array([[1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 2.62123419e-01],
           [1.00000000e+00, 1.00000000e+00, 2.00000000e+00, 1.17482060e-01],
           [1.00000000e+00, 1.00000000e+00, 3.00000000e+00, 1.69091395e-01],
           [1.00000000e+00, 2.00000000e+00, 1.00000000e+00, 1.84693092e-03],
           [1.00000000e+00, 2.00000000e+00, 2.00000000e+00, 4.49696127e-02],
           [1.00000000e+00, 2.00000000e+00, 3.00000000e+00, 3.14625030e-01],
           [1.00000000e+00, 3.00000000e+00, 1.00000000e+00, 3.83263684e+00],
           [1.00000000e+00, 3.00000000e+00, 2.00000000e+00, 3.61940358e+00],
           [1.00000000e+00, 3.00000000e+00, 3.00000000e+00, 6.37821131e-01],
           [2.00000000e+00, 1.00000000e+00, 1.00000000e+00, 3.03993263e-02],
           [2.00000000e+00, 1.00000000e+00, 2.00000000e+00, 8.79647355e-02],
           [2.00000000e+00, 1.00000000e+00, 3.00000000e+00, 3.16311325e-01],
           [2.00000000e+00, 2.00000000e+00, 1.00000000e+00, 2.99913074e-01],
           [2.00000000e+00, 2.00000000e+00, 2.00000000e+00, 6.50332822e-01],
           [2.00000000e+00, 2.00000000e+00, 3.00000000e+00, 4.07850079e+00],
           [2.00000000e+00, 3.00000000e+00, 1.00000000e+00, 1.18526227e+00],
           [2.00000000e+00, 3.00000000e+00, 2.00000000e+00, 2.20584711e+00],
           [2.00000000e+00, 3.00000000e+00, 3.00000000e+00, 1.45468555e-01]])

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
           [1., 3., 1.],
           [1., 1., 1.],
           [2., 2., 1.],
           [2., 3., 1.]])

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
    array([[1., 2., 1.],
           [1., 1., 1.],
           [1., 3., 1.],
           [2., 3., 1.],
           [2., 2., 1.],
           [2., 1., 1.]])

    Exact within-cluster permutations are not implemented, but there are typically
    too many to be worth attempting.
    
    >>> permute = Permuter(random_state=2)
    >>> permute.fit(test, col_to_permute=1, exact=True)
    Traceback (most recent call last):
        ...
    NotImplementedError: Exact permutation only available for col_to_permute = 0.
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

    def transform(self, data):
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
                data,
                self.col_to_permute,
                self.shuffled_col_values,
                self.keys,
                self.counts,
            )
        else:
            if self._len == len(data):
                data[:, self.col_to_permute] = next(self.iterator)
            else:
                iter_return(
                    data, self.col_to_permute, tuple(next(self.iterator)), self.counts
                )
        return data
