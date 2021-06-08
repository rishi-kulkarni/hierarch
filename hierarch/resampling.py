import numpy as np
from itertools import cycle
from numba import jit
from functools import lru_cache
from hierarch.internal_functions import (
    nb_reweighter,
    nb_reweighter_real,
    nb_unique,
    id_cluster_counts,
    msp,
    set_numba_random_state,
    _repeat,
    nb_fast_shuffle,
    nb_strat_shuffle,
)


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

    def __init__(self, random_state=None, kind="weights"):

        self.random_generator = np.random.default_rng(random_state)
        # this is a bit hacky, but we use the numpy generator to seed Numba
        # this makes it both reproducible and thread-safe enough
        nb_seed = self.random_generator.integers(low=2 ** 32 - 1)
        set_numba_random_state(nb_seed)
        if kind in self._BOOTSTRAP_ALGORITHMS:
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
        values, indexes, counts = np.unique(
            data[:, : col_to_permute + 2], return_index=True, return_counts=True, axis=0
        )

        if col_to_permute != 0 and exact is True:
            raise NotImplementedError(
                "Exact permutation only available for col_to_permute = 0."
            )

        # transform() is going to be called a lot, so generate a specialized version on the fly
        # this keeps us from having to do unnecessary flow control

        if exact is True:
            col_values = values[:, -2].copy()
            self.iterator = cycle(msp(col_values))
            if len(col_values) == len(data):
                self.transform = self._exact_return(col_to_permute, self.iterator)
            else:
                self.transform = self._exact_repeat_return(
                    col_to_permute, self.iterator, counts
                )

        else:
            try:
                values[:, -3]
                keys = nb_unique(values[:, :-2])[1]
                keys = np.append(keys, values[:, -3].shape[0])
            except IndexError:
                keys = np.zeros(1, dtype=np.int64)
                keys = np.append(keys, values[:, -2].shape[0])
            keys = tuple(keys.tolist())

            if indexes.size == len(data):
                self.transform = self._random_return(col_to_permute, keys)

            else:
                col_values = data[:, col_to_permute][indexes]
                col_values = tuple(col_values.tolist())
                counts = tuple(counts.tolist())
                self.transform = self._random_repeat_return(
                    col_to_permute, col_values, keys, counts
                )

    def transform(self, data):
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

        # this method is defined on the fly in fit() based one of the
        # four static methods defined below
        raise Exception("Use fit() before calling transform.")

    @staticmethod
    def _exact_return(col_to_permute, generator):
        """Transformer when exact is True and permutations are unrestricted.
        """

        def _exact_return_impl(data):
            data[:, col_to_permute] = next(generator)
            return data

        return _exact_return_impl

    @staticmethod
    def _exact_repeat_return(col_to_permute, generator, counts):
        """Transformer when exact is True and permutations are restricted by
        repetition of treated entities.
        """

        def _rep_iter_return_impl(data):
            data[:, col_to_permute] = _repeat(tuple(next(generator)), counts)
            return data

        return _rep_iter_return_impl

    @staticmethod
    @lru_cache()
    def _random_return(col_to_permute, keys):
        """Transformer when exact is False and repetition is not required.
        """

        if col_to_permute == 0:

            @jit(nopython=True)
            def _random_return_impl(data):
                nb_fast_shuffle(data[:, col_to_permute])
                return data

        else:

            @jit(nopython=True)
            def _random_return_impl(data):
                nb_strat_shuffle(data[:, col_to_permute], keys)
                return data

        return _random_return_impl

    @staticmethod
    @lru_cache()
    def _random_repeat_return(col_to_permute, col_values, keys, counts):
        """Transformer when exact is False and repetition is required.
        """
        col_values = np.array(col_values)
        counts = np.array(counts)
        if col_to_permute == 0:

            @jit(nopython=True)
            def _random_repeat_return_impl(data):
                shuffled_col_values = col_values.copy()
                nb_fast_shuffle(shuffled_col_values)
                data[:, col_to_permute] = np.repeat(shuffled_col_values, counts)
                return data

        else:

            @jit(nopython=True)
            def _random_repeat_return_impl(data):
                shuffled_col_values = col_values.copy()
                nb_strat_shuffle(shuffled_col_values, keys)
                data[:, col_to_permute] = np.repeat(shuffled_col_values, counts)
                return data

        return _random_repeat_return_impl

