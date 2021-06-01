import numpy as np
from hierarch import numba_overloads
import numba as nb
import pandas as pd

assert numba_overloads


@nb.jit(nopython=True, cache=True)
def set_numba_random_state(seed: int):
    """Helper function to set numba's RNG seed.

    Parameters
    ----------
    seed : int32
        Seed for Numba's internal MT PRNG.
    """
    np.random.seed(seed)


@nb.jit(nopython=True, cache=True)
def nb_data_grabber(data, col: int, treatment_labels):
    """Numba-accelerated fancy indexing. Assumes values to grab
    are in column index: -1.

    Parameters
    ----------
    data : 2D array
        Target data.
    col : int
        Index of target column.
    treatment_labels : 1D array or list
        Labels in target column to parse.

    Returns
    -------
    list of 1D arrays
        Values from col: -1 corresponding to the treatment_labels in target column.
    """

    ret_list = []

    for key in treatment_labels:

        # grab values from the data column for each label
        ret_list.append(data[:, -1][np.equal(data[:, col], key)])

    return ret_list


@nb.jit(nopython=True, cache=True)
def nb_unique(input_data, axis=0):
    """Numba-accelerated 2D np.unique(a, return_index=True, return_counts=True)

    Appears to asymptotically approach np.unique's speed
    when every row is unique, but otherwise runs faster.

    Parameters
    ----------
    input_data : 2D numeric array
    axis : int, optional
        axis along which to identify unique slices, by default 0

    Returns
    -------
    2D array
        unique rows (or columns) from the input array

    1D array of ints
        indices of unique rows (or columns) in input array

    1D array of ints
        number of instances of each unique row
    """

    # don't want to sort original data
    if axis == 1:
        data = input_data.T.copy()

    else:
        data = input_data.copy()

    # so we can remember the original indexes of each row
    orig_idx = np.array([i for i in range(data.shape[0])])

    # sort our data AND the original indexes
    for i in range(data.shape[1] - 1, -1, -1):
        sorter = data[:, i].argsort(kind="mergesort")

        # mergesort to keep associations
        data = data[sorter]
        orig_idx = orig_idx[sorter]
    # get original indexes
    idx = [0]

    if data.shape[1] > 1:
        bool_idx = ~np.all((data[:-1] == data[1:]), axis=1)
        additional_uniques = np.nonzero(bool_idx)[0] + 1

    else:
        additional_uniques = np.nonzero(~(data[:-1] == data[1:]))[0] + 1

    idx = np.append(idx, additional_uniques)
    # get counts for each unique row
    counts = np.append(idx[1:], data.shape[0])
    counts = counts - idx
    return data[idx], orig_idx[idx], counts


@nb.jit(nopython=True)
def bivar_central_moment(x, y, pow=1, ddof=1):
    """Computes the bivariate central moment.

    Default parameters compute sample covariance. Two-pass algorithm for stability.

    Parameters
    ----------
    x, y : 1D array-likes
        x and y values to be compared
    pow : int, optional
        Power to raise each sum to, by default 1
    ddof : int, optional
        Degrees of freedom correction, by default 1

    Returns
    -------
    float64
        Product central moment of x, y

    Notes
    -----
    This ddof correction is only valid for the first product central moment (covariance).
    Simply doing ddof=2 does not provide an unbiased estimator for higher order moments.
    """
    n = len(x)

    mean_x = mean_y = 0
    for i in range(n):
        mean_x += x[i]
        mean_y += y[i]
    mean_x /= n
    mean_y /= n

    sum_of_prods = 0
    for x_, y_ in zip(x, y):
        sum_of_prods += ((x_ - mean_x) ** pow) * ((y_ - mean_y) ** pow)

    moment = sum_of_prods / (n - ddof)

    return moment


@nb.jit(nopython=True, cache=True)
def studentized_covariance(x, y):
    """Studentized sample covariance between two variables.

    Sample covariance between two variables divided by standard error of
    sample covariance. Uses a bias-corrected approximation of standard error. 
    This computes an approximately pivotal test statistic.

    Parameters
    ----------
    x, y: numeric array-likes

    Returns
    -------
    float64
        Studentized covariance.

    Examples
    --------
    >>> x = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 
    ...               [1, 2, 3, 4, 5, 2, 3, 4, 5, 6]])
    >>> x.T
    array([[0, 1],
           [0, 2],
           [0, 3],
           [0, 4],
           [0, 5],
           [1, 2],
           [1, 3],
           [1, 4],
           [1, 5],
           [1, 6]])
    >>> studentized_covariance(x.T[:,0], x.T[:,1])
    1.0039690353154482

    This is approximately equal to the t-statistic.
    >>> import scipy.stats as stats    
    >>> a = np.array([2, 3, 4, 5, 6])
    >>> b = np.array([1, 2, 3, 4, 5])
    >>> stats.ttest_ind(a, b, equal_var=False)[0]
    1.0

    """
    n = len(x)

    # numerator is the sample covariance, or the first symmetric bivariate central moment
    numerator = bivar_central_moment(x, y, pow=1, ddof=1)

    # the denominator is the sample standard deviation of the sample covariance, aka
    # the standard error of sample covariance. the denominator has three terms.

    # first term is the second symmetric bivariate central moment. an approximate
    # bias correction of n - root(2) is applied
    denom_1 = bivar_central_moment(x, y, pow=2, ddof=2 ** 0.5)

    # second term is the product of the standard deviations of x and y over n - 1.
    # this term rapidly goes to 0 as n goes to infinity
    denom_2 = (
        bivar_central_moment(x, x, pow=1, ddof=1)
        * bivar_central_moment(y, y, pow=1, ddof=1)
    ) / (n - 1)

    # third term is the square of the covariance of x and y. an approximate bias
    # correction of n - root(3) is applied
    denom_3 = ((n - 2) * (bivar_central_moment(x, y, pow=1, ddof=1.75) ** 2)) / (n - 1)

    t = (numerator) / ((1 / (n - 1.5)) * (denom_1 + denom_2 - denom_3)) ** 0.5
    return t


@nb.jit(nopython=True, cache=True)
def welch_statistic(data, col: int, treatment_labels):
    """Calculates Welch's t statistic.

    Takes a 2D data matrix, a column to classify data by, and the labels
    corresponding to the data of interest. Assumes that the largest (-1)
    column in the data matrix is the dependent variable.

    Parameters
    ----------
    data : 2D array
        Data matrix. Assumes last column contains dependent variable values.
    col : int
        Target column to be used to divide the dependent variable into two groups.
    treatment_labels : 1D array-like
        Labels in target column to be used.

    Returns
    -------
    float64
        Welch's t statistic.

    Examples
    --------

    >>> x = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 
    ...               [1, 2, 3, 4, 5, 10, 11, 12, 13, 14]])
    >>> x.T
    array([[ 0,  1],
           [ 0,  2],
           [ 0,  3],
           [ 0,  4],
           [ 0,  5],
           [ 1, 10],
           [ 1, 11],
           [ 1, 12],
           [ 1, 13],
           [ 1, 14]])
    >>> welch_statistic(x.T, 0, (0, 1))
    -9.0

    This uses the same calculation as scipy's ttest function.
    >>> import scipy.stats as stats
    >>> a = [1, 2, 3, 4, 5]
    >>> b = [10, 11, 12, 13, 14]
    >>> stats.ttest_ind(a, b, equal_var=False)[0]
    -9.0
    
    
    Notes
    ----------
    Details on the validity of this test statistic can be found in
    "Studentized permutation tests for non-i.i.d. hypotheses and the
    generalized Behrens-Fisher problem" by Arnold Janssen.
    https://doi.org/10.1016/S0167-7152(97)00043-6.

    """
    # get our two samples from the data matrix
    sample_a, sample_b = nb_data_grabber(data, col, treatment_labels)
    len_a, len_b = len(sample_a), len(sample_b)

    # mean difference
    meandiff = np.mean(sample_a) - np.mean(sample_b)

    # weighted sample variances
    var_weight_one = bivar_central_moment(sample_a, sample_a, ddof=1) / len_a
    var_weight_two = bivar_central_moment(sample_b, sample_b, ddof=1) / len_b

    # compute t statistic
    t = meandiff / np.sqrt(var_weight_one + var_weight_two)

    return t


@nb.jit(nopython=True, cache=True)
def iter_return(data, col_to_permute, iterator):
    data[:, col_to_permute] = iterator
    return data


@nb.jit(nopython=True, cache=True)
def rep_iter_return(data, col_to_permute: int, iterator, counts):
    """In-place shuffles a column based on an input. Cannot be cluster-aware.

    Parameters
    ----------
    data : 2D numeric array
        Target data matrix.
    col_to_permute : int
        Index of the column whose values will be permuted.
    iterator : 1D array-like
        Values to replace target column with.
    counts : 1D array-like
        Number of times each value in iterator should appear in output.
    """

    # the shuffled column is defined by an input variable

    shuffled_col_values = np.repeat(np.array(iterator), counts)
    data[:, col_to_permute] = shuffled_col_values
    return data


@nb.jit(nopython=True, inline="always")
def bounded_uint(ub):
    rand = np.random.randint
    x = rand(low=2 ** 32)
    m = ub * x
    l = m & (2 ** 32 - 1)
    if l < ub:
        t = (2 ** 32 - ub) % ub
        while l < t:
            x = rand(low=2 ** 32)
            m = ub * x
            l = m & (2 ** 32 - 1)
    return m >> 32


@nb.jit(nopython=True, cache=True)
def nb_fast_shuffle(arr):
    i = arr.shape[0] - 1
    while i > 0:
        j = bounded_uint(i + 1)
        arr[i], arr[j] = arr[j], arr[i]
        i -= 1


@nb.jit(nopython=True, cache=True)
def nb_strat_shuffle(arr, stratification):
    for v, w in zip(stratification[:-1], stratification[1:]):
        i = w - v - 1
        while i > 0:
            j = bounded_uint(i + 1)
            arr[i + v], arr[j + v] = arr[j + v], arr[i + v]
            i -= 1


@nb.jit(nopython=True, cache=True)
def randomized_return(data, col_to_permute: int, keys):
    """Randomly shuffles a column in-place, in a cluster-aware fashion if necessary.

    Parameters
    ----------
    data : 2D numeric array
        Target data matrix.
    col_to_permute : int
        Index of the column whose values will be permuted.
    keys : 1D array-like
        Clusters to shuffle within (if col_to_permute > 0).
    """

    # if there are no clusters, just shuffle the columns

    if col_to_permute == 0:
        nb_fast_shuffle(data[:, col_to_permute])

    else:
        nb_strat_shuffle(data[:, col_to_permute], keys)

    return data


@nb.jit(nopython=True, cache=True)
def rep_randomized_return(data, col_to_permute: int, shuffled_col_values, keys, counts):
    """Randomly shuffles a column in-place, in a cluster-aware fashion if necessary.

    Parameters
    ----------
    data : 2D numeric array
        Target data matrix.
    col_to_permute : int
        Index of the column whose values will be permuted.
    shuffled_col_values : 1D array-like
        Labels in the column to be permuted.
    keys : 1D array-like
        Clusters to shuffle within (if col_to_permute > 0).
    counts : 1D array-like
        Number of times each value in shuffled_col_values should appear in output.
    """

    # if there are no clusters, just shuffle the columns

    if col_to_permute == 0:
        nb_fast_shuffle(shuffled_col_values)

    # if not, shuffle between the indices in keys
    else:
        nb_strat_shuffle(shuffled_col_values, keys)

    # check if the shuffled column needs to be repeated to fit back
    # into the original matrix

    shuffled_col_values = np.repeat(shuffled_col_values, counts)

    data[:, col_to_permute] = shuffled_col_values
    return data


@nb.jit(nopython=True, cache=True)
def id_cluster_counts(design):
    """Identifies the hierarchy in a design matrix.

    Constructs a Typed Dictionary from a tuple of arrays corresponding
    to number of values described by each cluster in a design matrix.
    This assumes that the design matrix is lexicographically sorted.

    Parameters
    ----------
    design : 2D numeric ndarray

    Returns
    -------
    TypedDict
        Each key corresponds to a column index and each value is the number
        of subclusters in each cluster in that column.
    """
    cluster_dict = {}
    to_analyze = design
    for i in range(to_analyze.shape[1] - 1, -1, -1):
        # equivalent to np.unique(to_analyze[:,:-1],
        # return_counts=True, axis=0)
        to_analyze, counts = nb_unique(to_analyze[:, :-1])[0::2]
        cluster_dict[i] = counts
    return cluster_dict


@nb.jit(nopython=True, cache=True)
def nb_reweighter(
    data, columns_to_resample, clusternum_dict, start: int, shape: int, indexes=True
):
    """Internal function for bootstrapping a design matrix with integer
    weights.

    Parameters
    ----------
    data : 2D array
        Target data to be bootstrapped.
    columns_to_resample : 1D bool array-like
        False for columns to be skipped in the resampling plan.
    clusternum_dict : TypedDict
        Hierarchy dictionary produced by id_cluster_counts
    start : int
        First column of the data matrix to resample
    shape : int
        Last column of the data matrix to resample
    indexes : bool, optional
        If True, returns a reindexed array. If False, returns
        a reweighted array, by default True.

    Returns
    -------
    2D array
        Nonparametrically bootstrapped sample from the input data array.
    """

    out = data.astype(np.float64)
    # at the start, everything is weighted equally
    weights = np.array([1 for i in clusternum_dict[start]])

    for key in range(start, shape):
        # fetch design matrix info for current column
        to_do = clusternum_dict[key]
        # preallocate the full array for new_weight
        new_weight = np.empty(to_do.sum(), np.int64)
        place = 0

        # if not resampling this column, new_weight is all 1
        if not columns_to_resample[key]:
            for idx, v in enumerate(to_do):
                num = np.array([1 for m in range(v.item())])
                # carry over resampled weights from previous columns
                num *= weights[idx]
                for idx_2, w in enumerate(num):
                    new_weight[place + idx_2] = w.item()
                place += v

        # else do a multinomial experiment to generate new_weight
        else:
            for idx, v in enumerate(to_do):
                num = v.item()
                # num*weights[idx] carries over weights from previous columns
                randos = np.random.multinomial(num * weights[idx], [1 / num] * num)
                for idx_2, w in enumerate(randos):
                    new_weight[place + idx_2] = w.item()
                place += v

        weights = new_weight

    if indexes is False:
        out[:, -1] = out[:, -1] * weights
        return out
    else:
        indexes = weights_to_index(weights)
        return out[indexes]


@nb.jit(nopython=True, cache=True)
def nb_reweighter_real(data, columns_to_resample, clusternum_dict, start, shape):
    """Internal function for bootstrapping a design matrix with real number
    weights.

    Parameters
    ----------
    data : 2D array
        Target data to be bootstrapped.
    columns_to_resample : 1D bool array-like
        False for columns to be skipped in the resampling plan.
    clusternum_dict : TypedDict
        Hierarchy dictionary produced by id_cluster_counts
    start : int
        First column of the data matrix to resample
    shape : int
        Last column of the data matrix to resample

    Returns
    -------
    2D array
        Nonparametrically bootstrapped sample from the input data array.
    """

    out = data.astype(np.float64)
    # at the start, everything is weighted equally
    # dype is float64 because weights can be any real number
    weights = np.array([1 for i in clusternum_dict[start]], dtype=np.float64)

    for key in range(start, shape):
        # fetch design matrix info for current column
        to_do = clusternum_dict[key]
        # preallocate the full array for new_weight
        new_weight = np.empty(to_do.sum(), np.float64)
        place = 0

        # if not resampling this column, new_weight is all 1
        if not columns_to_resample[key]:
            for idx, v in enumerate(to_do):
                num = np.array([1 for m in range(v.item())], dtype=np.float64)
                num *= weights[idx]
                for idx_2, w in enumerate(num):
                    new_weight[place + idx_2] = w.item()
                place += v

        # else do a dirichlet experiment to generate new_weight
        else:
            for idx, v in enumerate(to_do):
                num = [1 for a in range(v.item())]
                # multiplying by weights[idx] carries over prior columns
                randos = np.random.dirichlet(num, size=None) * weights[idx] * v.item()
                for idx_2, w in enumerate(randos):
                    new_weight[place + idx_2] = w.item()
                place += v

        weights = new_weight

    out[:, -1] = out[:, -1] * weights
    return out


@nb.jit(nopython=True, cache=True)
def weights_to_index(weights):
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


def msp(items):
    """Yield the permutations of `items`

    items is either a list of integers representing the actual items or a list of hashable items.
    The output are the unique permutations of the items.

    Parameters
    ----------
    items : sequence

    Yields
    -------
    list
        permutation of items


    Notes
    -----
    Reference: "An O(1) Time Algorithm for Generating Multiset Permutations",
    Tadao Takaoka.
    https://pdfs.semanticscholar.org/83b2/6f222e8648a7a0599309a40af21837a0264b.pdf

    Taken from @smichr
    """
    E = list(reversed(sorted([i for i in items])))

    def visit(head):
        (rv, j) = ([], head)
        for i in range(N):
            (dat, j) = E[j]
            rv.append(dat)
        return rv

    N = len(E)
    # put E into linked-list format
    (val, nxt) = (0, 1)
    for i in range(N):
        E[i] = [E[i], i + 1]
    E[-1][nxt] = None
    head = 0
    afteri = N - 1
    i = afteri - 1
    yield visit(head)

    while E[afteri][nxt] is not None or E[afteri][val] < E[head][val]:
        j = E[afteri][nxt]  # added to algorithm for clarity
        if j is not None and E[i][val] >= E[j][val]:
            beforek = afteri
        else:
            beforek = i
        k = E[beforek][nxt]
        E[beforek][nxt] = E[k][nxt]
        E[k][nxt] = head
        if E[k][val] < E[head][val]:
            i = k
        afteri = E[i][nxt]
        head = k
        yield visit(head)


def preprocess_data(data):
    """Performs label encoding without overwriting numerical variables.

    Parameters
    ----------
    data : 2D array or pandas DataFrame
        Data to be encoded.

    Returns
    -------
    2D array of float64s
        An array identical to data, but all elements that cannot be cast
        to np.float64s replaced with integer values.
    """
    # don't want to overwrite data
    if isinstance(data, np.ndarray):

        encoded = data.copy()

    # coerce dataframe to numpy array
    elif isinstance(data, pd.DataFrame):

        encoded = data.to_numpy()

    for idx, v in enumerate(encoded.T):
        # attempt to cast array as floats
        try:
            encoded = encoded.astype(np.float64)
            # if we can cast the array as floats, encoding is complete
            break

        except ValueError:
            # if we can't, attempt to cast one column as floats
            try:
                encoded[:, idx] = encoded[:, idx].astype(np.float64)
            # if we still can't, encode that column
            except ValueError:
                encoded[:, idx] = np.unique(v, return_inverse=True)[1]
    # stable sort sort the output by row
    encoded = np.unique(encoded, axis=0)

    return encoded


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
        """Combines fit() and transform() for convenience. See those methods for details.
        """
        if reference_data is None:
            reference_data = target
        self.fit(reference_data)
        return self.transform(target, iterations=iterations)


@nb.jit(nopython=True, cache=True)
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
