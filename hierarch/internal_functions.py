import numpy as np
from numpy.random import shuffle
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

    # weighted sample variances - might be able to speed this up

    a_bessel = sample_a.size / (sample_a.size - 1)
    var_weight_one = (np.var(sample_a) * a_bessel) / len_a

    b_bessel = sample_b.size / (sample_b.size - 1)
    var_weight_two = (np.var(sample_b) * b_bessel) / len_b

    # compute t statistic

    t = meandiff / np.sqrt(var_weight_one + var_weight_two)

    return t


@nb.jit(nopython=True, cache=True)
def iter_return(data, col_to_permute: int, iterator, counts):
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


@nb.jit(nopython=True, cache=True)
def randomized_return(data, col_to_permute: int, shuffled_col_values, keys, counts):
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

    if shuffled_col_values.size == data.shape[0]:

        if col_to_permute == 0:
            shuffle(data[:, col_to_permute])

        else:
            for idx in range(len(keys) - 1):

                shuffle(data[:, col_to_permute][keys[idx] : keys[idx + 1]])

    else:
        if col_to_permute == 0:
            shuffle(shuffled_col_values)

        # if not, shuffle between the indices in keys
        else:

            for idx in range(len(keys) - 1):

                shuffle(shuffled_col_values[keys[idx] : keys[idx + 1]])

        # check if the shuffled column needs to be repeated to fit back
        # into the original matrix

        shuffled_col_values = np.repeat(shuffled_col_values, counts)

        data[:, col_to_permute] = shuffled_col_values


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


def mean_agg(data, ref="None", groupby=-3):

    """
    Performs a "groupby" aggregation by taking the mean. Can only be used for
    quantities that can be calculated element-wise (such as mean.)
    Potentially workable with Numba guvectorized ufuncs, too.


    Parameters
    ----------

    data: array

        Data to perform operation on. Row to aggregate to must be sorted.


    groupby = int

        Column to groupby. The default of -3 assumes the last column is values
        and the second-to-last column is some kind of categorical
        label (technical replicate 1, 2, etc.)


    Returns
    ----------


    data_agg: array

        A reduced array such that the labels in column groupby
        (now column index -2) are no longer duplicated and column index -1
        contains averaged data values.


    """

    if isinstance(ref, str):

        key = hash((data[:, : groupby + 1].tobytes(), ref))

    else:

        key = hash((data[:, : groupby + 1].tobytes(), ref.tobytes()))

    try:

        unique_idx, unique_counts = mean_agg.__dict__[key]
    except KeyError:

        if isinstance(ref, str):

            unique_idx, unique_counts = nb_unique(data[:, : groupby + 1])[1:]

            mean_agg.__dict__[key] = unique_idx, unique_counts

            if len(mean_agg.__dict__.keys()) > 50:

                mean_agg.__dict__.pop(list(mean_agg.__dict__)[0])

        else:

            unique_idx = make_ufunc_list(data[:, : groupby + 1], ref)

            unique_counts = (
                np.append(unique_idx[1:], data[:, groupby + 1].size) - unique_idx
            )

            mean_agg.__dict__[key] = unique_idx, unique_counts

            if len(mean_agg.__dict__.keys()) > 50:

                mean_agg.__dict__.pop(list(mean_agg.__dict__)[0])

    avgcol = np.add.reduceat(data[:, -1], unique_idx) / unique_counts

    data_agg = data[unique_idx][:, :-1]

    data_agg[:, -1] = avgcol

    return data_agg


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

    def visit(head):

        (rv, j) = ([], head)

        for i in range(N):

            (dat, j) = E[j]

            rv.append(dat)

        return rv

    E = list(reversed(sorted([i for i in items])))

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


@nb.jit(nopython=True, cache=True)
def make_ufunc_list(target, ref):
    """Makes a list of indices to perform a ufunc.reduceat operation along.

    This is only necessary when an aggregation operation is performed
    while grouping by a column that was resampled using the "indices"
    bootstrapping algorithm.

    Parameters
    ----------
    target : 2D numeric array
        Array that will be the target of a reduceat operation.
    ref : 2D numeric array
        The array to use as a reference for building the list of indices.

    Returns
    -------
    1D array of ints
        Indices to reduceat along.
    """

    reference = ref

    for i in range(reference.shape[1] - target.shape[1]):

        reference, counts = nb_unique(reference[:, :-1])[0::2]

    counts = counts.astype(np.int64)

    ufunc_list = np.empty(0, dtype=np.int64)

    i = 0

    while i < target.shape[0]:

        ufunc_list = np.append(ufunc_list, i)

        i += counts[np.all((reference == target[i]), axis=1)][0]

    return ufunc_list


def preprocess_data(data):

    """

    Performs label encoding without overwriting numerical variables.


    Parameters
    ----------

    data: 2D array or pandas DataFrame

        Data to be encoded.


    Returns
    ----------

    encoded: 2D array of float64s

        The array underlying data, but all elements that cannot be cast
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
    def __init__(self):
        pass

    def fit(self, reference_data):

        self.reference_dict = {}

        self.cache_dict = {}

        reference = reference_data[:, :-1]

        for i in reversed(range(1, reference.shape[1])):

            reference, counts = nb_unique(reference[:, :-1])[0::2]

            self.reference_dict[i] = (reference, counts.astype(np.int64))

    def transform(self, target, iterations=1):

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


@nb.jit(nopython=True, cache=True)
def class_make_ufunc_list(target, reference, counts):

    """

    Makes a list of indices to perform a ufunc.reduceat operation along. This
    is necessary when an aggregation operation is performed
    while grouping by a column that was resampled.


    Parameters
    ----------


    target: 2D array, float64

        Array that the groupby-aggregate operation will be performed on.


    ref: 2D array, float64

        Array that the target array was resampled from.


    Output
    ----------


    ufunc_list: 1D array of ints

        Indices to reduceat along.


    """

    ufunc_list = np.empty(0, dtype=np.int64)

    i = 0

    if reference.shape[1] > 1:

        while i < target.shape[0]:

            ufunc_list = np.append(ufunc_list, i)

            i += counts[np.all((reference == target[i]), axis=1)][0]

    else:

        while i < target.shape[0]:

            ufunc_list = np.append(ufunc_list, i)

            i += counts[(reference == target[i]).flatten()].item()

    return ufunc_list
