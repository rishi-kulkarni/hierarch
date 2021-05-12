import numpy as np
from hierarch import numba_overloads
import numba as nb
import pandas as pd
assert numba_overloads


@nb.njit
def set_numba_random_state(seed):
    '''
    Helper function to set numba's RNG seed.

    Parameters
    ----------

    seed: int

    '''
    np.random.seed(seed)


@nb.jit(nopython=True, cache=True)
def nb_tuple(a, b):

    '''

    Makes a tuple from two numbers.


    Parameters
    ----------

    a, b: numbers of the same dtype (or can be cast to the same dtype).


    Returns
    ----------

    tuple(a, b)


    '''

    return tuple((a, b))


@nb.jit(nopython=True, cache=True)
def nb_data_grabber(data, col, treatment_labels):

    '''

    Numba-accelerated fancy indexing.


    Parameters
    ----------

    data: 2D array


    col: int


    treatment_labels: 1D array or list


    Returns
    ----------

    ret_list: list of 1D arrays


    '''

    ret_list = []

    for key in treatment_labels:

        # grab values from the data column for each label

        ret_list.append(data[:, -1][np.equal(data[:, col], key)])
    return ret_list


@nb.jit(nopython=True, cache=True)
def nb_unique(input_data, axis=0):

    '''

    Internal function that serves the same purpose as

    np.unique(a, return_index=True, return_counts=True)

    when called on a 2D arrays. Appears to asymptotically

    approach np.unique's speed when every row is unique,

    but otherwise runs faster.


    Parameters
    ----------

    input_data: 2D array


    axis: int

        0 for unique rows, 1 for unique columns


    Returns
    ----------

    data[idx]: unique rows (or columns) from the input array


    idx: index numbers of the unique rows (or columns)


    counts: number of instances of each unique row (or column)

    in the input array


    '''
    # don't want to sort original data

    if axis == 1:

        data = input_data.T.copy()

    else:

        data = input_data.copy()

    # so we can remember the original indexes of each row
    orig_idx = np.array([i for i in range(data.shape[0])])

    # sort our data AND the original indexes
    for i in range(data.shape[1]-1, -1, -1):

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
def welch_statistic(data, col, treatment_labels):

    '''

    Calculates Welch's t statistic.


    Takes a 2D data matrix, a column to classify data by, and the labels

    corresponding to the data of interest. Assumes that the largest (-1)

    column in the data matrix is the dependent variable.



    Parameters
    ----------

    data: 2D array

        Data matrix. The first n - 1 columns are the design matrix and the

        nth column is the dependent variable.


    col: int

        Columns of the design matrix used to classify the dependent variable

        into two groups.


    treatment_labels:

        The values of the elements in col.


    Returns
    ----------

    t: float64

        Welch's t statistic.


    Notes
    ----------

    Details on the validity of this test statistic can be found in

    "Studentized permutation tests for non-i.i.d. hypotheses and the

    generalized Behrens-Fisher problem"

    by Arnold Janssen. https://doi.org/10.1016/S0167-7152(97)00043-6.


    '''
    # get our two samples from the data matrix

    sample_a, sample_b = nb_data_grabber(data, col, treatment_labels)

    len_a, len_b = len(sample_a), len(sample_b)

    # mean difference

    meandiff = (np.mean(sample_a) - np.mean(sample_b))

    # weighted sample variances - might be able to speed this up

    a_bessel = (sample_a.size/(sample_a.size - 1))
    var_weight_one = (np.var(sample_a)*a_bessel) / len_a

    b_bessel = (sample_b.size/(sample_b.size - 1))
    var_weight_two = (np.var(sample_b)*b_bessel) / len_b

    # compute t statistic

    t = meandiff / np.sqrt(var_weight_one + var_weight_two)

    return t


@nb.jit(nopython=True, cache=True)
def iter_return(data, col_to_permute, iterator, counts):

    '''

    Shuffles a column based on an input. Cannot be cluster-aware.

    Parameters
    ----------

    data: 2D array of floats

        Original data matrix

    col_to_permute: int

        Column n, which is immediately right of the column that will be
        shuffled.


    iterator: 1D array

        Shuffled values to put into data.


    counts: 1D array of ints

        Number of times each value in iterator needs to be
        repeated (typically 1)


    Returns
    ----------


    permute: 2D array

        Array the same shape as data, but with column n-1 resampled
        with replacement.


    '''
    # the shuffled column is defined by an input variable
    shuffled_col_values = np.array(iterator)

    # check if the shuffled column needs to be duplicated to fit back into
    # the original matrix
    if len(shuffled_col_values) < data.shape[0]:

        shuffled_col_values = np.repeat(shuffled_col_values, counts)

    permute = data.copy()

    permute[:, col_to_permute-1] = shuffled_col_values
    return permute


@nb.jit(nopython=True, cache=True)
def randomized_return(data, col_to_permute, shuffled_col_values, keys, counts):

    '''

    Randomly shuffles a column in a cluster-aware fashion if necessary.


    Parameters
    ----------


    data: 2D array of floats

        Original data matrix


    col_to_permute: int

        Column n, which is immediately right of the column
        that will be shuffled.


    shuffled_col_values: 1D array

        Values in the column to be shuffled


    keys: 1D array of ints

        Indexes of the clusters in the shuffled column. If there is no
        clustering in the column to be shuffled, this still needs to
        be a 1D array to get Numba to behave properly.


    counts: 1D array of ints

        Number of times each value in shuffled_col_values needs to be
        repeated (typically 1)


    Returns
    ----------


    permute: 2D array

        Array the same shape as data, but with column n-1 resampled
        with replacement.


    '''

    # if there are no clusters, just shuffle the columns

    if col_to_permute == 1:

        np.random.shuffle(shuffled_col_values)

    # if not, shuffle between the indices in keys
    else:

        for idx, _ in enumerate(keys):

            if idx < len(keys)-1:

                np.random.shuffle(shuffled_col_values[keys[idx]:keys[idx+1]-1])

            else:

                np.random.shuffle(shuffled_col_values[keys[idx]:])

    # check if the shuffled column needs to be duplicated to fit back
    # into the original matrix
    if len(shuffled_col_values) < data.shape[0]:

        shuffled_col_values = np.repeat(shuffled_col_values, counts)

    permute = data.copy()
    permute[:, col_to_permute-1] = shuffled_col_values

    return permute


@nb.jit(nopython=True, cache=True)
def nb_reindexer(resampled_idx, data, columns_to_resample, cluster_dict,
                 randnos, start, shape):

    '''

    Internal function for shuffling a design matrix.


    Parameters
    ----------


    resampled_idx: 1D array of ints

        Indices of the unique rows of the design matrix up until the
        first column that needs shuffling.


    data: 2D array

        The original data


    columns_to_resample: 1D bool array

        False for columns that do not need resampling.


    cluster_dict: numba TypedDict

        This function uses the cluster_dict to quickly grab the indices in the
        column i+1 that correspond to a unique row in column i.


    randnos: 1D array of ints

        List of random numbers generated outside of numba to use for
        resampling with replacement. Generating these outside of numba
        is faster for now as numba np.random does not take the size argument.


    start: int

        First column of the data matrix to resample


    shape: int

        Last column of the data matrix to resample


    Output
    ----------


    resampled: 2D array

        Nested bootstrapped resample of the input data array



    '''

    rng_place = 0

    for i in range(start+1, shape):
        idx = np.empty(0, dtype=np.int64)

        for key in resampled_idx:
            idx_no = cluster_dict[nb_tuple(key, i)]

            if not columns_to_resample[i]:
                idx = np.hstack((idx, idx_no))
            else:
                idx_no = idx_no[randnos[rng_place:rng_place+idx_no.size]
                                % idx_no.size]
                idx_no.sort()
                rng_place += idx_no.size
                idx = np.hstack((idx, idx_no))

        resampled_idx = idx
    resampled = data[resampled_idx]
    return resampled


@nb.jit(nopython=True, cache=True)
def id_cluster_counts(unique_idx_list):
    '''
    Constructs a Typed Dictionary from a tuple of arrays corresponding
    to number of values described by each cluster in a design matrix.
    Again, this indirectly assumes that the design matrix is
    lexsorted starting from the last column.

    Parameters
    ----------
    unique_idx_list: tuple of 1D arrays (ints64)
        Tuple of arrays that identify the unique rows in
        columns 0:0, 0:1, 0:n, where n is the final column
        of the design matrix portion of the data.

    Outputs
    ----------
    cluster_dict: TypedDict of int64 : array (int64)
        Each int is the coordinate of the first
        index corresponding to a cluster in row m and
        each 1D array contains the indices of the nested
        clusters in row m + 1.

    '''
    cluster_dict = {}
    for i in range(1, len(unique_idx_list)):
        value = np.empty(0, dtype=np.int64)
        for j in range(unique_idx_list[i-1].size):
            if j < unique_idx_list[i-1].size-1:
                value = np.append(value, len(unique_idx_list[i]
                                  [(unique_idx_list[i] >=
                                    unique_idx_list[i-1][j]) &
                                  (unique_idx_list[i] <
                                   unique_idx_list[i-1][j+1])]))
            else:
                value = np.append(value, len(unique_idx_list[i]
                                  [(unique_idx_list[i] >=
                                    unique_idx_list[i-1][j])]))

        cluster_dict[i] = value
    return cluster_dict


@nb.njit
def nb_reweighter(data, columns_to_resample, clusternum_dict,
                  start, shape, indexes=True):
    out = data.astype(np.float64)
    weights = np.array([1 for i in clusternum_dict[start]])

    for key in range(start, shape):
        new_weight = np.empty(0, np.int64)
        to_do = clusternum_dict[key]

        if not columns_to_resample[key]:
            for idx, v in enumerate(to_do):
                num = np.array([1 for m in range(v.item())])
                num *= weights[idx]
                new_weight = np.append(new_weight, num)
        else:
            for idx, v in enumerate(to_do):
                num = v.item()
                randos = np.random.multinomial(num*weights[idx], [1/num]*num)
                new_weight = np.append(new_weight, randos)

        weights = new_weight

    if indexes is False:
        out[:, -1] = out[:, -1] * weights
        return out
    else:
        indexes = weights_to_index(weights)
        return out[indexes]


@nb.njit
def nb_reweighter_real(data, columns_to_resample,
                       clusternum_dict, start, shape):
    out = data.astype(np.float64)
    weights = np.array([1 for i in clusternum_dict[start]], dtype=np.float64)

    for key in range(start, shape):
        new_weight = np.empty(0, np.float64)
        to_do = clusternum_dict[key]

        if not columns_to_resample[key]:
            for idx, v in enumerate(to_do):
                num = np.array([1 for m in range(v.item())], dtype=np.float64)
                num *= weights[idx]
                new_weight = np.append(new_weight, num)

        else:
            for idx, v in enumerate(to_do):
                num = [1 for a in range(v.item())]
                randos = (np.random.dirichlet(num, size=1)
                          * weights[idx] * v.item())
                new_weight = np.append(new_weight, randos)

        weights = new_weight

    out[:, -1] = out[:, -1] * weights
    return out


@nb.njit
def weights_to_index(weights):
    indexes = np.empty(weights.sum(), dtype=np.int64)
    spot = 0
    for i, v in enumerate(weights):
        for j in range(v):
            indexes[spot] = i
            spot += 1
    return indexes


def mean_agg(data, ref='None', groupby=-3):

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

        key = hash((data[:, :groupby+1].tobytes(), ref))

    else:

        key = hash((data[:, :groupby+1].tobytes(), ref.tobytes()))

    try:

        unique_idx, unique_counts = mean_agg.__dict__[key]
    except KeyError:

        if isinstance(ref, str):

            unique_idx, unique_counts = nb_unique(data[:, :groupby+1])[1:]

            mean_agg.__dict__[key] = unique_idx, unique_counts

            if len(mean_agg.__dict__.keys()) > 50:

                mean_agg.__dict__.pop(list(mean_agg.__dict__)[0])

        else:

            unique_idx = make_ufunc_list(data[:, :groupby+1], ref)

            unique_counts = np.append(unique_idx[1:],
                                      data[:, groupby+1].size) - unique_idx

            mean_agg.__dict__[key] = unique_idx, unique_counts

            if len(mean_agg.__dict__.keys()) > 50:

                mean_agg.__dict__.pop(list(mean_agg.__dict__)[0])

    avgcol = np.add.reduceat(data[:, -1], unique_idx) / unique_counts

    data_agg = data[unique_idx][:, :-1]

    data_agg[:, -1] = avgcol

    return data_agg


def msp(items):

    '''Yield the permutations of `items` where items is either a list

    of integers representing the actual items or a list of hashable items.

    The output are the unique permutations of the items given as a list

    of integers 0, ..., n-1 that represent the n unique elements in

    `items`.


    Examples

    ========


    >>> for i in msp('xoxox'):

    ...   print(i)


    [1, 1, 1, 0, 0]

    [0, 1, 1, 1, 0]

    [1, 0, 1, 1, 0]

    [1, 1, 0, 1, 0]

    [0, 1, 1, 0, 1]

    [1, 0, 1, 0, 1]

    [0, 1, 0, 1, 1]

    [0, 0, 1, 1, 1]

    [1, 0, 0, 1, 1]

    [1, 1, 0, 0, 1]


    Reference: "An O(1) Time Algorithm for Generating Multiset Permutations",
    Tadao Takaoka

    https://pdfs.semanticscholar.org/83b2/6f222e8648a7a0599309a40af21837a0264b.pdf


    Taken from @smichr

    '''

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


def unique_idx_w_cache(data):

    '''
    Just np.unique(return_index=True, axis=0) with memoization, as np.unique
    is called a LOT in this package. Numpy arrays are not hashable,
    so this hashes the bytes of the array instead.

    '''

    key = hash(data.tobytes())
    try:

        unique_lists = unique_idx_w_cache.__dict__[key]

        return unique_lists

    except KeyError:

        unique_lists = []

        for i in range(0, data.shape[1] - 1):

            unique_lists += [np.unique(data[:, :i+1],
                             return_index=True, axis=0)[1]]

        unique_idx_w_cache.__dict__[key] = unique_lists

        if len(unique_idx_w_cache.__dict__.keys()) > 50:

            unique_idx_w_cache.__dict__.pop(list(
                unique_idx_w_cache.__dict__)[0])

        return unique_lists


@nb.jit(nopython=True, cache=True)
def make_ufunc_list(target, ref):

    '''

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


    '''

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


@nb.jit(nopython=True, cache=True)
def id_clusters(unique_idx_list):

    '''

    Constructs a Typed Dictionary from a tuple of arrays corresponding to
    unique cluster positions in a design matrix. Again, this indirectly
    assumes that the design matrix is lexsorted starting from the last column.


    Parameters
    ----------

    unique_idx_list: tuple of 1D arrays (ints64)

        Tuple of arrays that identify the unique rows in columns 0:0, 0:1, 0:n,
        where n is the final column of the design matrix portion of the data.


    Outputs
    ----------

    cluster_dict: TypedDict of UniTuples (int64 x 2) : 1D array (int64)

        Each tuple is the coordinate of the first index corresponding to a
        cluster in row m and each 1D array contains the indices of the nested
        clusters in row m + 1.


    '''

    cluster_dict = {}

    for i in range(1, len(unique_idx_list)):

        for j in range(unique_idx_list[i-1].size):

            if j < unique_idx_list[i-1].size-1:

                value = unique_idx_list[i][(unique_idx_list[i] >=
                                            unique_idx_list[i-1][j]) & (
                                                unique_idx_list[i] <
                                                unique_idx_list[i-1][j+1])]

            else:

                value = unique_idx_list[i][(unique_idx_list[i] >=
                                            unique_idx_list[i-1][j])]

            cluster_dict[nb_tuple(unique_idx_list[i-1][j], i)] = value
    return cluster_dict


def preprocess_data(data):

    '''

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


    '''

    if isinstance(data, np.ndarray):

        encoded = data.copy()

    elif isinstance(data, pd.DataFrame):

        encoded = data.to_numpy()

    for idx, v in enumerate(encoded.T):

        try:

            encoded = encoded.astype(np.float64)

            break

        except ValueError:

            try:

                encoded[:, idx] = encoded[:, idx].astype(np.float64)

            except ValueError:

                encoded[:, idx] = np.unique(v, return_inverse=True)[1]

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

                reduce_at_list = class_make_ufunc_list(target[:, :-2],
                                                       reference, counts)

                reduce_at_counts = (np.append(reduce_at_list[1:],
                                              target[:, -2].size) -
                                    reduce_at_list)

                self.cache_dict[key] = reduce_at_list, reduce_at_counts

                if len(self.cache_dict.keys()) > 50:

                    self.cache_dict.pop(list(self.cache_dict)[0])

            agg_col = (np.add.reduceat(target[:, -1], reduce_at_list) /
                       reduce_at_counts)

            target = target[reduce_at_list][:, :-1]

            target[:, -1] = agg_col

        return target


@nb.jit(nopython=True, cache=True)
def class_make_ufunc_list(target, reference, counts):

    '''

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


    '''

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
