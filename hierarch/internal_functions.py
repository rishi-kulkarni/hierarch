import numpy as np
from hierarch import numba_overloads
import numba as nb

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


def nb_unique(input_data, axis=0):
    """2D np.unique(a, return_index=True, return_counts=True).

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
    if axis == 1:
        input_data = input_data.T
    if input_data.shape[1] == 0:
        return (
            input_data[:1],
            np.zeros(1, dtype=np.intp),
            np.array([input_data.shape[0]]),
        )
    return np.unique(input_data, return_index=True, return_counts=True, axis=0)


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


def _repeat(target, counts):
    return np.repeat(np.array(target), counts)


@nb.jit(nopython=True, inline="always")
def bounded_uint(ub):
    """Produces an unbiased random integer within the half-open set of 0 to ub.

    Based on Daniel Lemire's implementation.

    Notes
    -----
    https://lemire.me/blog/2019/06/06/nearly-divisionless-random-integer-generation-on-various-systems/

    Parameters
    ----------
    ub : int
        The upper bound plus one.

    Returns
    -------
    int
    """
    x = np.random.randint(low=0, high=2**32)
    m = ub * x
    lower = np.uint32(m)
    if lower < ub:
        t = -np.uint32(ub)
        if t >= ub:
            t -= ub
            if t >= ub:
                t %= ub
        while lower < t:
            x = np.random.randint(low=0, high=2**32)
            m = ub * x
            lower = np.uint32(m)
    return m >> 32


@nb.jit(nopython=True, cache=True)
def nb_fast_shuffle(arr):
    """Reimplementation of Fisher-Yates shuffle using bounded_uint to generate random numbers."""
    i = arr.shape[0] - 1
    while i > 0:
        j = bounded_uint(i + 1)
        arr[i], arr[j] = arr[j], arr[i]
        i -= 1


@nb.jit(nopython=True, cache=True)
def nb_strat_shuffle(arr, stratification):
    """Stratified Fisher-Yates shuffle.

    Parameters
    ----------
    arr : 1D array-like
        Target array.
    stratification : 1D array-like
        Ranges to shuffle within. Must be sorted.
    """
    for v, w in zip(stratification[:-1], stratification[1:]):
        i = w - v - 1
        while i > 0:
            j = bounded_uint(i + 1)
            arr[i + v], arr[j + v] = arr[j + v], arr[i + v]
            i -= 1


def id_cluster_counts(design):
    """Identifies the hierarchy in a design matrix.

    Constructs a dictionary from a tuple of arrays corresponding
    to number of values described by each cluster in a design matrix.
    This assumes that the design matrix is lexicographically sorted.

    Parameters
    ----------
    design : 2D numeric ndarray

    Returns
    -------
    dict
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


def _row_positions_structured(reference, query):
    """Positions of query rows within lexsorted reference rows.

    Both arrays must be lexicographically sorted 2D arrays of the same dtype
    and width, and every query row must be present in reference. Fallback for
    when the mixed-radix row encoding would overflow int64.
    """
    ref_view = reference.view([("", reference.dtype)] * reference.shape[1]).ravel()
    query = np.ascontiguousarray(query)
    query_view = query.view([("", query.dtype)] * query.shape[1]).ravel()
    return np.searchsorted(ref_view, query_view)


class GroupbyMean:
    """Class for performing groupby reductions on numpy arrays.

    Currently only supports mean reduction. The fitted reference data
    (and any array passed to transform) must be lexicographically sorted.

    Because iterated mean-of-means over fixed groups is a linear map on the
    dependent-variable column, fit() precomputes a per-row coefficient vector
    for each reduction depth and transform() is a single weighted
    np.add.reduceat pass regardless of how many levels are aggregated.
    """

    def __init__(self):
        self._coefficient_cache = {}

    def fit(self, reference_data):
        """Fits the class to reference data.

        Parameters
        ----------
        reference_data : 2D numeric numpy array
            Reference data to use for the reduction. Must be
            lexicographically sorted.

        """
        reference = np.ascontiguousarray(reference_data[:, :-1])
        self.reference_keys = reference
        n, levels = reference.shape

        # _starts[k] holds the first-row index of each block when grouping
        # by the first k design columns
        starts = [np.zeros(1, dtype=np.intp)]
        for k in range(1, levels + 1):
            change = np.any(reference[1:, :k] != reference[:-1, :k], axis=1)
            starts.append(
                np.concatenate((np.zeros(1, dtype=np.intp), np.flatnonzero(change) + 1))
            )
        self._starts = starts
        self._coefficient_cache = {}

        # mixed-radix scalar codes for the reference rows, so resampled-row
        # lookup is an int64 searchsorted instead of a structured-dtype one
        uniques = [np.unique(reference[:, j]) for j in range(levels)]
        sizes = [len(u) for u in uniques]
        if int(np.prod([1] + sizes, dtype=object)) < 2**62:
            radix = np.ones(levels, dtype=np.int64)
            for j in range(levels - 2, -1, -1):
                radix[j] = radix[j + 1] * sizes[j + 1]
            codes = np.zeros(n, dtype=np.int64)
            for j in range(levels):
                codes += np.searchsorted(uniques[j], reference[:, j]) * radix[j]
            self._row_codec = (uniques, radix, codes)
        else:
            self._row_codec = None

    def _row_positions(self, query):
        """Positions of query rows within the fitted reference rows. Both are
        lexsorted and every query row must be present in the reference."""
        if self._row_codec is None:
            return _row_positions_structured(self.reference_keys, query)
        uniques, radix, ref_codes = self._row_codec
        codes = np.zeros(len(query), dtype=np.int64)
        for j in range(query.shape[1]):
            codes += np.searchsorted(uniques[j], query[:, j]) * radix[j]
        return np.searchsorted(ref_codes, codes)

    def _coefficients(self, iterations):
        """Per-row mean-of-means coefficients and output block starts for a
        given reduction depth."""
        try:
            return self._coefficient_cache[iterations]
        except KeyError:
            pass
        n, levels = self.reference_keys.shape
        coefficients = np.ones(n)
        child_starts = np.arange(n)
        for j in range(iterations):
            block_starts = self._starts[levels - 1 - j]
            edges = np.searchsorted(child_starts, np.append(block_starts, n))
            children_per_block = np.diff(edges)
            rows_per_block = np.diff(np.append(block_starts, n))
            coefficients *= np.repeat(1.0 / children_per_block, rows_per_block)
            child_starts = block_starts
        result = (coefficients, child_starts)
        self._coefficient_cache[iterations] = result
        return result

    def transform(self, target, iterations=1, resampled=False):
        """Performs iterative groupby reductions.

        Parameters
        ----------
        target : 2D numeric array
            Array to be reduced.
        iterations : int, optional
            Number of reductions to perform, by default 1
        resampled : bool, optional
            Set to True if target is an index-resampled version of the fitted
            reference data (rows repeated or dropped, as produced by
            Bootstrapper(kind="indexes")). Row multiplicities are then treated
            as bootstrap weights over the reference geometry, so the result
            agrees with aggregating the equivalent kind="weights" sample.
            By default False, which assumes target has the same row geometry
            as the fitted reference.

        Returns
        -------
        2D numeric array
            Array with one row per aggregated cluster and one fewer column
            for each iteration. Final column values are combined by taking
            the mean.
        """
        if iterations == 0:
            return target
        n, levels = self.reference_keys.shape
        coefficients, out_starts = self._coefficients(iterations)
        out = np.empty((len(out_starts), levels - iterations + 1))
        if not resampled:
            out[:, :-1] = target[out_starts, : levels - iterations]
            out[:, -1] = np.add.reduceat(coefficients * target[:, -1], out_starts)
        else:
            # collapse duplicate rows to (reference row, multiplicity), then
            # aggregate the multiplicity-weighted values over the reference
            # geometry; rows absent from the resample get zero weight
            keys = np.ascontiguousarray(target[:, :-1])
            change = np.any(keys[1:] != keys[:-1], axis=1)
            first = np.concatenate(
                (np.zeros(1, dtype=np.intp), np.flatnonzero(change) + 1)
            )
            multiplicity = np.diff(np.append(first, len(target)))
            positions = self._row_positions(keys[first])
            weighted = np.zeros(n)
            weighted[positions] = multiplicity * target[first, -1]
            out[:, :-1] = self.reference_keys[out_starts, : levels - iterations]
            out[:, -1] = np.add.reduceat(coefficients * weighted, out_starts)
        return out

    def fit_transform(self, target, reference_data=None, iterations=1):
        """Combines fit() and transform() for convenience. See those methods for details."""
        if reference_data is None:
            reference_data = target
        self.fit(reference_data)
        return self.transform(target, iterations=iterations)
