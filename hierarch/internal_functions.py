import numba as nb
import numpy as np


@nb.jit(nopython=True, cache=True)
def set_numba_random_state(seed: int):
    """Helper function to set numba's RNG seed.

    Parameters
    ----------
    seed : int32
        Seed for Numba's internal MT PRNG.
    """
    if type(seed) is nb.int64:
        np.random.seed(seed)
    else:
        raise ValueError("Numba seed must be an integer.")


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
def _repeat(target, counts):
    return np.repeat(target, counts)


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
    x = np.random.randint(low=2**32)
    m = ub * x
    l = np.uint32(m)
    if l < ub:
        t = -np.uint32(ub)
        if t >= ub:
            t -= ub
            if t >= ub:
                t %= ub
        while l < t:
            x = np.random.randint(low=2**32)
            m = ub * x
            l = np.uint32(m)
    return m >> 32


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
        return np.array(rv)

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
