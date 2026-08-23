import math
from functools import lru_cache
from itertools import combinations
from typing import Collection, Generator, Optional, Union
from warnings import simplefilter, warn

import numpy as np
import pandas as pd

from hierarch.internal_functions import (
    GroupbyMean,
    bivar_central_moment,
)
from hierarch.resampling import (
    Bootstrapper,
    Permuter,
    draw_bootstrap_weights_batch,
    draw_permuted_labels,
    exact_label_matrix,
    permutation_plan,
)


def _preprocess_data(data):
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
    encoded = encoded.astype(np.float64)

    return encoded


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
    >>> float(stats.ttest_ind(a, b, equal_var=False)[0])
    1.0

    """
    n = len(x)

    # numerator is the sample covariance, or the first symmetric bivariate central moment
    numerator = bivar_central_moment(x, y, pow=1, ddof=1)

    # the denominator is the sample standard deviation of the sample covariance, aka
    # the standard error of sample covariance. the denominator has three terms.

    # first term is the second symmetric bivariate central moment. an approximate
    # bias correction of n - root(2) is applied
    denom_1 = bivar_central_moment(x, y, pow=2, ddof=2**0.5)

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
    return float(t)


def jackknife_studentized_covariance(x, y):
    """Studentized sample covariance using jackknife variance estimate.

    Divides the sample covariance by the closed-form jackknife estimate
    of its standard error. This produces an approximately pivotal test
    statistic.

    Parameters
    ----------
    x, y: numeric array-likes

    Returns
    -------
    float64
        Jackknife studentized covariance.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    ab = (x - x.mean()) * (y - y.mean())
    S1 = ab.sum()
    S2 = (ab * ab).sum()
    return float(S1 * (n - 2) / ((n - 1) * (n * S2 - S1 * S1)) ** 0.5)


def welch_statistic(sample_a, sample_b):
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

    >>> import scipy.stats as stats
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> b = np.array([10, 11, 12, 13, 14])
    >>> welch_statistic(a, b)
    -9.0

    This uses the same calculation as scipy's ttest function.

    >>> import scipy.stats as stats
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> b = np.array([10, 11, 12, 13, 14])
    >>> float(stats.ttest_ind(a, b, equal_var=False)[0])
    -9.0


    Notes
    ----------
    Details on the validity of this test statistic can be found in
    "Studentized permutation tests for non-i.i.d. hypotheses and the
    generalized Behrens-Fisher problem" by Arnold Janssen.
    https://doi.org/10.1016/S0167-7152(97)00043-6.

    """

    len_a, len_b = len(sample_a), len(sample_b)

    # mean difference
    meandiff = np.mean(sample_a) - np.mean(sample_b)

    # weighted sample variances
    var_weight_one = bivar_central_moment(sample_a, sample_a, ddof=1) / len_a
    var_weight_two = bivar_central_moment(sample_b, sample_b, ddof=1) / len_b

    # compute t statistic
    t = meandiff / np.sqrt(var_weight_one + var_weight_two)

    return float(t)


def _wrap_custom_statistic(func):
    """Wrap a user-supplied test statistic with shape normalization and a
    loud contract check.

    Custom statistics receive a 2D array of permuted treatment columns with
    shape (permutations, n) and a row-aligned 2D array of dependent values
    with the same shape, and must return one statistic per row. A callable
    written for single columns is detected by its output shape and rejected
    with a migration hint rather than silently miscounted.
    """

    def batched(labels, y):
        y = np.asarray(y)
        if y.ndim == 1:
            y = np.broadcast_to(y, labels.shape)
        output = np.asarray(func(labels, y))
        if output.shape != (labels.shape[0],):
            raise TypeError(
                "Custom test statistics receive a (permutations, n) array of "
                "permuted treatment values and a matching array of dependent "
                "values, and must return one statistic per permutation (got "
                f"output shape {output.shape} for {labels.shape[0]} "
                "permutations). Vectorize over the last axis, or apply a "
                "single-column statistic row by row: "
                "np.array([f(x, y) for x, y in zip(treatments, values)])"
            )
        return output

    return batched


@lru_cache()
def _batched_stat_factory(treatment_col, compare):
    """Vectorized counterparts of the built-in test statistics.

    Returns a function mapping an (..., n) array of permuted treatment
    columns and a broadcast-compatible array of dependent values to one
    test statistic per row. Each statistic depends on the permuted labels
    only through the contractions sum(x * v) and sum(x^2 * w), so the
    labels array is read by einsum without materializing intermediates.
    Callers may pass (permutations, n) labels with an n-vector of values,
    or (chunk, permutations, n) labels with (chunk, 1, n) values to share
    one dependent-value row across a block of permutations.

    Parameters
    ----------
    treatment_col : 1D tuple
        Treatment column in the design matrix, as a tuple for lru_cache.
    compare : {'means', 'corr', 'jackknife_corr'}
    """
    if compare == "means":
        treatment_labels = np.unique(treatment_col)
        if treatment_labels.size != 2:
            raise ValueError("Needs 2 samples.")
        label_a, label_b = treatment_labels[0], treatment_labels[1]

        def _welch_batch(labels, y):
            n = labels.shape[-1]
            n_a = float((np.reshape(labels, (-1, n))[0] == label_a).sum())
            n_b = n - n_a
            # the group-a indicator is affine in the label values,
            # 1[x == a] = (b - x) / (b - a), so the group sums follow from
            # the contractions sum(x * y) and sum(x * y^2)
            span = label_b - label_a
            y_sq = y * y
            sum_y = np.sum(y, axis=-1)
            sum_y_sq = np.sum(y_sq, axis=-1)
            t1 = np.einsum("...m,...m->...", labels, y)
            t1_sq = np.einsum("...m,...m->...", labels, y_sq)
            sum_a = (label_b * sum_y - t1) / span
            sum_a_sq = (label_b * sum_y_sq - t1_sq) / span
            sum_b = sum_y - sum_a
            sum_b_sq = sum_y_sq - sum_a_sq
            mean_diff = sum_a / n_a - sum_b / n_b
            var_a = (sum_a_sq - sum_a**2 / n_a) / (n_a - 1)
            var_b = (sum_b_sq - sum_b**2 / n_b) / (n_b - 1)
            return mean_diff / np.sqrt(var_a / n_a + var_b / n_b)

        return _welch_batch

    elif compare == "corr":

        def _corr_batch(labels, y):
            n = labels.shape[-1]
            # constants come from the sorted multiset so every batch (and the
            # single-row observed-statistic call) computes them bit-identically
            x_sorted = np.sort(np.reshape(labels, (-1, n))[0])
            x_mean = x_sorted.mean()
            x_c_sorted = x_sorted - x_mean
            y_c = y - np.mean(y, axis=-1, keepdims=True)
            y_c2 = y_c * y_c
            sum_yc2 = y_c2.sum(axis=-1)
            s1 = np.einsum("...m,...m->...", labels, y_c) - x_mean * y_c.sum(
                axis=-1
            )
            t1 = np.einsum("...m,...m->...", labels, y_c2)
            t2 = np.einsum("...m,...m,...m->...", labels, labels, y_c2)
            s2 = t2 - 2.0 * x_mean * t1 + x_mean * x_mean * sum_yc2
            var_x = (x_c_sorted * x_c_sorted).sum() / (n - 1)
            var_y = sum_yc2 / (n - 1)
            numerator = s1 / (n - 1)
            denom_1 = s2 / (n - 2**0.5)
            denom_2 = var_x * var_y / (n - 1)
            denom_3 = (n - 2) * (s1 / (n - 1.75)) ** 2 / (n - 1)
            return numerator / ((1 / (n - 1.5)) * (denom_1 + denom_2 - denom_3)) ** 0.5

        return _corr_batch

    elif compare == "jackknife_corr":

        def _jackknife_batch(labels, y):
            n = labels.shape[-1]
            x_mean = np.sort(np.reshape(labels, (-1, n))[0]).mean()
            y_c = y - np.mean(y, axis=-1, keepdims=True)
            y_c2 = y_c * y_c
            s1 = np.einsum("...m,...m->...", labels, y_c) - x_mean * y_c.sum(
                axis=-1
            )
            t1 = np.einsum("...m,...m->...", labels, y_c2)
            t2 = np.einsum("...m,...m,...m->...", labels, labels, y_c2)
            s2 = t2 - 2.0 * x_mean * t1 + x_mean * x_mean * y_c2.sum(axis=-1)
            return s1 * (n - 2) / ((n - 1) * (n * s2 - s1 * s1)) ** 0.5

        return _jackknife_batch

    else:
        raise KeyError("No such comparison.")


def hypothesis_test(
    data_array,
    treatment_col,
    compare="corr",
    alternative="two-sided",
    skip=None,
    bootstraps=100,
    permutations=1000,
    kind="weights",
    return_null=False,
    random_state=None,
):
    """Two-tailed hierarchical permutation test for change in location
    with any number of samples.

    Equivalent to calculating a p-value for a slope coefficient in a linear model.

    Parameters
    ----------
    data_array : 2D numpy array or pandas DataFrame
        Array-like containing both the independent and dependent variables to
        be analyzed. It's assumed that the final (rightmost) column
        contains the dependent variable values.
    treatment_col : int or str
        The index number of the column containing "two samples" to be compared.
        Indexing starts at 0. If input data is a pandas DataFrame, this can be
        the name of the column.
    compare : {'corr', 'means', 'jackknife_corr'} or callable, optional
        The test statistic to use to perform the hypothesis test, by default "corr"
        which automatically calls the studentized covariance test statistic.
        "jackknife_corr" uses the jackknife studentized covariance test statistic.
        A callable receives a (permutations, n) array of permuted treatment
        columns and a row-aligned array of dependent values of the same
        shape, and must return one statistic per permutation (reduce over
        ``axis=-1``); it then runs at the speed of the built-in statistics.
        A single-column statistic can be applied row by row inside the
        callable: ``np.array([f(x, y) for x, y in zip(treatments, values)])``.
    alternative : {"two-sided", "less", "greater"}
        The alternative hypothesis for the test, "two-sided" by default.
    skip : list of ints, optional
        Columns to skip in the bootstrap. Skip columns that were sampled
        without replacement from the prior column, by default None
    bootstraps : int, optional
        Number of bootstraps to perform, by default 100. Can be set to 1 for a
        permutation test without any bootstrapping.
    permutations : int or "all", optional
        Number of permutations to perform PER bootstrap sample. "all"
        for exact test (only works if there are only two treatments), by default 1000
    kind : str, optional
        Bootstrap algorithm - see Bootstrapper class, by default "weights"
    return_null : bool, optional
        Return the null distribution as well as the p value, by default False
    random_state : int or numpy random Generator, optional
        Seedable for reproducibility, by default None

    Returns
    -------
    float64
        p-value for the hypothesis test

    list
        Empirical null distribution used to calculate the p-value

    Raises
    ------
    TypeError
        Raised if input data is not ndarray or DataFrame.
    KeyError
        If comparison is a string, it must be in the TEST_STATISTICS dictionary.
    AttributeError
        If comparison is a custom statistic, it must be a function.

    Examples
    --------
    Specify the parameters of a dataset with a difference of means of 2.

    >>> from hierarch.power import DataSimulator
    >>> import scipy.stats as stats
    >>> paramlist = [[0, 2], [stats.norm], [stats.norm]]
    >>> hierarchy = [2, 4, 3]
    >>> datagen = DataSimulator(paramlist, random_state=2)
    >>> datagen.fit(hierarchy)
    >>> data = datagen.generate()
    >>> print(data.shape)
    (24, 4)

    >>> hypothesis_test(data, treatment_col=0,
    ...                 bootstraps=1000, permutations='all',
    ...                 random_state=1)
    0.012514285714285714

    By setting compare to "means", this function will perform a permutation t-test.
    "corr", which is based on a studentized covariance test statistic, should give the
    same or a very similar p-value to the permutation t-test for datasets with two
    treatment groups.

    >>> hypothesis_test(data, treatment_col=0, compare='means',
    ...                 bootstraps=1000, permutations='all',
    ...                 random_state=1)
    0.012514285714285714

    This test can handle data with multiple treatment groups that have a
    hypothesized linear relationship.

    >>> paramlist = [[0, 2/3, 4/3, 2], [stats.norm], [stats.norm]]
    >>> hierarchy = [4, 2, 3]
    >>> datagen = DataSimulator(paramlist, random_state=2)
    >>> datagen.fit(hierarchy)
    >>> data = datagen.generate()
    >>> print(data.shape)
    (24, 4)

    There are 2,520 possible permutations, so choose a subset.

    >>> hypothesis_test(data, treatment_col=0,
    ...                 bootstraps=100, permutations=1000,
    ...                 random_state=1)
    0.0069


    """

    # turns the input array or dataframe into a float64 array
    if isinstance(data_array, (np.ndarray, pd.DataFrame)):
        if isinstance(data_array, pd.DataFrame) and isinstance(treatment_col, str):
            treatment_col = int(data_array.columns.get_loc(treatment_col))
        data = _preprocess_data(data_array)
    else:
        raise TypeError("Input data must be ndarray or DataFrame.")

    # set random state
    rng = np.random.default_rng(random_state)

    # enforce lower bound on skip
    if skip is not None:
        skip = list(skip)
        for v in reversed(skip):
            if v <= treatment_col + 1:
                warn("No need to include columns before treated columns in skip.")
                skip.remove(v)
    else:
        skip = []

    # enforce bounds on bootstraps and permutations
    if not isinstance(bootstraps, int) or bootstraps < 1:
        raise TypeError("bootstraps must be an integer greater than 0")
    if isinstance(permutations, str):
        if permutations != "all":
            raise TypeError("permutations must be 'all' or an integer greater than 0")
    elif not isinstance(permutations, int) or permutations < 1:
        raise TypeError("permutations must be 'all' or an integer greater than 0")

    # initialize and fit the bootstrapper to the data
    bootstrapper = Bootstrapper(random_state=rng, kind=kind)
    bootstrapper.fit(data, skip=skip)

    # fetch a vectorized test statistic from the built-in dictionary or, if
    # given a custom statistic, make sure it is callable and wrap it with
    # shape validation
    if isinstance(compare, str):
        batched_stat = _batched_stat_factory(
            tuple(data[:, treatment_col].tolist()), compare
        )
    elif callable(compare):
        batched_stat = _wrap_custom_statistic(compare)
    else:
        raise AttributeError("Custom test statistics must be callable.")

    # aggregate our data up to the treated level and determine the
    # observed test statistic
    aggregator = GroupbyMean()
    aggregator.fit(data)

    # determine the number of groupby reductions need to be done
    levels_to_agg = data.shape[1] - treatment_col - 3

    # if levels_to_agg = 0, there are no bootstrap samples to
    # generate.
    if (levels_to_agg - len(skip)) == 0 and bootstraps > 1:
        bootstraps = 1
        simplefilter("always", UserWarning)
        warn("No levels to bootstrap. Setting bootstraps to zero.")

    test = data
    test = aggregator.transform(test, iterations=levels_to_agg)

    # prepare the permuted treatment columns; if the test statistic is one of
    # the built-in comparisons, a whole batch of permutations is scored at
    # once with the vectorized implementations below
    if permutations == "all":
        # every distinct labeling, in multiset-permutation order; the total
        # number of permutations is C(n, n_0), and each bootstrap consumes
        # the next `permutations` rows of the (cycled) enumeration
        exact_labels = exact_label_matrix(test, treatment_col)
        counts = np.unique(test[:, 0], return_counts=True)[1]
        permutations = _binomial(counts.sum(), counts[0])
        perm_plan = None
    else:
        exact_labels = None
        perm_plan = permutation_plan(test, treatment_col)

    # the observed statistic is computed with the same (batched) arithmetic
    # as the null distribution: permutations that reproduce the observed
    # labeling then yield bit-identical statistics, so ties are counted as
    # extreme on both tails no matter how y has been transformed
    # contiguous copies, because einsum's accumulation order (and therefore
    # the exact result) differs between strided column views and the
    # contiguous rows scored in the null distribution
    truediff = batched_stat(
        np.ascontiguousarray(test[:, treatment_col])[None, :],
        np.ascontiguousarray(test[:, -1]),
    )[0]

    total = bootstraps * permutations

    # fully batched: all bootstrap weight sets, aggregations, permutations,
    # and statistics are drawn and scored in single vectorized passes. The
    # first y row is the original (unbootstrapped) aggregated data, which
    # prevents getting a p-value of 0. Index resamples are aggregated through
    # their weight representation, which is exactly how the sequential
    # resampled path computes them.
    y_matrix = np.empty((bootstraps, test.shape[0]))
    y_matrix[0] = test[:, -1]
    if bootstraps > 1:
        weights = draw_bootstrap_weights_batch(
            bootstrapper._plan, rng, treatment_col + 2, kind, bootstraps - 1
        )
        y_matrix[1:] = aggregator.transform_batch(
            weights * data[:, -1], iterations=levels_to_agg
        )
    if exact_labels is None:
        labels = draw_permuted_labels(perm_plan, rng, total)
    else:
        rows = (
            np.arange(bootstraps)[:, None] * permutations
            + np.arange(permutations)[None, :]
        ) % len(exact_labels)
        labels = exact_labels[rows.reshape(-1)]
    # built-in statistics broadcast one dependent-value row across each
    # bootstrap's block of permutations, so the whole null distribution is
    # scored in a single einsum-backed call; custom statistics keep the 2D
    # (permutations, n) contract, so their rows are materialized in
    # cache-sized chunks instead
    if isinstance(compare, str):
        null_distribution = batched_stat(
            labels.reshape(bootstraps, permutations, -1), y_matrix[:, None, :]
        ).reshape(-1)
    else:
        null_distribution = np.empty(total)
        group = max(1, 16384 // permutations + 1)
        for c0 in range(0, bootstraps, group):
            c1 = min(bootstraps, c0 + group)
            block = labels[c0 * permutations : c1 * permutations]
            y_block = np.repeat(y_matrix[c0:c1], permutations, axis=0)
            null_distribution[c0 * permutations : c1 * permutations] = batched_stat(
                block, y_block
            )

    # generate both one-tailed p-values, then two-tailed
    p_less = np.count_nonzero(truediff >= null_distribution) / len(null_distribution)
    p_greater = np.count_nonzero(truediff <= null_distribution) / len(null_distribution)
    p_two = 2 * np.min((p_less, p_greater))

    if alternative == "two-sided":
        pval = p_two
    elif alternative == "less":
        pval = p_less
    elif alternative == "greater":
        pval = p_greater

    if pval == 0:
        pval += 1 / (total)

    if return_null is True:
        return float(pval), null_distribution.tolist()

    else:
        return float(pval)


def multi_sample_test(
    data_array,
    treatment_col,
    hypotheses="all",
    correction="fdr",
    compare="means",
    skip=None,
    bootstraps=100,
    permutations=1000,
    kind="weights",
    random_state=None,
):
    """Two-tailed multiple-sample hierarchical permutation test.

    Equivalent to a post-hoc test after ANOVA. Results are more interpetable
    when the input data is in the form of a pandas dataframe or numpy object array.

    Parameters
    ----------
    data_array : 2D array or pandas DataFrame
        Array-like containing both the independent and dependent variables to
        be analyzed. It's assumed that the final (rightmost) column
        contains the dependent variable values.
    treatment_col : int or str
        The index number of the column containing labels to be compared.
        Indexing starts at 0. If input data is a pandas DataFrame, this can
        be the column name.
    hypotheses : list of two-element lists or "all", optional
        Hypotheses to be tested. If 'all' every pairwise comparison will be
        tested. Can be passed a list of lists to restrict comparisons, which
        will result in a less harsh multiple comparisons correction, by default "all"
    correction : str, optional
        Multiple comparisons question to be performed after p-values are
        calculated. 'fdr' performs the Benjamini-Hochberg procedure for
        controlling False Discovery Rate, by default "fdr"
    compare : function or str, optional
        The comparison to use to perform the hypothesis test, by default "means"
    skip : list of ints, optional
        Columns to skip in the bootstrap, by default None
    bootstraps : int, optional
        Number of bootstraps to perform, by default 100
    permutations : int or "all"
        Number of permutations to perform PER bootstrap sample. "all"
        for exact test, by default 1000
    kind : str, optional
        Bootstrapper algorithm. See Bootstrapper class, by default "weights"
    seed : int or numpy.random.Generator instance, optional
        Seedable for reproducibility, by default None

    Returns
    -------
    ndarray
        numpy ndarray with col 0, 1 corresponding to treatment labels, col 2 corresponding
        to an uncorrected p-value, and col 3 corresponding to a corrected p-value if a
        correction was specified.

    Raises
    ------
    KeyError
        Raised if passed correction is not valid.
    TypeError
        Raised if input data is not ndarray or DataFrame.
    KeyError
        Raised if specified comparison labels do not exist in the input data.

    Examples
    --------
    This function performs pairwise tests akin to a post-hoc test after one-way ANOVA.

    >>> from hierarch.power import DataSimulator
    >>> import scipy.stats as stats
    >>> paramlist = [[0, 1, 4, 0], [stats.norm], [stats.norm]]
    >>> hierarchy = [4, 3, 3]

    This dataset has four treatment conditions, two of which
    have the same mean (condition 1 and 4). Condition 2 has a
    slight mean difference from 1 and 4, so this experiment is
    likely not well-powered to detect it. Condition 3 has a
    large mean difference from the others, however, and should
    return a significant result against all three other conditions.

    >>> datagen = DataSimulator(paramlist, random_state=1)
    >>> datagen.fit(hierarchy)
    >>> data = datagen.generate()
    >>> data
    array([[ 1.        ,  1.        ,  1.        , -0.39086989],
           [ 1.        ,  1.        ,  2.        ,  0.18267424],
           [ 1.        ,  1.        ,  3.        , -0.13653512],
           [ 1.        ,  2.        ,  1.        ,  1.42046436],
           [ 1.        ,  2.        ,  2.        ,  0.86134025],
           [ 1.        ,  2.        ,  3.        ,  0.52916139],
           [ 1.        ,  3.        ,  1.        , -0.45147139],
           [ 1.        ,  3.        ,  2.        ,  0.07324484],
           [ 1.        ,  3.        ,  3.        ,  0.33857926],
           [ 2.        ,  1.        ,  1.        , -0.57876014],
           [ 2.        ,  1.        ,  2.        ,  0.99090658],
           [ 2.        ,  1.        ,  3.        ,  0.70356708],
           [ 2.        ,  2.        ,  1.        , -0.80580661],
           [ 2.        ,  2.        ,  2.        ,  0.01634262],
           [ 2.        ,  2.        ,  3.        ,  1.73058377],
           [ 2.        ,  3.        ,  1.        ,  1.02418416],
           [ 2.        ,  3.        ,  2.        ,  1.66001757],
           [ 2.        ,  3.        ,  3.        ,  1.6636965 ],
           [ 3.        ,  1.        ,  1.        ,  5.58088552],
           [ 3.        ,  1.        ,  2.        ,  2.351026  ],
           [ 3.        ,  1.        ,  3.        ,  3.08544176],
           [ 3.        ,  2.        ,  1.        ,  6.62388971],
           [ 3.        ,  2.        ,  2.        ,  5.2278211 ],
           [ 3.        ,  2.        ,  3.        ,  5.24418148],
           [ 3.        ,  3.        ,  1.        ,  3.85056602],
           [ 3.        ,  3.        ,  2.        ,  2.71649723],
           [ 3.        ,  3.        ,  3.        ,  4.53203714],
           [ 4.        ,  1.        ,  1.        ,  0.40314658],
           [ 4.        ,  1.        ,  2.        , -0.93321956],
           [ 4.        ,  1.        ,  3.        , -0.38909417],
           [ 4.        ,  2.        ,  1.        , -0.04362144],
           [ 4.        ,  2.        ,  2.        , -0.91632938],
           [ 4.        ,  2.        ,  3.        , -0.06984773],
           [ 4.        ,  3.        ,  1.        ,  0.64219601],
           [ 4.        ,  3.        ,  2.        ,  0.58229922],
           [ 4.        ,  3.        ,  3.        ,  0.04042133]])

    There are six total comparisons that can be made. Condition 1 and 2 are in the first
    two columns and the p-values are in the final column.

    >>> multi_sample_test(data, treatment_col=0, hypotheses="all",
    ...                   correction=None, bootstraps=1000,
    ...                   permutations="all", random_state=111)
      Condition 1 Condition 2 p-value
    0         3.0         4.0   0.035
    1         2.0         3.0  0.0353
    2         1.0         3.0  0.0414
    3         2.0         4.0  0.1504
    4         1.0         2.0  0.4029
    5         1.0         4.0  0.4519

    Multiple comparison correction to control False Discovery Rate is advisable in
    this situation. The final column now shows the q-values, or "adjusted" p-values
    following the Benjamini-Hochberg procedure.

    >>> multi_sample_test(data, treatment_col=0, hypotheses="all",
    ...                   correction='fdr', bootstraps=1000,
    ...                   permutations="all", random_state=111)
      Condition 1 Condition 2 p-value Corrected p-value
    0         3.0         4.0   0.035            0.0828
    1         2.0         3.0  0.0353            0.0828
    2         1.0         3.0  0.0414            0.0828
    3         2.0         4.0  0.1504            0.2256
    4         1.0         2.0  0.4029            0.4519
    5         1.0         4.0  0.4519            0.4519

    Perhaps the experimenter is not interested in every pairwise comparison - perhaps
    condition 2 is a control that all other conditions are meant to be compared to.
    The comparisons of interest can be specified using a list.

    >>> tests = [[2.0, 1.0], [2.0, 3.0], [2.0, 4.0]]
    >>> multi_sample_test(data, treatment_col=0, hypotheses=tests,
    ...                   correction='fdr', bootstraps=1000,
    ...                   permutations="all", random_state=222)
      Condition 1 Condition 2 p-value Corrected p-value
    0         2.0         3.0   0.035             0.105
    1         2.0         4.0  0.1521           0.22815
    2         2.0         1.0  0.4066            0.4066


    """

    MULTIPLE_COMPARISONS_CORRECTIONS = {
        "fdr": _false_discovery_adjust,
    }
    if correction is not None:
        try:
            multiple_correction = MULTIPLE_COMPARISONS_CORRECTIONS[correction]
        except KeyError:
            raise KeyError(
                correction + " is not a valid multiple comparisons correction."
            )

    random_state = np.random.default_rng(random_state)

    # coerce data into an object array
    if isinstance(data_array, pd.DataFrame):
        if isinstance(treatment_col, str):
            treatment_col = data_array.columns.get_loc(treatment_col)
        data = data_array.to_numpy()
    elif isinstance(data_array, np.ndarray):
        data = data_array
    else:
        raise TypeError("Input data must be ndarray or DataFrame")

    # if list of comparisons has been provided, make an array for output
    if isinstance(hypotheses, list):
        hypotheses = np.array(hypotheses, dtype="object")
        # if hypotheses were provided, check to make sure they're in the treatment_column.
        for label in iter(hypotheses.flat):
            if label not in data[:, treatment_col]:
                raise KeyError(label + " not found in specified column.")
        # make room to insert p-values
        output = np.empty(
            (hypotheses.shape[0], hypotheses.shape[1] + 1), dtype="object"
        )
        output[:, :-1] = hypotheses
    # otherwise, enumerate all possible comparisons and make output array
    else:
        output = _get_comparisons(data, treatment_col)

    # perform a two_sample_test for each comparison
    # no option to return null distributions because that would be a hassle
    for i in range(len(output)):
        test_idx = np.logical_or(
            (data[:, treatment_col] == output[i, 0]),
            (data[:, treatment_col] == output[i, 1]),
        )
        output[i, 2] = hypothesis_test(
            data[test_idx],
            treatment_col=treatment_col,
            compare=compare,
            skip=skip,
            bootstraps=bootstraps,
            permutations=permutations,
            kind=kind,
            random_state=random_state,
        )

    # sort the output array so that smallest p-values are on top
    ordered_idx = output[:, -1].argsort()
    output = output[ordered_idx]

    # perform multiple comparisons correction, if any
    if correction is not None:
        q_vals = multiple_correction(output[:, -1])
        out = np.empty((output.shape[0], output.shape[1] + 1), dtype="object")
        out[:, :-1] = output
        out[:, -1] = q_vals
        output = out
        output = pd.DataFrame(
            output,
            columns=["Condition 1", "Condition 2", "p-value", "Corrected p-value"],
        )
    else:
        output = pd.DataFrame(output, columns=["Condition 1", "Condition 2", "p-value"])

    return output


def _get_comparisons(data, treatment_col: int):
    """Generates a list of pairwise comparisons for a k-sample test.

    Parameters
    ----------
    data : 2D array or pandas DataFrame
        Target data.
    treatment_col : int
        Target column.

    Returns
    -------
    list of lists
        list of two-member lists containing each pairwise comparison.
    """

    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    comparisons = []
    for i, j in combinations(np.unique(data[:, treatment_col]), 2):
        comparisons.append([i, j])
    comparisons = np.array(comparisons, dtype="object")
    out = np.empty((comparisons.shape[0], comparisons.shape[1] + 1), dtype="object")
    out[:, :-1] = comparisons
    return out


def _binomial(x: int, y: int):
    """Calculates a binomial coefficient.

    Parameters
    ----------
    x : int
        Total number of elements
    y : int
        Elements to choose

    Returns
    -------
    int
        x choose y
    """

    try:
        return math.factorial(x) // math.factorial(y) // math.factorial(x - y)
    except ValueError:
        return 0


def _false_discovery_adjust(pvals, return_index=False):
    """Performs the Benjamini-Hochberg procedure for controlling false discovery rate.

    Parameters
    ----------
    pvals : 1D array-like
        p-values to be adjusted
    return_index : bool, optional
        If true, will return the indices to sort the original p-value list, by default False

    Returns
    -------
    q_vals : 1D array
        q-values aka "adjusted" p-values

    sort_key : 1D array
        indices used to sort pvals

    Notes
    -----
    Refererence: Benjamini, Y. & Hochberg, Y. Controlling the false discovery
    rate: a practical and powerful approach to multiple testing.
    Journal of the Royal Statistical Society.
    Series B (Methodological) 289–300 (1995).

    The q-values, or "adjusted p-values," are not really p-values.
    Rather, each q-value is the minimum FDR you must accept to regard
    the result of that hypothesis test significant. In that sense, each q-value
    represents the minimum posterior probability that the null hypothesis is
    true for the comparison of interest. However, q-values are often called
    adjusted p-values in practice, so we do so here.

    Examples
    --------
    >>> p_vals = np.arange(0.05, 1.05, step=0.1)
    >>> p_vals
    array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

    >>> _false_discovery_adjust(p_vals)
    array([0.5    , 0.75   , 0.83333, 0.875  , 0.9    , 0.91667, 0.92857,
           0.9375 , 0.94444, 0.95   ])

    A large number of rejections "remain" rejected.

    >>> p_vals = np.arange(0.01, 0.05, step=0.01)
    >>> _false_discovery_adjust(p_vals)
    array([0.04, 0.04, 0.04, 0.04])

    """

    # argsort so we can sort a list of hypotheses, if need be
    sort_key = np.argsort(pvals)
    # q-value adjustment
    q_vals = np.array(pvals)[sort_key] * len(pvals)
    q_vals /= np.array(range(1, len(pvals) + 1))
    q_vals = np.around(q_vals.astype(float), decimals=5)
    # list of q values must be strictly non-decreasing
    for i in range(len(q_vals) - 1, 0, -1):
        if q_vals[i] < q_vals[i - 1]:
            q_vals[i - 1] = q_vals[i]

    if return_index is True:
        return q_vals, sort_key
    else:
        return q_vals


def confidence_interval(
    data_array,
    treatment_col,
    interval=95.0,
    iterations=7,
    tolerance=1,
    compare="corr",
    skip=None,
    bootstraps=50,
    permutations=100,
    kind="bayesian",
    random_state=None,
):
    """Compute a confidence inverval via test inversion.

    Confidence interval can be calculated by inverting the acceptance region of a hypothesis test.
    Using a test statistic that is approximately normally distributed under the null makes this
    much easier.

    Parameters
    ----------
    data_array : 2D numpy array or pandas DataFrame
        Array-like containing both the independent and dependent variables to
        be analyzed. It's assumed that the final (rightmost) column
        contains the dependent variable values.
    treatment_col : int or str
        The index number of the column containing "two samples" to be compared.
        Indexing starts at 0. If input data is a pandas DataFrame, this can be
        the column name.
    interval : float, optional
        Percentage value indicating the confidence interval's coverage, by default 95
    iterations : int, optional
        Maximum number of times the interval will be refined, by default 7
    tolerance : float, optional
        If the delta between the current bounds and the target interval is less than
        this value, refinement will stop. Setting this number too close to the Monte Carlo
        error of the underlying hypothesis test will have a negative effect on coverage.
    compare : {'corr', 'means', 'jackknife_corr'} or callable, optional
        The test statistic to use to perform the hypothesis test, by default "corr"
        which automatically calls the studentized covariance test statistic.
        "jackknife_corr" uses the jackknife studentized covariance test statistic.
    skip : list of ints, optional
        Columns to skip in the bootstrap. Skip columns that were sampled
        without replacement from the prior column, by default None
    bootstraps : int, optional
        Number of bootstraps to perform, by default 100. Can be set to 1 for a
        permutation test without any bootstrapping.
    permutations : int or "all", optional
        Number of permutations to perform PER bootstrap sample. "all"
        for exact test (only works if there are only two treatments), by default 1000
    kind : str, optional
        Bootstrap algorithm - see Bootstrapper class, by default "bayesian"
    random_state : int or numpy random Generator, optional
        Seedable for reproducibility., by default None

    Returns
    -------
    tuple of floats
        Confidence interval spanning the specified interval.

    Notes
    -----
    While the Efron bootstrap is the default in most of hierarch's statistical functions,
    using the Bayesian bootstrap here helps get tighter confidence intervals with the
    correct coverage without having to massively increase the number of resamples.

    The inversion procedure performed by this function is described in detail in
    "Randomization, Bootstrap and Monte Carlo Methods in Biology" by Bryan FJ Manly.
    https://doi.org/10.1201/9781315273075.

    Examples
    --------
    Specify the parameters of a dataset with a difference of means of 2.

    >>> from hierarch.power import DataSimulator
    >>> import scipy.stats as stats
    >>> paramlist = [[0, 2], [stats.norm], [stats.norm]]
    >>> hierarchy = [2, 4, 3]
    >>> datagen = DataSimulator(paramlist, random_state=2)
    >>> datagen.fit(hierarchy)
    >>> data = datagen.generate()
    >>> print(data.shape)
    (24, 4)

    >>> confidence_interval(data, treatment_col=0, interval=95,
    ...    bootstraps=1000, permutations='all', random_state=1)
    (1.339105823544274, 6.100359929247537)

    The true difference is 2, which falls within the interval. We can examine
    the p-value for the corresponding dataset:

    >>> from hierarch.stats import hypothesis_test
    >>> hypothesis_test(data, treatment_col=0, compare='corr',
    ...                 bootstraps=1000, permutations='all',
    ...                 random_state=1)
    0.012514285714285714

    This suggests that while the 95% confidence interval does not contain 0, the 99.5%
    confidence interval should.

    >>> confidence_interval(data, treatment_col=0, interval=99.5,
    ...    bootstraps=1000, permutations='all', random_state=1)
    (-0.15334319814776087, 7.592808950939554)

    A permutation t-test can be used to generate the null distribution by
    specifying compare = "means". This should return the same or a very
    similar interval.

    >>> confidence_interval(data, treatment_col=0, interval=95,
    ...    compare='means', bootstraps=1000,
    ...    permutations='all', random_state=1)
    (1.339105823544274, 6.100359929247537)

    Setting compare = "corr" will generate a confidence interval for the slope
    in a regression equation.

    >>> paramlist = [[0, 1, 2, 3], [stats.norm], [stats.norm]]
    >>> hierarchy = [4, 4, 3]
    >>> datagen = DataSimulator(paramlist, random_state=2)
    >>> datagen.fit(hierarchy)
    >>> data = datagen.generate()

    >>> confidence_interval(data, treatment_col=0, interval=95,
    ...                 compare='corr', bootstraps=100,
    ...                 permutations=1000, random_state=1)
    (0.8329725571205298, 1.6195336227023125)

    The dataset was specified to have a true slope of 1, which is within the interval.

    """

    rng = np.random.default_rng(random_state)

    alpha = (100 - interval) / 200

    # turns the input array or dataframe into a float64 array
    if isinstance(data_array, (np.ndarray, pd.DataFrame)):
        if isinstance(data_array, pd.DataFrame) and isinstance(treatment_col, str):
            treatment_col = int(data_array.columns.get_loc(treatment_col))
        data = _preprocess_data(data_array)
    else:
        raise TypeError("Input data must be ndarray or DataFrame.")

    # first compute the null distribution against the null that the effect size is equal to the MLE
    null_imposed_data = data.copy()
    levels_to_agg = data.shape[1] - treatment_col - 3

    grouper = GroupbyMean()
    grouper.fit(data)
    test = grouper.transform(data, iterations=levels_to_agg)
    start_slope = bivar_central_moment(
        test[:, treatment_col], test[:, -1]
    ) / bivar_central_moment(test[:, treatment_col], test[:, treatment_col])

    # subtract the observed covariance out
    correction = start_slope * null_imposed_data[:, treatment_col]
    null_imposed_data[:, -1] -= correction

    # compute the null distribution for the null hypothesis that the true effect
    # size is equal to the MLE
    _, null = hypothesis_test(
        null_imposed_data,
        treatment_col,
        skip=skip,
        bootstraps=bootstraps,
        permutations=permutations,
        kind=kind,
        return_null=True,
        random_state=rng,
    )

    # make a guess as to the lower and upper bounds of the confidence interval

    if compare == "jackknife_corr":
        std_error_fn = _jackknife_cov_std_error
    else:
        std_error_fn = _cov_std_error

    null_agg = grouper.transform(null_imposed_data, iterations=levels_to_agg)

    current_lower = _compute_interval(
        np.array(null), null_agg, treatment_col, alpha, std_error_fn
    )
    current_upper = _compute_interval(
        np.array(null), null_agg, treatment_col, 1 - alpha, std_error_fn
    )

    # refine the bounds via iterative hypothesis testing
    # each bound needs to be found separately

    # find lower bound

    if compare == "means":
        alternative_lower, alternative_upper = "greater", "less"
    else:
        alternative_lower, alternative_upper = "less", "greater"

    for i in range(iterations):
        bound_imposed_data = null_imposed_data.copy()
        bound_imposed_data[:, -1] += (current_lower) * bound_imposed_data[
            :, treatment_col
        ]
        current_p, null = hypothesis_test(
            bound_imposed_data,
            treatment_col,
            compare=compare,
            alternative=alternative_lower,
            skip=skip,
            bootstraps=bootstraps,
            permutations=permutations,
            kind=kind,
            return_null=True,
            random_state=rng,
        )

        if np.abs(100 * (alpha - current_p)) < tolerance:
            break

        bound_agg = grouper.transform(bound_imposed_data, iterations=levels_to_agg)

        current_lower = _compute_interval(
            np.array(null), bound_agg, treatment_col, alpha, std_error_fn
        )

    else:
        warn(
            " ".join(["lower tail:", str(current_p), "failed to converge"]),
            ConvergenceWarning,
            stacklevel=2,
        )

    for i in range(iterations):
        bound_imposed_data = null_imposed_data.copy()
        bound_imposed_data[:, -1] += (current_upper) * bound_imposed_data[
            :, treatment_col
        ]
        current_p, null = hypothesis_test(
            bound_imposed_data,
            treatment_col,
            compare=compare,
            alternative=alternative_upper,
            skip=skip,
            bootstraps=bootstraps,
            permutations=permutations,
            kind=kind,
            return_null=True,
            random_state=rng,
        )

        if np.abs(100 * (alpha - current_p)) < tolerance:
            break
        bound_agg = grouper.transform(bound_imposed_data, iterations=levels_to_agg)

        current_upper = _compute_interval(
            np.array(null), bound_agg, treatment_col, 1 - alpha, std_error_fn
        )

    else:
        warn(
            " ".join(["upper tail:", str(current_p), "failed to converge"]),
            ConvergenceWarning,
            stacklevel=2,
        )

    return float(current_lower + start_slope), float(current_upper + start_slope)


class ConvergenceWarning(Warning):
    """Arises when iterative search for confidence intervals fails.

    Can typically be solved by upping the number of permutations or search iterations.
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


def _compute_interval(null, null_data, treatment_col, quantile, std_error_fn):
    """Unpivots a test statistic to a slope.

    Parameters
    ----------
    null : 1D array
    null_data : 2D array
        Data used to compute the null distribution.
    treatment_col : int
    quantile : float between 0 and 1
        Quantile of the null distribution to pull test statistic from.
    std_error_fn : callable
        Function to compute the standard error of the covariance.

    Returns
    -------
    float

    Examples
    --------
    >>> from hierarch.power import DataSimulator
    >>> import scipy.stats as stats
    >>> paramlist = [[0, 2], [stats.norm], [stats.norm]]
    >>> hierarchy = [2, 4, 3]
    >>> datagen = DataSimulator(paramlist, random_state=2)
    >>> datagen.fit(hierarchy)
    >>> data = datagen.generate()
    >>> null = np.array(hypothesis_test(data, 0, return_null=True, random_state=5)[1])
    >>> _compute_interval(null, data, 0, 0.025, _cov_std_error)
    -1.6154082371193446

    The test statistic distribution is essentially symmetric about 0.

    >>> _compute_interval(null, data, 0, 0.975, _cov_std_error)
    1.594832054941255

    """
    x = null_data[:, treatment_col]
    y = null_data[:, -1]

    denom = std_error_fn(x, y)
    bound = np.quantile(null, quantile) * denom / bivar_central_moment(x, x)

    return float(bound)


def _cov_std_error(x, y):
    """Computes an estimate of the standard error of the covariance between
    two variables.

    Parameters
    ----------
    x, y : 1D numeric arrays

    Returns
    -------
    float

    Examples
    --------
    >>> x = np.arange(10)
    >>> y = np.arange(10)
    >>> _cov_std_error(x, y)
    2.683675672629574

    More data with an identical relationship causes the standard error to decrease.

    >>> x = np.arange(10).repeat(5)
    >>> y = np.arange(10).repeat(5)
    >>> _cov_std_error(x, y)
    1.0574158590055294

    >>> x = np.arange(10).repeat(50)
    >>> y = np.arange(10).repeat(50)
    >>> _cov_std_error(x, y)
    0.32587563558526406

    """
    n = len(x)
    # first term is the second symmetric bivariate central moment. an approximate
    # bias correction of n - root(2) is applied
    denom_1 = bivar_central_moment(x, y, pow=2, ddof=2**0.5)

    # second term is the product of the standard deviations of x and y over n - 1.
    # this term rapidly goes to 0 as n goes to infinity
    denom_2 = (
        bivar_central_moment(x, x, pow=1, ddof=1)
        * bivar_central_moment(y, y, pow=1, ddof=1)
    ) / (n - 1)

    # third term is the square of the covariance of x and y. an approximate bias
    # correction of n - root(3) is applied
    denom_3 = ((n - 2) * (bivar_central_moment(x, y, pow=1, ddof=1.75) ** 2)) / (n - 1)
    return float(((1 / (n - 1.5)) * (denom_1 + denom_2 - denom_3)) ** 0.5)


def _jackknife_cov_std_error(x, y):
    """Computes the jackknife standard error of the covariance between
    two variables.

    This extracts the denominator of ``jackknife_studentized_covariance``
    so that it can be used by ``_compute_interval`` to unpivot the test
    statistic back to a slope.

    Parameters
    ----------
    x, y : 1D numeric arrays

    Returns
    -------
    float

    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    ab = (x - x.mean()) * (y - y.mean())
    S1 = ab.sum()
    S2 = (ab * ab).sum()
    return float(((n - 1) * (n * S2 - S1 * S1)) ** 0.5 / ((n - 1) * (n - 2)))


def hierarchical_randomization(
    data_array: Union[np.ndarray, pd.DataFrame],
    treatment_col: Union[int, str],
    skip: Optional[Collection[int]] = None,
    bootstraps: int = 100,
    permutations: int = 1000,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Generator[np.ndarray, None, None]:
    """Yields permuted datasets for a hierarchical randomization test.

    Parameters
    ----------
    data_array : 2D numpy array or pandas DataFrame
        Array-like containing both the independent and dependent variables to
        be analyzed. It's assumed that the final (rightmost) column
        contains the dependent variable values.
    treatment_col : int or str
        The index number of the column containing "N samples" to be compared.
        Indexing starts at 0. If input data is a pandas DataFrame, this can be
        the column name.
    skip : list of ints, optional
        Columns to skip in the bootstrap. Skip columns that were sampled
        without replacement from the prior column, by default None
    bootstraps : int, optional
        Number of bootstraps to perform, by default 100. Can be set to 1 for a
        permutation test without any bootstrapping.
    permutations : int, optional
        Number of permutations to perform PER bootstrap sample. "all"
        for exact test (only works if there are only two treatments), by default 1000
    random_state : int or numpy random Generator, optional
        Seedable for reproducibility., by default None

    Yields
    ------
    Generator[np.ndarray, None, None]
        Permuted data for a hierarchical randomization test.

    """
    # turns the input array or dataframe into a float64 array
    if isinstance(data_array, (np.ndarray, pd.DataFrame)):
        if isinstance(data_array, pd.DataFrame) and isinstance(treatment_col, str):
            treatment_col = int(data_array.columns.get_loc(treatment_col))
        data = _preprocess_data(data_array)
    else:
        raise TypeError("Input data must be ndarray or DataFrame.")

    assert isinstance(treatment_col, int)  # mypy

    # set random state
    rng = np.random.default_rng(random_state)

    # enforce lower bound on skip
    if skip is not None:
        skip = list(skip)
        for v in reversed(skip):
            if v <= treatment_col + 1:
                warn("No need to include columns before treated columns in skip.")
                skip.remove(v)
    else:
        skip = []

    # enforce bounds on bootstraps and permutations
    if not isinstance(bootstraps, int) or bootstraps < 1:
        raise TypeError("bootstraps must be an integer greater than 0")
    if isinstance(permutations, str):
        if permutations != "all":
            raise TypeError("permutations must be 'all' or an integer greater than 0")
    elif not isinstance(permutations, int) or permutations < 1:
        raise TypeError("permutations must be 'all' or an integer greater than 0")

    # initialize and fit the bootstrapper to the data
    bootstrapper = Bootstrapper(random_state=rng, kind="indexes")
    bootstrapper.fit(data, skip=skip)

    for i in range(bootstraps):
        # get a bootstrap sample
        bootstrapped_sample = bootstrapper.transform(data, start=treatment_col + 2)

        # initialize and fit the permuter to the aggregated data
        permuter = Permuter(random_state=rng)

        if permutations == "all":
            permuter.fit(bootstrapped_sample, treatment_col, exact=True)

            # in the exact case, determine and set the total number of
            # possible permutations
            counts = np.unique(bootstrapped_sample[:, 0], return_counts=True)[1]
            permutations = _binomial(counts.sum(), counts[0])

        else:
            # just fit the permuter if this is a randomized test
            permuter.fit(bootstrapped_sample, treatment_col)

        for j in range(permutations):
            # yield a permuted sample
            yield permuter.transform(bootstrapped_sample)
