import numpy as np
import math
from itertools import combinations
import pandas as pd
from hierarch.internal_functions import welch_statistic, preprocess_data, GroupbyMean
from hierarch.resampling import Bootstrapper, Permuter
from warnings import warn, simplefilter

TEST_STATISTICS = {
    "means": welch_statistic,
}


def two_sample_test(
    data_array,
    treatment_col: int,
    compare="means",
    skip=None,
    bootstraps=100,
    permutations=1000,
    kind="weights",
    return_null=False,
    seed=None,
):
    """Two-tailed two-sample hierarchical permutation test.

    Parameters
    ----------
    data_array : 2D numpy array or pandas DataFrame
        Array-like containing both the independent and dependent variables to
        be analyzed. It's assumed that the final (rightmost) column
        contains the dependent variable values.
    treatment_col : int
        The index number of the column containing "two samples" to be compared.
        Indexing starts at 0.
    compare : str, optional
        The test statistic to use to perform the hypothesis test. "means" automatically 
        calls the Welch t-statistic for a difference of means test, by default "means"
    skip : list of ints, optional
        Columns to skip in the bootstrap. Skip columns that were sampled
        without replacement from the prior column, by default None
    bootstraps : int, optional
        Number of bootstraps to perform, by default 100. Can be set to 1 for a
        permutation test without any bootstrapping.
    permutations : int or "all", optional
        Number of permutations to perform PER bootstrap sample. "all"
        for exact test, by default 1000
    kind : str, optional
        Bootstrap algorithm - see Bootstrapper class, by default "weights"
    return_null : bool, optional
        Return the null distribution as well as the p value, by default False
    seed : int or numpy random Generator, optional
        Seedable for reproducibility., by default None

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
    IndexError
        Raised if skip index is out of bounds for the array or not an integer.
    ValueError
        Raised if treatment_col has more than two different labels in it.
    KeyError
        If comparison is a string, it must be in the TEST_STATISTICS dictionary.
    AttributeError
        If comparison is a custom statistic, it must be a function.
    """

    # turns the input array or dataframe into a float64 array
    if isinstance(data_array, (np.ndarray, pd.DataFrame)):
        data = preprocess_data(data_array)
    else:
        raise TypeError("Input data must be ndarray or DataFrame.")

    # set random state
    rng = np.random.default_rng(seed)

    # enforce lower bound on skip
    if skip is not None:
        skip = list(skip)
        for v in reversed(skip):
            if v <= treatment_col + 1:
                warn("No need to include columns before treated columns in skip.")
                skip.remove(v)
    else:
        skip = []

    # initialize and fit the bootstrapper to the data
    bootstrapper = Bootstrapper(random_state=rng, kind=kind)
    bootstrapper.fit(data, skip=skip)

    # gather labels and raise an exception if there are more than two
    try:
        treatment_labels = np.unique(data[:, treatment_col])
    except IndexError:
        print("treatment_col must be an integer")
        raise
    if treatment_labels.size != 2:
        raise ValueError("Needs 2 samples.")

    # fetch test statistic from dictionary or, if given a custom test statistic, make sure it is callable
    if isinstance(compare, str):
        try:
            teststat = TEST_STATISTICS[compare]
        except KeyError:
            print(
                "Invalid comparison. Available comparisons are: "
                + "".join(stat for stat in TEST_STATISTICS.keys())
            )
            raise
    elif not callable(compare):
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
    truediff = teststat(test, treatment_col, treatment_labels)

    # initialize and fit the permuter to the aggregated data
    # don't need to seed this, as numba's PRNG state is shared
    permuter = Permuter()

    if permutations == "all":
        permuter.fit(test, treatment_col, exact=True)

        # in the exact case, determine and set the total number of
        # possible permutations
        counts = np.unique(test[:, 0], return_counts=True)[1]
        permutations = _binomial(counts.sum(), counts[0])

    else:
        # just fit the permuter if this is a randomized test
        permuter.fit(test, treatment_col)

    # skip the dot on the permute function
    call_permute = permuter.transform

    # initialize empty null distribution list
    null_distribution = []

    # first set of permutations is on the original data
    # this helps to prevent getting a p-value of 0
    for k in range(permutations):
        permute_resample = call_permute(test, treatment_col)
        null_distribution.append(
            teststat(permute_resample, treatment_col, treatment_labels)
        )

    # already did one set of permutations
    bootstraps -= 1

    for j in range(bootstraps):
        # generate a bootstrapped sample and aggregate it up to the
        # treated level
        bootstrapped_sample = bootstrapper.transform(data, start=treatment_col + 2)
        bootstrapped_sample = aggregator.transform(
            bootstrapped_sample, iterations=levels_to_agg
        )

        # generate permuted samples, calculate test statistic,
        # append to null distribution

        for k in range(permutations):
            permute_resample = call_permute(bootstrapped_sample, treatment_col)
            null_distribution.append(
                teststat(permute_resample, treatment_col, treatment_labels)
            )

    # two tailed test, so check where absolute values of the null distribution
    # are greater or equal to the absolute value of the observed difference
    pval = np.where((np.array(np.abs(null_distribution)) >= np.abs(truediff)))[
        0
    ].size / len(null_distribution)

    if return_null is True:
        return pval, null_distribution

    else:
        return pval


def multi_sample_test(
    data_array,
    treatment_col: int,
    hypotheses="all",
    correction="fdr",
    compare="means",
    skip=None,
    bootstraps=100,
    permutations=1000,
    kind="weights",
    seed=None,
):
    """Two-tailed multiple-sample hierarchical permutation test.

    Equivalent to a post-hoc test after ANOVA. Results are more interpetable when the input
    data is in the form of a pandas dataframe or numpy object array.

    Parameters
    ----------
    data_array : 2D array or pandas DataFrame
        Array-like containing both the independent and dependent variables to
        be analyzed. It's assumed that the final (rightmost) column
        contains the dependent variable values.
    treatment_col : int
        The index number of the column containing labels to be compared.
        Indexing starts at 0.
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
        numpy ndarray with col 0, 1 corresponding to treatment labels, col 2 corresponding to an uncorrected p-value, 
        and col 3 corresponding to a corrected p-value if a correction was specified.

    Raises
    ------
    KeyError
        Raised if passed correction is not valid.
    TypeError
        Raised if input data is not ndarray or DataFrame.
    KeyError
        Raised if specified comparison labels do not exist in the input data.
    """

    MULTIPLE_COMPARISONS_CORRECTIONS = {
        "fdr": _false_discovery_adjust,
    }
    if correction is not None:
        try:
            multiple_correction = MULTIPLE_COMPARISONS_CORRECTIONS[correction]
        except KeyError:
            print(correction + " is not a valid multiple comparisons correction.")
            raise

    seed = np.random.default_rng(seed)

    # coerce data into an object array
    if isinstance(data_array, pd.DataFrame):
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
            (data[:, 1] == output[i, 0]), (data[:, 1] == output[i, 1])
        )
        output[i, 2] = two_sample_test(
            data[test_idx],
            treatment_col=treatment_col,
            compare=compare,
            skip=skip,
            bootstraps=bootstraps,
            permutations=permutations,
            kind=kind,
            seed=seed,
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

    """

    # argsort so we can sort a list of hypotheses, if need be
    sort_key = np.argsort(pvals)
    # q-value adjustment
    q_vals = np.array(pvals)[sort_key] * len(pvals)
    q_vals /= np.array(range(1, len(pvals) + 1))

    # list of q values must be strictly non-decreasing
    for i in range(len(q_vals) - 1, 0, -1):
        if q_vals[i] < q_vals[i - 1]:
            q_vals[i - 1] = q_vals[i]

    if return_index is True:
        return q_vals, sort_key
    else:
        return q_vals
