import numpy as np
import math
from itertools import combinations
import pandas as pd
from hierarch.internal_functions import GroupbyMean
from hierarch.internal_functions import welch_statistic, preprocess_data
from hierarch.resampling import Bootstrapper, Permuter
from warnings import warn, simplefilter


def two_sample_test(data_array, treatment_col, teststat="welch", skip=[],
                    bootstraps=100, permutations=1000, kind='weights',
                    return_null=False, seed=None):

    '''
    Two-tailed two-sample hierarchical permutation test.

    Parameters
    -----------
    data_array: 2D array or pandas DataFrame
        Array-like containing both the independent and dependent variables to
        be analyzed. It's assumed that the final (rightmost) column
        contains the dependent variable values.

    treatment_col: int
        The index number of the column containing "two samples" to be compared.
        Indexing starts at 0.

    teststat: function or string
        The test statistic to use to perform the hypothesis test. "Welch"
        automatically calls the Welch t-statistic for a
        difference of means test.

    skip: list of ints
        Columns to skip in the bootstrap. Skip columns that were sampled
        without replacement from the prior column.

    bootstraps: int
        Number of bootstraps to perform.

    permutations: int or "all"
        Number of permutations to perform PER bootstrap sample. "all"
        for exact test.

    kind: str = "weights" or "bayesian" or "indexes"
        Specifies the bootstrapping algorithm. See Bootstrapper class
        for details.

    return_null: bool
        Set to true to return the null distribution as well as the p value.

    seed: int or numpy random Generator
        Seedable for reproducibility.

    Returns
    ---------
    pval: float64
        Two-tailed p-value.

    null_distribution: list of floats
        The empirical null distribution used to calculate the p-value.

    '''

    # turns the input array or dataframe into a float64 array
    data = preprocess_data(data_array)

    # set random state
    rng = np.random.default_rng(seed)

    # enforce bounds on skip
    for v in reversed(skip):
        if v <= treatment_col+1:
            warn('No need to include columns before treated columns in skip.')
            skip.remove(v)
        if v >= data.shape[1] - 1:
            raise IndexError('skip index out of bounds for this array.')

    # initialize and fit the bootstrapper to the data
    bootstrapper = Bootstrapper(random_state=rng, kind=kind)
    bootstrapper.fit(data, skip)

    # gather labels
    treatment_labels = np.unique(data[:, treatment_col])

    # raise an exception if there are more than two treatment labels
    if treatment_labels.size != 2:
        raise Exception("Needs 2 samples.")

    # shorthand for welch_statistic
    if teststat == "welch":
        teststat = welch_statistic

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
        warn('No levels to bootstrap. Setting bootstraps to zero.')

    test = data
    test = aggregator.transform(test, iterations=levels_to_agg)
    truediff = teststat(test, treatment_col, treatment_labels)

    # initialize and fit the permuter to the aggregated data
    # don't need to seed this, as numba's PRNG state is shared
    permuter = Permuter()

    if permutations == "all":
        permuter.fit(test, treatment_col+1, exact=True)

        # in the exact case, determine and set the total number of
        # possible permutations
        counts = np.unique(test[:, 0], return_counts=True)[1]
        permutations = binomial(counts.sum(), counts[0])

    else:
        # just fit the permuter if this is a randomized test
        permuter.fit(test, treatment_col+1)

    # initialize empty null distribution list
    null_distribution = []

    # first set of permutations is on the original data
    # this helps to prevent getting a p-value of 0
    for k in range(permutations):
        permute_resample = permuter.transform(test)
        null_distribution.append(teststat(permute_resample,
                                          treatment_col,
                                          treatment_labels))

    # already did one set of permutations
    bootstraps -= 1

    for j in range(bootstraps):
        # generate a bootstrapped sample and aggregate it up to the
        # treated level
        bootstrapped_sample = bootstrapper.transform(data,
                                                     start=treatment_col+2)
        bootstrapped_sample = aggregator.transform(bootstrapped_sample,
                                                   iterations=levels_to_agg)

        # generate permuted samples, calculate test statistic,
        # append to null distribution
        for k in range(permutations):
            permute_resample = permuter.transform(bootstrapped_sample)
            null_distribution.append(teststat(permute_resample,
                                     treatment_col, treatment_labels))

    # two tailed test, so check where absolute values of the null distribution
    # are greater or equal to the absolute value of the observed difference
    pval = np.where((np.array(np.abs(null_distribution)) >=
                     np.abs(truediff)))[0].size / len(null_distribution)

    if return_null is True:
        return pval, null_distribution

    else:
        return pval


def multi_sample_test(data_array, treatment_col, hypotheses='all',
                      correction='fdr', teststat='welch', skip=[],
                      bootstraps=100, permutations=1000, kind='weights',
                      seed=None):
    '''
    Two-tailed multiple-sample hierarchical permutation test. Equivalent to a
    post-hoc test after ANOVA. Results are more interpetable when the input
    data is in the form of a pandas dataframe.

    Parameters
    -----------
    data_array: 2D array or pandas DataFrame
        Array-like containing both the independent and dependent variables to
        be analyzed. It's assumed that the final (rightmost) column
        contains the dependent variable values.

    treatment_col: int
        The index number of the column containing labels to be compared.
        Indexing starts at 0.

    hypotheses: 'all' or list of two-element lists
        Hypotheses to be tested. If 'all' every pairwise comparison will be
        tested. Can be passed a list of lists to restrict comparisons, which
        will result in a less harsh multiple comparisons correction.

    correction: 'fdr' or 'none'
        Multiple comparisons question to be performed after p-values are
        calculated. 'fdr' performs the Benjamini-Hochberg procedure for
        controlling False Discovery Rate.

    teststat: function or string
        The test statistic to use to perform the hypothesis test. "Welch"
        automatically calls the Welch t-statistic for a
        difference of means test.

    skip: list of ints
        Columns to skip in the bootstrap. Skip columns that were sampled
        without replacement from the prior column.

    bootstraps: int
        Number of bootstraps to perform.

    permutations: int or "all"
        Number of permutations to perform PER bootstrap sample. "all"
        for exact test.

    kind: str = "weights" or "bayesian" or "indexes"
        Specifies the bootstrapping algorithm. See Bootstrapper class
        for details.

    seed: int or numpy random Generator
        Seedable for reproducibility.

    Returns
    ---------
    pval: float64
        Two-tailed p-value.

    '''
    seed = np.random.default_rng(seed)

    # if list of comparisons has been provided, make an array for output
    if isinstance(hypotheses, list):
        hypotheses = np.array(hypotheses, dtype='object')
        output = np.empty((hypotheses.shape[0], hypotheses.shape[1] + 1),
                          dtype="object")
        output[:, :-1] = hypotheses
    # otherwise, enumerate all possible comparisons and make output array
    else:
        output = _get_comparisons(data_array, treatment_col)

    # coerce data into an object array
    if isinstance(data_array, pd.DataFrame):
        data = data_array.to_numpy()
    else:
        data = data_array

    # perform a two_sample_test for each comparison
    # no option to return null distributions because that would be a hassle
    for i in range(len(output)):
        test_idx = np.logical_or((data[:, 1] == output[i, 0]),
                                 (data[:, 1] == output[i, 1]))
        output[i, 2] = two_sample_test(data[test_idx], treatment_col,
                                       teststat, skip, bootstraps,
                                       permutations, kind, seed=seed)

    # sort the output array so that smallest p-values are on top
    ordered_idx = output[:, -1].argsort()
    output = output[ordered_idx]

    # perform multiple comparisons correction, if any
    if correction == 'fdr':
        q_vals = _false_discovery_adjust(output[:, -1])
        out = np.empty((output.shape[0], output.shape[1] + 1), dtype="object")
        out[:, :-1] = output
        out[:, -1] = q_vals
        output = out

    return output


def _get_comparisons(data, treatment_col):
    '''
    Generates a list of pairwise comparisons for a k-sample test.

    Parameters
    ----------
    data: 2D array or pd.DataFrame

    treatment_col: int
        The column of interest

    Returns
    ----------
    comparisons: list of lists
        list of two-member lists containing each comparison

    '''
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    comparisons = []
    for i, j in combinations(np.unique(data[:, treatment_col]), 2):
        comparisons.append([i, j])
    comparisons = np.array(comparisons, dtype='object')
    out = np.empty((comparisons.shape[0], comparisons.shape[1] + 1),
                   dtype="object")
    out[:, :-1] = comparisons
    return out


def binomial(x, y):
    try:
        return math.factorial(x) // math.factorial(y) // math.factorial(x - y)
    except ValueError:
        return 0


def _false_discovery_adjust(pvals, return_index=False):
    '''
    Performs the Benjamini-Hochberg method to control false discovery rate.

    Parameters
    ----------
    pvals: 1D array-like
        p-values to be adjusted

    return_index: bool, default=False
        If true, will return the indices to sort the original p-value list.

    Returns
    ----------
    q_vals: 1D array
        q-values, or "adjusted" p-values.

    sort_key: 1D array
        indices to sort pvals

    Notes
    ----------
    Refererence: Benjamini, Y. & Hochberg, Y. Controlling the false discovery
    rate: a practical and powerful approach to multiple testing.
    Journal of the Royal Statistical Society.
    Series B (Methodological) 289–300 (1995).

    The q-values, or "adjusted p-values," are not really p-values and should
    not be interpreted as such. Rather, each q-value is the minimum FDR you
    must accept to regard the result of that hypothesis test significant.
    However, q-values are often called adjusted p-values in practice, so
    we do so here.
    '''
    # argsort so we can sort a list of hypotheses, if need be
    sort_key = np.argsort(pvals)
    # q-value adjustment
    q_vals = np.array(pvals)[sort_key] * len(pvals)
    q_vals /= np.array(range(1, len(pvals)+1))

    # list of q values must be strictly non-decreasing
    for i in range(len(q_vals)-1, 0, -1):
        if q_vals[i] < q_vals[i-1]:
            q_vals[i-1] = q_vals[i]

    if return_index is True:
        return q_vals, sort_key
    else:
        return q_vals
