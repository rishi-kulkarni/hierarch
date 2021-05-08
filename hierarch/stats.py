import numpy as np
import math
import hierarch.internal_functions as internal_functions
from hierarch.internal_functions import GroupbyMean
import hierarch.resampling as resampling
import sympy.utilities.iterables as iterables



def two_sample_test(data_array, treatment_col, teststat="welch", skip=[], bootstraps=100, permutations=1000, return_null=False, seed=None):
    
    '''
    Two-tailed two-sample hierarchical permutation test.
    
    Parameters
    -----------
    data_array: 2D array or pandas DataFrame
        Array-like containing both the independent and dependent variables to be analyzed. It's assumed that the final (rightmost) column contains the dependent variable values.
        
    treatment_col: int
        The index number of the column containing "two samples" to be compared. Indexing starts at 0.
        
    teststat: function or string
        The test statistic to use to perform the hypothesis test. "Welch" automatically calls the Welch t-statistic for a difference of means test.
        
    skip: list of ints
        Columns to skip in the bootstrap. Skip columns that were sampled without replacement from the prior column. 
        
    bootstraps: int
        Number of bootstraps to perform.
        
    permutations: int or "all"
        Number of permutations to perform PER bootstrap sample. "all" for exact test.
        
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
    
    #turns the input array or dataframe into a float64 array
    data = internal_functions.preprocess_data(data_array)
    
    #set random state
    rng = np.random.default_rng(seed)
    
    #initialize and fit the bootstrapper to the data
    bootstrapper = resampling.Bootstrapper(random_state=rng)
    bootstrapper.fit(data, skip)
    
    #gather labels 
    treatment_labels = np.unique(data[:,treatment_col])
    
    #raise an exception if there are more than two treatment labels
    if treatment_labels.size != 2:
        raise Exception("Needs 2 samples.")        
    
    #shorthand for welch_statistic
    if teststat == "welch":
        teststat = internal_functions.welch_statistic
         
    
    #aggregate our data up to the treated level and determine the observed test statistic
    aggregator = GroupbyMean()
    aggregator.fit(data)
    
    levels_to_agg = data.shape[1] - treatment_col - 3
    test = data
    
    test = aggregator.transform(test, iterations=levels_to_agg)

    truediff = teststat(test, treatment_col, treatment_labels)

    #initialize and fit the permuter to the aggregated data
    permuter = resampling.Permuter()
    
    if permutations == "all":
        permuter.fit(test, treatment_col+1, exact=True)
        
        #in the exact case, determine and set the total number of possible permutations 
        counts = np.unique(test[:,0], return_counts=True)[1]
        permutations = binomial(counts.sum(), counts[0])
        
    else:
        #just fit the permuter if this is a randomized test
        permuter.fit(test, treatment_col+1)
        
    #initialize empty null distribution list
    null_distribution = []
   
    for j in range(bootstraps):
        #generate a bootstrapped sample and aggregate it up to the treated level
        bootstrapped_sample = bootstrapper.transform(data, start=treatment_col+1)
        bootstrapped_sample = aggregator.transform(bootstrapped_sample, iterations=levels_to_agg)
        
        #generate permuted samples, calculate test statistic, append to null distribution
        for k in range(permutations):
            permute_resample = permuter.transform(bootstrapped_sample)
            null_distribution.append(teststat(permute_resample, treatment_col, treatment_labels))

    #two tailed test, so check where absolute values of the null distribution are greater or equal to the absolute value of the observed difference
    pval = np.where((np.array(np.abs(null_distribution)) >= np.abs(truediff)))[0].size / len(null_distribution)
    
    
    if return_null==True:
        return pval, null_distribution
    
    else:
        return pval

def two_sample_test_jackknife(data, treatment_col, permutations='all', teststat='welch'):
    
    treatment_labels = np.unique(data[:,treatment_col])
    if treatment_labels.size != 2:
        raise Exception("Needs 2 samples.")
        
    if teststat == "welch":
        teststat = internal_functions.welch_statistic
        
    means = []

    levels_to_agg = data.shape[1] - treatment_col - 3
    test = internal_functions.mean_agg(data)
    for m in range(levels_to_agg - 1):
        test = internal_functions.mean_agg(test)    

    truediff = np.abs(teststat(test[test[:,treatment_col] == treatment_labels[0]][:,-1], test[test[:,treatment_col] == treatment_labels[1]][:,-1]))

    pre_colu_values = data[:,0][internal_functions.nb_unique(data[:,:2])[1]]
    it_list = list(internal_functions.msp(pre_colu_values))

    for indexes in iterables.cartes(*np.split(internal_functions.nb_unique(data[:,:-1])[1], internal_functions.nb_unique(data[:,:-2])[1])[1:]):
        jacknifed = np.delete(data, indexes, axis=0)
        jacknifed = internal_functions.mean_agg(jacknifed)

        for shuffle in it_list:
            permute_resample = internal_functions.permute_column(jacknifed, 1, shuffle)
            means.append(teststat(permute_resample[permute_resample[:,treatment_col] == treatment_labels[0]][:,-1], permute_resample[permute_resample[:,treatment_col] == treatment_labels[1]][:,-1]))

    pval = np.where((np.array(np.abs(means)) >= np.abs(truediff)))[0].size / len(means)
    
    return pval

def binomial(x, y):
    try:
        return math.factorial(x) // math.factorial(y) // math.factorial(x - y)
    except ValueError:
        return 0