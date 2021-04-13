import numpy as np
import hierarch.internal_functions as internal_functions
import sympy.utilities.iterables as iterables



def two_sample_test(data, treatment_col, teststat="welch", skip=[], bootstraps=500, permutations=100, seed=None):

    treatment_labels = np.unique(data[:,treatment_col])
    
    if treatment_labels.size != 2:
        raise Exception("Needs 2 samples.")
    
    rng = np.random.default_rng(seed)
    
    if teststat == "welch":
        teststat = internal_functions.welch_statistic

    if permutations == "all":
        #determine total number of possible level 0 permutations 
        indexes = np.unique(data[:,:treatment_col+2],return_index=True,axis=0)[1]
        pre_col_values = data[:,treatment_col][indexes]
        it_list = list(internal_functions.msp(pre_col_values))

    levels_to_agg = data.shape[1] - treatment_col - 3
    test = internal_functions.mean_agg(data)
    for m in range(levels_to_agg - 1):
        test = internal_functions.mean_agg(test)
    
    truediff = np.abs(teststat(test[test[:,treatment_col] == treatment_labels[0]][:,-1], test[test[:,treatment_col] == treatment_labels[1]][:,-1]))


    means = []
    for j in range(bootstraps):

        #resample level 1 data from level 2s, using our generator for reproducible rng

        bootstrapped_sample = internal_functions.bootstrap_sample(data, start=treatment_col+1, skip=skip, seed=rng)
        for m in range(levels_to_agg):
            bootstrapped_sample = internal_functions.mean_agg(bootstrapped_sample)




        if permutations == "all":
            #we are sampling all 20 permutations, so no need for rng. 
            for k in it_list:
                permute_resample = internal_functions.permute_column(bootstrapped_sample, treatment_col+1, k)
                means.append(teststat(permute_resample[permute_resample[:,treatment_col] == treatment_labels[0]][:,-1], permute_resample[permute_resample[:,treatment_col] == treatment_labels[1]][:,-1]))
        
        else:
            for k in range(permutations):
                permute_resample = internal_functions.permute_column(bootstrapped_sample, treatment_col+1)
                means.append(teststat(permute_resample[permute_resample[:,treatment_col] == treatment_labels[0]][:,-1], permute_resample[permute_resample[:,treatment_col] == treatment_labels[1]][:,-1]))

    pval = np.where((np.array(np.abs(means)) >= truediff))[0].size / len(means)
    
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