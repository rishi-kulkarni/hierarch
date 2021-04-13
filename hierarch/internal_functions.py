import numpy as np
import numba as nb
import scipy.stats as stats
from collections import Counter
import sympy.utilities.iterables as iterables

@nb.jit(nopython=True)
def nb_unique(input_data, axis=0):
    '''
    Internal function that serves the same purpose as np.unique(a, return_index=True, return_counts=True) when called on a 2D arraya. Appears to asymptotically approach np.unique's speed when every row is unique, but otherwise runs faster.
    
    Parameters
    ----------
    input_data: 2D array
    
    axis: int
        0 for unique rows, 1 for unique columns
    
    Returns
    ----------
    data[idx]: unique rows (or columns) from the input array
    
    idx: index numbers of the unique rows (or columns)
    
    counts: number of instances of each unique row (or column) in the input array 
    
    '''
    if axis == 1:
        data = input_data.T.copy()
    else:
        data = input_data.copy()
        
    for i in range(data.shape[1]-1,-1,-1):
        data = data[data[:,i].argsort(kind="mergesort")]

    idx = np.zeros(1, dtype=np.int64)
    counts = np.ones(1, dtype=np.int64)
    
    additional_uniques = np.where(~np_all_axis1(data[:-1] == data[1:]))[0] + 1
    
    idx = np.append(idx, additional_uniques)
    counts = idx[1:].copy()
    counts = np.append(counts, data.shape[0])
    counts = counts - idx
       
    return data[idx], idx, counts

@nb.jit(nopython=True)
def welch_statistic(sample_a, sample_b):
    '''
    Internal function that calculates Welch's t statistic.
    
    Parameters
    ----------
    
    sample_a, sample_b: 1D arrays
    
    Returns
    ----------
    t: float64
    
    Note: The formula for Welch's t reduces to Student's t when sample_a and sample_b are the same size, so use this function whenever you need a t statistic.
    
    '''
    meandiff = (np.mean(sample_a) - np.mean(sample_b))
    
    var_weight_one = (np.var(sample_a)*(sample_a.size/(sample_a.size - 1))) / len(sample_a)
    
    var_weight_two = (np.var(sample_b)*(sample_b.size/(sample_b.size - 1))) / len(sample_b)
    
    t = meandiff / np.sqrt(var_weight_one + var_weight_two)
    return t
        
@nb.jit()
def randomize_chunks(values, keys):
    '''
    Internal function for permuting a column a data while paying attention to the dependency structure of the prior column. Numba's implementation of np.random.permutation is faster than numpy's, so we're using this.
    
    Parameters
    ----------
    values: 2D array
        2D array of unique rows, second-to-last column is the column to be permuted
    
    keys: 2D array
        2D array of unique rows of the values array, not including the final row
        
    Returns
    ----------
    append_col: list
        List of permuted values of the second-to-last column of the values array
    
    '''
    append_col=[]
    for i in keys:
        append_col.append(np.random.permutation(values[np_all_axis1(values[:,:-2] == values[i][:-2])][:,-2]))
    return append_col

@nb.njit(cache=True)
def np_all_axis1(x):
    """
    Numba compatible version of np.all(x, axis=1)
    
    Parameters
    ----------
    x: 2D array
    
    Returns
    ----------
    out: 1D array
        Boolean array the same length as x in axis 0
    
    """
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out

@nb.jit
def nb_reindexer(i, resampled, data, columns_to_resample, unique_idx_list, randnos, rng_place):
    '''
    Internal function for numba-accelerated iterating through a numpy array. 
    
    
    '''
    idx = np.empty(0, dtype=np.int64)

    for key in resampled[:,:i]:
        
        idx_no = np.where(np_all_axis1(data[unique_idx_list][:,:i] == key))[0]
        
        if not columns_to_resample[i]:
            idx = np.hstack((idx, idx_no))
        else:
            idx_no = idx_no[randnos[rng_place:rng_place+idx_no.size] % idx_no.size]
            rng_place += idx_no.size
            idx = np.hstack((idx,idx_no))

    return idx, rng_place

def permute_column(data, col_to_permute=-2, iterator=None):

    """
    This function takes column n and permutes the column n - 1 while accounting for the clustering in column n - 2. This function is memoized based on the hash of the data and col_to_permute variable, which improves performance significantly. 
    
    Parameters
    ----------
    data: arrays
        The numpy array that contains the data of interest.
        
    col_to_permute: int
        Column n, which is immediately right of the column that will be shuffled. 
    
    iterator: an iterator representing the multiset of permutations of column n - 1
        In very small samples (n = 3 or 4), it is worthwhile to iterate through every permutation rather than hoping to randomly sample all of them. To use this, construct a multiset of permutations and iterate through using a for loop. Even at n = 5, it's faster to generate the multiset and randomly choose 100 elements from it than it is to approach it randomly. 
        
    Returns
    ---------
    permuted: an array the same size as data with column n - 1 permuted within column n - 2's clusters.
    
    
    
    
    """
    
    key = hash(tuple((data[:,:col_to_permute].tobytes(),col_to_permute)))
    
    try:
        values, indexes, counts = permute_column.__dict__[key]
    except:
        values, indexes, counts = np.unique(data[:,:col_to_permute+1],return_index=True, return_counts=True,axis=0)   
        permute_column.__dict__[key] = values, indexes, counts
        if len(permute_column.__dict__.keys()) > 50:
            permute_column.__dict__.pop(list(permute_column.__dict__)[0])
    

    if iterator == None:
        try:
            keys = unique_idx_w_cache(values)[-2]
            append_col = randomize_chunks(values, keys)        
            shuffled_col_values = np.concatenate(append_col)
            
        except:
    
            pre_col_values = data[:,col_to_permute-1][indexes]
            shuffled_col_values = np.random.permutation(pre_col_values)
    else:
            shuffled_col_values = iterator
    new_col = np.repeat(shuffled_col_values, counts)
    permuted = np.copy(data)
    permuted[:,col_to_permute-1] = new_col
    return permuted
    
def bootstrap_agg(bootstrap_sample, func=np.nanmean, agg_to=-2, first_data_col=-1):
    
    '''
    Groupby_aggregate function, but slower than mean_agg below. Probably not worth calling in most cases unless you have a strange aggregation to do, but you will probably save a lot of time in the end by jitting your aggregation in numba. 
    
    '''
    
    iterations = first_data_col - agg_to
    for i in range(iterations):
        append_col = np.empty(0)
        for key in np.unique(bootstrap_sample[:,first_data_col - 1]):
            append_col = np.append(append_col, func(bootstrap_sample[bootstrap_sample[:, first_data_col - 1] == key][:,first_data_col]))
        append_col = append_col.reshape(append_col.size,1)
        bootstrap_sample = bootstrap_sample[:,:-1]
        idx = np.unique(bootstrap_sample, axis=0, return_index=True)[1]
        bootstrap_sample = bootstrap_sample[np.sort(idx)]
        bootstrap_sample = bootstrap_sample[:,:-1]
        bootstrap_sample = np.append(bootstrap_sample, append_col,axis=1)
    return bootstrap_sample

def mean_agg(data, groupby=-3):

    """
    Performs a "groupby" aggregation by taking the mean. Much better performance than bootstrap_agg above, but can only be used for quantities that can be calculated element-wise (such as mean.)
    
    Parameters
    ----------
    data: array
        Data to perform operation on. 
    
    groupby = int
        Column to groupby. The default of -3 assumes the last column is values and the second-to-last column is some kind of categorical label (technical replicate 1, 2, etc.)
        
    Returns
    ----------
    
    permute: array
        A reduced array such that the labels in column groupby (now column index -2) are no longer duplicated and column index -1 contains averaged data values. 
    
    
    """   
    
    
    key = hash(data[:,:groupby+1].tobytes())
    
    try:
        unique_idx, unique_counts = mean_agg.__dict__[key]
    except:
        unique_idx, unique_counts = np.unique(data[:,:groupby+1], return_index=True, return_counts=True,axis=0)[1:]
        mean_agg.__dict__[key] = unique_idx, unique_counts
        if len(mean_agg.__dict__.keys()) > 50:
            mean_agg.__dict__.pop(list(mean_agg.__dict__)[0])
    
    
    avgcol = np.add.reduceat(data[:,-1], unique_idx) / unique_counts
    permute = data[unique_idx][:,:-1]
    permute[:,-1] = avgcol
    
    
    return permute



def bootstrap_sample(data, start=0, data_col=-1, skip=[], seed=None):
    
    '''
    Performs a numba-accelerated multi-level bootstrap of input data. 
    
    Parameters
    ----------
    data: 2D array
        The input array containing your data in the final (or more) columns and categorical variables classifying the data in every prior column. Each column is resampled based on the column prior, so make sure your column ordering reflects that.
    
    start: int
        This is the first column corresponding to a level that you want to resample. Note: this column won't be resampled, but the next one will be resampled based on this column.
        
    data_col: int
        This is the first column that has your measured values (and therefore shouldn't be resampled). Default assumes you have a single column of measured values.
        
    skip: list of ints
        Column indices provided here will be sampled WITHOUT replacement. Ideally, you should skip columns that do not represent randomly sampled data (i.e. rather than having a random sample from that level, you have all the data).
        
    seed: int or numpy.random.Generator object
        Enables seeding of the random resampling for reproducibility. The function runs much faster in a loop if it does not have to initialize a generator every time, so passing it something is good for performance.
        
    Returns
    ----------
    resampled: 2D array
        Data array the same number of columns as data, might be longer or shorter if your experimental data is imbalanced.
    
    
    '''
    
    rng = np.random.default_rng(seed)

    if data_col < 0:
        shape = data.shape[1] + data_col
    else:
        shape = data_col - 1
        
    columns_to_resample = np.array([True for k in range(shape)])
    for key in skip:
        columns_to_resample[key] = False


    randnos = rng.integers(low=2**32,size=data[:,:data_col].size)


    unique_idx_list = unique_idx_w_cache(data)
    rng_place=0

    resampled = data[unique_idx_list[start]].copy()
        
    for i in range(start+1, shape):
        
        idx, rng_place = nb_reindexer(i, resampled, data, columns_to_resample, unique_idx_list[i], randnos, rng_place)
        
        idx = unique_idx_list[i][idx]
        
        resampled = data[idx]
    
    return resampled



def relabel_col(resample_data, original_data, col):
    
    """
    Relabels in-place duplicate clusters for later calculations that require aggregation based on cluster. This function requires input of the raw (unresampled) data to determine how large each cluster should be. 

    Parameters
    ----------
    
    resample_data, original_data: arrays (dtype='float')
        The column to relabel must have the same column index in both data sets. 
        
    col: int
        Column index of the column to relabel
        
    Returns
    ----------
    
    Modifies resample_data in-place by adding multiples of 0.01 to repeated clusters. If the test statistic of interest does not require aggregation by cluster, this function is not doing much aside from adding a little overhead. 
    
    """
    
    a = resample_data[:,col]
    b = original_data[:,col]
    
    unique, counts = np.unique(a, return_counts=True)
    ca = dict(zip(unique, counts))
    ca = Counter(ca)
    
    unique, counts = np.unique(b, return_counts=True)
    cb = dict(zip(unique, counts))
    cb = Counter(cb)

    
    relabel = 0.01

    while any((ca - cb) & cb):
        idx=[]
        for key in (ca - cb) & cb:
            idx.append(list(np.where(a == key)[0][-(cb)[key]:]))

        np.add.at(resample_data[:,col], np.concatenate(idx), relabel)
        ca = Counter(resample_data[:,col])
        relabel += 0.01

        
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

    Reference: "An O(1) Time Algorithm for Generating Multiset Permutations", Tadao Takaoka
    https://pdfs.semanticscholar.org/83b2/6f222e8648a7a0599309a40af21837a0264b.pdf
    
    Taken from @smichr 
    '''

    def visit(head):
        (rv, j) = ([], head)
        for i in range(N):
            (dat, j) = E[j]
            rv.append(dat)
        return rv

    u = list(set(items))
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
    Just np.unique(return_index=True, axis=0) with memoization, as np.unique is called a LOT in this package. Numpy arrays are not hashable, so this hashes the bytes of the array instead.
    
    
    '''
    
    key = hash(data.tobytes())
    
    try:
        unique_lists = unique_idx_w_cache.__dict__[key]
        return unique_lists
    except:
        unique_lists = []
        for i in range(0, data.shape[1] - 1):
            unique_lists += [np.unique(data[:,:i+1],return_index=True,axis=0)[1]]
        unique_idx_w_cache.__dict__[key] = unique_lists
        if len(unique_idx_w_cache.__dict__.keys()) > 50:
            unique_idx_w_cache.__dict__.pop(list(unique_idx_w_cache.__dict__)[0])
        
    
        return unique_lists        



