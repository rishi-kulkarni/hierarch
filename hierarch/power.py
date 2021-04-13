import numpy as np
import scipy.stats as stats



def gen_fake_data(reference, params, seed=None):
    
    rng = np.random.default_rng(seed)

    fakedata = np.copy(reference)
    fakedata = fakedata.astype('float64')
    for i in range(reference.shape[1] - 1):

        if type(params[i][0]) is not int:

            idx, replicates = np.unique(reference[:,:i+1], return_counts=True, axis=0)
            ranlist = []
            if not isinstance(params[i][0],list):
                for j in range(len(idx)):
                    ranlist.append(params[i][0].rvs(*params[i][1:], random_state=rng))
                ranlist = np.repeat(ranlist, replicates)
            else: 
                for j in range(len(idx)):
                    ranlist.append(params[i][j][0].rvs(*params[i][j][1:], random_state=rng))
                ranlist = np.repeat(ranlist, replicates)
            
            
            np.put(fakedata[:,i], np.where(fakedata[:,i]), ranlist)

        else:
            idx, replicates = np.unique(reference[:,:i+1], return_counts=True, axis=0)
            ranlist = np.repeat(params[i], replicates)
            np.put(fakedata[:,i], np.where(fakedata[:,i] > -1), ranlist)

        if i > 0: fakedata[:,i] = fakedata[:,i] + fakedata[:,i-1]    

    return fakedata[:,-2]

def make_ref_container(samples_per_level=[]):
    
    iterator = samples_per_level
    
    hcontainer = np.arange(1,iterator[0] + 1).reshape(iterator[0], 1)
    
    for i in range(1, len(iterator)):
        if type(iterator[i]) is not list: iterator[i] = [iterator[i]] * hcontainer.shape[0]
        hcontainer = np.repeat(hcontainer, iterator[i],axis=0)
        append = []
        for j in iterator[i]:
            append += list(range(1,j+1))
        hcontainer = np.append(hcontainer, np.array(append).reshape(len(append),1),axis=1)
    hcontainer = np.append(hcontainer, np.zeros_like(hcontainer[:,0]).reshape(hcontainer[:,0].shape[0],1),axis=1)
    
    return hcontainer.astype(float)