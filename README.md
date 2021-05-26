# hierarch

## A Hierarchical Resampling Package for Python

Version 0.2.1

hierarch is a package for hierarchical resampling (bootstrapping, permutation) of datasets in Python. Because for loops are ultimately intrinsic to cluster-aware resampling, hierarch uses Numba to accelerate many of its key functions.

hierarch has several functions to assist in performing resampling-based hypothesis tests on hierarchical data. Additionally, hierarch can be used to construct power analyses for hierarchical experimental designs. 

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Using hierarch](#using-hierarch)
    1. [Importing Data](#import-data)
    2. [Hierarchical Hypothesis Testing](#hypothesis-testing)
    3. [Sample Size Determination via Power Analysis](#power-analysis)

<a name="introduction"></a>
## Introduction 

Design-based randomization tests represents the platinum standard for significance analyses [[1, 2, 3]](#1) - that is, they produce probability statements that depend only on the experimental design, not at all on less-than-verifiable assumptions about the probability distributions of the data-generating process. Researchers can use hierarch to quickly perform automated design-based randomization tests for experiments with arbitrary levels of hierarchy.


<a id="1">[1]</a> Tukey, J.W. (1993). Tightening the Clinical Trial. Controlled Clinical Trials, 14(4), 266-285.

<a id="1">[2]</a> Millard, S.P., Krause, A. (2001). Applied Statistics in the Pharmaceutical Industry. Springer.

<a id="1">[3]</a> Berger, V.W. (2000). Pros and cons of permutation tests in clinical trials. Statistics in Medicine, 19(10), 1319-1328.


<a name="setup"></a>
## Setup 

### Dependencies
* numpy
* pandas (for importing data)
* numba
* scipy (for power analysis)
* sympy (for jackknifing)

### Installation

The easiest way to install hierarch is via PyPi. 

```pip install hierarch```

Alternatively, you can install from Anaconda.

```conda install -c rkulk111 hierarch```


<a name="using-hierarch"></a>
## Using hierarch 

<a name="import-data"></a>
### Importing Data

Hierarch is compatible with pandas DataFrames and numpy arrays. Pandas is capable of conveniently importing data from a wide variety of formats, including Excel files.

```python
import pandas as pd

data = pd.read_excel(filepath)
```

<a name="hypothesis-testing"></a>
### Hierarchical Hypothesis Testing 

Performing a hierarchical permutation test for difference of means is simple. Consider an imaging experiment with two treatment groups, three coverslips in each group, and three images (fields of view) within each coverslip. If you have the data stored in an Excel file, you can use pandas to either directly read the file or copy it in from the clipboard, as below.

```python
import pandas as pd
import numpy as np
import hierarch as ha

data = pd.read_clipboard()

print(data)
```
|  Condition | Coverslip | Field of View |  Mean Fluorescence  |
|:----------:|:-----------------:|:-------------------:|:--------:|
|    Control    |         1         |          1          | 5.202258 |
|    Control    |         1         |          2          | 5.136128 |
|    Control    |         1         |          3          | 5.231401 |
|    Control    |         2         |          1          | 5.336643 |
|    Control    |         2         |          2          | 5.287973 |
|    Control    |         2         |          3          | 5.375359 |
|    Control    |         3         |          1          | 5.350692 |
|    Control    |         3         |          2          | 5.465206 |
|    Control    |         3         |          3          | 5.422602 |
| +Treatment |         4         |          1          | 5.695427 |
| +Treatment |         4         |          2          | 5.668457 |
| +Treatment |         4         |          3          | 5.752592 |
| +Treatment |         5         |          1          | 5.583562 |
| +Treatment |         5         |          2          | 5.647895 |
| +Treatment |         5         |          3          | 5.618315 |
| +Treatment |         6         |          1          | 5.642983 |
| +Treatment |         6         |          2          |  5.47072 |
| +Treatment |         6         |          3          | 5.486654 |


It is important to note that the ordering of the columns from left to right reflects the experimental design scheme. This is necessary for hierarch to infer the clustering within your dataset. In case your data is not ordered properly, pandas makes it easy enough to index your data in the correct order.

```python
columns = ['Condition', 'Coverslip', 'Field of View', 'Mean Fluorescence']

data[columns]
```
Next, you can call two_sample_test from hierarch's stats module, which will calculate the p-value. You have to specify what column is the treatment column - in this case, "Condition." Indexing starts at 0, so you input treatment_col=0. In this case, there are only 6c3 = 20 ways to permute the treatment labels, so you should specify "all" permutations be used.

```python
p_val = ha.stats.two_sample_test(data, treatment_col=0, bootstraps=500, permutations='all', random_state=1)

print('p-value =', p_val)

#out: p-value = 0.0406

```
There are a number of parameters that can be used to modify two_sample_test.

```python
ha.stats.two_sample_test(data_array, 
                         treatment_col, 
                         compare="means", 
                         skip=None, 
                         bootstraps=100, 
                         permutations=1000, 
                         kind='weights', 
                         return_null=False,
                         random_state=None)

```
**compare**: The default "means" assumes that you are testing for a difference in means, so it uses the Welch t-statistic. For flexibility, two_sample_test can take a test statistic function as an argument. 

**skip**: indicates the indices of columns that should be skipped in the bootstrapping procedure. 

A simple rule of thumb is that columns should be resampled with replacement only if they were originally sampled with replacement (or effectively sampled with replacement). For example, consider an imaging experiment in which you image several fields of view in a well, which each contain several cells. While you can consider the cells sampled with replacement from the well (there are so many more cells in the population that this assumption is fair), the cells are not sampled with replacement from the field of view (as you are measuring ALL cells in each field of view). The 10% condition is a reasonable rule of thumb here - if your sample represents less than 10% of the population, you can treat it as sampled with replacement.

**bootstraps**: indicates the number of bootstrapped samples to be drawn from data_array. 

Generally, as the number of possible permutations of your data increases, the number of bootstraps should decrease. If the goal of bootstrapping is to include the standard error in the biological samples in the null distribution, only 50-100 bootstraps is sufficient given a large enough set of possible permutations.

**permutations**: indicates the number of permutations of the treatment label PER bootstrapped sample.

Inputting "all" will enumerate all of the possible permutations and iterate through them one by one. This is done using a generator, so the permutations are not stored in memory, but is still excessively time consuming for large datasets. 

**kind**: "weights" or "indexes" or "bayesian" specifies the bootstrapping algorithm. "weights" returns an array the same size as the input array, but with the data reweighted according to the Efron bootstrap procedure. "indexes" uses the same algorithm, but returns a reindexed array. "bayesian" also returns a reweighted array, but the weights are allowed to be any real number rather than just integers.

**return_null**: setting this to True will also return the empirical null distribution as a list.

**seed**: allows you to specify a random seed for reproducibility. 

<a name="power-analysis"></a>
### Sample Size Determination via Power Analysis

Researchers can also use hierarch to determine the appropriate sample size for a future experiment. hierarch.power provides a class, DataSimulator, to assist in power analyses. DataSimulator is initialized with a list specifying the probability distributions generating the data and an optional random_state for reproducibility. 

In this case, consider an experiment similar to the one above - two treatment conditions, but the sample size at each level of hierarchy is yet to be determined. First, you must posit a data-generating process for the analysis.

Suppose you assume that the column 1 values are normally distributed with mean 0 and variance 1. From past experience, you believe that the column 2 values follow a right-tailed distribution, so you choose to model it as a lognormal distribution with a scale parameter of 0.75. Finally, you decide that you want to achieve 80% power for a mean difference equal to one standard deviation. You calculate that the summed standard deviation of the two distributions you specified is 1.525 and input that as a parameter, as well.

```python
from hierarch.power import DataSimulator

parameters = [[0, 1.525], #difference in means due to treatment
             [stats.norm, 0, 1], #column 1 distribution - stats.norm(loc=0, scale=1)
             [stats.lognorm, 0.75]] #column 2 distribution - stats.lognorm(s = 0.75)

sim = DataSimulator(parameters, random_state=1)
```

Next, you choose a experimental design to simulate. Perhaps, like above, you decide to start with three samples per treatment condition and three measurements within each sample. Calling the .fit() function will ready the DataSimulator to produce randomly-generated data according to this experimental scheme.

```python
import scipy.stats as stats

hierarchy = [2, #treatments
             3, #samples
             3] #within-sample measurements

sim.fit(hierarchy)

```
By calling the .generate() function, DataSimulator uses the prespecified parameters to generate a simulated dataset. 

```python
print(sim.generate())
```

|    |    |    |         |
|----:|----:|----:|---------:|
|   1 |   1 |   1 | 1.01409  |
|   1 |   1 |   2 | 1.89184  |
|   1 |   1 |   3 | 1.66005  |
|   1 |   2 |   1 | 2.06844  |
|   1 |   2 |   2 | 1.84316  |
|   1 |   2 |   3 | 2.32849  |
|   1 |   3 |   1 | 0.906038 |
|   1 |   3 |   2 | 1.21542  |
|   1 |   3 |   3 | 1.02701  |
|   2 |   1 |   1 | 2.2638   |
|   2 |   1 |   2 | 1.72708  |
|   2 |   1 |   3 | 1.49989  |
|   2 |   2 |   1 | 3.46166  |
|   2 |   2 |   2 | 3.72993  |
|   2 |   2 |   3 | 3.91148  |
|   2 |   3 |   1 | 3.25964  |
|   2 |   3 |   2 | 5.08576  |
|   2 |   3 |   3 | 4.57408  |

You can use this to set up a simple power analysis. The following code performs a hierarchical permutation test with 50,000 total permutations (though this is overkill in the n = 3, 3 case) on each of 100 simulated datasets and prints the fraction of them that return a significant result, assuming a p-value cutoff of 0.05.

```python
pvalues = []
loops = 100
for i in range(loops):
    data = sim.generate()
    pvalues.append(ha.stats.two_sample_test(data, 0, bootstraps=500, permutations=100))
    
print(np.less(pvalues, 0.05).sum() / loops) 

#out: 0.29

```

The targeted power is 0.8, so you can fit the DataSimulator with a larger sample size. You can run the following code block with different sample sizes until you determine the column 1 sample size that achieves at least 80% power.

```python
sim.fit([2,10,3])

pvalues = []
loops = 100
for i in range(loops):
    data = sim.generate()
    pvalues.append(ha.stats.two_sample_test(data, 0, bootstraps=500, permutations=100))
    
print(np.less(pvalues, 0.05).sum() / loops)

#out: 0.81
```
You note, however, that increasing the number of column 1 samples is much more laborious than increasing the number of column 2 samples. For example, perhaps the column 1 samples represent mice, while column 2 represents multiple measurements of some feature from each mouse's cells. You have posited that the slight majority of your observed variance comes from the column 2 samples - indeed, in biological samples, within-sample variance can be equal to or greater than between-sample variance. After all, that is why we make multiple measurements within the same biological sample! Given that this is a reasonable assumption, perhaps 80% power can be achieved with an experimental design that makes more column 2 measurements.

```python
sim.fit([2,8,30])

pvalues = []
loops = 100
for i in range(loops):
    data = sim.generate()
    pvalues.append(ha.stats.two_sample_test(data, 0, bootstraps=500, permutations=100))
    
print(np.less(pvalues, 0.05).sum() / loops)

#out: 0.84
```

Of course, adding column 2 samples has a much smaller and limited influence on power compared to adding column 1 samples - with infinite column 2 samples, the standard error for the difference of means is still dependent on the variance of the column 1 data-generating process. This is illustrated with an excessive example of 300 column 2 samples per column 1 sample, which shows no improvement in power over using only 30 column 2 samples.

```python
sim.fit([2,8,300])

pvalues = []
loops = 100
for i in range(loops):
    data = sim.generate()
    pvalues.append(ha.stats.two_sample_test(data, 0, bootstraps=500, permutations=100))
    
print(np.less(pvalues, 0.05).sum() / loops)

#out: 0.83
```

On the other hand, adding only four column 1 samples to each treatment group (rather than 270 to each column 1 sample) brings the power to 97%. 

Finally, to ensure that hierarchical permutation is valid for the posited data-generating process, you can do another power analysis under the null hypothesis - that there is no difference between groups. To compensate for Monte Carlo error, you should increase the number of loops - at 100 loops, the error for an event that happens 5% probability is +/- 2%, but at 1000 loops, it is only +/- 0.7%. 

```python
parameters = [[0, 0], #no difference in means because we are sampling under the null hypothesis
             [stats.norm, 0, 1], #column 1 probability distribution  
             [stats.lognorm, 0.75]] #column 2 probability distribution
sim = ha.power.DataSimulator(parameters, random_state=1)
sim.fit([2,12,30])

pvalues = []
loops = 1000
for i in range(loops):
    data = sim.generate()
    pvalues.append(ha.stats.two_sample_test(data, 0, bootstraps=500, permutations=100))
    
print(np.less(pvalues, 0.05).sum() / loops)

#out: 0.05
```
Hierarchical permutation experiences no size distortion for this experimental design and is therefore a valid test.  

Note: because these power calculations are subject to Monte Carlo error, so you should consider upping the number of loops if the precise value for power is of extreme importance. In nonclinical settings, however, small-scale power analyses are sufficient and can be a valuable guide for choosing the sample size for your study. 