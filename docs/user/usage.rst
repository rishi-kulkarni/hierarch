Usage 
=====

Importing Data
--------------
Hierarch is compatible with pandas DataFrames and numpy arrays. 
Pandas is capable of conveniently importing data from a wide variety 
of formats, including Excel files. ::

    import pandas as pd
    data = pd.read_excel(filepath)

Two-Sample Hypothesis Tests
---------------------------
Performing a hierarchical permutation test for difference of means is simple. 
Consider an imaging experiment with two treatment groups, three coverslips in 
each group, and three images (fields of view) within each coverslip. If you have 
the data stored in an Excel file, you can use pandas to either directly read the 
file or copy it in from the clipboard, as below. ::

    import pandas as pd
    import numpy as np
    import hierarch as ha

    data = pd.read_clipboard()

    print(data)

+------------+------+-------------+----------+
|  Condition | Well | Measurement |  Values  |
+============+======+=============+==========+
|    None    |   1  |      1      | 5.202258 |
+------------+------+-------------+----------+
|    None    |   1  |      2      | 5.136128 |
+------------+------+-------------+----------+
|    None    |   1  |      3      | 5.231401 |
+------------+------+-------------+----------+
|    None    |   2  |      1      | 5.336643 |
+------------+------+-------------+----------+
|    None    |   2  |      2      | 5.287973 |
+------------+------+-------------+----------+
|    None    |   2  |      3      | 5.375359 |
+------------+------+-------------+----------+
|    None    |   3  |      1      | 5.350692 |
+------------+------+-------------+----------+
|    None    |   3  |      2      | 5.465206 |
+------------+------+-------------+----------+
|    None    |   3  |      3      | 5.422602 |
+------------+------+-------------+----------+
| +Treatment |   4  |      1      | 5.695427 |
+------------+------+-------------+----------+
| +Treatment |   4  |      2      | 5.668457 |
+------------+------+-------------+----------+
| +Treatment |   4  |      3      | 5.752592 |
+------------+------+-------------+----------+
| +Treatment |   5  |      1      | 5.583562 |
+------------+------+-------------+----------+
| +Treatment |   5  |      2      | 5.647895 |
+------------+------+-------------+----------+
| +Treatment |   5  |      3      | 5.618315 |
+------------+------+-------------+----------+
| +Treatment |   6  |      1      | 5.642983 |
+------------+------+-------------+----------+
| +Treatment |   6  |      2      |  5.47072 |
+------------+------+-------------+----------+
| +Treatment |   6  |      3      | 5.686654 |
+------------+------+-------------+----------+

It is important to note that the ordering of the columns from left to right 
reflects the experimental design scheme. This is necessary for hierarch 
to infer the clustering within your dataset. In case your data is not 
ordered properly, pandas makes it easy enough to index your data in the 
correct order. ::


    columns = ['Condition', 'Coverslip', 'Field of View', 'Mean Fluorescence']

    data[columns]

Next, you can call two_sample_test from hierarch's stats module, which will 
calculate the p-value. You have to specify what column is the treatment 
column - in this case, "Condition." Indexing starts at 0, so you input 
treatment_col=0. In this case, there are only 6c3 = 20 ways to permute the 
treatment labels, so you should specify "all" permutations be used. ::

    p_val = ha.stats.two_sample_test(data, treatment_col=0, 
                                     bootstraps=500, permutations='all', 
                                     random_state=1)

    print('p-value =', p_val)

    #out: p-value = 0.0406

There are a number of parameters that can be used to modify two_sample_test. ::

    ha.stats.two_sample_test(data_array, 
                            treatment_col, 
                            compare="means", 
                            skip=None, 
                            bootstraps=100, 
                            permutations=1000, 
                            kind='weights', 
                            return_null=False,
                            random_state=None)

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

Many-Sample Hypothesis Tests - Several Hypotheses
-------------------------------------------------
Researchers may want to perform a series of hypothesis tests to determine 
whether there are significant differences between some parameter in three 
or more unrelated groups. This is similar to the goal of one-way ANOVA. To 
this end, hierarch includes the multi_sample_test function, which performs
multiple two-sample tests in the vein of post-hoc tests after ANOVA. The 
researcher can also choose to make a multiple-comparison correction in the 
form of the Benjamini-Hochberg procedure, which controls for False Discovery
Rate.

Consider an experiment with four treatment groups. We can simulate a dataset
as follows. ::

    from hierarch.power import DataSimulator
    import scipy.stats as stats

    paramlist = [[0, 1, 4, 0], [stats.norm], [stats.norm]]
    hierarchy = [4, 3, 3]

    datagen = DataSimulator(paramlist, random_state=1)
    datagen.fit(hierarchy)
    data = datagen.generate()
    data    

+---+---+---+----------+
| 0 | 1 | 2 | 3        |
+===+===+===+==========+
| 1 | 1 | 1 | -0.39087 |
+---+---+---+----------+
| 1 | 1 | 2 | 0.182674 |
+---+---+---+----------+
| 1 | 1 | 3 | -0.13654 |
+---+---+---+----------+
| 1 | 2 | 1 | 1.420464 |
+---+---+---+----------+
| 1 | 2 | 2 | 0.86134  |
+---+---+---+----------+
| 1 | 2 | 3 | 0.529161 |
+---+---+---+----------+
| 1 | 3 | 1 | -0.45147 |
+---+---+---+----------+
| 1 | 3 | 2 | 0.073245 |
+---+---+---+----------+
| 1 | 3 | 3 | 0.338579 |
+---+---+---+----------+
| 2 | 1 | 1 | -0.57876 |
+---+---+---+----------+
| 2 | 1 | 2 | 0.990907 |
+---+---+---+----------+
| 2 | 1 | 3 | 0.703567 |
+---+---+---+----------+
| 2 | 2 | 1 | -0.80581 |
+---+---+---+----------+
| 2 | 2 | 2 | 0.016343 |
+---+---+---+----------+
| 2 | 2 | 3 | 1.730584 |
+---+---+---+----------+
| 2 | 3 | 1 | 1.024184 |
+---+---+---+----------+
| 2 | 3 | 2 | 1.660018 |
+---+---+---+----------+
| 2 | 3 | 3 | 1.663697 |
+---+---+---+----------+
| 3 | 1 | 1 | 5.580886 |
+---+---+---+----------+
| 3 | 1 | 2 | 2.351026 |
+---+---+---+----------+
| 3 | 1 | 3 | 3.085442 |
+---+---+---+----------+
| 3 | 2 | 1 | 6.62389  |
+---+---+---+----------+
| 3 | 2 | 2 | 5.227821 |
+---+---+---+----------+
| 3 | 2 | 3 | 5.244181 |
+---+---+---+----------+
| 3 | 3 | 1 | 3.850566 |
+---+---+---+----------+
| 3 | 3 | 2 | 2.716497 |
+---+---+---+----------+
| 3 | 3 | 3 | 4.532037 |
+---+---+---+----------+
| 4 | 1 | 1 | 0.403147 |
+---+---+---+----------+
| 4 | 1 | 2 | -0.93322 |
+---+---+---+----------+
| 4 | 1 | 3 | -0.38909 |
+---+---+---+----------+
| 4 | 2 | 1 | -0.04362 |
+---+---+---+----------+
| 4 | 2 | 2 | -0.91633 |
+---+---+---+----------+
| 4 | 2 | 3 | -0.06985 |
+---+---+---+----------+
| 4 | 3 | 1 | 0.642196 |
+---+---+---+----------+
| 4 | 3 | 2 | 0.582299 |
+---+---+---+----------+
| 4 | 3 | 3 | 0.040421 |
+---+---+---+----------+

This dataset has been generated such that treatments 1 and 4 have the same mean, while
treatment 2 represents a slight difference and treatment 4 represents a large difference.
There are six total comparisons that can be made, which can be performed automatically
using multi_sample_test as follows. ::

    multi_sample_test(data, treatment_col=0, hypotheses="all",
                    correction=None, bootstraps=1000,
                    permutations="all", random_state=111)
    
    array([[2.0, 3.0, 0.0355],
           [1.0, 3.0, 0.0394],
           [3.0, 4.0, 0.0407],
           [2.0, 4.0, 0.1477],
           [1.0, 2.0, 0.4022],
           [1.0, 4.0, 0.4559]], dtype=object)

The first two columns indicate the conditions being compared, while the last column indicates
the uncorrected p-value. Because there are several hypotheses being tested, it is advisable
to make a multiple comparisons correction. Currently, hierarch can automatically perform the
Benjamini-Hochberg procedure, which controls False Discovery Rate. By indicating the "fdr"
correction, the output array has an additional column showing the q-values, or adjusted p-values. ::

    multi_sample_test(data, treatment_col=0, hypotheses="all",
                    correction='fdr', bootstraps=1000,
                    permutations="all", random_state=111)
    array([[2.0, 3.0, 0.0355, 0.0814],
           [1.0, 3.0, 0.0394, 0.0814],
           [3.0, 4.0, 0.0407, 0.0814],
           [2.0, 4.0, 0.1477, 0.22155],
           [1.0, 2.0, 0.4022, 0.4559],
           [1.0, 4.0, 0.4559, 0.4559]], dtype=object)

Testing more hypotheses necessarily lowers the p-value required to call a result significant. However,
we are not always interested in performing every comparison - perhaps condition 2 is a control that all
other conditions are meant to be compared to. The comparisons of interest can be specified using a list. ::

    tests = [[2.0, 1.0], [2.0, 3.0], [2.0, 4.0]]
    multi_sample_test(data, treatment_col=0, hypotheses=tests,
                      correction='fdr', bootstraps=1000,
                      permutations="all", random_state=222)
    array([[2.0, 3.0, 0.036, 0.108],
           [2.0, 4.0, 0.1506, 0.2259],
           [2.0, 1.0, 0.4036, 0.4036]], dtype=object)

Many-Sample Hypothesis Tests - Single Hypothesis
------------------------------------------------
One-way ANOVA and similar tests (like multi_sample_test) are inappropriate when
you have several samples meant to test a single hypothesis. For example, perhaps
you have several samples with different concentrations of the same drug treatment.
In this case, hierarch provides linear_regression_test, which is equivalent to
performing a hypothesis test on a linear model against the null hypothesis that
the slope coefficient is equal to 0.

This hypothesis test uses a studentized covariance test statistic - essentially,
the sample covariance divided by the standard error of the sample covariance. This
test statistic is approximately normally distributed and in the two-sample case, 
this test gives the same result as two_sample_test.

First, consider a dataset with two treatment groups, four samples each, and three
measurements on each sample. ::

    from hierarch.power import DataSimulator
    import scipy.stats as stats

    paramlist = [[0, 2], [stats.norm], [stats.norm]]
    hierarchy = [2, 4, 3]

    datagen = DataSimulator(paramlist, random_state=2)
    datagen.fit(hierarchy)
    data = datagen.generate()
    data

+---+---+---+----------+
| 0 | 1 | 2 | 3        |
+===+===+===+==========+
| 1 | 1 | 1 | 0.470264 |
+---+---+---+----------+
| 1 | 1 | 2 | -0.36477 |
+---+---+---+----------+
| 1 | 1 | 3 | 1.166621 |
+---+---+---+----------+
| 1 | 2 | 1 | -0.8333  |
+---+---+---+----------+
| 1 | 2 | 2 | -0.85157 |
+---+---+---+----------+
| 1 | 2 | 3 | -1.3149  |
+---+---+---+----------+
| 1 | 3 | 1 | 0.041895 |
+---+---+---+----------+
| 1 | 3 | 2 | -0.51226 |
+---+---+---+----------+
| 1 | 3 | 3 | 0.132225 |
+---+---+---+----------+
| 1 | 4 | 1 | -3.04865 |
+---+---+---+----------+
| 1 | 4 | 2 | -2.31464 |
+---+---+---+----------+
| 1 | 4 | 3 | -3.33374 |
+---+---+---+----------+
| 2 | 1 | 1 | 4.641172 |
+---+---+---+----------+
| 2 | 1 | 2 | 3.987742 |
+---+---+---+----------+
| 2 | 1 | 3 | 4.130278 |
+---+---+---+----------+
| 2 | 2 | 1 | 3.55467  |
+---+---+---+----------+
| 2 | 2 | 2 | 2.133408 |
+---+---+---+----------+
| 2 | 2 | 3 | 3.927347 |
+---+---+---+----------+
| 2 | 3 | 1 | 3.73128  |
+---+---+---+----------+
| 2 | 3 | 2 | 0.036135 |
+---+---+---+----------+
| 2 | 3 | 3 | -0.05483 |
+---+---+---+----------+
| 2 | 4 | 1 | 1.268975 |
+---+---+---+----------+
| 2 | 4 | 2 | 3.615265 |
+---+---+---+----------+
| 2 | 4 | 3 | 2.902522 |
+---+---+---+----------+

Performing linear_regression_test and two_sample_test on this dataset should
give very similar p-values. ::

    linear_regression_test(data, treatment_col=0,
                        bootstraps=1000, permutations='all',
                        random_state=1)
    0.013714285714285714

    two_sample_test(data, treatment_col=0,
                    bootstraps=1000, permutations='all',
                    random_state=1)
    0.013714285714285714

However, unlike two_sample_test, this test can handle any number of conditions. Consider instead
a dataset with four treatment conditions that have a linear relationship. ::

    paramlist = [[0, 2/3, 4/3, 2], [stats.norm], [stats.norm]]
    hierarchy = [4, 2, 3]
    datagen = DataSimulator(paramlist, random_state=2)
    datagen.fit(hierarchy)
    data = datagen.generate()
    data

+---+---+---+----------+
| 0 | 1 | 2 | 3        |
+===+===+===+==========+
| 1 | 1 | 1 | 0.470264 |
+---+---+---+----------+
| 1 | 1 | 2 | -0.36477 |
+---+---+---+----------+
| 1 | 1 | 3 | 1.166621 |
+---+---+---+----------+
| 1 | 2 | 1 | -0.8333  |
+---+---+---+----------+
| 1 | 2 | 2 | -0.85157 |
+---+---+---+----------+
| 1 | 2 | 3 | -1.3149  |
+---+---+---+----------+
| 2 | 1 | 1 | 0.708561 |
+---+---+---+----------+
| 2 | 1 | 2 | 0.154405 |
+---+---+---+----------+
| 2 | 1 | 3 | 0.798892 |
+---+---+---+----------+
| 2 | 2 | 1 | -2.38199 |
+---+---+---+----------+
| 2 | 2 | 2 | -1.64797 |
+---+---+---+----------+
| 2 | 2 | 3 | -2.66707 |
+---+---+---+----------+
| 3 | 1 | 1 | 3.974506 |
+---+---+---+----------+
| 3 | 1 | 2 | 3.321076 |
+---+---+---+----------+
| 3 | 1 | 3 | 3.463612 |
+---+---+---+----------+
| 3 | 2 | 1 | 2.888003 |
+---+---+---+----------+
| 3 | 2 | 2 | 1.466742 |
+---+---+---+----------+
| 3 | 2 | 3 | 3.26068  |
+---+---+---+----------+
| 4 | 1 | 1 | 3.73128  |
+---+---+---+----------+
| 4 | 1 | 2 | 0.036135 |
+---+---+---+----------+
| 4 | 1 | 3 | -0.05483 |
+---+---+---+----------+
| 4 | 2 | 1 | 1.268975 |
+---+---+---+----------+
| 4 | 2 | 2 | 3.615265 |
+---+---+---+----------+
| 4 | 2 | 3 | 2.902522 |
+---+---+---+----------+

For this dataset, there are 8! / (2!^4) = 2,520 total permutations. We will choose a random
subset of them to compute the p-value. ::

    linear_regression_test(data, treatment_col=0,
                        bootstraps=100, permutations=1000,
                        random_state=1)
    0.00767

Between these three tests, researchers can address a large variety of experimental designs. Unfortunately,
interaction effects are outside the scope of permutation tests - it is not possible to construct an
exact test for interaction effects in general. However, an asymptotic test for interaction effects
may be implemented in the future.

Power Analysis
--------------
Researchers can also use hierarch to determine the appropriate sample size 
for a future experiment. hierarch.power provides a class, DataSimulator, 
to assist in power analyses. DataSimulator is initialized with a list 
specifying the probability distributions generating the data and an optional 
random_state for reproducibility. 

In this case, consider an experiment similar to the one above - two treatment 
conditions, but the sample size at each level of hierarchy is yet to be 
determined. First, you must posit a data-generating process for the analysis.

Suppose you assume that the column 1 values are normally distributed with 
mean 0 and variance 1. From past experience, you believe that the column 2 
values follow a right-tailed distribution, so you choose to model it as a 
lognormal distribution with a scale parameter of 0.75. Finally, you decide 
that you want to achieve 80% power for a mean difference equal to one standard 
deviation. You calculate that the summed standard deviation of the two 
distributions you specified is 1.525 and input that as a parameter, as well. ::

    from hierarch.power import DataSimulator

    parameters = [[0, 1.525], #difference in means due to treatment
                [stats.norm, 0, 1], #column 1 distribution - stats.norm(loc=0, scale=1)
                [stats.lognorm, 0.75]] #column 2 distribution - stats.lognorm(s = 0.75)

    sim = DataSimulator(parameters, random_state=1)

Next, you choose a experimental design to simulate. Perhaps, like above, you 
decide to start with three samples per treatment condition and three measurements 
within each sample. Calling the .fit() function will ready the DataSimulator to 
produce randomly-generated data according to this experimental scheme. ::

    import scipy.stats as stats

    hierarchy = [2, #treatments
                3, #samples
                3] #within-sample measurements

    sim.fit(hierarchy)

    By calling the .generate() function, DataSimulator uses the prespecified 
    parameters to generate a simulated dataset. ::

    print(sim.generate())

+---+---+---+----------+
| 0 | 1 | 2 | 3        |
+===+===+===+==========+
| 1 | 1 | 1 | 1.014087 |
+---+---+---+----------+
| 1 | 1 | 2 | 1.891843 |
+---+---+---+----------+
| 1 | 1 | 3 | 1.660049 |
+---+---+---+----------+
| 1 | 2 | 1 | 2.068442 |
+---+---+---+----------+
| 1 | 2 | 2 | 1.843164 |
+---+---+---+----------+
| 1 | 2 | 3 | 2.328488 |
+---+---+---+----------+
| 1 | 3 | 1 | 0.906038 |
+---+---+---+----------+
| 1 | 3 | 2 | 1.215424 |
+---+---+---+----------+
| 1 | 3 | 3 | 1.027005 |
+---+---+---+----------+
| 2 | 1 | 1 | 1.788798 |
+---+---+---+----------+
| 2 | 1 | 2 | 1.252083 |
+---+---+---+----------+
| 2 | 1 | 3 | 1.024889 |
+---+---+---+----------+
| 2 | 2 | 1 | 2.986665 |
+---+---+---+----------+
| 2 | 2 | 2 | 3.254925 |
+---+---+---+----------+
| 2 | 2 | 3 | 3.436481 |
+---+---+---+----------+
| 2 | 3 | 1 | 2.784636 |
+---+---+---+----------+
| 2 | 3 | 2 | 4.610765 |
+---+---+---+----------+
| 2 | 3 | 3 | 4.099078 |
+---+---+---+----------+    

You can use this to set up a simple power analysis. The following 
code performs a hierarchical permutation test with 50,000 total 
permutations (though this is overkill in the 2, 3, 3 case) on each 
of 100 simulated datasets and prints the fraction of them that return 
a significant result, assuming a p-value cutoff of 0.05. ::

    pvalues = []
    loops = 100
    for i in range(loops):
        data = sim.generate()
        pvalues.append(ha.stats.two_sample_test(data, 0, bootstraps=500, permutations=100))
        
    print(np.less(pvalues, 0.05).sum() / loops) 

    #out: 0.29

The targeted power is 0.8, so you can fit the DataSimulator with a larger sample 
size. You can run the following code block with different sample sizes until 
you determine the column 1 sample size that achieves at least 80% power. ::

    sim.fit([2,10,3])

    pvalues = []
    loops = 100
    for i in range(loops):
        data = sim.generate()
        pvalues.append(ha.stats.two_sample_test(data, 0, bootstraps=500, permutations=100))
        
    print(np.less(pvalues, 0.05).sum() / loops)

    #out: 0.81

You note, however, that increasing the number of column 1 samples is much 
more laborious than increasing the number of column 2 samples. For example, 
perhaps the column 1 samples represent mice, while column 2 represents 
multiple measurements of some feature from each mouse's cells. You have 
posited that the slight majority of your observed variance comes from the 
column 2 samples - indeed, in biological samples, within-sample variance 
can be equal to or greater than between-sample variance. After all, that 
is why we make multiple measurements within the same biological sample! 
Given that this is a reasonable assumption, perhaps 80% power can be 
achieved with an experimental design that makes more column 2 measurements. ::

    sim.fit([2,8,30])

    pvalues = []
    loops = 100
    for i in range(loops):
        data = sim.generate()
        pvalues.append(ha.stats.two_sample_test(data, 0, bootstraps=500, permutations=100))
        
    print(np.less(pvalues, 0.05).sum() / loops)

    #out: 0.84

Of course, adding column 2 samples has a much more limited 
influence on power compared to adding column 1 samples - with infinite 
column 2 samples, the standard error for the difference of means is 
still dependent on the variance of the column 1 data-generating process. 
This is illustrated with an excessive example of 300 column 2 samples 
per column 1 sample, which shows no improvement in power over using 
only 30 column 2 samples. ::

    sim.fit([2,8,300])

    pvalues = []
    loops = 100
    for i in range(loops):
        data = sim.generate()
        pvalues.append(ha.stats.two_sample_test(data, 0, bootstraps=500, permutations=100))
        
    print(np.less(pvalues, 0.05).sum() / loops)
    
    #out: 0.83

On the other hand, adding only four column 1 samples to each treatment group 
(rather than 270 to each column 1 sample) brings the power to 97%. 

Finally, to ensure that hierarchical permutation is valid for the posited 
data-generating process, you can do another power analysis under the null 
hypothesis - that there is no difference between groups. To compensate for 
Monte Carlo error, you should increase the number of loops - at 100 loops, 
the error for an event that happens 5% probability is +/- 2%, but at 
1000 loops, it is only +/- 0.7%. ::

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

Hierarchical permutation experiences no size distortion for this experimental 
design and is therefore a valid test.  

Note: because these power calculations are subject to Monte Carlo error, 
so you should consider upping the number of loops if the precise value for 
power is of extreme importance. In nonclinical settings, however, small-scale 
power analyses are sufficient and can be a valuable guide for choosing the 
sample size for your study. 