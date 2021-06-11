Hypothesis Testing
==================

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

Next, you can call hypothesis_test from hierarch's stats module, which will 
calculate the p-value. You have to specify what column is the treatment 
column - in this case, "Condition." Indexing starts at 0, so you input 
treatment_col=0. In this case, there are only 6c3 = 20 ways to permute the 
treatment labels, so you should specify "all" permutations be used. ::

    from hierarch.stats import hypothesis_test

    p_val = hypothesis_test(data, treatment_col=0, compare='means',
                                     bootstraps=500, permutations='all', 
                                     random_state=1)

    print('p-value =', p_val)

    #out: p-value = 0.0406

There are a number of parameters that can be used to modify hypothesis_test. ::

    hypothesis_test(data_array, 
                    treatment_col, 
                    compare="means", 
                    skip=None, 
                    bootstraps=100, 
                    permutations=1000, 
                    kind='weights', 
                    return_null=False,
                    random_state=None)

**compare**: The default "means" assumes that you are testing for a difference in means, so it uses the Welch t-statistic. 
"corr" uses a studentized covariance based test statistic which gives the same result as the Welch t-statistic for two-sample
datasets, but can be used on datasets with any number of related treatment groups. For flexibility, hypothesis_test can 
also take a test statistic function as an argument.

**alternative** : "two-sided" or "less" or "greater" specifies the alternative hypothesis. "two-sided" conducts
a two-tailed test, while "less" or "greater" conduct the appropriate one-tailed test.

**skip**: indicates the indices of columns that should be skipped in the bootstrapping procedure. 

A simple rule of thumb is that columns should be resampled with replacement only if they were originally sampled with replacement 
(or effectively sampled with replacement). For example, consider an imaging experiment in which you image several fields of view in a well, 
which each contain several cells. While you can consider the cells sampled with replacement from the well (there are so many more cells in the 
population that this assumption is fair), the cells are not sampled with replacement from the field of view (as you are measuring ALL cells 
in each field of view). The 10% condition is a reasonable rule of thumb here - if your sample represents less than 10% of the population, 
you can treat it as sampled with replacement.

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

    from hierarch.stats import multi_sample_test

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
In this case, you can set compare to "corr", which is equivalent to
performing a hypothesis test on a linear model against the null hypothesis that
the slope coefficient is equal to 0.

This hypothesis test uses a studentized covariance test statistic - essentially,
the sample covariance divided by the standard error of the sample covariance. This
test statistic is approximately normally distributed and in the two-sample case, 
this test gives the same result as setting compare="means".

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

Using studentized covariance or the Welch t statistic on this dataset should
give very similar p-values. ::

    hypothesis_test(data, treatment_col=0, compare="corr",
                        bootstraps=1000, permutations='all',
                        random_state=1)
    0.013714285714285714

    hypothesis_test(data, treatment_col=0, compare="means",
                    bootstraps=1000, permutations='all',
                    random_state=1)
    0.013714285714285714

However, unlike the Welch t-statistic, studentized covariance can handle any number of conditions. Consider instead
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

    hypothesis_test(data, treatment_col=0,
                        bootstraps=100, permutations=1000,
                        random_state=1)
    0.00767

Between these three tests, researchers can address a large variety of experimental designs. Unfortunately,
interaction effects are outside the scope of permutation tests - it is not possible to construct an
exact test for interaction effects in general. However, an asymptotic test for interaction effects
may be implemented in the future.
