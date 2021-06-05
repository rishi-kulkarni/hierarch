Confidence Intervals
====================

Two-Sample Effect Sizes 
-----------------------
Researchers can use hierarch to compute confidence intervals for effect sizes. 
These intervals are computed via test inversion and, as a result, have the advantage
of essentially always achieving the nominal coverage. 

To put it another way, hierarch computes a 95% confidence interval by performing a 
permutation test against the null hypothesis that true effect size is exactly equal 
to the observed effect size. Then, the bounds of the acceptance region at alpha = 0.05
are the bounds of the confidence interval. Let's consider the dataset from earlier. ::

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

You can use the confidence_interval function in hierarch.stats to compute the 
confidence interval. ::

    from hierarch.stats import confidence_interval

    ha.stats.confidence_interval(
    data,
    treatment_col=0,
    compare='means',
    interval=95,
    bootstraps=500,
    permutations="all",
    random_state=1,
    )

    (-0.5373088054909549, -0.12010079984237881)

This interval does not cross 0, so it is consistent with significance at the alpha = 0.05
level.

Because ha.stats.confidence_interval is based on a hypothesis test, it requires
the same input parameters as two_sample_test or linear_regression_test. However, 
the new **interval** parameter determines the width of the interval. ::

    ha.stats.confidence_interval(
    data,
    treatment_col=0,
    compare='means',
    interval=99,
    bootstraps=500,
    permutations="all",
    random_state=1,
    )

    (-0.9086402840632387, 0.25123067872990457)

    ha.stats.confidence_interval(
    data,
    treatment_col=0,
    compare='means',
    interval=68,
    bootstraps=500,
    permutations="all",
    random_state=1,
    )

    (-0.40676489798778065, -0.25064470734555316)

The 99% confidence interval does indeed cross 0, so we could not reject the null hypothesis
at the alpha = 0.01 level.

To build your confidence, you can perform a simulation analysis to ensure 
the confidence interval achieves the nominal coverage. You can set up a 
DataSimulator using the functions in hierarch.power as follows. ::

    from hierarch.power import DataSimulator

    parameters = [[0, 1.525], #difference in means due to treatment
                [stats.norm, 0, 1], #column 1 distribution - stats.norm(loc=0, scale=1)
                [stats.lognorm, 0.75]] #column 2 distribution - stats.lognorm(s = 0.75)

    sim = DataSimulator(parameters, random_state=1)

    import scipy.stats as stats

    hierarchy = [2, #treatments
                3, #samples
                3] #within-sample measurements

    sim.fit(hierarchy)

The "true" difference between the two samples is 1.525 according to the simulation
parameters, so 95% of 95% confidence intervals that hierarch calculates should contain
this value. You can test this with the following code. ::

    true_difference = 1.525
    coverage = 0
    loops = 1000

    for i in range(loops):
        data = sim.generate()
        lower, upper = ha.stats.confidence_interval(data, 0, interval=95, bootstraps=100, permutations='all')
        if lower <= true_difference <= upper:
            coverage += 1

    print("Coverage:", coverage/loops)
    
    Coverage: 0.946

This is within the Monte Carlo error of the simulation (+/- 0.7%) of 95%, so we can feel
confident in this method of interval computation.

Regression Coefficient Confidence Intervals
-------------------------------------------
The confidence_interval function can also be used on many-sample datasets that represent
a hypothesized linear relationship. Let's generate a dataset with a "true" slope of 
2/3. ::

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

You can compute a confidence interval in the same manner as above. This time, you should set the
**compare** keyword argument to "corr" for clarity, but "corr" is also the default setting
for **compare** when computing a confidence interval. ::

    from hierarch.stats import confidence_interval

    ha.stats.confidence_interval(
    data,
    treatment_col=0,
    compare='corr',
    interval=95,
    bootstraps=500,
    permutations="all",
    random_state=1,
    )

    (0.3410887712843298, 1.7540918236455125)

This confidence interval corresponds to the slope in a linear model. You can check this by
computing the slope coefficient via Ordinary Least Squares. ::

    import scipy.stats
    from hierarch.internal_functions import GroupbyMean

    grouper = GroupbyMean()
    test = grouper.fit_transform(data)
    stats.linregress(test[:,0], test[:,-1])

    LinregressResult(slope=1.0515132531203024, intercept=-1.6658194480556106, 
    rvalue=0.6444075548383587, pvalue=0.08456152533094284, 
    stderr=0.5094006523081002, intercept_stderr=1.3950511403849626)

The slope, 1.0515, is indeed in the center of our computed interval (within Monte Carlo error).

Again, it is worthwhile to check that confidence_interval is performing adequately. You can
set up a simulation as above to check the coverage of the 95% confidence interval. ::

    true_difference = 2/3
    coverage = 0
    loops = 1000

    for i in range(loops):
        data = datagen.generate()
        lower, upper = ha.stats.confidence_interval(data, 0, interval=95, bootstraps=100, permutations='all')
        if lower <= true_difference <= upper:
            coverage += 1

    print(coverage/loops)

    0.956

This is within the Monte Carlo error of the simulation (+/- 0.7%) of 95% and therefore
acceptable. You can check the coverage of other intervals by changing the **interval** keyword
argument, though be aware that Monte Carlo error depends on the probability of the event of
interest. ::

    true_difference = 2/3
    coverage = 0
    loops = 1000

    for i in range(loops):
        data = datagen.generate()
        lower, upper = ha.stats.confidence_interval(data, 0, interval=99, bootstraps=100, permutations='all')
        if lower <= true_difference <= upper:
            coverage += 1

    print(coverage/loops)

    0.99

Using the confidence_interval function, researchers can rapidly calculate confidence intervals for
effect sizes that maintain nominal coverage without worrying about distributional assumptions. 