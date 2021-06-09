Power Analysis
==============

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
        pvalues.append(ha.stats.hypothesis_test(data, 0, bootstraps=500, permutations=100))
        
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
        pvalues.append(ha.stats.hypothesis_test(data, 0, bootstraps=500, permutations=100))
        
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
        pvalues.append(ha.stats.hypothesis_test(data, 0, bootstraps=500, permutations=100))
        
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
        pvalues.append(ha.stats.hypothesis_test(data, 0, bootstraps=500, permutations=100))
        
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
        pvalues.append(ha.stats.hypothesis_test(data, 0, bootstraps=500, permutations=100))
        
    print(np.less(pvalues, 0.05).sum() / loops)

    #out: 0.05

Hierarchical permutation experiences no size distortion for this experimental 
design and is therefore a valid test.  

Note: because these power calculations are subject to Monte Carlo error, 
so you should consider upping the number of loops if the precise value for 
power is of extreme importance. In nonclinical settings, however, small-scale 
power analyses are sufficient and can be a valuable guide for choosing the 
sample size for your study. 