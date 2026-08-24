Overview
========

hierarch is a package for hierarchical resampling (bootstrapping, permutation) of datasets in Python. 
Resampling is batched and vectorized with numpy, so cluster-aware bootstraps and permutations
are drawn without Python-level loops over the hierarchy.

hierarch has several functions to assist in performing resampling-based hypothesis tests on hierarchical data. 
Additionally, hierarch can be used to construct power analyses for hierarchical experimental designs.

Here is the sort of data that hierarch is designed to perform hypothesis tests on:

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

Describe the design with a formula and hierarch builds the matrix its tests expect.
The code to perform a hierarchical permutation t-test on this dataset looks like::

    from hierarch.design import design_matrix
    from hierarch.stats import hypothesis_test

    matrix, treatment_col = design_matrix(
        data, "Values ~ Condition/Well/Measurement", treatment="Condition"
    )
    hypothesis_test(matrix, treatment_col, bootstraps=1000, permutations='all')

The formula's right-hand side is the nesting chain, written outermost-first: wells are
nested within conditions and measurements within wells. See :doc:`importing` for the
column layout this produces and for passing pre-arranged arrays directly.

If you find hierarch useful for analyzing your data, please consider citing it. 

Analyzing Nested Experimental Designs – A User-Friendly Resampling Method to Determine Experimental Significance
Rishikesh U. Kulkarni, Catherine L. Wang, Carolyn R. Bertozzi
bioRxiv 2021.06.29.450439; doi: https://doi.org/10.1101/2021.06.29.450439