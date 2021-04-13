# hierarch

## A Hierarchical Resampling Package for Python

Version 0.1.2

hierarch is a package for hierarchical resampling (bootstrapping, permutation, jackknifing) datasets in Python. Because for loops are ultimately intrinsic to cluster-aware resampling, hierarch uses Numba to accelerate many of its key functions.

hierarch has several functions for performing resampling-based hypothesis tests on hierarchical data. Additionally, hierarch can be used to construct power analyses for hierarchical experimental designs. 

### Dependencies
* numpy
* numba
* scipy (for power analysis)
* sympy (for jackknifing)

