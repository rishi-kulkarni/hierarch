# Hierarch

## A Hierarchical Resampling Package for Python

Version 0.1

Hierarch is a package for hierarchical resampling (bootstrapping, permutation, jackknifing) datasets in Python. Because for loops are ultimately intrinsinc to cluster-aware resampling, Hierarch uses Numba to accelerate many of its key functions.

Hierarch has several functions for performing resampling-based hypothesis tests on hierarchical data. Additionally, Hierarch can be used to construct power analyses for hierarchical experimental designs. 

### Dependencies
* numpy
* numba
* scipy (for power analysis)
* sympy (for jackknifing)

