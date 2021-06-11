# hierarch

## A Hierarchical Resampling Package for Python

Version 1.1.1

hierarch is a package for hierarchical resampling (bootstrapping, permutation) of datasets in Python. Because for loops are ultimately intrinsic to cluster-aware resampling, hierarch uses Numba to accelerate many of its key functions.

hierarch has several functions to assist in performing resampling-based (and therefore distribution-free) hypothesis tests, confidence interval calculations, and power analyses on hierarchical data.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Documentation](#documentation)


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

### Installation

The easiest way to install hierarch is via PyPi. 

```pip install hierarch```

Alternatively, you can install from Anaconda.

```conda install -c rkulk111 hierarch```


<a name="documentation"></a>
## Documentation
Check out our user guide at [readthedocs](https://hierarch.readthedocs.io/).