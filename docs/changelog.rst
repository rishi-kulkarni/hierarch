Changelog
=========

2.0.0 (2026-08-26)
-------------------

- Removed Numba. Resampling and statistics are now vectorized numpy: batched
  permutations and einsum-based statistic scoring, built on per-instance
  ``PCG64`` generators. No more JIT warmup on first call. Warm-to-warm,
  ``hypothesis_test`` measured about 5x faster than 1.1.3.
- Breaking: seeded ``random_state`` output is no longer bit-identical to
  pre-2.0 releases. Results pinned to a specific seed will differ after
  upgrading.
- Breaking: removed the ``Bootstrapper`` and ``Permuter`` classes from
  ``hierarch.resampling``. Their functionality is the functional
  ``bootstrap_plan``/``draw_bootstrap_weights``/``draw_bootstrap_weights_batch``
  and ``permutation_plan``/``draw_permuted_labels``/``exact_label_matrix``,
  which is what ``hypothesis_test``, ``confidence_interval``, and
  ``hierarchical_randomization`` already used internally.
- Fixed a bug in ``hierarchical_randomization``: with ``permutations="all"``
  and ``bootstraps`` greater than 1, only the first bootstrap sample was
  exhaustively enumerated; every subsequent bootstrap silently fell back to
  a random (non-exact) permutation draw.
- Removed the unused, undocumented ``hierarch.internal_functions.nb_data_grabber``
  helper (dead code left over from Numba removal; not referenced anywhere in
  the current codebase).
- Added :func:`hierarch.design.design_matrix`, which builds the data layout
  ``hypothesis_test`` and ``confidence_interval`` expect from a Wilkinson
  formula. Adds ``formulaic`` as a dependency.
- Added ``jackknife_corr`` as a ``compare`` option and a jackknife
  studentized covariance function.
- Fixed a ``GroupbyMean`` bug in indexes aggregation, found while removing
  Numba.
- Added a contract/invariance test suite (brute-force exact tests,
  reference implementations, seeded-output checks) covering the resampling
  primitives, ``GroupbyMean``, ``hypothesis_test``, ``confidence_interval``,
  ``multi_sample_test``, and ``hierarchical_randomization``.

1.2.1 (2026-02-21)
-------------------

- Migrated build tooling from Poetry to uv. No API changes.

1.2.0 (2026-01-11)
-------------------

- Maintenance release: CI configuration and dependency updates only. No API
  changes.

1.1.6 (2023-08-23)
-------------------

- Added ``hierarch.stats.hierarchical_randomization``, a generator that
  yields resampled datasets.

1.1.5 (2023-05-24)
-------------------

- Republished 1.1.4. No code changes.

1.1.4 (2023-05-24)
-------------------

- Fixed Numba 0.57 compatibility.
- Migrated packaging and docs build to Poetry.
- Added a test suite covering ``Bootstrapper``, ``Permuter``,
  ``GroupbyMean``, ``hypothesis_test``, ``confidence_interval``, and
  ``hierarch.power``.

1.1.3 (2021-07-07)
-------------------

- ``treatment_col`` can now be specified by column name when the input is a
  DataFrame.
- Cleaned up ``hierarch.resampling`` and improved initial compile time.

1.1.2 (2021-06-21)
-------------------

- Improved stability of ``confidence_interval``'s initial guess and its
  behavior for very asymmetric null distributions.

1.1.1 (2021-06-10)
-------------------

- Fixed a bug in ``hypothesis_test`` involving ``>=`` and ``<=``
  alternatives.

1.1 (2021-06-08)
-----------------

- Combined ``two_sample_test`` and ``linear_regression_test`` into a single
  ``hypothesis_test``.
- Fixed a ``Permuter`` memory leak caused by ``lru_cache``.
- ``confidence_interval`` now finds its upper and lower bounds separately.

1.0.1 (2021-06-05)
-------------------

- Fixed Anaconda package dependencies.

1.0.0 (2021-06-05)
-------------------

- Added ``confidence_interval``; test statistics moved into
  ``hierarch.stats``.
- Confidence interval bounds are now found via an iterative search.
- Non-exact permutation tests can no longer return a p-value of 0.

0.3 (2021-05-26)
-----------------

- Implemented the Bayesian bootstrap and other reweighting-based bootstrap
  algorithms.
- Implemented multiple-sample (post-hoc) tests.
- Added the studentized covariance test statistic.
- Added Sphinx-based documentation.

0.2 (2021-05-05)
-----------------

- Initial release. ``hierarch.resampling`` with Numba-accelerated
  ``Bootstrapper`` and ``Permuter`` classes.
