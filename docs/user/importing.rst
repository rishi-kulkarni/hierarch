Importing Data
==============

Hierarch is compatible with pandas DataFrames and numpy arrays.
Pandas is capable of conveniently importing data from a wide variety
of formats, including Excel files. ::

    import pandas as pd
    data = pd.read_excel(filepath)

Building a Design Matrix from a Formula
---------------------------------------

hierarch expects columns ordered from the outermost level of the
hierarchy to the innermost, with the dependent variable last.
:func:`hierarch.design.design_matrix` builds this layout from a
long-format DataFrame. Consider an experiment where a treatment is
applied to coverslips within each mouse and several cells are measured
per coverslip, one row per cell (first mouse shown):

+-------+-----------+-----------+--------------+
| Mouse | Treatment | Coverslip | Fluorescence |
+=======+===========+===========+==============+
| m1    | ctrl      | c1        | 4.83961      |
+-------+-----------+-----------+--------------+
| m1    | ctrl      | c1        | 4.73513      |
+-------+-----------+-----------+--------------+
| m1    | ctrl      | c2        | 4.95033      |
+-------+-----------+-----------+--------------+
| m1    | ctrl      | c2        | 5.08409      |
+-------+-----------+-----------+--------------+
| m1    | drug      | c1        | 5.52721      |
+-------+-----------+-----------+--------------+
| m1    | drug      | c1        | 5.32194      |
+-------+-----------+-----------+--------------+
| m1    | drug      | c2        | 5.18947      |
+-------+-----------+-----------+--------------+
| m1    | drug      | c2        | 5.14304      |
+-------+-----------+-----------+--------------+

The right-hand side of the formula is the nesting chain, written
outermost-first, and must contain the treatment variable at the level
where it was applied::

    from hierarch.design import design_matrix
    from hierarch.stats import hypothesis_test

    matrix, treatment_col = design_matrix(
        data, "Fluorescence ~ Mouse/Treatment/Coverslip", treatment="Treatment"
    )
    hypothesis_test(matrix, treatment_col, bootstraps=100, permutations=1000)

This produces the numeric layout hierarch's tests expect:

+-------+-----------+-----------+--------------+--------------+
| Mouse | Treatment | Coverslip | _measurement | Fluorescence |
+=======+===========+===========+==============+==============+
| 0     | 0         | 0         | 0            | 4.83961      |
+-------+-----------+-----------+--------------+--------------+
| 0     | 0         | 0         | 1            | 4.73513      |
+-------+-----------+-----------+--------------+--------------+
| 0     | 0         | 1         | 0            | 4.95033      |
+-------+-----------+-----------+--------------+--------------+
| 0     | 0         | 1         | 1            | 5.08409      |
+-------+-----------+-----------+--------------+--------------+
| 0     | 1         | 2         | 0            | 5.52721      |
+-------+-----------+-----------+--------------+--------------+
| 0     | 1         | 2         | 1            | 5.32194      |
+-------+-----------+-----------+--------------+--------------+
| 0     | 1         | 3         | 0            | 5.18947      |
+-------+-----------+-----------+--------------+--------------+
| 0     | 1         | 3         | 1            | 5.14304      |
+-------+-----------+-----------+--------------+--------------+

``y ~ treatment/mouse`` describes mice nested within treatment groups.
``y ~ mouse/treatment`` describes treatments applied within each mouse.

Levels are integer-coded by their composite key, so ids reused across
enclosing clusters are treated as distinct: coverslip "c1" under drug
became coverslip 2, not a repeat of coverslip 0. Rows are sorted into
hierarchical order.

The innermost replicate level does not need an id column. If the
innermost named cells contain more than one row, a replicate index
column is synthesized. If the data does have an id column for that level,
name it in the chain instead; both forms produce the same test.

The Underlying Column Layout
----------------------------

design_matrix is a convenience layer over the layout the tests actually consume,
and a numpy array or DataFrame already in that layout can be passed directly.
The contract is:

* One row per measurement.
* Grouping columns ordered from the outermost level of the hierarchy to the
  innermost, left to right, with the dependent variable in the last column.
* The treatment column identified by position (``treatment_col=0``) or, for a
  DataFrame, by name (``treatment_col="Condition"``).

::

    from hierarch.stats import hypothesis_test

    hypothesis_test(data, treatment_col=0, bootstraps=1000, permutations="all")

Non-numeric columns are label encoded and rows are sorted into hierarchical
order internally, so cluster ids only need to be unique within their enclosing
cluster. Note that this is the one guarantee design_matrix strengthens: it
relabels every level by its composite key, so an id reused under two different
parents stays distinct in the output you read back.

Since design_matrix returns a ``(data, treatment_col)`` named tuple, it can also
be splatted straight into any of the tests::

    hypothesis_test(
        *design_matrix(data, "Values ~ Condition/Well/Measurement", treatment="Condition"),
        bootstraps=1000,
        permutations="all",
    )
