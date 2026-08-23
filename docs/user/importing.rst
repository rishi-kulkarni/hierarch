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