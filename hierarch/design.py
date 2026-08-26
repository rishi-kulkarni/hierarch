"""Construct design matrices for hierarch from Wilkinson formulas.

The :func:`design_matrix` helper translates a long-format DataFrame with
named columns into the positional numeric layout that
:func:`hierarch.stats.hypothesis_test` and friends expect, using a
Wilkinson formula to describe the nesting structure of the experiment.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from formulaic import Formula
from formulaic.formula import SimpleFormula


class DesignMatrix(NamedTuple):
    """A design matrix in hierarch's column layout.

    Can be splatted directly into a test:
    ``hypothesis_test(*design_matrix(df, formula, treatment))``.
    """

    data: pd.DataFrame
    treatment_col: str


def design_matrix(data, formula, treatment):
    """Build a hierarch design matrix from a Wilkinson formula.

    The formula's right-hand side is a nesting chain written
    outermost-first, mirroring the column order of the output:
    ``y ~ a/b/c`` declares that ``b`` is nested within ``a`` and ``c``
    within ``b``. The chain must contain the treatment variable, whose
    position declares the level at which it was applied: ``y ~
    treatment/mouse`` describes mice nested within treatment groups,
    while ``y ~ mouse/treatment`` describes treatments applied within
    each mouse.

    Every level is relabeled by its composite key, so cluster ids that
    repeat across enclosing clusters (subject "1" in family 1 vs. family
    2) are kept distinct. Rows are sorted into hierarchical order, and if
    the innermost cells hold more than one measurement, a replicate index
    column is synthesized; a replicate id column can be named in the chain
    instead. The dependent variable named on the left-hand side becomes
    the final column.

    Parameters
    ----------
    data : pandas DataFrame
        Long-format data, one row per measurement.
    formula : str
        Wilkinson formula ``y ~ a/b/...`` naming columns in data. The
        right-hand side must be a pure nesting chain.
    treatment : str
        Name of the treatment variable. Must appear in the nesting chain.

    Returns
    -------
    DesignMatrix
        Named tuple of (data, treatment_col): the design matrix as a
        DataFrame in hierarchical column order, and the treatment column's
        name. Suitable for ``hypothesis_test(*design_matrix(...))``.

    Raises
    ------
    ValueError
        If the formula is not a pure nesting chain with a single
        dependent variable, if the treatment does not appear in the
        chain, or if the treatment is constant within the level declared
        above it (a misdeclared between-cluster design).
    KeyError
        If a name in the formula is not a column in data.

    Examples
    --------
    Mice nested within treatments (a between-mouse design):

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "treatment": ["ctrl"] * 4 + ["drug"] * 4,
    ...         "mouse": ["m1", "m1", "m2", "m2", "m3", "m3", "m4", "m4"],
    ...         "y": [1.0, 1.2, 0.9, 1.1, 2.0, 2.1, 1.9, 2.2],
    ...     }
    ... )
    >>> matrix, treatment_col = design_matrix(df, "y ~ treatment/mouse", "treatment")
    >>> matrix
       treatment  mouse  _measurement    y
    0        0.0    0.0           0.0  1.0
    1        0.0    0.0           1.0  1.2
    2        0.0    1.0           0.0  0.9
    3        0.0    1.0           1.0  1.1
    4        1.0    2.0           0.0  2.0
    5        1.0    2.0           1.0  2.1
    6        1.0    3.0           0.0  1.9
    7        1.0    3.0           1.0  2.2
    >>> treatment_col
    'treatment'

    Treatments nested within mice (a within-mouse design) put the
    treatment column after the mouse column instead:

    >>> df["treatment"] = ["ctrl", "drug"] * 4
    >>> matrix, treatment_col = design_matrix(df, "y ~ mouse/treatment", "treatment")
    >>> list(matrix.columns)
    ['mouse', 'treatment', 'y']
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    parsed = Formula(formula)
    if not hasattr(parsed, "lhs"):
        raise ValueError("formula must have a dependent variable: 'y ~ ...'")
    lhs = [str(term) for term in parsed.lhs]
    if len(lhs) != 1:
        raise ValueError("formula must have exactly one dependent variable")
    y_name = lhs[0]

    if not isinstance(parsed.rhs, SimpleFormula):
        raise ValueError(
            "formula right-hand side must be a single nesting chain " "like 'y ~ a/b/c'"
        )

    # formulaic expands a/b/c -> a, a:b, a:b:c in degree order; verify each
    # term extends the previous by one factor, i.e. a pure nesting chain
    chain = sorted(
        (term for term in parsed.rhs if str(term) != "1"),
        key=lambda term: len(term.factors),
    )
    if not chain:
        raise ValueError("formula must name at least one grouping variable")
    factor_lists = [[str(factor) for factor in term.factors] for term in chain]
    for prev, cur in zip(factor_lists, factor_lists[1:]):
        if cur[: len(prev)] != prev or len(cur) != len(prev) + 1:
            raise ValueError(
                "formula right-hand side must be a pure nesting chain "
                f"like 'y ~ a/b/c', got terms {[str(t) for t in chain]}"
            )
    level_names = [factors[-1] for factors in factor_lists]

    missing = [name for name in level_names + [y_name] if name not in data.columns]
    if missing:
        raise KeyError(f"columns named in formula not found in data: {missing}")

    if treatment not in level_names:
        raise ValueError(
            f"treatment={treatment!r} does not appear in the nesting chain "
            f"{level_names}"
        )
    treatment_idx = level_names.index(treatment)

    labels = data[treatment].to_numpy()
    try:
        y_values = data[y_name].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dependent variable {y_name!r} must be numeric") from exc

    # integer-code each level by its composite key (mouse:treatment:well),
    # so ids reused across enclosing clusters stay distinct; the treatment
    # column itself keeps its own label codes, since those are what get
    # permuted. sort=True makes the codes row-order invariant.
    columns = []
    for level, factors in enumerate(factor_lists):
        if level == treatment_idx:
            if np.issubdtype(np.asarray(labels).dtype, np.number):
                columns.append(labels.astype(np.float64))
            else:
                columns.append(pd.factorize(labels, sort=True)[0].astype(np.float64))
        else:
            codes = pd.MultiIndex.from_frame(data[factors]).factorize(sort=True)[0]
            columns.append(codes.astype(np.float64))

    # the data validates the declared treatment position: a treatment below
    # some grouping level must actually vary inside those blocks
    if treatment_idx > 0:
        blocks = columns[treatment_idx - 1]
        per_block = pd.Series(labels).groupby(blocks).nunique()
        if (per_block == 1).all():
            raise ValueError(
                f"{treatment!r} is constant within every "
                f"{level_names[treatment_idx - 1]!r} cluster; move it above "
                f"that level in the formula"
            )

    matrix = pd.DataFrame(dict(zip(level_names, columns)))
    matrix[y_name] = y_values
    matrix = matrix.sort_values(level_names, kind="stable", ignore_index=True)

    # synthesize an innermost replicate index if cells hold >1 measurement
    replicates = matrix.groupby(level_names, sort=False).cumcount()
    if replicates.max() > 0:
        index_name = "_measurement"
        while index_name in matrix.columns:
            index_name = "_" + index_name
        matrix.insert(len(level_names), index_name, replicates.astype(np.float64))

    return DesignMatrix(matrix, treatment)
