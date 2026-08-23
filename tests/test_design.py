"""design_matrix: formula -> hierarch column layout.

Contract: the emitted DataFrame has one column per nesting-chain level in
formula order (plus a synthesized replicate index when innermost cells hold
more than one measurement), with the dependent variable last; the treatment's
position in the chain is its position in the layout; composite relabeling
keeps reused ids distinct; output is invariant to input row order; the result
splats into hypothesis_test and reproduces the hand-built matrix bit-for-bit.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import hierarch.stats as hs
from hierarch.design import DesignMatrix, design_matrix


def _quiet(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return hs.hypothesis_test(*args, **kwargs)


@pytest.fixture
def between_df():
    """Two treatments, three donors each, two measurements per donor."""
    rng = np.random.default_rng(7)
    rows = []
    for t, treatment in enumerate(["ctrl", "drug"]):
        for donor in range(3):
            for _ in range(2):
                rows.append((treatment, f"D{t}{donor}", rng.normal(loc=t)))
    return pd.DataFrame(rows, columns=["treatment", "donor", "y"])


@pytest.fixture
def within_df():
    """Four mice, both treatments within each, three wells per cell."""
    rng = np.random.default_rng(3)
    rows = [
        (mouse, treatment, well, rng.normal())
        for mouse in ["m1", "m2", "m3", "m4"]
        for treatment in ["ctrl", "drug"]
        for well in ["w1", "w2", "w3"]
    ]
    return pd.DataFrame(rows, columns=["mouse", "treatment", "well", "y"])


def test_between_layout(between_df):
    matrix, treatment_col = design_matrix(
        between_df, "y ~ treatment/donor", "treatment"
    )
    assert treatment_col == "treatment"
    assert list(matrix.columns) == ["treatment", "donor", "_measurement", "y"]
    assert (matrix.dtypes == np.float64).all()
    # six distinct donors, relabeled 0..5 in hierarchical order
    assert sorted(matrix["donor"].unique()) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert set(matrix["_measurement"]) == {0.0, 1.0}


def test_within_layout(within_df):
    matrix, treatment_col = design_matrix(
        within_df, "y ~ mouse/treatment/well", "treatment"
    )
    assert list(matrix.columns) == ["mouse", "treatment", "well", "y"]
    assert treatment_col == "treatment"
    # well ids repeat across arms: the composite key must keep the arms'
    # wells distinct, six per mouse
    per_mouse = matrix.groupby("mouse")["well"].nunique()
    assert (per_mouse == 6).all()


def test_within_named_wells_match_auto_indexed(within_df):
    """Naming the replicate level and omitting it give the same test."""
    named = design_matrix(within_df, "y ~ mouse/treatment/well", "treatment")
    auto = design_matrix(within_df, "y ~ mouse/treatment", "treatment")
    assert list(auto.data.columns) == ["mouse", "treatment", "_measurement", "y"]
    p_named = _quiet(*named, bootstraps=1, permutations="all", random_state=1)
    p_auto = _quiet(*auto, bootstraps=1, permutations="all", random_state=1)
    assert p_named == p_auto


def test_reused_subject_ids_relabeled():
    """Subject '1' in family 1 and family 2 are different subjects."""
    df = pd.DataFrame(
        {
            "treatment": ["a"] * 4 + ["b"] * 4,
            "family": [1, 1, 2, 2, 3, 3, 4, 4],
            "subject": [1, 2, 1, 2, 1, 2, 1, 2],
            "y": np.arange(8.0),
        }
    )
    matrix, _ = design_matrix(df, "y ~ treatment/family/subject", "treatment")
    assert matrix["subject"].nunique() == 8


def test_roundtrip_matches_hand_built_matrix(between_df):
    """The formula path reproduces the hand-built matrix bit-for-bit."""
    by_hand = np.column_stack(
        [
            pd.factorize(between_df["treatment"], sort=True)[0],
            pd.factorize(between_df["donor"], sort=True)[0],
            np.tile([0.0, 1.0], 6),
            between_df["y"].to_numpy(),
        ]
    ).astype(np.float64)
    matrix, treatment_col = design_matrix(
        between_df, "y ~ treatment/donor", "treatment"
    )
    assert np.array_equal(matrix.to_numpy(), by_hand)
    p_formula = _quiet(
        matrix, treatment_col, bootstraps=50, permutations="all", random_state=5
    )
    p_hand = _quiet(by_hand, 0, bootstraps=50, permutations="all", random_state=5)
    assert p_formula == p_hand


def test_row_order_invariance(within_df):
    sorted_matrix = design_matrix(
        within_df, "y ~ mouse/treatment/well", "treatment"
    ).data
    shuffled = within_df.sample(frac=1, random_state=11).reset_index(drop=True)
    shuffled_matrix = design_matrix(
        shuffled, "y ~ mouse/treatment/well", "treatment"
    ).data
    assert np.array_equal(sorted_matrix.to_numpy(), shuffled_matrix.to_numpy())


def test_numeric_treatment_passthrough():
    df = pd.DataFrame(
        {
            "dose": [0.0, 0.0, 10.0, 10.0, 50.0, 50.0],
            "plate": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "y": np.arange(6.0),
        }
    )
    matrix, _ = design_matrix(df, "y ~ dose/plate", "dose")
    assert sorted(matrix["dose"].unique()) == [0.0, 10.0, 50.0]


def test_no_replicate_column_when_cells_unique():
    df = pd.DataFrame(
        {
            "treatment": ["a", "a", "b", "b"],
            "mouse": ["m1", "m2", "m3", "m4"],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    matrix, _ = design_matrix(df, "y ~ treatment/mouse", "treatment")
    assert list(matrix.columns) == ["treatment", "mouse", "y"]


def test_result_is_named_tuple(between_df):
    result = design_matrix(between_df, "y ~ treatment/donor", "treatment")
    assert isinstance(result, DesignMatrix)
    assert result.treatment_col == "treatment"
    assert result.data is result[0]


@pytest.mark.parametrize(
    "formula, treatment, message",
    [
        ("y ~ treatment + donor", "treatment", "pure nesting chain"),
        ("y ~ treatment*donor", "treatment", "pure nesting chain"),
        ("y ~ donor", "treatment", "does not appear in the nesting chain"),
        ("y ~ treatment/donor | donor", "treatment", "single nesting chain"),
        ("treatment/donor", "treatment", "dependent variable"),
        ("y + treatment ~ donor", "treatment", "one dependent variable"),
        ("y ~ 1", "treatment", "at least one grouping variable"),
    ],
)
def test_formula_errors(between_df, formula, treatment, message):
    with pytest.raises(ValueError, match=message):
        design_matrix(between_df, formula, treatment)


def test_misdeclared_treatment_position_errors(between_df):
    """A between-donor treatment declared below the donor level."""
    with pytest.raises(ValueError, match="constant within every 'donor'"):
        design_matrix(between_df, "y ~ donor/treatment", "treatment")


def test_missing_column_errors(between_df):
    with pytest.raises(KeyError, match="mouse"):
        design_matrix(between_df, "y ~ treatment/mouse", "treatment")


def test_non_numeric_dependent_errors(between_df):
    df = between_df.assign(y=["low"] * 6 + ["high"] * 6)
    with pytest.raises(ValueError, match="must be numeric"):
        design_matrix(df, "y ~ treatment/donor", "treatment")


def test_non_dataframe_errors():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        design_matrix(np.ones((4, 3)), "y ~ a/b", "a")


# A sweep over design shapes: chain depths 2-4, balanced and unbalanced,
# 2 and 3 arms, with the treatment column at every feasible level. Each
# case takes a make_design matrix (within-parent codes) as the reference,
# dresses it up as a shuffled string-labeled DataFrame with ids reused
# across parents, and requires the formula path to reproduce the expected
# recoded matrix and the reference matrix's seeded p-value bit-for-bit.
SWEEP = [
    ("2lvl_between", [2, 5], 0),
    ("3lvl_treatment_top", [2, 4, 3], 0),
    ("3lvl_treatment_mid", [2, 3, 2], 1),
    ("3lvl_unbalanced", [2, (3, 5), (2, 4)], 0),
    ("4lvl_3arm_unbalanced", [3, (2, 4), 2, 2], 0),
    ("4lvl_blocked", [3, 2, 3, 2], 1),
    ("4lvl_treatment_inner", [4, 2, 2, 2], 2),
]


@pytest.mark.parametrize("name, hierarchy, t_idx", SWEEP, ids=[c[0] for c in SWEEP])
def test_design_sweep(name, hierarchy, t_idx):
    from tests._reference import make_design

    reference = make_design(hierarchy, rng=np.random.default_rng(hash(name) % 2**32))
    n_levels = len(hierarchy)
    level_names = ["treatment" if i == t_idx else f"lvl{i}" for i in range(n_levels)]

    # string labels reuse the within-parent codes, so e.g. every parent has
    # a child named "lvl2_01"; arms share names across blocks
    df = pd.DataFrame(
        {
            level_names[i]: [
                f"arm{int(c)}" if i == t_idx else f"lvl{i}_{int(c):02d}"
                for c in reference[:, i]
            ]
            for i in range(n_levels)
        }
    )
    df["y"] = reference[:, -1]
    df = df.sample(frac=1, random_state=13).reset_index(drop=True)

    formula = "y ~ " + "/".join(level_names)
    matrix, treatment_col = design_matrix(df, formula, "treatment")
    assert list(matrix.columns) == level_names + ["y"]
    assert treatment_col == "treatment"

    # expected recoding: treatment keeps its own (0-based) labels, other
    # levels get global sorted-composite codes
    expected = np.empty_like(reference)
    for i in range(n_levels):
        if i == t_idx:
            expected[:, i] = reference[:, i] - 1
        else:
            expected[:, i] = np.unique(
                reference[:, : i + 1], axis=0, return_inverse=True
            )[1].ravel()
    expected[:, -1] = reference[:, -1]
    assert np.array_equal(matrix.to_numpy(), expected)

    # the recoding is a relabeling of the same structure: seeded p-values
    # must match the reference matrix exactly
    p_formula = _quiet(
        matrix, treatment_col, bootstraps=10, permutations=50, random_state=42
    )
    p_reference = _quiet(
        reference, t_idx, bootstraps=10, permutations=50, random_state=42
    )
    assert p_formula == p_reference
