from dataclasses import dataclass
from typing import Iterable, Tuple, Union

import numpy as np

from hierarch.internal_functions import (
    id_cluster_counts,
    msp,
)


BOOTSTRAP_KINDS = ("weights", "indexes", "bayesian")


@dataclass(frozen=True)
class BootstrapPlan:
    """Precomputed cluster geometry for nested bootstrapping.

    ``children_per_level[k]`` holds the number of level-(k+1) subclusters
    within each level-k cluster (rows are the deepest level's children), in
    lexicographic order. ``resampled_levels[k]`` is False for levels listed
    in ``skip`` at plan time. ``block_starts_per_level`` and
    ``multinomial_groups_per_level`` are shape-derived lookup tables so that
    drawing weights spends no time re-deriving geometry: each multinomial
    group is one ``(size, cluster_indices, scatter_indices, pvals)`` batch of
    same-sized clusters that can be drawn in a single Generator call.
    """

    children_per_level: Tuple[np.ndarray, ...]
    resampled_levels: Tuple[bool, ...]
    block_starts_per_level: Tuple[np.ndarray, ...]
    multinomial_groups_per_level: Tuple[tuple, ...]


def bootstrap_plan(design: np.ndarray, skip: Iterable[int] = ()) -> BootstrapPlan:
    """Analyze the hierarchy of a lexsorted design matrix.

    Parameters
    ----------
    design : 2D numeric ndarray
        Design columns only (no dependent-variable column). Must be
        lexicographically sorted.
    skip : iterable of ints, optional
        Levels that should not be resampled when drawing weights.

    Returns
    -------
    BootstrapPlan
    """
    cluster_dict = id_cluster_counts(design)
    children = tuple(reversed(list(cluster_dict.values())))
    skip = frozenset(skip)
    resampled = tuple(level not in skip for level in range(len(children)))

    starts, groups = [], []
    for level_children in children:
        block_starts = np.concatenate(([0], np.cumsum(level_children[:-1])))
        starts.append(block_starts)
        level_groups = []
        for size in np.unique(level_children):
            clusters = np.flatnonzero(level_children == size)
            scatter = block_starts[clusters][:, None] + np.arange(size)
            level_groups.append((int(size), clusters, scatter, [1 / size] * size))
        groups.append(tuple(level_groups))
    return BootstrapPlan(children, resampled, tuple(starts), tuple(groups))


def draw_bootstrap_weights(
    plan: BootstrapPlan,
    rng: np.random.Generator,
    start: int,
    kind: str,
) -> np.ndarray:
    """Draw one set of per-row bootstrap weights for a fitted plan.

    Parameters
    ----------
    plan : BootstrapPlan
    rng : numpy.random.Generator
        Source of randomness; advanced by this call.
    start : int
        First level to resample.
    kind : { "weights", "indexes", "bayesian" }
        "weights" and "indexes" draw integer (Efron) weights; "bayesian"
        draws continuous Dirichlet weights.

    Returns
    -------
    1D array of per-row weights
        Integer dtype for the Efron bootstrap, float64 for the Bayesian.
    """
    if kind == "bayesian":
        weights = np.ones(len(plan.children_per_level[start]))
    else:
        weights = np.ones(len(plan.children_per_level[start]), dtype=np.int64)
    for level in range(start, len(plan.children_per_level)):
        children = plan.children_per_level[level]
        if not plan.resampled_levels[level]:
            weights = np.repeat(weights, children)
        elif kind == "bayesian":
            weights = _draw_dirichlet_level(
                weights, children, plan.block_starts_per_level[level], rng
            )
        else:
            weights = _draw_multinomial_level(
                weights,
                plan.multinomial_groups_per_level[level],
                int(children.sum()),
                rng,
            )
    return weights


def draw_bootstrap_weights_batch(
    plan: BootstrapPlan,
    rng: np.random.Generator,
    start: int,
    kind: str,
    size: int,
) -> np.ndarray:
    """Draw ``size`` independent sets of per-row bootstrap weights at once.

    Equivalent to ``size`` calls of :func:`draw_bootstrap_weights` (with a
    different stream consumption), but each resampling level is drawn for
    every replicate in a single Generator call.

    Parameters
    ----------
    plan : BootstrapPlan
    rng : numpy.random.Generator
        Source of randomness; advanced by this call.
    start : int
        First level to resample.
    kind : { "weights", "indexes", "bayesian" }
    size : int
        Number of independent weight sets to draw.

    Returns
    -------
    2D array of shape (size, number of rows)
    """
    if kind == "bayesian":
        weights = np.ones((size, len(plan.children_per_level[start])))
    else:
        weights = np.ones((size, len(plan.children_per_level[start])), dtype=np.int64)
    for level in range(start, len(plan.children_per_level)):
        children = plan.children_per_level[level]
        total = int(children.sum())
        if not plan.resampled_levels[level]:
            weights = np.repeat(weights, children, axis=1)
        elif kind == "bayesian":
            gammas = rng.standard_gamma(1.0, size=(size, total))
            sums = np.add.reduceat(gammas, plan.block_starts_per_level[level], axis=1)
            scale = weights * children
            weights = (
                gammas
                / np.repeat(sums, children, axis=1)
                * np.repeat(scale, children, axis=1)
            )
        else:
            out = np.empty((size, total), dtype=np.int64)
            for csize, clusters, scatter, pvals in plan.multinomial_groups_per_level[
                level
            ]:
                flat_n = (csize * weights[:, clusters]).ravel()
                draws = rng.multinomial(flat_n, pvals)
                out[:, scatter] = draws.reshape(size, len(clusters), csize)
            weights = out
    return weights


def _draw_multinomial_level(weights, groups, total, rng):
    """One nested Efron resampling step: each cluster's weight is split
    among its children by a uniform multinomial draw."""
    out = np.empty(total, dtype=np.int64)
    for size, clusters, scatter, pvals in groups:
        out[scatter] = rng.multinomial(size * weights[clusters], pvals)
    return out


def _draw_dirichlet_level(weights, children, block_starts, rng):
    """One nested Bayesian resampling step: each cluster's weight is split
    among its children by a flat Dirichlet draw."""
    gammas = rng.standard_gamma(1.0, size=int(children.sum()))
    sums = np.add.reduceat(gammas, block_starts)
    scale = weights * children
    return gammas / np.repeat(sums, children) * np.repeat(scale, children)


@dataclass(frozen=True)
class PermutationPlan:
    """Precomputed structure for cluster-aware permutation of one column.

    A permutable *unit* is a distinct row of the design columns up to and
    including the column after the target column: whole clusters move
    together when the target column is above the row level. ``unit_values``
    holds each unit's target-column value; ``stratum_starts`` bounds the
    runs of units that may exchange values (a single [0, n] stratum when
    the target column is column 0); ``row_repeats`` expands unit labels
    back to data rows, or None when units and rows coincide.
    """

    col: int
    unit_values: np.ndarray
    stratum_starts: np.ndarray
    row_repeats: Union[np.ndarray, None]


def permutation_plan(data: np.ndarray, col_to_permute: int) -> PermutationPlan:
    """Analyze the target data for cluster-aware permutation.

    Parameters
    ----------
    data : 2D numeric ndarray
        Target data. Must be lexicographically sorted.
    col_to_permute : int
        Index of the column to be permuted.

    Returns
    -------
    PermutationPlan
    """
    values, indexes, counts = np.unique(
        data[:, : col_to_permute + 2], return_index=True, return_counts=True, axis=0
    )
    if col_to_permute == 0:
        stratum_starts = np.array([0, len(values)])
    else:
        ids = np.unique(values[:, :-2], axis=0, return_inverse=True)[1].ravel()
        changes = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        stratum_starts = np.concatenate(([0], changes, [len(values)]))
    unit_values = values[:, -2].copy()
    row_repeats = None if indexes.size == len(data) else counts
    return PermutationPlan(col_to_permute, unit_values, stratum_starts, row_repeats)


def draw_permuted_labels(
    plan: PermutationPlan, rng: np.random.Generator, size: int
) -> np.ndarray:
    """Draw a batch of cluster-aware permutations of the target column.

    Each output row is one independent uniform (stratified) permutation of
    the unit labels, expanded to data rows if units span multiple rows.

    Parameters
    ----------
    plan : PermutationPlan
    rng : numpy.random.Generator
        Source of randomness; advanced by this call.
    size : int
        Number of permutations to draw.

    Returns
    -------
    2D array of shape (size, number of data rows)
    """
    labels = np.tile(plan.unit_values, (size, 1))
    for start, stop in zip(plan.stratum_starts[:-1], plan.stratum_starts[1:]):
        if stop - start > 1:
            block = labels[:, start:stop]
            rng.permuted(block, axis=1, out=block)
    if plan.row_repeats is not None:
        labels = np.repeat(labels, plan.row_repeats, axis=1)
    return labels


def exact_label_matrix(data: np.ndarray, col_to_permute: int) -> np.ndarray:
    """Enumerate every distinct permutation of the target column's unit
    labels, in multiset-permutation order, expanded to data rows.

    Parameters
    ----------
    data : 2D numeric ndarray
        Target data. Must be lexicographically sorted.
    col_to_permute : int
        Index of the column to be permuted. Must be 0.

    Returns
    -------
    2D array of shape (number of distinct permutations, number of data rows)
    """
    values, counts = np.unique(
        data[:, : col_to_permute + 2], return_counts=True, axis=0
    )
    col_values = values[:, -2].tolist()
    labels = np.array(list(msp(col_values)))
    if len(col_values) != len(data):
        labels = np.repeat(labels, counts, axis=1)
    return labels
