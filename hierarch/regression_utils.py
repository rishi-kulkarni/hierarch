from functools import lru_cache
from itertools import repeat
from typing import Tuple
import numpy as np
from numba import jit


@jit(nopython=True)
def intercepts(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y)[0]


def encoded_last_level(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One-hot encoding of the last level of hierarchy.

    Delegates to _encoded_last_level, which is cached.

    Parameters
    ----------
    X : np.ndarray
        design matrix where each column is a level of
        hierarchy

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        one-hot encoded final level,
        unique rows in the final level

    Examples
    --------

    Consider a simple design matrix that has three units
    in its lowest level:

    >>> design = np.array([[1, 1], [1, 2], [1, 3]])
    >>> encoded_last_level(design)
    (array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]]), array([[1, 1],
           [1, 2],
           [1, 3]]))

    >>> repeated_design = design.repeat(2, axis=0)
    >>> encoded_last_level(repeated_design)
    (array([[1., 0., 0.],
           [1., 0., 0.],
           [0., 1., 0.],
           [0., 1., 0.],
           [0., 0., 1.],
           [0., 0., 1.]]), array([[1., 1.],
           [1., 2.],
           [1., 3.]]))
    """
    hashable = tuple(map(tuple, X.tolist()))
    return _encoded_last_level(hashable)


@lru_cache
def _encoded_last_level(X_hashable: Tuple[Tuple]) -> np.ndarray:
    uniq, inverse = np.unique(X_hashable, axis=0, return_inverse=True)
    return np.eye(len(uniq))[inverse], uniq


def collapse_hierarchy(
    design: np.ndarray, y: np.ndarray, weights: np.ndarray, levels_to_collapse: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Collapses a level of hierarchy by computing regression coefficients and
    passing them up to the next level.

    Parameters
    ----------
    design : np.ndarray
    y : np.ndarray
    weights : np.ndarray
        Bootstrapped weights.
    levels_to_collapse : int
        Number of times to repeat the process.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Collapsed X and y

    Examples
    --------

    This function implements a single step of two-stage regression.

    >>> design = np.array([[1., 1.], [1., 2.], [1., 3.]]).repeat(2, axis=0)
    >>> y = np.array([1., 2., 3., 4., 5., 6.])
    >>> weights = np.array([1. for i in y])

    Calling collapse_hierarchy with level_to_collapse=1 should yield us a design
    matrix with a single column and y-values representing the average of every
    pair of y-values.

    >>> collapse_hierarchy(design, y, weights, 1)
    (array([[1.],
           [1.],
           [1.]]), array([1.5, 3.5, 5.5]))

    This function computes weighted averages for each intercept, as well.

    >>> weights = np.array([2, 0, 2, 0, 2, 0])
    >>> collapse_hierarchy(design, y, weights, 1)
    (array([[1.],
           [1.],
           [1.]]), array([1.5, 3.5, 5.5]))

    Several levels can be collapsed at once:
    >>> design = np.array([[1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2]])
    >>> y = np.array([1., 2., 3., 4.])
    >>> weights = np.array([1 for i in y])
    >>> collapse_hierarchy(design, y, weights, 2)
    (array([[1],
           [1]]), array([1.5, 3.5]))

    """
    out_x = design
    # this is only allowed when we're only computing
    # intercepts - it's not a general approach to
    # weighted regression
    out_y = y * weights
    for _ in repeat(None, levels_to_collapse):
        regressor, out_x = encoded_last_level(out_x)
        out_y = intercepts(regressor, out_y)
        out_x = out_x[:, :-1]
    return out_x, out_y
