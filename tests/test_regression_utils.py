import unittest

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from hierarch.regression_utils import pseudo_inverse_last_level, collapse_hierarchy


class TestEncodedLastLevel(unittest.TestCase):
    def test_encoded_last_level_no_rep(self):

        design = np.array([[1, 1], [1, 2], [1, 3]])

        expected_encoding = np.eye(3, dtype=np.float64)
        expected_uniques = np.array([1, 1, 1])[:, None]

        generated_encoding, generated_uniques = pseudo_inverse_last_level(design)

        assert_equal(generated_encoding, expected_encoding)
        assert_equal(generated_uniques, expected_uniques)

    def test_encoded_last_level_reps(self):

        design = np.array([[1, 1], [1, 2], [1, 3]]).repeat(3, axis=0)

        enc = np.eye(3, dtype=np.float64).repeat(3, axis=0)
        expected_pseudo_inverse = np.linalg.inv(enc.T @ enc) @ enc.T
        expected_uniques = np.array([1, 1, 1])[:, None]

        generated_encoding, generated_uniques = pseudo_inverse_last_level(design)

        assert_equal(generated_encoding, expected_pseudo_inverse)
        assert_equal(generated_uniques, expected_uniques)


class TestCollapseHierarchy(unittest.TestCase):
    def test_single_collapse(self):
        design = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]).repeat(2, axis=0)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        weights = np.array([1.0 for i in y])

        expected_x = np.array([1, 1, 1]).reshape((-1, 1))
        expected_y = np.array([1.5, 3.5, 5.5])

        generated_x, generated_y = collapse_hierarchy(design, y, weights, 1)

        assert_almost_equal(generated_x, expected_x)
        assert_almost_equal(generated_y, expected_y)

    def test_multiple_collapse(self):
        design = np.array([[1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2]]).repeat(
            3, axis=0
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).repeat(2)
        weights = np.array([1.0 for i in y])

        expected_x = np.array([1, 1]).reshape((-1, 1))
        expected_y = np.array([2, 5])

        generated_x, generated_y = collapse_hierarchy(design, y, weights, 2)

        assert_almost_equal(generated_x, expected_x)
        assert_almost_equal(generated_y, expected_y)

    def test_weighted_collapse(self):
        design = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]).repeat(2, axis=0)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        weights = np.array([2.0, 0.0, 2.0, 0, 0, 2.0])

        expected_x = np.array([1, 1, 1]).reshape((-1, 1))
        expected_y = np.array([1.0, 3.0, 6.0])

        generated_x, generated_y = collapse_hierarchy(design, y, weights, 1)

        assert_almost_equal(generated_x, expected_x)
        assert_almost_equal(generated_y, expected_y)
