import unittest
from hierarch import internal_functions
from hierarch.power import DataSimulator
import scipy.stats as stats
import numpy as np


class TestSetRandomState(unittest.TestCase):
    def _try_seed(self, seed):
        internal_functions.set_numba_random_state(seed)

    def test_set_random_state(self):
        """
        Test normal behavior
        """
        seeds = (1, 1000, 2**32)
        for seed in seeds:
            self._try_seed(seed)

    def test_set_random_state_exceptions(self):
        """
        Checks that exceptions are raised when seed is not an integer.
        """
        seeds = (1.5,)
        for seed in seeds:
            with self.assertRaises(ValueError) as raises:
                self._try_seed(seed)
            self.assertIn("Numba seed must be an integer.", str(raises.exception))


class TestDataGrabber(unittest.TestCase):
    def _check_samples(self, data, treatment_col, treatment_labels, ret):
        """
        Check lengths of grabbed samples.
        """
        for idx, key in enumerate(treatment_labels):
            self.assertEqual(ret[idx].size, (data[:, treatment_col] == key).sum())

    def test_data_grabber(self):
        hierarchies = ([2, 3, 3], [2, [4, 3], 3], [2, 3, [10, 11, 5, 6, 4, 3]])
        parameters = [[stats.norm, 0, 0], [stats.norm, 0, 0], [stats.norm, 0, 0]]
        sim = DataSimulator(parameters)

        for hierarchy in hierarchies:
            sim.fit(hierarchy)
            data = sim.generate()

            for treatment_col in range(data.shape[1] - 1):
                treatment_labels = np.unique(data[:, treatment_col])
                ret = internal_functions.nb_data_grabber(
                    data, treatment_col, treatment_labels
                )
                self._check_samples(data, treatment_col, treatment_labels, ret)


class TestBivarCentralMoment(unittest.TestCase):
    def test_var(self):
        """
        Check that bivar_central_moment(x, x) matches np.var(x)
        """
        rng = np.random.default_rng(123)
        x = rng.random(size=10)
        ddofs = (0, 1, 2)
        for ddof in ddofs:
            ret = internal_functions.bivar_central_moment(x, x, pow=1, ddof=ddof)
            self.assertAlmostEqual(ret, np.var(x, ddof=ddof))

    def test_cov(self):
        """
        Check that bivar_central_moment(x, y) matches np.cov(x, y)[0, 1] (because np.cov makes
        a covariance matrix)
        """
        rng = np.random.default_rng(456)
        x = rng.random(size=10)
        y = rng.random(size=10)
        ddofs = (0, 1, 2)
        for ddof in ddofs:
            ret = internal_functions.bivar_central_moment(x, y, pow=1, ddof=ddof)
            self.assertAlmostEqual(ret, np.cov(x, y, ddof=ddof)[0, 1])


class TestBoundedUInt(unittest.TestCase):
    def _check_bound(self, ub, ret):
        """
        Check that all generated numbers are smaller than the upper bound.
        """
        for v in iter(ret):
            self.assertLess(v, ub)

    def test_bounded_int(self):
        ubs = (10, 100, 1000, 10000)
        for ub in ubs:
            size = 10000
            ret = np.empty(size)
            for idx, v in enumerate(ret):
                ret[idx] = internal_functions.bounded_uint(ub)
            self._check_bound(ub, ret)


class TestMultisetPermutations(unittest.TestCase):
    def test_msp(self):
        """
        Tests to make sure the number of permutations is equal to the expected amount based
        on the unique elements in the list.
        """
        lists = (
            [1, 1, 1, 2, 2, 2],
            [1, 1, 2, 2, 3, 3],
            [1, 2, 3, 4, 5, 6],
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
        )
        for set in lists:
            num = np.math.factorial(len(set))
            denom = 1
            for v in iter(np.unique(set, return_counts=True)[1]):
                denom *= np.math.factorial(v)
            expected_length = num / denom

            self.assertEqual(expected_length, len(list(internal_functions.msp(set))))


if __name__ == "__main__":
    unittest.main()
