import unittest
import hierarch.resampling as resampling
from hierarch.power import DataSimulator
import scipy.stats as stats
import numpy as np
import pandas as pd


class TestIDClusters(unittest.TestCase):
    def test_id_clusters(self):
        hierarchies = ([2, 3, 3], [2, [4, 3], 3], [2, 3, [10, 11, 5, 6, 4, 3]])
        parameters = [[stats.norm, 0, 0], [stats.norm, 0, 0], [stats.norm, 0, 0]]
        sim = DataSimulator(parameters)

        for hierarchy in hierarchies:
            sim.fit(hierarchy)
            ret = resampling._id_cluster_counts(sim.container[:, :-2])
            for idx, level in enumerate(ret):
                if isinstance(hierarchy[idx], int):
                    for v in iter(level):
                        self.assertEqual(v, hierarchy[idx])
                elif isinstance(hierarchy[idx], list):
                    for idx_2, v in enumerate(level):
                        self.assertEqual(hierarchy[idx][idx_2], v)


class TestBootstrapper(unittest.TestCase):
    sim = DataSimulator([[stats.norm], [stats.norm], [stats.norm]])
    sim.fit([2, 3, 3])
    data_1 = sim.generate()

    def test_seeding(self):
        """
        Tests that setting the random_state generates the same bootstrapped sample.
        """

        boot = resampling.Bootstrapper(n_resamples=1, random_state=1, kind="weights")
        test_1 = next(boot.resample(self.data_1[:, :-2]))

        boot = resampling.Bootstrapper(n_resamples=1, random_state=1, kind="weights")
        test_2 = next(boot.resample(self.data_1[:, :-2]))

        for idx, v in enumerate(test_1):
            self.assertAlmostEqual(v, test_2[idx])

    def test_efron_bootstrapper(self):
        """
        Tests the Efron bootstrap "contract" - the final weights should sum
        to the sum of the original weights within each column.
        """
        boot = resampling.Bootstrapper(n_resamples=10, kind="weights")
        self.data_1[:, -1] = 1
        starts = (0, 1, 2)
        for start in starts:
            for test in boot.resample(self.data_1, start_col=start):
                self.assertAlmostEqual(self.data_1[:, -1].sum(), test.sum())

    def test_bayesian_bootstrapper(self):
        """
        Tests the Efron bootstrap "contract" - the final weights should sum
        to the sum of the original weights within each column.
        """
        boot = resampling.Bootstrapper(n_resamples=10, kind="bayesian")
        self.data_1[:, -1] = 1
        starts = (0, 1, 2)
        for start in starts:
            for test in boot.resample(self.data_1, start_col=start):
                self.assertAlmostEqual(self.data_1[:, -1].sum(), test.sum())

    def test_bootstrapper_exceptions(self):
        with self.assertRaises(KeyError) as raises:
            boot = resampling.Bootstrapper(n_resamples=10, kind="blah")
        self.assertIn("Invalid 'kind' argument.", str(raises.exception))

        boot = resampling.Bootstrapper(n_resamples=10, kind="weights")
        with self.assertRaises(ValueError) as raises:
            next(boot.resample(np.array(["str"])))
        self.assertIn(
            "Bootstrapper can only handle numeric datatypes. Please pre-process your data.",
            str(raises.exception),
        )

        with self.assertRaises(AttributeError) as raises:
            next(boot.resample(pd.DataFrame(self.data_1)))
        self.assertIn(
            "Bootstrapper can only handle numpy arrays. Please pre-process your data.",
            str(raises.exception),
        )

        with self.assertRaises(IndexError) as raises:
            next(boot.resample(self.data_1, skip=[2.3]))
        self.assertIn(
            "skip values must be integers corresponding to column indices.",
            str(raises.exception),
        )

        with self.assertRaises(IndexError) as raises:
            next(boot.resample(self.data_1, skip=[5]))
        self.assertIn("skip index out of bounds for this array.", str(raises.exception))


class TestWeightstoIndex(unittest.TestCase):
    def test_weights_to_index(self):
        """
        Tests that weights_to_index gives same array as np.arange(weights.size).repeat(weights)
        """
        weights = np.random.randint(10, size=10)
        indexes = resampling._weights_to_index(weights)
        np_indexes = np.arange(weights.size).repeat(weights)
        for idx, v in enumerate(indexes):
            self.assertEqual(v, np_indexes[idx])


class TestPermuter(unittest.TestCase):
    orig_data = np.arange(100).reshape((50, 2))

    def test_random(self):
        """
        Tests random permutation.
        """
        shuf_1 = self.orig_data.copy()
        shuf_2 = self.orig_data.copy()

        permuter = resampling.Permuter(random_state=1)
        permuter.fit(shuf_1, 0, exact=False)
        permuter.transform(shuf_1)

        permuter = resampling.Permuter(random_state=1)
        permuter.fit(shuf_2, 0, exact=False)
        permuter.transform(shuf_2)

        for idx, v in enumerate(shuf_1[:, 0]):
            self.assertEqual(v, shuf_2[:, 0][idx])

        for i in range(10):
            permuter.transform(shuf_1)
            shuf_1[:, 0].sort()

            for idx, v in enumerate(shuf_1[:, 0]):
                self.assertEqual(v, self.orig_data[:, 0][idx])

    def test_exact(self):
        """
        Test that exact permutations are generated in the same order.
        """
        shuf_1 = self.orig_data.copy()
        shuf_2 = self.orig_data.copy()

        permuter = resampling.Permuter()
        permuter.fit(shuf_1, 0, exact=True)
        permuter.transform(shuf_1)

        permuter = resampling.Permuter()
        permuter.fit(shuf_2, 0, exact=True)
        permuter.transform(shuf_2)

        for idx, v in enumerate(shuf_1[:, 0]):
            self.assertEqual(v, shuf_2[:, 0][idx])

        for i in range(10):
            permuter.transform(shuf_1)
            shuf_1[:, 0].sort()

            for idx, v in enumerate(shuf_1[:, 0]):
                self.assertEqual(v, self.orig_data[:, 0][idx])

    def test_permuter_exceptions(self):
        permuter = resampling.Permuter()
        with self.assertRaises(Exception) as raises:
            permuter.transform(self.orig_data)
        self.assertIn("Use fit() before using transform().", str(raises.exception))


if __name__ == "__main__":
    unittest.main()
