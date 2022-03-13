import unittest
import hierarch.resampling as resampling
from hierarch.power import DataSimulator
import scipy.stats as stats
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd


class TestIDClusters(unittest.TestCase):
    def test_id_clusters(self):
        hierarchies = ([2, 3, 3], [2, [4, 3], 3], [2, 3, [10, 11, 5, 6, 4, 3]])
        parameters = [[stats.norm, 0, 0], [stats.norm, 0, 0], [stats.norm, 0, 0]]
        sim = DataSimulator(parameters)

        for hierarchy in hierarchies:
            sim.fit(hierarchy)
            ret = resampling.id_cluster_counts(sim.container[:, :-2])
            for idx, level in enumerate(ret):
                if isinstance(hierarchy[idx], int):
                    for v in iter(level):
                        self.assertEqual(v, hierarchy[idx])
                elif isinstance(hierarchy[idx], list):
                    for idx_2, v in enumerate(level):
                        self.assertEqual(hierarchy[idx][idx_2], v)


class TestBootstrapper(unittest.TestCase):
    def setUp(self):
        sim = DataSimulator([[stats.norm], [stats.norm], [stats.norm]])
        sim.fit([2, 3, 3])
        self.X = sim.generate()[:, :-2]

    def test_seeding(self):
        """
        Tests that setting the random_state generates the same bootstrapped sample.
        """

        boot_1 = resampling.Bootstrapper(n_resamples=10, random_state=1, kind="weights")
        test_1_list = [sample for sample in boot_1.resample(self.X)]

        boot_2 = resampling.Bootstrapper(n_resamples=10, random_state=1, kind="weights")
        test_2_list = [sample for sample in boot_2.resample(self.X)]

        for test_1, test_2 in zip(
            test_1_list,
            test_2_list,
        ):
            assert_equal(test_1, test_2)

    def test_efron_bootstrapper_weights(self):
        """
        Tests the Efron bootstrap "contract" - the final weights should sum
        to the sum of the original weights within each column.

        Checks that sums of weights are as expected based on what columns
        are being resampled, i.e. if starting at column 0 in this test,
        all weights should sum to 18. If starting at column 1, there should
        be two groups of weights that sum to 9, etc.
        """
        boot = resampling.Bootstrapper(n_resamples=10, kind="weights")

        starts = (0, 1, 2)
        for start in starts:
            _, index_bins, bin_sums = np.unique(
                self.X[:, :start], axis=0, return_index=True, return_counts=True
            )
            for test in boot.resample(self.X, start_col=start):
                generated_bin_sums = np.add.reduceat(test, index_bins)
                assert_equal(generated_bin_sums, bin_sums)

    def test_efron_bootstrapper_weights_with_y(self):
        """
        Tests the Efron bootstrap "contract" - the final weights should sum
        to the sum of the original weights within each column.

        Checks that sums of weights are as expected based on what columns
        are being resampled, i.e. if starting at column 0 in this test,
        all weights should sum to 18. If starting at column 1, there should
        be two groups of weights that sum to 9, etc.
        """
        boot = resampling.Bootstrapper(n_resamples=10, kind="weights")
        y = np.array([1 for row in self.X])

        starts = (0, 1, 2)
        for start in starts:
            _, index_bins, bin_sums = np.unique(
                self.X[:, :start], axis=0, return_index=True, return_counts=True
            )
            for design, reweighted_y in boot.resample(self.X, y, start_col=start):
                generated_bin_sums = np.add.reduceat(reweighted_y, index_bins)
                assert_equal(generated_bin_sums, bin_sums)

    def test_efron_bootstrapper_indexes(self):
        """
        Tests the Efron bootstrap "contract" - counts of unique values of
        indices should sum to the the original weights within each column.
        """
        boot = resampling.Bootstrapper(n_resamples=10, kind="indexes")

        starts = (0, 1, 2)
        for start in starts:
            _, index_bins, bin_sums = np.unique(
                self.X[:, :start], axis=0, return_index=True, return_counts=True
            )
            index_bins = np.append(index_bins, len(self.X))
            for test in boot.resample(self.X, start_col=start):
                generated_bin_sums = [
                    np.sum((test >= low) & (test < high))
                    for low, high in zip(index_bins[:-1], index_bins[1:])
                ]
                assert_equal(generated_bin_sums, bin_sums)

    def test_efron_bootstrapper_indexes_with_y(self):
        """
        Tests the Efron bootstrap "contract" - counts of unique values of
        indices should sum to the the original weights within each column.
        """
        boot = resampling.Bootstrapper(n_resamples=10, kind="indexes")

        y = np.array([idx for idx, row in enumerate(self.X)])
        starts = (0, 1, 2)
        for start in starts:
            _, index_bins, bin_sums = np.unique(
                self.X[:, :start], axis=0, return_index=True, return_counts=True
            )
            index_bins = np.append(index_bins, len(self.X))
            for design, reindexed_y in boot.resample(self.X, y, start_col=start):
                generated_bin_sums = [
                    np.sum((reindexed_y >= low) & (reindexed_y < high))
                    for low, high in zip(index_bins[:-1], index_bins[1:])
                ]
                assert_equal(generated_bin_sums, bin_sums)

    def test_bayesian_bootstrapper(self):
        """
        Tests the Bayesian bootstrap "contract" - the final weights should sum
        to the sum of the original weights within each column. Uses
        assert_almost_equal because bayesian bootstrap weights are floats.
        """
        boot = resampling.Bootstrapper(n_resamples=10, kind="bayesian")

        starts = (0, 1, 2)
        for start in starts:
            _, index_bins, bin_sums = np.unique(
                self.X[:, :start], axis=0, return_index=True, return_counts=True
            )
            for test in boot.resample(self.X, start_col=start):
                generated_bin_sums = np.add.reduceat(test, index_bins)
                assert_almost_equal(generated_bin_sums, bin_sums)

    def test_bayesian_bootstrapper_weights_with_y(self):
        """
        Tests the Efron bootstrap "contract" - the final weights should sum
        to the sum of the original weights within each column.

        Checks that sums of weights are as expected based on what columns
        are being resampled, i.e. if starting at column 0 in this test,
        all weights should sum to 18. If starting at column 1, there should
        be two groups of weights that sum to 9, etc.
        """
        boot = resampling.Bootstrapper(n_resamples=10, kind="bayesian")

        y = np.array([1 for row in self.X])
        starts = (0, 1, 2)
        for start in starts:
            _, index_bins, bin_sums = np.unique(
                self.X[:, :start], axis=0, return_index=True, return_counts=True
            )
            for design, reweighted_y in boot.resample(self.X, y, start_col=start):
                generated_bin_sums = np.add.reduceat(reweighted_y, index_bins)
                assert_almost_equal(generated_bin_sums, bin_sums)

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
            next(boot.resample(pd.DataFrame(self.X)))
        self.assertIn(
            "Bootstrapper can only handle numpy arrays. Please pre-process your data.",
            str(raises.exception),
        )

        with self.assertRaises(IndexError) as raises:
            next(boot.resample(self.X, skip=[2.3]))
        self.assertIn(
            "skip contains invalid column indexes for X:",
            str(raises.exception),
        )

        with self.assertRaises(IndexError) as raises:
            next(boot.resample(self.X, skip=[5]))
        self.assertIn(
            "skip contains invalid column indexes for X:", str(raises.exception)
        )


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
    def setUp(self):
        self.orig_data = np.arange(100).reshape((50, 2))

    def test_random_seeding(self):
        """
        Tests random permutation.
        """
        shuf_1 = self.orig_data.copy()

        permuter = resampling.Permuter(n_resamples=10, exact=False, random_state=1)
        sample_1 = [x for x in permuter.resample(shuf_1, 0)]

        permuter = resampling.Permuter(n_resamples=10, exact=False, random_state=1)
        sample_2 = [x for x in permuter.resample(shuf_1, 0)]

        for arr_1, arr_2 in zip(sample_1, sample_2):
            assert_equal(arr_1, arr_2)

    def test_stratified(self):
        """Test that within-cluster permutation works as expected."""
        data = np.array(
            [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [2, 3], [2, 4]]
        )

        permuter = resampling.Permuter(n_resamples=10, exact=False, random_state=1)

        for permutation in permuter.resample(data, col_to_permute=1):
            expected_per_cluster = set((1, 2, 3, 4))
            self.assertEqual(
                expected_per_cluster,
                expected_per_cluster.intersection(permutation[:4, 1]),
            )
            self.assertEqual(
                expected_per_cluster,
                expected_per_cluster.intersection(permutation[4:, 1]),
            )

    def test_repeat(self):
        """Test that subclusters move together."""

        data = np.array([[1, 1], [1, 2]]).repeat(3, axis=0)

        permuter = resampling.Permuter(n_resamples=10, exact=False, random_state=1)

        for permutation in permuter.resample(data, col_to_permute=1):
            condition = all(permutation[:, 1] == [1, 1, 1, 2, 2, 2]) or all(
                permutation[:, 1] == [2, 2, 2, 1, 1, 1]
            )

            self.assertTrue(condition)

    def test_exact(self):
        """
        Test that exact permutations are generated in the same order.
        """
        shuf_1 = self.orig_data.copy()

        permuter_1 = resampling.Permuter(n_resamples=10, exact=True)
        permuter_2 = resampling.Permuter(n_resamples=10, exact=True)

        for i in range(20):
            arr_1 = next(permuter_1.resample(shuf_1, 0))
            arr_2 = next(permuter_2.resample(shuf_1, 0))
            assert_equal(arr_1, arr_2)

    def test_permuter_exceptions(self):
        permuter = resampling.Permuter(n_resamples=1, exact=True)
        with self.assertRaises(NotImplementedError) as raises:
            next(permuter.resample(self.orig_data, col_to_permute=1))
        self.assertIn(
            "Exact permutation only available for col_to_permute = 0.",
            str(raises.exception),
        )
