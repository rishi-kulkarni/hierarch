import unittest
from hierarch import internal_functions
from hierarch.power import DataSimulator
import scipy.stats as stats
import numpy as np
import pandas as pd


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


class TestNumbaUnique(unittest.TestCase):
    def _check_unique(self, data, col, ret):
        """
        Check that nb_unique returns the same unique values, indices, and counts give
        the same values as np.unique.
        """
        np_ret = np.unique(data[:, :col], return_index=True, return_counts=True, axis=0)

        for idx, v in enumerate(np_ret):
            np_flat = v.flat
            nb_flat = ret[idx].flat

            for idx_2, w in enumerate(np_flat):
                self.assertEqual(w, nb_flat[idx_2])

    def test_nb_unique(self):
        hierarchies = ([2, 3, 3], [2, [4, 3], 3], [2, 3, [10, 11, 5, 6, 4, 3]])
        parameters = [[stats.norm, 0, 0], [stats.norm, 0, 0], [stats.norm, 0, 0]]
        sim = DataSimulator(parameters)

        for hierarchy in hierarchies:
            sim.fit(hierarchy)
            data = sim.generate()

            for treatment_col in range(1, data.shape[1]):
                ret = internal_functions.nb_unique(data[:, :treatment_col])
                self._check_unique(data, treatment_col, ret)


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


class TestFastShuffle(unittest.TestCase):
    def _check_shuffle(self, original, shuffled):
        """
        Check that the shuffled array is the same length as the original one
        and contains all of the same unique entries.
        """
        self.assertEqual(original.size, shuffled.size)
        orig_unique = np.unique(original)
        shuffled_unique = np.unique(shuffled)
        for idx, v in enumerate(orig_unique):
            self.assertEqual(v, shuffled_unique[idx])

    def test_shuffle(self):
        lengths = (5, 10, 100, 500)
        for length in lengths:
            original = np.arange(length)
            shuffled = np.arange(length)
            for i in range(50):
                internal_functions.nb_fast_shuffle(shuffled)
                self._check_shuffle(original, shuffled)

    def test_strat_shuffle(self):
        original = np.arange(20)
        original[10:] = 0
        shuffled = np.arange(20)
        shuffled[10:] = 0
        for i in range(50):
            internal_functions.nb_strat_shuffle(shuffled, (0, 10, 20))
            self._check_shuffle(original, shuffled)
            self.assertEqual(shuffled[10:].sum(), 0)


class TestIDClusters(unittest.TestCase):
    def test_id_clusters(self):
        hierarchies = ([2, 3, 3], [2, [4, 3], 3], [2, 3, [10, 11, 5, 6, 4, 3]])
        parameters = [[stats.norm, 0, 0], [stats.norm, 0, 0], [stats.norm, 0, 0]]
        sim = DataSimulator(parameters)

        for hierarchy in hierarchies:
            sim.fit(hierarchy)
            ret = internal_functions.id_cluster_counts(sim.container[:, :-1])
            for idx, level in enumerate(reversed(list(ret.values()))):
                if isinstance(hierarchy[idx], int):
                    for v in iter(level):
                        self.assertEqual(v, hierarchy[idx])
                elif isinstance(hierarchy[idx], list):
                    for idx_2, v in enumerate(level):
                        self.assertEqual(hierarchy[idx][idx_2], v)


class TestWeightstoIndex(unittest.TestCase):
    def test_weights_to_index(self):
        """
        Tests that weights_to_index gives same array as np.arange(weights.size).repeat(weights)
        """
        weights = np.random.randint(10, size=10)
        indexes = internal_functions.weights_to_index(weights)
        np_indexes = np.arange(weights.size).repeat(weights)
        for idx, v in enumerate(indexes):
            self.assertEqual(v, np_indexes[idx])


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


class TestGroupByMean(unittest.TestCase):
    def _compare_results(self, pd_agg, groupby_agg):
        for idx, v in enumerate(pd_agg):
            self.assertAlmostEqual(v, groupby_agg[idx])

    def test_groupby_mean(self):
        """
        Checks that GroupbyMean produces the same values as pandas
        groupby.mean() operation.
        """
        sim = DataSimulator([[stats.norm], [stats.norm], [stats.norm]])
        hierarchies = ([2, 3, 3], [2, [4, 3], 3], [2, [4, 3], [10, 11, 2, 5, 6, 4, 3]])
        for hierarchy in hierarchies:
            sim.fit(hierarchy)
            data = sim.generate()
            pd_data = pd.DataFrame(data, columns=["Col 1", "Col 2", "Col 3", "Value"])
            grouper = internal_functions.GroupbyMean()
            # reduce by one column
            groupby_agg = grouper.fit_transform(data, iterations=1)[:, -1]
            pd_agg = pd_data.groupby(["Col 1", "Col 2"]).mean()["Value"].to_numpy()
            self._compare_results(pd_agg, groupby_agg)
            # reduce by two columns
            groupby_agg = grouper.fit_transform(data, iterations=2)[:, -1]
            pd_agg = (
                pd_data.groupby(["Col 1", "Col 2"])
                .mean()
                .groupby(["Col 1"])
                .mean()["Value"]
                .to_numpy()
            )
            self._compare_results(pd_agg, groupby_agg)

    def test_groupby_mean_ref(self):
        sim = DataSimulator([[stats.norm], [stats.norm], [stats.norm]])
        hierarchies = ([2, 3, 3], [2, [4, 3], 3], [3, [10, 4, 3], 7])
        for hierarchy in hierarchies:
            sim.fit(hierarchy)
            data = sim.generate()
            grouper = internal_functions.GroupbyMean()
            # check that using a reference array works
            grouper_2 = internal_functions.GroupbyMean()
            grouper_2.fit(data)
            ordinary_agg = grouper.fit_transform(data, iterations=1)[:, -1]
            data[:, 1] = 1
            ref_agg = grouper_2.transform(data)[:, -1]
            self._compare_results(ordinary_agg, ref_agg)


if __name__ == "__main__":
    unittest.main()
