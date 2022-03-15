import unittest

import hierarch.stats
import numpy as np
import scipy.stats as stats
from hierarch.power import DataSimulator
from numpy.testing import assert_almost_equal


class TestPreprocessData(unittest.TestCase):
    def test_string_encoding(self):
        # check that strings get encoded
        data = np.array(["a", "b", "c"]).reshape((3, 1))
        y = np.array([1, 2, 3])
        processed_x, processed_y = hierarch.stats._preprocess_data(data, y)
        self.assertTrue(processed_x.dtype, np.float64)
        self.assertEqual(data.shape, processed_x.shape)

    def test_float_encoding(self):

        # check that floats do not get encoded
        data = np.arange(10, step=0.5, dtype="object").reshape((10, 2))
        y = np.array([1.0 for i in data])
        processed, y_proc = hierarch.stats._preprocess_data(data, y)
        for idx, v in enumerate(processed.flat):
            self.assertEqual(v, data.flat[idx])

    def test_mixed_encoding(self):
        # check that when there is a mix of numerical and string columns,
        # the numerical columns do not get encoded
        data = np.arange(3, step=0.5, dtype="object").reshape((3, 2))
        data[:, 0] = np.array(["a", "b", "c"])
        y = np.array([1.0 for i in data])
        processed, y_proc = hierarch.stats._preprocess_data(data, y)
        self.assertTrue(processed.dtype, np.float64)
        for idx, v in enumerate(processed[:, 1]):
            self.assertEqual(v, data[:, 1][idx])

    def test_sorting(self):
        data = np.array([[1, 2], [1, 1], [0, 1]])
        y = np.array([1, 2, 3])

        expected_data = np.array([[0, 1], [1, 1], [1, 2]])
        expected_y = np.array([3, 2, 1])

        generated_data, generated_y = hierarch.stats._preprocess_data(data, y)

        assert_almost_equal(generated_data, expected_data)
        assert_almost_equal(generated_y, expected_y)


class TestStudentizedCovariance(unittest.TestCase):
    def test_cov(self):
        """
        Checks studentized_covariance against expected value.
        """
        x = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 2, 3, 4, 5, 6]]).T

        self.assertAlmostEqual(
            hierarch.stats.studentized_covariance(x[:, 0], x[:, 1]), 1.0039690353154482
        )


class TestWelch(unittest.TestCase):
    def test_welch(self):
        """
        Checks welch_statistic against expected value from scipy.stats.ttest_ind.
        """

        a = np.random.randint(10, size=10)
        b = np.random.randint(10, size=10)

        self.assertAlmostEqual(
            hierarch.stats.welch_statistic(a, b),
            stats.ttest_ind(a, b, equal_var=False)[0],
        )


class TestHypothesisTest(unittest.TestCase):
    import scipy.stats as stats

    paramlist = [[0, 2], [stats.norm], [stats.norm]]
    hierarchy = [2, 4, 3]
    datagen = DataSimulator(paramlist, random_state=2)
    datagen.fit(hierarchy)
    data = datagen.generate()

    def test_corr_vs_means(self):
        # check exact test
        corr_p = hierarch.stats.hypothesis_test(
            self.data[:, :-2],
            self.data[:, -1],
            treatment_col=0,
            compare="corr",
            bootstraps=1000,
            permutations="all",
            random_state=1,
        )
        t_p = hierarch.stats.hypothesis_test(
            self.data[:, :-2],
            self.data[:, -1],
            treatment_col=0,
            compare="means",
            bootstraps=1000,
            permutations="all",
            random_state=1,
        )
        self.assertAlmostEqual(corr_p, t_p)

        # check approximate test
        corr_p = hierarch.stats.hypothesis_test(
            self.data[:, :-2],
            self.data[:, -1],
            treatment_col=0,
            compare="corr",
            bootstraps=1000,
            permutations=70,
            random_state=1,
        )
        t_p = hierarch.stats.hypothesis_test(
            self.data[:, :-2],
            self.data[:, -1],
            treatment_col=0,
            compare="means",
            bootstraps=1000,
            permutations=70,
            random_state=1,
        )
        self.assertAlmostEqual(corr_p, t_p)

    def test_hypothesis_exceptions(self):
        with self.assertRaises(TypeError) as raises:
            hierarch.stats.hypothesis_test("ah", 0)
        self.assertIn("Input data must be ndarray or DataFrame.", str(raises.exception))

        with self.assertWarns(Warning) as warning:
            hierarch.stats.hypothesis_test(self.data, 0, skip=[0])
        self.assertIn(
            "No need to include columns before treated columns in skip.",
            str(warning.warning),
        )

        with self.assertRaises(TypeError) as raises:
            hierarch.stats.hypothesis_test(self.data, 0, bootstraps=1.5)
        self.assertIn(
            "bootstraps must be an integer greater than 0", str(raises.exception)
        )

        with self.assertRaises(TypeError) as raises:
            hierarch.stats.hypothesis_test(self.data, 0, permutations="a")
        self.assertIn(
            "permutations must be 'all' or an integer greater than 0",
            str(raises.exception),
        )

        with self.assertRaises(TypeError) as raises:
            hierarch.stats.hypothesis_test(self.data, 0, permutations=1.5)
        self.assertIn(
            "permutations must be 'all' or an integer greater than 0",
            str(raises.exception),
        )

        with self.assertRaises(AttributeError) as raises:
            hello = 5
            hierarch.stats.hypothesis_test(self.data, 0, compare=hello)
        self.assertIn("Custom test statistics must be callable.", str(raises.exception))

        with self.assertWarns(Warning) as warning:
            hierarch.stats.hypothesis_test(self.data, 1)
        self.assertIn(
            "No levels to bootstrap. Setting bootstraps to zero.", str(warning.warning)
        )


class TestMultiTest(unittest.TestCase):
    import scipy.stats as stats

    paramlist = [[0, 2, 4, 6], [stats.norm], [stats.norm]]
    hierarchy = [4, 4, 3]
    datagen = DataSimulator(paramlist, random_state=2)
    datagen.fit(hierarchy)
    data = datagen.generate()

    def test_get_comparisons(self):
        # check that all hypotheses are grabbed
        test = hierarch.stats._get_comparisons(self.data, 0)
        self.assertEqual(len(test), 6)

        # check that every hypothesis is tested
        out = hierarch.stats.multi_sample_test(self.data, 0).to_numpy()
        self.assertEqual(len(out), 6)

    def test_fdr_adjustment(self):
        p_vals = np.arange(0.05, 1.05, step=0.1)
        adjusted = hierarch.stats._false_discovery_adjust(p_vals)
        standard = np.array(
            [0.5, 0.75, 0.83333, 0.875, 0.9, 0.91667, 0.92857, 0.9375, 0.94444, 0.95]
        )
        for idx, v in enumerate(adjusted):
            self.assertAlmostEqual(v, standard[idx])

    def test_exceptions(self):
        with self.assertRaises(KeyError) as raises:
            hierarch.stats.multi_sample_test(self.data, 0, correction="ben")
        self.assertIn(
            "ben is not a valid multiple comparisons correction", str(raises.exception)
        )

        with self.assertRaises(TypeError) as raises:
            hierarch.stats.multi_sample_test("hi", 0)
        self.assertIn("Input data must be ndarray or DataFrame", str(raises.exception))


class TestConfidenceInterval(unittest.TestCase):
    paramlist = [[0, 2], [stats.norm], [stats.norm]]
    hierarchy = [2, 4, 3]
    datagen = DataSimulator(paramlist, random_state=2)
    datagen.fit(hierarchy)
    data = datagen.generate()

    def test_conf(self):
        interval_95 = hierarch.stats.confidence_interval(self.data, 0, interval=95)
        self.assertEqual(len(interval_95), 2)

        interval_68 = hierarch.stats.confidence_interval(self.data, 0, interval=68)

        # check that a 95% interval is wider than a 68% interval
        self.assertLess(interval_95[0], interval_68[0])
        self.assertGreater(interval_95[1], interval_68[1])


if __name__ == "__main__":
    unittest.main()
