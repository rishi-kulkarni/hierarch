import unittest
import hierarch.stats
from hierarch.power import DataSimulator
import scipy.stats as stats
import numpy as np
import pandas as pd

class TestPreprocessData(unittest.TestCase):
    def test_label_encoding(self):
        # check that strings get encoded
        data = np.array(["a", "b", "c"]).reshape((3, 1))
        processed = hierarch.stats._preprocess_data(data)
        self.assertTrue(processed.dtype, np.float64)
        self.assertEqual(data.shape, processed.shape)

        # check that floats do not get encoded
        data = np.arange(10, step=0.5, dtype='object').reshape((10, 2))
        processed = hierarch.stats._preprocess_data(data)
        for idx, v in enumerate(processed.flat):
            self.assertEqual(v, data.flat[idx])

        # check that when there is a mix of numerical and string columns,
        # the numerical columns do not get encoded
        data = np.arange(3, step=0.5, dtype='object').reshape((3, 2))
        data[:,0] = np.array(["a", "b", "c"])
        processed = hierarch.stats._preprocess_data(data)
        self.assertTrue(processed.dtype, np.float64)
        for idx, v in enumerate(processed[:,1]):
            self.assertEqual(v, data[:,1][idx])        

class TestStudentizedCovariance(unittest.TestCase):
    def test_cov(self):
        '''
        Checks studentized_covariance against expected value.
        '''
        x = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 
                      [1, 2, 3, 4, 5, 2, 3, 4, 5, 6]]).T

        self.assertAlmostEqual(hierarch.stats.studentized_covariance(x[:,0], x[:,1]), 1.0039690353154482)

class TestWelch(unittest.TestCase):
    def test_welch(self):
        '''
        Checks welch_statistic against expected value from scipy.stats.ttest_ind.
        '''

        a = np.random.randint(10, size=10)
        b = np.random.randint(10, size=10)

        self.assertAlmostEqual(hierarch.stats.welch_statistic(a, b), stats.ttest_ind(a, b, equal_var=False)[0])

class TestHypothesisTest(unittest.TestCase):
    import scipy.stats as stats
    paramlist = [[0, 2], [stats.norm], [stats.norm]]
    hierarchy = [2, 4, 3]
    datagen = DataSimulator(paramlist, random_state=2)
    datagen.fit(hierarchy)
    data = datagen.generate()

    def test_corr_vs_means(self):
        # check exact test
        corr_p = hierarch.stats.hypothesis_test(self.data, treatment_col=0, compare='corr', bootstraps=1000, permutations='all', random_state=1)
        t_p = hierarch.stats.hypothesis_test(self.data, treatment_col=0, compare='means', bootstraps=1000, permutations='all', random_state=1)
        self.assertAlmostEqual(corr_p, t_p)

        # check approximate test
        corr_p = hierarch.stats.hypothesis_test(self.data, treatment_col=0, compare='corr', bootstraps=1000, permutations=70, random_state=1)
        t_p = hierarch.stats.hypothesis_test(self.data, treatment_col=0, compare='means', bootstraps=1000, permutations=70, random_state=1)
        self.assertAlmostEqual(corr_p, t_p)

    def test_hypothesis_exceptions(self):
        with self.assertRaises(TypeError) as raises:
            hierarch.stats.hypothesis_test("ah", 0)
        self.assertIn("Input data must be ndarray or DataFrame.", str(raises.exception))

        with self.assertWarns(Warning) as warning:
            hierarch.stats.hypothesis_test(self.data, 0, skip=[0])
        self.assertIn("No need to include columns before treated columns in skip.", str(warning.warning))

        with self.assertRaises(TypeError) as raises:
            hierarch.stats.hypothesis_test(self.data, 0, bootstraps=1.5)
        self.assertIn("bootstraps must be an integer greater than 0", str(raises.exception))

        with self.assertRaises(TypeError) as raises:
            hierarch.stats.hypothesis_test(self.data, 0, permutations='a')
        self.assertIn("permutations must be 'all' or an integer greater than 0", str(raises.exception))

        with self.assertRaises(TypeError) as raises:
            hierarch.stats.hypothesis_test(self.data, 0, permutations=1.5)
        self.assertIn("permutations must be 'all' or an integer greater than 0", str(raises.exception))

        with self.assertRaises(AttributeError) as raises:
            hello = 5
            hierarch.stats.hypothesis_test(self.data, 0, compare=hello)
        self.assertIn("Custom test statistics must be callable.", str(raises.exception))

        with self.assertWarns(Warning) as warning:
            hierarch.stats.hypothesis_test(self.data, 1)
        self.assertIn("No levels to bootstrap. Setting bootstraps to zero.", str(warning.warning))



if __name__ == "__main__":
    unittest.main()
