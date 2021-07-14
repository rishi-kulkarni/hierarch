from multiprocessing.sharedctypes import Value
import unittest
import hierarch.resampling
from hierarch.power import DataSimulator
import scipy.stats as stats
import numpy as np
import pandas as pd

class TestBootstrapper(unittest.TestCase):
    sim = DataSimulator([[stats.norm], [stats.norm], [stats.norm]])
    sim.fit([2, 3, 3])
    data_1  = sim.generate()

    def test_seeding(self):
        '''
        Tests that setting the random_state generates the same bootstrapped sample.
        '''

        boot = hierarch.resampling.Bootstrapper(random_state = 1, kind='weights')
        boot.fit(self.data_1)
        test_1 = boot.transform(self.data_1, start=0)

        boot = hierarch.resampling.Bootstrapper(random_state = 1, kind='weights')
        boot.fit(self.data_1)
        test_2 = boot.transform(self.data_1, start=0)

        for idx, v in enumerate(test_1[:,-1]):
            self.assertAlmostEqual(v, test_2[:,-1][idx])

    def test_efron_bootstrapper(self):
        '''
        Tests the Efron bootstrap "contract" - the final weights should sum
        to the sum of the original weights within each column.
        '''
        boot = hierarch.resampling.Bootstrapper(kind = 'weights')
        boot.fit(self.data_1)
        self.data_1[:,-1] = 1
        starts = (0, 1, 2)
        for start in starts:
            test_1 = boot.transform(self.data_1, start=start)
            self.assertAlmostEqual(self.data_1[:,-1].sum(), test_1[:,-1].sum())
    
    def test_bayesian_bootstrapper(self):
        '''
        Tests the Efron bootstrap "contract" - the final weights should sum
        to the sum of the original weights within each column.
        '''
        boot = hierarch.resampling.Bootstrapper(kind = 'bayesian')
        boot.fit(self.data_1)
        self.data_1[:,-1] = 1
        starts = (0, 1, 2)
        for start in starts:
            test_1 = boot.transform(self.data_1, start=start)
            self.assertAlmostEqual(self.data_1[:,-1].sum(), test_1[:,-1].sum())
    
    def test_bootstrapper_exceptions(self):
        with self.assertRaises(KeyError) as raises:
            boot = hierarch.resampling.Bootstrapper(kind = "blah")
        self.assertIn("Invalid 'kind' argument.", str(raises.exception))

        boot = hierarch.resampling.Bootstrapper(kind = 'weights')
        with self.assertRaises(ValueError) as raises:
            boot.fit(np.array(["str"]))
        self.assertIn("Bootstrapper can only handle numeric datatypes. Please pre-process your data.", str(raises.exception)
)

        with self.assertRaises(AttributeError) as raises:
            boot.fit(pd.DataFrame(self.data_1))
        self.assertIn("Bootstrapper can only handle numpy arrays. Please pre-process your data.", str(raises.exception)
)

        with self.assertRaises(IndexError) as raises:
            boot.fit(self.data_1, skip=[2.3])
        self.assertIn("skip values must be integers corresponding to column indices.", str(raises.exception))

        with self.assertRaises(IndexError) as raises:
            boot.fit(self.data_1, skip=[5])
        self.assertIn("skip index out of bounds for this array.", str(raises.exception))

        with self.assertRaises(Exception) as raises:
            boot.transform(self.data_1, start=1)
        self.assertIn("Use fit() before using transform().", str(raises.exception))

if __name__ == "__main__":
    unittest.main()
