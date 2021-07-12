import enum
import unittest
from hierarch import internal_functions
from hierarch.power import DataSimulator
import scipy.stats as stats
import numpy as np

class TestSetRandomState(unittest.TestCase):
    def _try_seed(self, seed):
        internal_functions.set_numba_random_state(seed)

    def test_set_random_state(self):
        '''
        Test normal behavior
        '''
        seeds = (1, 1000, 2**32)
        for seed in seeds:
            self._try_seed(seed)

    def test_set_random_state_exceptions(self):
        '''
        Checks that exceptions are raised when seed is not an integer.
        '''
        seeds = (1.5,)
        for seed in seeds:
            with self.assertRaises(ValueError) as raises:
                self._try_seed(seed)
            self.assertIn("Numba seed must be an integer.", str(raises.exception))

class TestDataGrabber(unittest.TestCase):
   
    def _check_samples(self, data, treatment_col, treatment_labels, ret):
        '''
        Check lengths of grabbed samples.
        '''
        lengths = []
        for idx, key in enumerate(treatment_labels):
            self.assertEqual(ret[idx].size, (data[:,treatment_col] == key).sum())

    def test_data_grabber(self):
        hierarchies = ([2, 3, 3], [2, [4, 3], 3], [2, 3, [10, 11, 5, 6, 4, 3]])
        parameters = [[stats.norm, 0, 0], [stats.norm, 0, 0], [stats.norm, 0, 0]]
        sim = DataSimulator(parameters)

        for hierarchy in hierarchies:
            sim.fit(hierarchy)
            data = sim.generate()

            for treatment_col in range(data.shape[1] - 1):
                treatment_labels = np.unique(data[:,treatment_col])
                ret = internal_functions.nb_data_grabber(data, treatment_col, treatment_labels)
                self._check_samples(data, treatment_col, treatment_labels, ret)
            
class TestNumbaUnique(unittest.TestCase):

    def _check_unique(self, data, col, ret):
        '''
        Check that nb_unique returns the same unique values, indices, and counts give
        the same values as np.unique.
        '''
        np_ret = np.unique(data[:,:col], return_index=True, return_counts=True, axis=0)

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

if __name__ == "__main__":
    unittest.main()