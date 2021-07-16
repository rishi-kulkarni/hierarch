from symbol import parameters
import unittest
import hierarch.power
import scipy.stats as stats
import numpy as np
import pandas as pd


class TestDataSimulator(unittest.TestCase):
    def test_rng(self):
        """
        Check that specifying the random_state generates reproducible datasets
        and that the datasets have the specified properties.
        """
        parameters = [[stats.norm], [stats.norm], [stats.norm]]
        sim_1 = hierarch.power.DataSimulator(parameters, random_state=789)
        sim_2 = hierarch.power.DataSimulator(parameters, random_state=789)

        hierarchy = [2, 3, 3]

        sim_1.fit(hierarchy)
        sim_2.fit(hierarchy)

        data_1 = sim_1.generate()
        data_2 = sim_2.generate()

        for idx, v in enumerate(data_1[:, -1]):
            self.assertAlmostEqual(v, data_2[:, -1][idx])

        self.assertEqual(len(data_1), np.array(hierarchy).prod())

    def test_exceptions(self):
        parameters = [[stats.norm], [stats.norm], [stats.norm]]
        sim_1 = hierarch.power.DataSimulator(parameters, random_state=789)
        with self.assertRaises(ValueError) as raises:
            sim_1.fit([2, 3, 3, 3])
        self.assertIn(
            "hierarchy and parameters should be the same length.", str(raises.exception)
        )

        with self.assertRaises(TypeError) as raises:
            sim_1.fit([2, 3.5, 3])
        self.assertIn("hierarchy must be a list of integers.", str(raises.exception))


if __name__ == "__main__":
    unittest.main()
