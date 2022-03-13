import unittest

import pandas as pd
import scipy.stats as stats
from hierarch.power import DataSimulator
from hierarch.regression_utils import GroupbyMean


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
            grouper = GroupbyMean()
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
            grouper = GroupbyMean()
            # check that using a reference array works
            grouper_2 = GroupbyMean()
            grouper_2.fit(data)
            ordinary_agg = grouper.fit_transform(data, iterations=1)[:, -1]
            data[:, 1] = 1
            ref_agg = grouper_2.transform(data)[:, -1]
            self._compare_results(ordinary_agg, ref_agg)
