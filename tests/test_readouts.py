import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("readouts", Path(__file__).parents[1] / "scripts/reanalyze_readouts.py")
readouts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readouts)


class ReadoutTests(unittest.TestCase):
    def test_average_ranks_preserve_ties(self):
        np.testing.assert_allclose(readouts.average_ranks(np.array([2, 1, 2])), [1.5, 0., 1.5])

    def test_spearman_tracks_order_with_ties(self):
        x = np.array([1, 1, 4, 8])
        self.assertAlmostEqual(readouts.spearman(x, x*3), 1.)
        self.assertAlmostEqual(readouts.spearman(x, -x), -1.)
