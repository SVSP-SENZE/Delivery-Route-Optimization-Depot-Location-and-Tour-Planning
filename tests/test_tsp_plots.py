import os
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from tsp_plots import plot_gap
from tsp_experiments import _gap


class TestTspGapPlot(unittest.TestCase):
    def test_proven_optimum_gap_cannot_be_negative(self):
        self.assertEqual(_gap(10.0, 10.0), 0.0)
        self.assertEqual(_gap(10.0 - 1e-13, 10.0), 0.0)
        with self.assertRaisesRegex(ValueError, "materially negative"):
            _gap(9.0, 10.0)

    def test_gap_plot_pairs_proven_exact_rows(self):
        rows = [
            {"n": 5, "seed": 0, "method": "exact", "route_length_km": 10.0, "optimality_proven": True},
            {"n": 5, "seed": 0, "method": "nearest_neighbour", "route_length_km": 11.0, "optimality_proven": ""},
            {"n": 5, "seed": 0, "method": "two_opt", "route_length_km": 10.0, "optimality_proven": ""},
            {"n": 5, "seed": 1, "method": "exact", "route_length_km": 8.0, "optimality_proven": False},
            {"n": 5, "seed": 1, "method": "nearest_neighbour", "route_length_km": 7.0, "optimality_proven": ""},
            {"n": 5, "seed": 1, "method": "two_opt", "route_length_km": 7.0, "optimality_proven": ""},
        ]
        with tempfile.TemporaryDirectory() as directory:
            csv_path = os.path.join(directory, "scaling.csv")
            png_path = os.path.join(directory, "gap.png")
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            self.assertEqual(plot_gap(csv_path, png_path), png_path)
            self.assertTrue(os.path.isfile(png_path))
            self.assertGreater(os.path.getsize(png_path), 0)
