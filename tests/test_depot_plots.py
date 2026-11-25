"""Smoke tests for the saved optimizer convergence plot."""

import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from depot_plots import plot_depot_heatmap, plot_optimizer_convergence


class TestDepotPlots(unittest.TestCase):
    """Verify that a valid saved history produces a non-empty PNG."""

    def test_plot_optimizer_convergence_smoke(self):
        rows = []
        for method, offset in (("gradient_descent", 0.0), ("newton", 0.2), ("bfgs", 0.4)):
            for iteration in range(3):
                rows.append(
                    {
                        "dataset": "smoke",
                        "seed": 11,
                        "epsilon": 0.01,
                        "method": method,
                        "iteration": iteration,
                        "x": 1.0 + offset + 0.1 * iteration,
                        "y": 2.0 + offset - 0.1 * iteration,
                        "objective": 10.0 - iteration,
                        "grad_norm": 3.0 / (iteration + 1),
                        "step_size": 0.1,
                    }
                )

        delivery_points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
        with tempfile.TemporaryDirectory() as directory:
            history_path = os.path.join(directory, "optimizer_history.csv")
            output_path = os.path.join(directory, "optimizer_convergence.png")
            pd.DataFrame(rows).to_csv(history_path, index=False)

            returned_path = plot_optimizer_convergence(
                history_path,
                delivery_points,
                output_path,
                dataset="smoke",
                epsilon=0.01,
            )

            self.assertEqual(returned_path, output_path)
            self.assertTrue(os.path.isfile(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_heatmap_rejects_empty_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            grid_path = os.path.join(directory, "grid.csv")
            effect_path = os.path.join(directory, "effect.csv")
            pd.DataFrame(columns=["grid_x", "grid_y", "tour_length_km"]).to_csv(grid_path, index=False)
            pd.DataFrame([{"depot_type": "stage1", "depot_x": 0, "depot_y": 0, "tour_length_km": 1}]).to_csv(effect_path, index=False)
            with self.assertRaisesRegex(ValueError, "no rows"):
                plot_depot_heatmap(grid_path, effect_path, output_path=os.path.join(directory, "out.png"))

    def test_heatmap_rejects_nonfinite_grid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            grid_path = os.path.join(directory, "grid.csv")
            effect_path = os.path.join(directory, "effect.csv")
            pd.DataFrame([{"grid_x": 0, "grid_y": 0, "tour_length_km": "bad"}]).to_csv(grid_path, index=False)
            pd.DataFrame([{"depot_type": "stage1", "depot_x": 0, "depot_y": 0, "tour_length_km": 1}]).to_csv(effect_path, index=False)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                plot_depot_heatmap(grid_path, effect_path, output_path=os.path.join(directory, "out.png"))


if __name__ == "__main__":
    unittest.main()
