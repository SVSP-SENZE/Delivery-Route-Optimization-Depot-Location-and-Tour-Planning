"""Smoke tests for the saved optimizer convergence plot."""

import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from depot_plots import plot_optimizer_convergence


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


if __name__ == "__main__":
    unittest.main()
