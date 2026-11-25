"""Numerical derivative checks for depot objective functions."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize
import depot_experiments

from depot import (
    bfgs_method,
    gradient,
    gradient_descent,
    hessian,
    newton_method,
    objective,
)


class DepotDerivativeTests(unittest.TestCase):
    """Test analytical derivatives against deterministic numerical checks."""

    def setUp(self):
        self.points = np.array(
            [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0], [5.0, 4.0]]
        )
        self.center = np.array([1.2, 1.7])
        self.eps = 1e-6
        self.step = 1e-5

    def test_gradient_matches_central_finite_differences(self):
        """Central differences of the objective approximate its gradient."""
        numerical_gradient = np.zeros(2)
        for index in range(2):
            basis = np.zeros(2)
            basis[index] = 1.0
            # Differentiate the objective along each coordinate direction.
            numerical_gradient[index] = (
                objective(self.center + self.step * basis, self.points, self.eps)
                - objective(self.center - self.step * basis, self.points, self.eps)
            ) / (2.0 * self.step)

        np.testing.assert_allclose(
            gradient(self.center, self.points, self.eps),
            numerical_gradient,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_hessian_matches_gradient_central_finite_differences(self):
        """Central differences of the gradient approximate Hessian columns."""
        numerical_hessian = np.zeros((2, 2))
        for index in range(2):
            basis = np.zeros(2)
            basis[index] = 1.0
            # Differentiate the full gradient along the selected coordinate.
            numerical_hessian[:, index] = (
                gradient(self.center + self.step * basis, self.points, self.eps)
                - gradient(self.center - self.step * basis, self.points, self.eps)
            ) / (2.0 * self.step)

        np.testing.assert_allclose(
            hessian(self.center, self.points, self.eps),
            numerical_hessian,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_hessian_is_symmetric(self):
        """The analytical Hessian equals its transpose."""
        analytical_hessian = hessian(self.center, self.points, self.eps)
        np.testing.assert_allclose(
            analytical_hessian, analytical_hessian.T, rtol=1e-12, atol=1e-12
        )

    def test_hessian_is_positive_definite(self):
        """Smoothing makes every Hessian eigenvalue strictly positive."""
        eigenvalues = np.linalg.eigvalsh(hessian(self.center, self.points, self.eps))
        self.assertTrue(np.all(eigenvalues > 0.0))


class DepotOptimizerTests(unittest.TestCase):
    """Test deterministic optimizer behavior against a SciPy reference."""

    def setUp(self):
        self.points = np.array(
            [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0], [5.0, 4.0]]
        )
        self.start = np.array([3.0, 3.0])
        self.eps = 1e-6

    def _scipy_reference(self):
        """Return a BFGS reference using the project objective and gradient."""
        return minimize(
            objective,
            self.start,
            args=(self.points, self.eps),
            jac=gradient,
            method="BFGS",
            options={"gtol": 1e-8, "maxiter": 1000},
        )

    def _project_results(self):
        """Run every project optimizer from the same explicit start."""
        return {
            "gradient_descent": gradient_descent(
                self.points, start=self.start, eps=self.eps
            ),
            "newton_method": newton_method(
                self.points, start=self.start, eps=self.eps
            ),
            "bfgs_method": bfgs_method(self.points, start=self.start, eps=self.eps),
        }

    def test_optimizers_converge_and_agree_with_scipy(self):
        """All solvers should reach the same solution as SciPy BFGS."""
        reference = self._scipy_reference()
        self.assertTrue(reference.success, reference.message)

        for name, result in self._project_results().items():
            with self.subTest(optimizer=name):
                self.assertTrue(result["converged"], result["message"])
                self.assertLess(
                    np.linalg.norm(result["depot_xy"] - reference.x), 1e-5
                )
                np.testing.assert_allclose(
                    result["objective"], reference.fun, rtol=1e-8, atol=1e-8
                )

    def test_optimizer_history_is_complete_and_consistent(self):
        """Each accepted update contributes exactly one history row."""
        for name, result in self._project_results().items():
            with self.subTest(optimizer=name):
                history = result["history"]
                self.assertEqual(history[0]["iteration"], 0)
                self.assertEqual(history[-1]["iteration"], result["iterations"])
                self.assertEqual(len(history), result["iterations"] + 1)
                self.assertEqual(history[0]["step_size"], 0.0)

    def test_iteration_limit_is_not_reported_as_convergence(self):
        """A zero update budget must retain only the initial, non-converged state."""
        result = gradient_descent(
            self.points, start=self.start, eps=self.eps, tol=1e-30, max_iter=0
        )
        self.assertFalse(result["converged"])
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(result["history"][0]["iteration"], 0)
        self.assertIn("maximum iterations", result["message"].lower())

class DepotExperimentTests(unittest.TestCase):
    def test_failed_scipy_reference_is_rejected(self):
        failed = type("Result", (), {
            "success": False, "status": 2, "message": "forced failure",
            "x": np.array([999.0, 999.0]),
        })()
        points = np.array([[0.0, 0.0], [1.0, 0.0]])
        with patch.object(depot_experiments, "minimize", return_value=failed):
            result = depot_experiments.run_optimizer_experiments(
                points, random_seeds=[], eps_values=(1e-6,)
            )
        summary = result["summary_df"]
        self.assertFalse(summary["scipy_success"].any())
        self.assertTrue(summary["scipy_distance"].isna().all())
        self.assertEqual(set(summary["scipy_status"]), {2})
        self.assertEqual(set(summary["scipy_message"]), {"forced failure"})
        self.assertEqual(len(summary), 3)


if __name__ == "__main__":
    unittest.main()
