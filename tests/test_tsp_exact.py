import types
import unittest

from tsp_exact import _is_proven_optimal


class TestTspOptimalityStatus(unittest.TestCase):
    def setUp(self):
        self.pulp = types.SimpleNamespace(
            LpStatusOptimal=1,
            LpSolutionOptimal=1,
            LpSolution={1: "Optimal", 2: "Solution Found"},
        )

    def test_feasible_solution_found_is_not_proven_optimal(self):
        self.assertFalse(_is_proven_optimal(1, "Solution Found", self.pulp))

    def test_explicit_optimal_solution_is_proven_optimal(self):
        self.assertTrue(_is_proven_optimal(1, "Optimal", self.pulp))

