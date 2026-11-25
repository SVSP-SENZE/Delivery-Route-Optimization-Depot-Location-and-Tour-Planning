"""Unit tests for heuristics.py. Run from the project root:  pytest tests/test_heuristics.py"""

import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heuristics import tour_length, nearest_neighbour, two_opt  # noqa: E402


def make_D(n, seed):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, size=(n, 2))
    return np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)


def brute_force(D):
    n = D.shape[0]
    best = np.inf
    for perm in itertools.permutations(range(1, n)):
        best = min(best, tour_length([0, *perm], D))
    return best


# ---------------------------- tour_length ----------------------------------
def test_tour_length_includes_return_edge():
    D = np.array([[0, 1, 4], [1, 0, 2], [4, 2, 0]], dtype=float)
    assert tour_length([0, 1, 2], D) == pytest.approx(1 + 2 + 4)


def test_tour_length_example_route_convention():
    D = make_D(5, 0)
    r = [0, 2, 4, 1, 3]
    expected = sum(D[a, b] for a, b in zip(r, r[1:] + [0]))
    assert tour_length(r, D) == pytest.approx(expected)


@pytest.mark.parametrize("bad_route", [
    [0, 1, 1], [0, 1], [0, 1, 2, 0], [0, 1, 5], [0, "a", 2], [0, 1.5, 2],
])
def test_tour_length_rejects_malformed_routes(bad_route):
    with pytest.raises(ValueError):
        tour_length(bad_route, make_D(3, 0))


@pytest.mark.parametrize("bad_D", [
    np.ones((3, 4)), np.ones(3), np.array([[0, np.nan], [np.nan, 0]]),
    np.array([[0, np.inf], [np.inf, 0]]),
])
def test_tour_length_rejects_invalid_matrices(bad_D):
    with pytest.raises(ValueError):
        tour_length([0, 1], bad_D)


# -------------------------- nearest_neighbour ------------------------------
@pytest.mark.parametrize("n", [3, 5, 12, 30])
def test_nn_visits_each_node_once_and_starts_at_depot(n):
    route = nearest_neighbour(make_D(n, n))
    assert route[0] == 0
    assert sorted(route) == list(range(n))


def test_nn_custom_depot():
    route = nearest_neighbour(make_D(8, 1), depot=3)
    assert route[0] == 3 and sorted(route) == list(range(8))


def test_nn_picks_closest_unvisited():
    D = np.array([[0, 5, 1], [5, 0, 2], [1, 2, 0]], dtype=float)
    assert nearest_neighbour(D) == [0, 2, 1]


def test_nn_rejects_bad_depot():
    with pytest.raises(ValueError):
        nearest_neighbour(make_D(4, 0), depot=9)


# ------------------------------- two_opt -----------------------------------
@pytest.mark.parametrize("n", [4, 6, 10, 25, 50])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_two_opt_valid_depot_fixed_and_never_worse(n, seed):
    D = make_D(n, seed)
    nn = nearest_neighbour(D)
    opt = two_opt(nn, D)
    assert opt[0] == 0
    assert sorted(opt) == list(range(n))
    assert tour_length(opt, D) <= tour_length(nn, D) + 1e-9


def test_two_opt_improves_a_crossing_tour():
    # Square corners visited in a crossing order: 0->2->1->3 (diagonals).
    xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    D = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    bad = [0, 2, 1, 3]
    opt = two_opt(bad, D)
    assert tour_length(opt, D) == pytest.approx(4.0)


def test_two_opt_does_not_mutate_input_and_handles_tiny():
    D = make_D(5, 3)
    r = [0, 3, 1, 4, 2]
    copy = list(r)
    two_opt(r, D)
    assert r == copy
    assert two_opt([0, 1, 2], make_D(3, 0)) == [0, 1, 2]


def test_two_opt_rejects_invalid_route():
    with pytest.raises(ValueError):
        two_opt([0, 1, 1, 3], make_D(4, 0))


# ------------------ exact <= 2-opt <= NN on small cases --------------------
@pytest.mark.parametrize("n", [5, 6, 7, 8])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_exact_le_two_opt_le_nn(n, seed):
    D = make_D(n, seed)
    nn = nearest_neighbour(D)
    l_nn = tour_length(nn, D)
    l_2opt = tour_length(two_opt(nn, D), D)
    l_exact = brute_force(D)  # proven optimal by exhaustive search
    assert l_exact <= l_2opt + 1e-9
    assert l_2opt <= l_nn + 1e-9


def test_mtz_solver_matches_brute_force_if_available():
    pytest.importorskip("pulp")
    tsp_exact = pytest.importorskip("tsp_exact")
    D = make_D(7, 11)
    res = tsp_exact.solve_tsp(D, time_limit=30)
    assert res["optimal"]
    assert tour_length(res["route"], D) == pytest.approx(brute_force(D), abs=1e-6)
