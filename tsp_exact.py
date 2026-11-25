"""Exact symmetric TSP solver using the Miller-Tucker-Zemlin formulation."""

from itertools import permutations
import time

import numpy as np

from heuristics import tour_length

__all__ = ["solve_tsp", "brute_force_tsp"]


def _validate_distance_matrix(D):
    try:
        matrix = np.asarray(D, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("D must be a finite numeric square distance matrix.") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 3:
        raise ValueError(f"D must be square with n >= 3; got shape {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("D must contain only finite distances.")
    if np.any(matrix < 0):
        raise ValueError("D must contain non-negative distances.")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-10):
        raise ValueError("D must be symmetric.")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-10):
        raise ValueError("D must have a zero diagonal.")
    return matrix


def _validate_time_limit(time_limit):
    if isinstance(time_limit, (bool, np.bool_)):
        raise ValueError("time_limit must be finite and positive.")
    try:
        limit = float(time_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("time_limit must be finite and positive.") from exc
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("time_limit must be finite and positive.")
    return limit


def solve_tsp(D, time_limit=60):
    """Solve a symmetric TSP, returning an extracted route and solver diagnostics."""
    matrix = _validate_distance_matrix(D)
    limit = _validate_time_limit(time_limit)
    started = time.perf_counter()
    try:
        import pulp
    except ImportError as exc:
        raise ImportError("solve_tsp requires PuLP and its CBC solver; install requirements.txt.") from exc

    n = matrix.shape[0]
    problem = pulp.LpProblem("delivery_tsp_mtz", pulp.LpMinimize)
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]
    x = pulp.LpVariable.dicts("x", arcs, lowBound=0, upBound=1, cat=pulp.LpBinary)
    u = {
        i: pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n - 1, cat=pulp.LpContinuous)
        for i in range(1, n)
    }
    problem += pulp.lpSum(matrix[i, j] * x[i, j] for i, j in arcs)
    for i in range(n):
        problem += pulp.lpSum(x[i, j] for j in range(n) if j != i) == 1
        problem += pulp.lpSum(x[j, i] for j in range(n) if j != i) == 1
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                problem += u[i] - u[j] + n * x[i, j] <= n - 1

    try:
        # PuLP 4 exposes its bundled/CLI CBC adapter as COIN_CMD; older PuLP
        # releases provide PULP_CBC_CMD. Both use CBC and accept timeLimit.
        solver_factory = getattr(pulp, "PULP_CBC_CMD", None) or getattr(pulp, "COIN_CMD", None)
        if solver_factory is None:
            raise RuntimeError("This PuLP installation does not expose a CBC command solver.")
        solver = solver_factory(msg=False, timeLimit=limit)
        problem.solve(solver)
    except (pulp.PulpSolverError, OSError) as exc:
        raise RuntimeError(f"CBC could not solve the TSP model: {exc}") from exc

    solver_status = str(pulp.LpStatus.get(problem.status, problem.status))
    solution_code = getattr(problem, "sol_status", None)
    solution_status = str(pulp.LpSolution.get(solution_code, "No Solution"))
    selected = {}
    for i in range(n):
        outgoing = [j for j in range(n) if i != j and x[i, j].value() is not None and x[i, j].value() > 0.5]
        if len(outgoing) != 1:
            selected = {}
            break
        selected[i] = outgoing[0]

    route = None
    length = None
    feasible = False
    if len(selected) == n:
        candidate = [0]
        seen = {0}
        current = 0
        for _ in range(n):
            current = selected[current]
            if current == 0:
                break
            if current in seen or current not in range(n):
                break
            candidate.append(current)
            seen.add(current)
        if current == 0 and len(candidate) == n and seen == set(range(n)):
            route = candidate
            length = tour_length(route, matrix)
            feasible = True
            objective_value = pulp.value(problem.objective)
            if objective_value is not None and not np.isclose(length, objective_value, rtol=1e-7, atol=1e-7):
                raise RuntimeError("Extracted tour length does not match the MTZ objective.")

    return {
        "route": route,
        "length_km": length,
        "solver_status": solver_status,
        "solution_status": solution_status,
        "feasible": bool(feasible),
        "optimal": bool(feasible and problem.status == pulp.LpStatusOptimal),
        "runtime_sec": float(time.perf_counter() - started),
    }


def brute_force_tsp(D):
    """Find a proven-optimal tour by enumeration (intended only for small n)."""
    matrix = _validate_distance_matrix(D)
    n = matrix.shape[0]
    best_route = None
    best_length = float("inf")
    for order in permutations(range(1, n)):
        route = [0, *order]
        length = tour_length(route, matrix)
        if length < best_length:
            best_route, best_length = route, length
    return {"route": best_route, "length_km": float(best_length), "optimal": True}

