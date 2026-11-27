"""TSP heuristics: tour length, Nearest Neighbour and 2-opt.

Route convention (shared contract)
----------------------------------
A route is a list such as [0, 2, 4, 1, 3]. The depot appears once, at the
start. The return edge to the depot is implicit, i.e. the route above means
0 -> 2 -> 4 -> 1 -> 3 -> 0.

This module depends only on NumPy. It never imports data.py or any
experiment/plotting module.
"""

import operator

import numpy as np


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_matrix(D):
    """Return D as a float array after checking it is a finite square matrix."""
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be a square 2D matrix, got shape {D.shape}.")
    if D.shape[0] == 0:
        raise ValueError("D must contain at least one node.")
    if not np.all(np.isfinite(D)):
        raise ValueError("D contains NaN or infinite values.")
    return D


def _validate_route(route, n):
    """Return route as a list of Python ints after checking it is a permutation
    of 0..n-1 (every node exactly once, depot not repeated at the end)."""
    try:
        route = [operator.index(r) for r in route]
    except TypeError as exc:
        raise ValueError("Route must be a sequence of integer node indices.") from exc
    if len(route) != n:
        raise ValueError(
            f"Route has {len(route)} entries but D has {n} nodes. Every node must "
            "appear exactly once and the depot must not be repeated at the end."
        )
    if sorted(route) != list(range(n)):
        raise ValueError(
            "Route must contain every node 0..n-1 exactly once (no repeats, "
            "no missing or out-of-range nodes)."
        )
    return route


# ---------------------------------------------------------------------------
# 1. Tour length
# ---------------------------------------------------------------------------
def tour_length(route, D):
    """Length of the closed tour, INCLUDING the final edge back to route[0].

    Parameters
    ----------
    route : list of int, depot first, each node exactly once.
    D     : (n, n) distance matrix in km.

    Returns
    -------
    float : tour length in km.
    """
    D = _validate_matrix(D)
    route = _validate_route(route, D.shape[0])
    idx = np.asarray(route, dtype=int)
    # D[r_k, r_{k+1}] for every k, where np.roll supplies the closing edge.
    return float(D[idx, np.roll(idx, -1)].sum())


# ---------------------------------------------------------------------------
# 2. Nearest Neighbour
# ---------------------------------------------------------------------------
def nearest_neighbour(D, depot=0):
    """Greedy Nearest Neighbour tour starting from `depot`.

    Repeatedly moves to the closest unvisited node. Ties are broken by the
    smallest node index so results are deterministic.

    Returns
    -------
    list of int : route with the depot first and appearing only once.
    """
    D = _validate_matrix(D)
    n = D.shape[0]
    try:
        depot = operator.index(depot)
    except TypeError as exc:
        raise ValueError("depot must be an integer node index.") from exc
    if not 0 <= depot < n:
        raise ValueError(f"depot {depot} is out of range for {n} nodes.")

    visited = np.zeros(n, dtype=bool)
    visited[depot] = True
    route = [depot]
    current = depot
    for _ in range(n - 1):
        row = np.where(visited, np.inf, D[current])
        nxt = int(np.argmin(row))
        route.append(nxt)
        visited[nxt] = True
        current = nxt
    return route


# ---------------------------------------------------------------------------
# 3. 2-opt
# ---------------------------------------------------------------------------
def two_opt(route, D, improvement_tol=1e-9, max_passes=1000):
    """2-opt local search (assumes a symmetric distance matrix).

    For tour edges (a,b) = (route[i], route[i+1]) and (c,d) = (route[j],
    route[j+1 mod n]) with non-adjacent edges, the change in length from
    replacing them by (a,c) and (b,d) is

        delta = D[a,c] + D[b,d] - D[a,b] - D[c,d]

    The exchange is accepted only if delta < -improvement_tol, and is done by
    reversing route[i+1 .. j]. Because i >= 0, position 0 is never touched, so
    the depot stays fixed at the start. The closing edge (last node -> depot)
    is included as the edge with j = n-1.

    Parameters
    ----------
    route : valid initial route (normally Nearest Neighbour output).
    D : (n, n) distance matrix.
    improvement_tol : minimum improvement (km) needed to accept a move.
    max_passes : safety cap on full sweeps over all edge pairs.

    Returns
    -------
    list of int : improved route, same convention as the input.
    """
    D = _validate_matrix(D)
    n = D.shape[0]
    r = _validate_route(route, n)
    if improvement_tol < 0:
        raise ValueError("improvement_tol must be non-negative.")
    if n < 4:  # no pair of non-adjacent edges exists
        return r

    for _ in range(max_passes):
        improved = False
        for i in range(0, n - 2):
            a, b = r[i], r[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # edges (r[0],r[1]) and (r[n-1],r[0]) share the depot
                c, d = r[j], r[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -improvement_tol:
                    r[i + 1:j + 1] = r[i + 1:j + 1][::-1]
                    b = r[i + 1]  # segment reversed, so the edge's end changed
                    improved = True
        if not improved:
            break
    return r
