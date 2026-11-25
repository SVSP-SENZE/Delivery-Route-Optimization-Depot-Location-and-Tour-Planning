"""Experiment A: TSP scaling and solution quality (exact vs NN vs 2-opt).

Owner: Prerana Kurra

Imports: data.py, tsp_exact.py, heuristics.py, config.py (one-way dependency).
"""

import json
import os
import time

import numpy as np
import pandas as pd

from config import TSP_TIME_LIMIT_SEC, TSP_N_VALUES, RANDOM_SEEDS, TWO_OPT_TOL
from data import random_instance, dist_matrix
from heuristics import nearest_neighbour, two_opt, tour_length
from tsp_exact import solve_tsp

CSV_COLUMNS = [
    "n", "seed", "method", "route_length_km", "runtime_sec", "solver_status",
    "feasible", "optimality_proven", "gap_percent",
]


GAP_NEGATIVE_TOL = 1e-10


def _gap(length, exact_length):
    """Return a proven-optimum gap, rejecting non-numerical negatives.

    The tolerance is in percentage points and only absorbs floating-point
    roundoff; a materially negative gap indicates inconsistent route data.
    """
    gap = 100.0 * (float(length) - float(exact_length)) / float(exact_length)
    if gap < -GAP_NEGATIVE_TOL:
        raise ValueError(
            "Heuristic gap is materially negative relative to the proven exact "
            f"tour: {gap:.12g}% (heuristic={length}, exact={exact_length})."
        )
    return 0.0 if gap < 0.0 else gap


def _row(n, seed, method, length, runtime, solver_status=None,
         feasible=None, optimality_proven=None, gap=None):
    return {
        "n": n, "seed": seed, "method": method,
        "route_length_km": length, "runtime_sec": runtime,
        "solver_status": solver_status, "feasible": feasible,
        "optimality_proven": optimality_proven,
        "gap_percent": np.nan if gap is None else gap,
    }


def run_tsp_scaling_experiment(
    n_values=None,
    seeds=None,
    time_limit=TSP_TIME_LIMIT_SEC,
    output_path="results/scaling.csv",
    verbose=True,
):
    """Run Experiment A and save the results to `output_path`.

    For every (n, seed): one random point set (node 0 = depot), one distance
    matrix D shared by all methods, then exact MTZ, Nearest Neighbour and
    NN + 2-opt. Heuristic gaps are computed only when the exact solver PROVED
    optimality.

    Timing boundaries: points and D are built before any timer starts;
    exact = model build + CBC + route extraction; NN = nearest_neighbour();
    2-opt = full NN + 2-opt pipeline. CSV writing/printing is never timed.

    Returns
    -------
    pandas.DataFrame with the columns in CSV_COLUMNS.
    """
    n_values = list(TSP_N_VALUES if n_values is None else n_values)
    seeds = list(RANDOM_SEEDS if seeds is None else seeds)
    rows = []

    for n in n_values:
        for seed in seeds:
            # --- instance + distance matrix (NOT timed) -----------------------
            xy = random_instance(n, seed)
            D = dist_matrix(xy)

            # --- exact MTZ ----------------------------------------------------
            t0 = time.perf_counter()
            exact = solve_tsp(D, time_limit=time_limit)
            exact_time = time.perf_counter() - t0

            exact_len = None
            if exact.get("feasible") and exact.get("route") is not None:
                exact_len = tour_length(exact["route"], D)  # independent recompute
            proven = bool(exact.get("optimal")) and exact_len is not None

            # --- Nearest Neighbour -------------------------------------------
            t0 = time.perf_counter()
            nn_route = nearest_neighbour(D, depot=0)
            nn_time = time.perf_counter() - t0
            nn_len = tour_length(nn_route, D)

            # --- 2-opt (timed as NN start + 2-opt) -----------------------------
            t0 = time.perf_counter()
            start = nearest_neighbour(D, depot=0)
            opt_route = two_opt(start, D, improvement_tol=TWO_OPT_TOL)
            opt_time = time.perf_counter() - t0
            opt_len = tour_length(opt_route, D)

            rows.append(_row(
                n, seed, "exact",
                np.nan if exact_len is None else exact_len, exact_time,
                solver_status=exact.get("solver_status"),
                feasible=bool(exact.get("feasible")),
                optimality_proven=proven,
                gap=0.0 if proven else None,
            ))
            rows.append(_row(n, seed, "nearest_neighbour", nn_len, nn_time,
                             gap=_gap(nn_len, exact_len) if proven else None))
            rows.append(_row(n, seed, "two_opt", opt_len, opt_time,
                             gap=_gap(opt_len, exact_len) if proven else None))

            if verbose:
                tag = "optimal" if proven else ("feasible" if exact_len is not None else "none")
                print(f"[Exp A] n={n:>3} seed={seed}: exact={exact_len} ({tag}, "
                      f"{exact_time:.2f}s) NN={nn_len:.2f} 2opt={opt_len:.2f}")

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def summarize_scaling(df, output_path=None):
    """Mean and standard deviation across seeds for each (n, method).

    Gap statistics only use rows where the optimum was proven. Optionally
    saved to `output_path`.
    """
    g = df.groupby(["n", "method"])
    summary = g.agg(
        runs=("seed", "count"),
        length_mean=("route_length_km", "mean"),
        length_std=("route_length_km", "std"),
        runtime_mean=("runtime_sec", "mean"),
        runtime_std=("runtime_sec", "std"),
        gap_mean=("gap_percent", "mean"),
        gap_std=("gap_percent", "std"),
        gap_runs=("gap_percent", "count"),
    ).reset_index()
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        summary.to_csv(output_path, index=False)
    return summary


# ---------------------------------------------------------------------------
# Optional helper for the real-instance route comparison (Figure 2)
# ---------------------------------------------------------------------------
def compute_route_comparison(tsp_points, time_limit=TSP_TIME_LIMIT_SEC):
    """Solve one real instance with exact, NN and NN+2-opt.

    `tsp_points` is vstack([depot_xy, delivery_xy]) in km. Returns a
    JSON-serialisable dict that `tsp_plots.plot_route_comparison_from_json`
    can read. Shubhan can call this and save it with save_route_comparison().
    """
    pts = np.asarray(tsp_points, dtype=float)
    D = dist_matrix(pts)
    exact = solve_tsp(D, time_limit=time_limit)
    nn_route = nearest_neighbour(D, depot=0)
    opt_route = two_opt(nn_route, D, improvement_tol=TWO_OPT_TOL)
    return {
        "tsp_points": pts.tolist(),
        "exact": {
            "route": exact.get("route"),
            "length_km": exact.get("length_km"),
            "feasible": bool(exact.get("feasible")),
            "optimal": bool(exact.get("optimal")),
            "solver_status": exact.get("solver_status"),
        },
        "nn_route": [int(i) for i in nn_route],
        "two_opt_route": [int(i) for i in opt_route],
    }


def save_route_comparison(result, path="results/route_comparison.json"):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path
