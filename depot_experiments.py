"""Experiments B and C for depot optimization and depot-choice effects."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import EPSILON, GRAD_TOL, GRID_SIZE, MAX_OPT_ITER, RANDOM_SEEDS
from data import random_instance, dist_matrix
from depot import bfgs_method, gradient, gradient_descent, newton_method, objective
from heuristics import nearest_neighbour, tour_length, two_opt

METHODS = {
    "gradient_descent": gradient_descent,
    "newton_method": newton_method,
    "bfgs_method": bfgs_method,
}


def _validate_points(points, name):
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) == 0:
        raise ValueError(f"{name} must have shape (m, 2) with m >= 1.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite coordinates.")
    return array


def _write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_optimizer_experiments(
    real_delivery_points,
    random_seeds=RANDOM_SEEDS,
    random_n=15,
    eps_values=(1e-6, 1e-5, 1e-4),
    output_dir="results",
):
    """Compare depot optimizers and SciPy BFGS on real and seeded data."""
    real_points = _validate_points(real_delivery_points, "real_delivery_points")
    datasets = [("real", None, real_points)]
    for seed in random_seeds:
        datasets.append(("random", int(seed), random_instance(random_n, int(seed))))

    summary_rows, history_rows = [], []
    for dataset, seed, points in datasets:
        start = np.mean(points, axis=0)
        for eps in eps_values:
            reference = minimize(
                objective, start, args=(points, eps), jac=gradient, method="BFGS",
                options={"gtol": GRAD_TOL, "maxiter": MAX_OPT_ITER},
            )
            if not reference.success:
                raise RuntimeError(
                    f"SciPy BFGS reference failed for dataset={dataset!r}, "
                    f"seed={seed!r}, epsilon={eps}: {reference.message}"
                )
            for method_name, method in METHODS.items():
                result = method(points, start=start, eps=eps, tol=GRAD_TOL, max_iter=MAX_OPT_ITER)
                summary_rows.append({
                    "method": method_name, "dataset": dataset, "seed": seed,
                    "epsilon": eps, "iterations": result["iterations"],
                    "runtime_sec": result["runtime_sec"], "objective": result["objective"],
                    "depot_x": result["depot_xy"][0], "depot_y": result["depot_xy"][1],
                    "grad_norm": result["grad_norm"],
                    "scipy_distance": float(np.linalg.norm(result["depot_xy"] - reference.x)),
                    "converged": result["converged"],
                })
                for record in result["history"]:
                    history_rows.append({
                        "dataset": dataset, "seed": seed, "epsilon": eps,
                        "method": method_name, **record,
                    })

    summary_df = pd.DataFrame(summary_rows)
    history_df = pd.DataFrame(history_rows)
    _write_csv(summary_df, Path(output_dir) / "optimizers.csv")
    _write_csv(history_df, Path(output_dir) / "optimizer_history.csv")
    return {"summary_df": summary_df, "history_df": history_df}


def run_depot_effect_experiment(
    delivery_points,
    baseline_depot_xy,
    grid_size=GRID_SIZE,
    output_dir="results",
):
    """Compare baseline, subset-specific Stage 1 and best-grid depots."""
    deliveries = _validate_points(delivery_points, "delivery_points")
    baseline = np.asarray(baseline_depot_xy, dtype=float)
    if baseline.shape != (2,) or not np.all(np.isfinite(baseline)):
        raise ValueError("baseline_depot_xy must be a finite coordinate pair.")
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)) or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least 2.")

    stage1 = bfgs_method(
        deliveries, start=np.mean(deliveries, axis=0), eps=EPSILON,
        tol=GRAD_TOL, max_iter=MAX_OPT_ITER,
    )
    minimum = deliveries.min(axis=0)
    maximum = deliveries.max(axis=0)
    margin = np.maximum((maximum - minimum) * 0.05, 0.5)
    xs = np.linspace(minimum[0] - margin[0], maximum[0] + margin[0], int(grid_size))
    ys = np.linspace(minimum[1] - margin[1], maximum[1] + margin[1], int(grid_size))

    def evaluate(depot_xy):
        points = np.vstack((depot_xy, deliveries))
        matrix = dist_matrix(points)
        initial = nearest_neighbour(matrix, depot=0)
        route = two_opt(initial, matrix)
        return tour_length(route, matrix)

    baseline_length = evaluate(baseline)
    stage1_xy = np.asarray(stage1["depot_xy"], dtype=float)
    stage1_length = evaluate(stage1_xy)
    grid_rows = []
    for x in xs:
        for y in ys:
            candidate = np.array([x, y])
            grid_rows.append({
                "grid_x": float(x), "grid_y": float(y),
                "tour_length_km": evaluate(candidate), "algorithm": "nearest_neighbour+2_opt",
            })
    grid_df = pd.DataFrame(grid_rows)
    best_row = grid_df.loc[grid_df["tour_length_km"].idxmin()]
    grid_best_xy = np.array([best_row["grid_x"], best_row["grid_y"]], dtype=float)
    grid_best_length = float(best_row["tour_length_km"])

    candidates = [
        ("baseline", baseline, baseline_length),
        ("stage1", stage1_xy, stage1_length),
        ("grid_best", grid_best_xy, grid_best_length),
    ]
    summary_rows = []
    for depot_type, coordinate, length in candidates:
        summary_rows.append({
            "depot_type": depot_type, "depot_x": float(coordinate[0]),
            "depot_y": float(coordinate[1]), "algorithm": "nearest_neighbour+2_opt",
            "tour_length_km": float(length),
            "percentage_reduction": 0.0 if depot_type == "baseline" else
                float(100.0 * (baseline_length - length) / baseline_length),
        })
    summary_df = pd.DataFrame(summary_rows)
    _write_csv(summary_df, Path(output_dir) / "depot_effect.csv")
    _write_csv(grid_df, Path(output_dir) / "depot_grid.csv")
    return {
        "summary_df": summary_df, "grid_df": grid_df,
        "stage1_result": stage1,
    }

