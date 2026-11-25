"""Plots for saved depot-optimizer histories.

This module only reads optimizer results that have already been written to CSV;
it never imports or invokes an optimization routine.
"""

import os

import matplotlib

matplotlib.use("Agg")  # Support rendering in headless environments.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = {
    "dataset",
    "seed",
    "epsilon",
    "method",
    "iteration",
    "x",
    "y",
    "objective",
    "grad_norm",
    "step_size",
}

_METHODS = {
    "gradient descent": {
        "label": "Gradient Descent",
        "color": "tab:blue",
        "marker": "o",
    },
    "newton": {"label": "Newton", "color": "tab:orange", "marker": "s"},
    "bfgs": {"label": "BFGS", "color": "tab:green", "marker": "^"},
}

_METHOD_ALIASES = {
    "gradient descent": "gradient descent",
    "gradient_descent": "gradient descent",
    "gd": "gradient descent",
    "newton": "newton",
    "newton_method": "newton",
    "bfgs": "bfgs",
    "bfgs_method": "bfgs",
}


def _validated_delivery_points(delivery_points):
    """Return delivery points as a finite two-dimensional float array."""
    try:
        points = np.asarray(delivery_points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("delivery_points must be a numeric array with shape (m, 2).") from exc

    if points.ndim != 2 or points.shape[1:] != (2,) or points.shape[0] == 0:
        raise ValueError("delivery_points must be a non-empty array with shape (m, 2).")
    if not np.isfinite(points).all():
        raise ValueError("delivery_points must contain only finite coordinates.")
    return points


def _read_history(history_csv_path):
    """Load and validate the numerical fields used by the figure."""
    try:
        history = pd.read_csv(history_csv_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read optimizer-history CSV: {history_csv_path!s}") from exc

    missing = sorted(_REQUIRED_COLUMNS.difference(history.columns))
    if missing:
        raise ValueError("Optimizer-history CSV is missing required columns: " + ", ".join(missing))
    if history.empty:
        raise ValueError("Optimizer-history CSV contains no rows.")

    numeric_columns = ["seed", "epsilon", "iteration", "x", "y", "objective", "grad_norm", "step_size"]
    for column in numeric_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    required_numeric = [column for column in numeric_columns if column != "seed"]
    if history[required_numeric].isna().any().any() or not np.isfinite(
        history[required_numeric].to_numpy(dtype=float)
    ).all():
        raise ValueError("Optimizer-history CSV has missing, non-numeric, or non-finite numeric values.")
    if history[["dataset", "method"]].isna().any().any():
        raise ValueError("Optimizer-history CSV has missing dataset or method values.")
    # Real-data rows have no random seed; randomized rows must record one.
    real_rows = history["dataset"].astype(str).str.strip().str.lower() == "real"
    random_seeds = history.loc[~real_rows, "seed"]
    if (random_seeds.isna().any()
            or not np.isfinite(random_seeds.to_numpy(dtype=float)).all()
            or not np.isfinite(history.loc[history["seed"].notna(), "seed"].to_numpy(dtype=float)).all()):
        raise ValueError("Optimizer-history CSV must record a finite seed for every randomized dataset.")
    if (history["grad_norm"] < 0).any():
        raise ValueError("Optimizer-history CSV grad_norm values must be non-negative.")

    history = history.copy()
    history["_method_key"] = history["method"].astype(str).str.strip().str.lower().map(_METHOD_ALIASES)
    return history


def plot_optimizer_convergence(history_csv_path, delivery_points, output_path, dataset=None, epsilon=None):
    """Plot saved Gradient Descent, Newton, and BFGS convergence and depot paths."""
    points = _validated_delivery_points(delivery_points)
    history = _read_history(history_csv_path)

    if dataset is not None:
        history = history[history["dataset"] == dataset]
    if epsilon is not None:
        try:
            epsilon_value = float(epsilon)
        except (TypeError, ValueError) as exc:
            raise ValueError("epsilon must be a finite numeric value when provided.") from exc
        if not np.isfinite(epsilon_value):
            raise ValueError("epsilon must be a finite numeric value when provided.")
        history = history[np.isclose(history["epsilon"], epsilon_value, rtol=1e-10, atol=1e-12)]
    if history.empty:
        filters = []
        if dataset is not None:
            filters.append(f"dataset={dataset!r}")
        if epsilon is not None:
            filters.append(f"epsilon={epsilon!r}")
        detail = " and ".join(filters) if filters else "the supplied history"
        raise ValueError(f"No optimizer-history rows remain after filtering by {detail}.")

    available = set(history["_method_key"].dropna())
    missing_methods = [style["label"] for key, style in _METHODS.items() if key not in available]
    if missing_methods:
        raise ValueError("Optimizer-history CSV is missing required methods: " + ", ".join(missing_methods))

    centroid = points.mean(axis=0)
    fig, (convergence_ax, path_ax) = plt.subplots(1, 2, figsize=(14, 6))

    convergence_ax.set_title("Optimizer convergence")
    path_ax.set_title("Optimizer depot-coordinate paths")
    path_ax.scatter(points[:, 0], points[:, 1], c="tab:blue", s=45, alpha=0.75,
                    label="Delivery points", zorder=2)
    path_ax.scatter(*centroid, c="black", marker="*", s=240, edgecolor="white",
                    linewidth=0.8, label="Starting centroid", zorder=5)

    for method_key, style in _METHODS.items():
        method_history = history[history["_method_key"] == method_key]
        first_series = True
        # Separate seeds before sorting so multi-seed CSVs never draw a path between runs.
        for _, series in method_history.groupby("seed", sort=True, dropna=False):
            series = series.sort_values("iteration")
            grad_norm = series["grad_norm"].to_numpy(dtype=float)
            positive_floor = np.finfo(float).tiny
            plotted_norm = np.maximum(grad_norm, positive_floor)
            label = style["label"] if first_series else "_nolegend_"
            convergence_ax.semilogy(
                series["iteration"], plotted_norm, color=style["color"], marker=style["marker"],
                markersize=4, linewidth=1.5, label=label,
            )
            path_ax.plot(
                series["x"], series["y"], color=style["color"], marker=style["marker"],
                markersize=4, linewidth=1.5, label=label, zorder=3,
            )
            final = series.iloc[-1]
            path_ax.scatter(final["x"], final["y"], c=style["color"], marker=style["marker"],
                            s=95, edgecolor="black", linewidth=0.6, zorder=4)
            first_series = False

    convergence_ax.set_xlabel("Iteration")
    convergence_ax.set_ylabel("Gradient norm (log scale)")
    convergence_ax.grid(alpha=0.3, which="both")
    convergence_ax.legend(loc="best")

    path_ax.set_xlabel("x (km)")
    path_ax.set_ylabel("y (km)")
    path_ax.set_aspect("equal", adjustable="datalim")
    path_ax.grid(alpha=0.3)
    path_ax.legend(loc="best")

    fig.tight_layout()
    output_path = os.fspath(output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=200, format="png")
    plt.close(fig)
    return output_path


def plot_depot_heatmap(grid_csv="results/depot_grid.csv",
                       effect_csv="results/depot_effect.csv",
                       delivery_points=None,
                       output_path="results/figure6_heatmap.png"):
    """Plot grid tour lengths and mark deliveries, Stage 1 and grid-best depots."""
    required_grid = {"grid_x", "grid_y", "tour_length_km"}
    required_effects = {"depot_type", "depot_x", "depot_y", "tour_length_km"}
    try:
        grid = pd.read_csv(grid_csv)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read depot-grid CSV: {grid_csv!s}") from exc
    try:
        effects = pd.read_csv(effect_csv)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read depot-effect CSV: {effect_csv!s}") from exc
    missing_grid = sorted(required_grid.difference(grid.columns))
    if missing_grid:
        raise ValueError("Depot grid CSV is missing required columns: " + ", ".join(missing_grid))
    missing_effects = sorted(required_effects.difference(effects.columns))
    if missing_effects:
        raise ValueError("Depot effect CSV is missing required columns: " + ", ".join(missing_effects))
    if grid.empty:
        raise ValueError("Depot grid CSV contains no rows.")
    if effects.empty:
        raise ValueError("Depot effect CSV contains no rows.")
    for column in required_grid:
        grid[column] = pd.to_numeric(grid[column], errors="coerce")
    for column in {"depot_x", "depot_y", "tour_length_km"}:
        effects[column] = pd.to_numeric(effects[column], errors="coerce")
    if grid[list(required_grid)].isna().any().any() or not np.isfinite(grid[list(required_grid)]).all().all():
        raise ValueError("Depot grid CSV has missing, non-numeric, or non-finite values.")
    if effects[["depot_x", "depot_y", "tour_length_km"]].isna().any().any() or not np.isfinite(effects[["depot_x", "depot_y", "tour_length_km"]]).all().all():
        raise ValueError("Depot effect CSV has missing, non-numeric, or non-finite values.")
    if effects["depot_type"].isna().any():
        raise ValueError("Depot effect CSV contains missing depot_type values.")
    if grid[["grid_x", "grid_y"]].duplicated().any():
        raise ValueError("Depot grid CSV contains duplicate grid coordinates.")
    effects = effects.set_index("depot_type")
    x_values = np.sort(grid["grid_x"].unique())
    y_values = np.sort(grid["grid_y"].unique())
    pivot = grid.pivot(index="grid_y", columns="grid_x", values="tour_length_km").reindex(
        index=y_values, columns=x_values
    )
    if pivot.isna().any().any():
        raise ValueError("Depot grid CSV does not contain a complete rectangular grid.")
    fig, ax = plt.subplots(figsize=(8, 7))
    mesh = ax.pcolormesh(x_values, y_values, pivot.to_numpy(), shading="auto", cmap="viridis_r")
    fig.colorbar(mesh, ax=ax, label="NN + 2-opt tour length (km)")
    if delivery_points is not None:
        points = _validated_delivery_points(delivery_points)
        ax.scatter(points[:, 0], points[:, 1], c="white", edgecolor="black", s=30,
                   label="Delivery points", zorder=3)
    for depot_type, label, marker, color in [
        ("stage1", "Stage 1 depot", "D", "tab:red"),
        ("grid_best", "Grid-best depot", "*", "tab:cyan"),
    ]:
        if depot_type in effects.index:
            row = effects.loc[depot_type]
            ax.scatter(row["depot_x"], row["depot_y"], marker=marker, c=color,
                       edgecolor="black", s=130, label=label, zorder=4)
    ax.set_xlabel("Depot x (km)")
    ax.set_ylabel("Depot y (km)")
    ax.set_title("Figure 6: Delivery-tour length across depot grid")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path = os.fspath(output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=200, format="png")
    plt.close(fig)
    return output_path
