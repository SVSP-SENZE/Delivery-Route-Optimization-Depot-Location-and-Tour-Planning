"""Figures 5 and 6, rendered from the saved depot experiment outputs."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def plot_convergence(history_csv="results/optimizer_history.csv",
                     output_path="results/figure5_convergence.png",
                     delivery_points=None):
    """Plot gradient-norm convergence and coordinate paths for real data."""
    history = pd.read_csv(history_csv)
    history = history[(history["dataset"] == "real") & (history["epsilon"] == history["epsilon"].min())]
    if history.empty:
        raise ValueError("No real-dataset optimizer history is available to plot.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"gradient_descent": "tab:blue", "newton_method": "tab:orange", "bfgs_method": "tab:green"}
    labels = {"gradient_descent": "Gradient Descent", "newton_method": "Newton", "bfgs_method": "BFGS"}
    for method, group in history.groupby("method"):
        group = group.sort_values("iteration")
        color = colors.get(method)
        axes[0].semilogy(group["iteration"], np.maximum(group["grad_norm"], 1e-16),
                         marker=".", label=labels.get(method, method), color=color)
        axes[1].plot(group["x"], group["y"], marker=".", label=labels.get(method, method), color=color)
        axes[1].scatter(group["x"].iloc[0], group["y"].iloc[0], marker="s", color=color, s=45)
        axes[1].scatter(group["x"].iloc[-1], group["y"].iloc[-1], marker="*", color=color, s=110)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Gradient norm")
    axes[0].set_title("Gradient convergence")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend()
    axes[1].set_xlabel("x (km)")
    axes[1].set_ylabel("y (km)")
    axes[1].set_title("Optimizer coordinate paths")
    if delivery_points is not None:
        points = np.asarray(delivery_points, dtype=float)
        axes[1].scatter(points[:, 0], points[:, 1], marker="o", facecolors="none",
                        edgecolors="black", s=35, label="Delivery points", zorder=2)
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.suptitle("Figure 5: Depot optimizer convergence")
    fig.tight_layout()
    _prepare(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_depot_heatmap(grid_csv="results/depot_grid.csv",
                       effect_csv="results/depot_effect.csv",
                       delivery_points=None,
                       output_path="results/figure6_heatmap.png"):
    """Plot grid tour lengths and mark deliveries, Stage 1 and grid-best depots."""
    grid = pd.read_csv(grid_csv)
    effects = pd.read_csv(effect_csv).set_index("depot_type")
    if not {"grid_x", "grid_y", "tour_length_km"}.issubset(grid.columns):
        raise ValueError("Depot grid CSV is missing the required coordinate/length columns.")
    x_values = np.sort(grid["grid_x"].unique())
    y_values = np.sort(grid["grid_y"].unique())
    pivot = grid.pivot(index="grid_y", columns="grid_x", values="tour_length_km").reindex(
        index=y_values, columns=x_values
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    mesh = ax.pcolormesh(x_values, y_values, pivot.to_numpy(), shading="auto", cmap="viridis_r")
    fig.colorbar(mesh, ax=ax, label="NN + 2-opt tour length (km)")
    if delivery_points is not None:
        points = np.asarray(delivery_points, dtype=float)
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
    _prepare(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path

