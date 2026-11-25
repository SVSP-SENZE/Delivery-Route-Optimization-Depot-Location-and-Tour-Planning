"""Figures 1-4 (locations, route comparison, runtime, optimality gap).

Owner: Prerana Kurra

All plotting functions read saved data (CSV / JSON / arrays). None of them
runs any optimization, so changing a title or axis never re-solves anything.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # headless backend: works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_STYLE = {
    "exact": {"label": "Exact (MTZ)", "color": "tab:green", "marker": "o"},
    "nearest_neighbour": {"label": "Nearest Neighbour", "color": "tab:red", "marker": "s"},
    "two_opt": {"label": "NN + 2-opt", "color": "tab:blue", "marker": "^"},
}


def _prepare(output_path):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def _route_km(points, route):
    pts = np.asarray(points, dtype=float)
    idx = np.asarray(route, dtype=int)
    return float(np.linalg.norm(pts[idx] - pts[np.roll(idx, -1)], axis=1).sum())


# ---------------------------------------------------------------------------
# Figure 1: locations
# ---------------------------------------------------------------------------
def plot_locations(all_xy, names, baseline_depot_xy, output_path="results/figure1_locations.png",
                   optimized_depot_xy=None):
    """Projected delivery points with labels and the baseline depot.

    `all_xy` has every real location (depot included), in km. The optimized
    depot is only drawn if `optimized_depot_xy` is supplied.
    """
    all_xy = np.asarray(all_xy, dtype=float)
    depot = np.asarray(baseline_depot_xy, dtype=float)
    is_depot = np.all(np.isclose(all_xy, depot, atol=1e-9), axis=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(all_xy[~is_depot, 0], all_xy[~is_depot, 1], c="tab:blue", s=50,
               label="Delivery points", zorder=3)
    ax.scatter(*depot, c="tab:red", marker="*", s=300, edgecolor="k",
               label="Baseline depot", zorder=4)
    if optimized_depot_xy is not None:
        opt = np.asarray(optimized_depot_xy, dtype=float)
        ax.scatter(*opt, c="tab:green", marker="D", s=120, edgecolor="k",
                   label="Stage 1 optimized depot", zorder=4)
    for (x, y), name in zip(all_xy, names):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("x (km, east of reference)")
    ax.set_ylabel("y (km, north of reference)")
    ax.set_title("Figure 1: Bengaluru delivery locations (projected, km)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    _prepare(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Figure 2: route comparison
# ---------------------------------------------------------------------------
def _draw_route(ax, pts, route, title, color):
    pts = np.asarray(pts, dtype=float)
    if route is not None:
        order = list(route) + [route[0]]
        for a, b in zip(order[:-1], order[1:]):
            ax.annotate("", xy=pts[b], xytext=pts[a],
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.4), zorder=2)
    ax.scatter(pts[1:, 0], pts[1:, 1], c="tab:blue", s=40, zorder=3)
    ax.scatter(*pts[0], c="tab:red", marker="*", s=250, edgecolor="k", zorder=4)
    for i, (x, y) in enumerate(pts):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)


def plot_route_comparison(tsp_points, exact_result, nn_route, two_opt_route,
                          output_path="results/figure2_routes.png"):
    """Exact, NN and 2-opt tours side by side on the same instance.

    `exact_result` is the dict from solve_tsp (needs route/optimal/feasible).
    A time-limited route is labelled feasible, not optimal.
    """
    pts = np.asarray(tsp_points, dtype=float)
    exact_route = exact_result.get("route") if exact_result else None

    if exact_route is None:
        exact_title = "Exact (MTZ)\nno feasible solution found"
    else:
        label = "optimal" if exact_result.get("optimal") else "feasible, NOT proven optimal"
        exact_title = f"Exact (MTZ): {_route_km(pts, exact_route):.2f} km\n({label})"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _draw_route(axes[0], pts, exact_route, exact_title, "tab:green")
    _draw_route(axes[1], pts, nn_route,
                f"Nearest Neighbour: {_route_km(pts, nn_route):.2f} km", "tab:red")
    _draw_route(axes[2], pts, two_opt_route,
                f"NN + 2-opt: {_route_km(pts, two_opt_route):.2f} km", "tab:blue")
    fig.suptitle("Figure 2: Route comparison on the same instance (node 0 = depot)")
    fig.tight_layout()
    _prepare(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_route_comparison_from_json(json_path="results/route_comparison.json",
                                    output_path="results/figure2_routes.png"):
    """Redraw Figure 2 from results/route_comparison.json (no solving)."""
    with open(json_path) as f:
        data = json.load(f)
    return plot_route_comparison(
        np.array(data["tsp_points"]), data["exact"],
        data["nn_route"], data["two_opt_route"], output_path)


# ---------------------------------------------------------------------------
# Figure 3: runtime vs n
# ---------------------------------------------------------------------------
def plot_runtime(csv_path="results/scaling.csv", output_path="results/figure3_runtime.png"):
    """Mean runtime (± std across seeds) vs total node count, log y-axis."""
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, style in METHOD_STYLE.items():
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        g = sub.groupby("n")["runtime_sec"].agg(["mean", "std"]).fillna(0.0).reset_index()
        mean = np.maximum(g["mean"].to_numpy(), 1e-7)  # keep positive for log axis
        ax.errorbar(g["n"], mean, yerr=g["std"], marker=style["marker"], color=style["color"],
                    label=style["label"], capsize=3)
    ax.set_yscale("log")
    ax.set_xlabel("Total nodes n (including depot)")
    ax.set_ylabel("Runtime (seconds, log scale)")
    ax.set_title("Figure 3: Runtime vs problem size (mean ± std over seeds)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    _prepare(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Figure 4: optimality gap
# ---------------------------------------------------------------------------
def plot_gap(csv_path="results/scaling.csv", output_path="results/figure4_gap.png"):
    """NN and 2-opt gap against PROVEN optimal solutions only.

    Rows without a proven optimum have a blank gap_percent and are omitted.
    """
    df = pd.read_csv(csv_path)
    df = df[df["method"].isin(["nearest_neighbour", "two_opt"])]
    df = df[df["gap_percent"].notna()]

    fig, ax = plt.subplots(figsize=(8, 6))
    if df.empty:
        ax.text(0.5, 0.5, "No proven-optimal instances available", ha="center",
                va="center", transform=ax.transAxes)
    else:
        for method in ["nearest_neighbour", "two_opt"]:
            sub = df[df["method"] == method]
            if sub.empty:
                continue
            style = METHOD_STYLE[method]
            g = sub.groupby("n")["gap_percent"].agg(["mean", "std", "count"]).fillna(0.0).reset_index()
            ax.errorbar(g["n"], g["mean"], yerr=g["std"], marker=style["marker"],
                        color=style["color"], label=style["label"], capsize=3)
            for n, m, c in zip(g["n"], g["mean"], g["count"]):
                ax.annotate(f"{int(c)}", (n, m), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7, color=style["color"])
        ax.legend()
    ax.set_xlabel("Total nodes n (including depot)")
    ax.set_ylabel("Gap to proven optimum (%)")
    ax.set_title("Figure 4: Heuristic optimality gap (proven optima only)\n"
                 "numbers = seeds with a proven optimum")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _prepare(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
