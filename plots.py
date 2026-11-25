"""Thin orchestration layer for all six figures; plotting stays in owner modules."""

from pathlib import Path

import numpy as np

from config import RESULTS_DIR


def generate_all_plots(
    all_xy,
    names,
    baseline_depot_xy,
    route_comparison_path=None,
    results_dir=RESULTS_DIR,
    optimized_depot_xy=None,
):
    """Generate figures 1-6 from the saved experiment files and coordinates."""
    try:
        from tsp_plots import (
            plot_locations,
            plot_route_comparison_from_json,
            plot_runtime,
            plot_gap,
        )
    except ImportError as exc:
        raise RuntimeError("TSP figures need tsp_plots.py and its dependencies.") from exc
    try:
        from depot_plots import plot_optimizer_convergence, plot_depot_heatmap
    except ImportError as exc:
        raise RuntimeError("Figures 5/6 need Jiya's depot_plots.py in the project root.") from exc

    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    route_path = Path(route_comparison_path or output / "route_comparison.json")
    paths = {
        "figure1": plot_locations(
            all_xy, names, baseline_depot_xy,
            str(output / "figure1_locations.png"), optimized_depot_xy=optimized_depot_xy,
        ),
        "figure2": plot_route_comparison_from_json(
            str(route_path), str(output / "figure2_routes.png")
        ),
        "figure3": plot_runtime(str(output / "scaling.csv"), str(output / "figure3_runtime.png")),
        "figure4": plot_gap(str(output / "scaling.csv"), str(output / "figure4_gap.png")),
        "figure5": plot_optimizer_convergence(
            str(output / "optimizer_history.csv"),
            np.asarray(all_xy, dtype=float)[
                ~np.all(np.isclose(np.asarray(all_xy, dtype=float), baseline_depot_xy), axis=1)
            ],
            str(output / "figure5_convergence.png"), dataset="real", epsilon=1e-6,
        ),
        "figure6": plot_depot_heatmap(
            str(output / "depot_grid.csv"),
            str(output / "depot_effect.csv"),
            np.asarray(all_xy, dtype=float)[
                ~np.all(np.isclose(np.asarray(all_xy, dtype=float), baseline_depot_xy), axis=1)
            ],
            str(output / "figure6_heatmap.png"),
        ),
    }
    return paths

