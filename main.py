"""Run the complete delivery depot and tour-planning project."""

import argparse
import importlib.util
import os

import numpy as np

from config import EPSILON, GRAD_TOL, MAX_OPT_ITER, PLACES_CSV, RESULTS_DIR, TSP_TIME_LIMIT_SEC

# Keep Matplotlib's writable font cache inside this project workspace.
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib"))

from data import load_places, to_km
from depot import bfgs_method
from tsp_experiments import compute_route_comparison, save_route_comparison


def _require_integrated_modules():
    required = ["depot_experiments", "depot_plots"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "The full pipeline is waiting for these team modules: "
            + ", ".join(missing)
            + ". Add the files to the project root, then rerun main.py."
        )


def _prepare_real_data(places_path=PLACES_CSV):
    places = load_places(places_path)
    reference = np.mean(places["latlon"], axis=0)
    all_xy = to_km(places["latlon"], reference)
    delivery_xy = all_xy[places["delivery_indices"]]
    baseline_depot_xy = all_xy[places["baseline_depot_index"]]
    return places, all_xy, delivery_xy, baseline_depot_xy


def _experiment_subset(delivery_xy, max_points=12):
    """Choose a deterministic, spatially distributed subset for Experiment C."""
    if len(delivery_xy) <= max_points:
        return delivery_xy.copy()
    indices = np.linspace(0, len(delivery_xy) - 1, num=max_points, dtype=int)
    return delivery_xy[indices]


def run(quick=False, places_path=PLACES_CSV):
    """Execute data prep, Stage 1, real route comparison, A/B/C and plots."""
    _require_integrated_modules()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    places, all_xy, delivery_xy, baseline_depot_xy = _prepare_real_data(places_path)

    # Stage 1 locates the depot for the full real delivery set. Experiment C
    # independently reruns Stage 1 on its fixed subset before comparing depots.
    experiment_c_points = _experiment_subset(delivery_xy)
    depot_result = bfgs_method(
        delivery_xy, start=np.mean(delivery_xy, axis=0),
        eps=EPSILON, tol=GRAD_TOL, max_iter=MAX_OPT_ITER,
    )

    tsp_points = np.vstack((baseline_depot_xy, delivery_xy))
    route_limit = min(TSP_TIME_LIMIT_SEC, 10) if quick else TSP_TIME_LIMIT_SEC
    route_result = compute_route_comparison(tsp_points, time_limit=route_limit)
    route_path = save_route_comparison(route_result, str(RESULTS_DIR / "route_comparison.json"))

    from experiments import run_all_experiments
    from plots import generate_all_plots

    experiment_outputs = run_all_experiments(
        real_delivery_points=delivery_xy,
        baseline_depot_xy=baseline_depot_xy,
        experiment_c_delivery_points=experiment_c_points,
        quick=quick,
        output_dir=RESULTS_DIR,
    )
    plot_paths = generate_all_plots(
        all_xy=all_xy,
        names=places["names"],
        baseline_depot_xy=baseline_depot_xy,
        route_comparison_path=route_path,
        results_dir=RESULTS_DIR,
        optimized_depot_xy=depot_result["depot_xy"],
    )

    print(f"Stage 1 depot: {depot_result['depot_xy']} km; converged={depot_result['converged']}")
    exact = route_result["exact"]
    print(f"Real-route exact solver: {exact['solver_status']}; "
          f"length={exact['length_km']:.2f} km, feasible={exact['feasible']}, "
          f"optimal={exact['optimal']}")
    print(f"Results saved under: {RESULTS_DIR}")
    for name, path in plot_paths.items():
        print(f"{name}: {path}")
    return {"experiments": experiment_outputs, "depot": depot_result,
            "route_comparison": route_result, "plots": plot_paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run reduced experiment settings.")
    parser.add_argument("--places", default=str(PLACES_CSV),
                        help="Places CSV path (default: data/places.csv).")
    args = parser.parse_args(argv)
    try:
        run(quick=args.quick, places_path=args.places)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

