"""Thin orchestration layer for the three project experiments."""

from config import GRID_SIZE, RANDOM_SEEDS, RESULTS_DIR, TSP_N_VALUES


def _experiment_functions():
    try:
        from tsp_experiments import run_tsp_scaling_experiment
    except ImportError as exc:
        raise RuntimeError("Experiment A needs tsp_experiments.py and its dependencies.") from exc
    try:
        from depot_experiments import run_optimizer_experiments, run_depot_effect_experiment
    except ImportError as exc:
        raise RuntimeError(
            "Experiments B/C need Jiya's depot_experiments.py in the project root."
        ) from exc
    return run_tsp_scaling_experiment, run_optimizer_experiments, run_depot_effect_experiment


def run_all_experiments(
    real_delivery_points,
    baseline_depot_xy,
    experiment_c_delivery_points,
    n_values=None,
    seeds=None,
    quick=False,
    output_dir=RESULTS_DIR,
):
    """Run A/B/C through their owning modules and return each module's results."""
    import numpy as np

    run_a, run_b, run_c = _experiment_functions()
    sizes = list(TSP_N_VALUES if n_values is None else n_values)
    seed_values = list(RANDOM_SEEDS if seeds is None else seeds)
    if quick:
        sizes = sizes[:3]
        seed_values = seed_values[:1]
        random_n = max(5, min(8, len(real_delivery_points)))
        eps_values = (1e-6,)
        grid_size = 5
    else:
        random_n, eps_values, grid_size = 15, (1e-6, 1e-5, 1e-4), GRID_SIZE

    output_dir = str(output_dir)
    scaling_path = f"{output_dir}/scaling.csv"
    experiment_a = run_a(n_values=sizes, seeds=seed_values, output_path=scaling_path)
    experiment_b = run_b(
        np.asarray(real_delivery_points, dtype=float),
        random_seeds=seed_values,
        random_n=random_n,
        eps_values=eps_values,
        output_dir=output_dir,
    )
    experiment_c = run_c(
        np.asarray(experiment_c_delivery_points, dtype=float),
        np.asarray(baseline_depot_xy, dtype=float),
        grid_size=grid_size,
        output_dir=output_dir,
    )
    return {"A": experiment_a, "B": experiment_b, "C": experiment_c}

