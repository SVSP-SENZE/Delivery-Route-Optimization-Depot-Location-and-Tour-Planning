# Delivery Route Optimization: Depot Location and Tour Planning

Team: Bharatiya Optimization Party

Members: Shubhan Pokuri, Jiya Merja, Prerana Kurra

This project compares a continuous depot-location model with discrete delivery-tour planning. The depot optimizer minimizes the sum of smoothed Euclidean distances to the delivery points. The TSP stage compares an exact Miller-Tucker-Zemlin integer program with Nearest Neighbour and 2-opt. The experiments measure scaling and route quality, optimizer convergence, and the effect of depot choice.

## Shared conventions

- Geographic source coordinates are `[latitude, longitude]` in degrees. All real locations are projected together into local east/north kilometres using one reference point: the mean latitude/longitude of the selected locations.
- The baseline depot is one row in the places input and is never included in the Stage 1 delivery-point array.
- A TSP point array has the depot first (node `0`), followed by deliveries. A route lists each node once; the final edge back to node `0` is implicit and included in route length.
- Distances and optimizer coordinates use kilometres. The objective smoothing `EPSILON` is in square kilometres.
- An exact TSP result reports feasibility separately from a proof of optimality. A feasible time-limited route is not described as optimal.

## Input data

Create `data/places.csv` with one row for each selected Bengaluru location and these columns:

```csv
name,latitude,longitude,is_baseline_depot
Example depot,12.9716,77.5946,true
Example delivery,12.9352,77.6245,false
```

Use the team’s verified locations and coordinates; the example above only illustrates the file format. Exactly one row must be marked as the baseline depot. All other rows are treated as delivery destinations. The loader checks names, coordinate ranges, finite values and the unique depot marker.

## Algorithms and modules

- `config.py`: shared numerical settings and paths.
- `data.py`: CSV loading, local geographic projection, Haversine spot checks, reproducible random instances and pairwise Euclidean distances.
- `depot.py`: smoothed geometric-median objective and Gradient Descent, Newton and BFGS optimizers.
- `tsp_exact.py`: PuLP/CBC MTZ model and a small-instance brute-force reference solver.
- `heuristics.py`: tour length, Nearest Neighbour and 2-opt.
- `depot_experiments.py`, `tsp_experiments.py`: experiment implementations owned by the depot and TSP members.
- `experiments.py`, `plots.py`, `main.py`: top-level project orchestration; algorithms remain in their owning modules.
- `tsp_plots.py`, `depot_plots.py`: plotting from saved experiment outputs.

## Setup and run

From the project root, install dependencies and ensure PuLP can access CBC:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

After adding `data/places.csv` and integrating all team-owned modules:

```bash
python main.py --quick
python main.py
```

The default run uses the agreed full random TSP sizes, five seeds, and a 25 by 25 grid. `--quick` reduces the random TSP study and grid resolution for a faster integration pass.

For a clearly synthetic end-to-end demonstration, run `python main.py --quick --places data/demo_places.csv`. Replace that demo input with the team's verified location data before using results in the project report.

## Generated outputs

The pipeline writes results under `results/`, creating the directory when needed:

- `scaling.csv`: exact, Nearest Neighbour and 2-opt runtimes, route lengths, solver status and proven-optimum gaps.
- `route_comparison.json`: saved projected coordinates and routes for the real-instance comparison.
- `optimizers.csv`, `optimizer_history.csv`: optimizer summary and convergence history.
- `depot_effect.csv`, `depot_grid.csv`: baseline/Stage 1/grid-best comparison and every grid candidate.
- `figure1_locations.png` through `figure6_heatmap.png`: location, route, scaling, gap, convergence and depot-grid figures.

Plots should be regenerated from saved results; plotting must not rerun optimization. Experiment C uses a fixed delivery subset and evaluates baseline, Stage 1 and grid candidates with the same Nearest Neighbour plus 2-opt pipeline.

## Limitations

The local projection is appropriate for a compact Bengaluru-area study; it approximates great-circle distances, which can be checked with `haversine_km`. The depot objective uses straight-line distance and does not model road networks, traffic, capacities or time windows. MTZ is exact but may become slow as node count grows. Nearest Neighbour and 2-opt are heuristics and do not guarantee a globally shortest tour. A grid-best depot is best among the evaluated grid points only.
