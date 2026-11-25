"""Shared project settings. Keep algorithm defaults centralized here."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
PLACES_CSV = DATA_DIR / "places.csv"

EARTH_RADIUS_KM = 6371.0
EPSILON = 1e-6
GRAD_TOL = 1e-8
MAX_OPT_ITER = 1000
ARMIJO_C1 = 1e-4
BACKTRACKING_BETA = 0.5
MAX_BACKTRACKS = 100
TSP_TIME_LIMIT_SEC = 60
TWO_OPT_TOL = 1e-9
GRID_SIZE = 25
BOX_SIZE_KM = 100.0
TSP_N_VALUES = [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50]
RANDOM_SEEDS = [0, 1, 2, 3, 4]

