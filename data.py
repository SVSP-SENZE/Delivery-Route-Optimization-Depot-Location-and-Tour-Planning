"""Project data loading, geographic projection, and distance utilities.

The real-location source is a CSV at ``data/places.csv`` (or an explicit path).
It must have columns ``name,latitude,longitude,is_baseline_depot`` and exactly
one row marked as the baseline depot. Keeping raw coordinates in a data file
lets the team update/verify its selected Bengaluru locations without editing
the shared projection and optimization code.
"""

import csv
from pathlib import Path

import numpy as np

from config import BOX_SIZE_KM, EARTH_RADIUS_KM, PLACES_CSV

__all__ = ["load_places", "to_km", "haversine_km", "random_instance", "dist_matrix"]


def _coordinates(value, name, allowed_shapes):
    try:
        coords = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numeric latitude/longitude values.") from exc
    if coords.shape not in allowed_shapes:
        expected = " or ".join(map(str, allowed_shapes))
        raise ValueError(f"{name} must have shape {expected}; got {coords.shape}.")
    if not np.all(np.isfinite(coords)):
        raise ValueError(f"{name} must contain only finite values.")
    if coords.ndim == 1:
        latitudes, longitudes = coords[0], coords[1]
    else:
        latitudes, longitudes = coords[:, 0], coords[:, 1]
    if np.any(np.abs(latitudes) > 90) or np.any(np.abs(longitudes) > 180):
        raise ValueError(f"{name} must use latitude/longitude degrees in that order.")
    return coords


def load_places(path=PLACES_CSV):
    """Load selected real locations from CSV and identify the baseline depot.

    Returns a dictionary with ``names``, ``latlon`` (latitude, longitude in
    degrees), ``baseline_depot_index``, and ``delivery_indices``. The baseline
    depot is exactly one of the listed locations and is excluded from delivery
    indices.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Places CSV not found: {source}. Create it with columns "
            "name,latitude,longitude,is_baseline_depot and mark exactly one depot."
        )
    with source.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"name", "latitude", "longitude", "is_baseline_depot"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Places CSV must include columns: name, latitude, longitude, "
                "is_baseline_depot."
            )
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("Places CSV must contain a depot and at least one delivery location.")
    names = [row["name"].strip() for row in rows]
    if any(not name for name in names):
        raise ValueError("Every place must have a non-empty name.")
    if len(set(names)) != len(names):
        raise ValueError("Place names must be unique.")
    try:
        latlon = np.array(
            [[float(row["latitude"]), float(row["longitude"])] for row in rows],
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Every place must have numeric latitude and longitude values.") from exc
    _coordinates(latlon, "latlon", ((len(rows), 2),))
    flags = []
    for row in rows:
        value = row["is_baseline_depot"].strip().lower()
        if value not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError("is_baseline_depot must be true/false, yes/no, or 1/0.")
        flags.append(value in {"true", "1", "yes"})
    depot_indices = np.flatnonzero(flags)
    if len(depot_indices) != 1:
        raise ValueError(f"Exactly one baseline depot is required; found {len(depot_indices)}.")
    depot_index = int(depot_indices[0])
    delivery_indices = [i for i in range(len(rows)) if i != depot_index]
    return {
        "names": names,
        "latlon": latlon,
        "baseline_depot_index": depot_index,
        "delivery_indices": delivery_indices,
    }


def to_km(latlon, reference_latlon):
    """Project latitude/longitude degrees to local east/north kilometres."""
    raw = np.asarray(latlon)
    allowed = ((2,), (raw.shape[0], 2)) if raw.ndim == 2 else ((2,),)
    points = _coordinates(latlon, "latlon", allowed)
    reference = _coordinates(reference_latlon, "reference_latlon", ((2,),))
    point_radians = np.radians(points)
    reference_radians = np.radians(reference)
    lat0, lon0 = reference_radians
    x = EARTH_RADIUS_KM * np.cos(lat0) * (point_radians[..., 1] - lon0)
    y = EARTH_RADIUS_KM * (point_radians[..., 0] - lat0)
    projected = np.stack((x, y), axis=-1)
    return projected


def haversine_km(point1, point2):
    """Great-circle distance between two [latitude, longitude] pairs in degrees."""
    first = _coordinates(point1, "point1", ((2,),))
    second = _coordinates(point2, "point2", ((2,),))
    lat1, lon1, lat2, lon2 = np.radians([first[0], first[1], second[0], second[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def random_instance(n, seed, box_size_km=BOX_SIZE_KM):
    """Generate ``n`` reproducible 2D points in a square kilometre box."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")
    try:
        size = float(box_size_km)
    except (TypeError, ValueError) as exc:
        raise ValueError("box_size_km must be finite and positive.") from exc
    if not np.isfinite(size) or size <= 0:
        raise ValueError("box_size_km must be finite and positive.")
    return np.random.default_rng(int(seed)).uniform(0.0, size, size=(int(n), 2))


def dist_matrix(xy):
    """Return the Euclidean pairwise distance matrix for finite ``(n, 2)`` points."""
    try:
        points = np.asarray(xy, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("xy must be a finite numeric array with shape (n, 2).") from exc
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
        raise ValueError(f"xy must have shape (n, 2), n >= 1; got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError("xy must contain only finite values.")
    differences = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.linalg.norm(differences, axis=2)

