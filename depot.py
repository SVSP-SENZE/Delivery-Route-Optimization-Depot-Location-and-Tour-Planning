"""Smooth geometric-median objective derivatives for depot placement."""

import time

import numpy as np

__all__ = [
    "objective",
    "gradient",
    "hessian",
    "backtracking_line_search",
    "gradient_descent",
    "newton_method",
    "bfgs_method",
]


def _finite_real_array(value, name, shape):
    """Validate and return a finite real-valued array of the requested shape."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array with shape {shape}.") from exc

    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {raw.shape}.")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must contain finite real numeric values.")

    array = raw.astype(float, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_eps(eps):
    """Validate and return a finite positive scalar smoothing value."""
    try:
        raw = np.asarray(eps)
    except (TypeError, ValueError) as exc:
        raise ValueError("eps must be a finite positive scalar.") from exc

    if raw.ndim != 0:
        raise ValueError("eps must be a finite positive scalar.")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        raise ValueError("eps must be a finite positive scalar.")

    value = float(raw)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("eps must be a finite positive scalar.")
    return value


def _validated_inputs(c, points, eps):
    """Validate common function inputs."""
    center = _finite_real_array(c, "c", (2,))

    try:
        raw_points = np.asarray(points)
    except (TypeError, ValueError) as exc:
        raise ValueError("points must be a finite numeric array with shape (m, 2).") from exc
    if raw_points.ndim != 2 or raw_points.shape[1:] != (2,) or raw_points.shape[0] < 1:
        raise ValueError("points must have shape (m, 2) with at least one delivery point.")
    delivery_points = _finite_real_array(
        raw_points, "points", raw_points.shape
    )
    smoothing = _validate_eps(eps)
    return center, delivery_points, smoothing


def _finite_real_scalar(value, name):
    """Validate and return a finite real scalar."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar.") from exc

    if raw.ndim != 0 or not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must be a finite real scalar.")

    scalar = float(raw)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar.")
    return scalar


def _validate_max_backtracks(max_backtracks):
    """Validate and return a non-negative integer backtracking limit."""
    try:
        raw = np.asarray(max_backtracks)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_backtracks must be a non-negative integer.") from exc
    if raw.ndim != 0 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("max_backtracks must be a non-negative integer.")

    value = int(raw)
    if value < 0:
        raise ValueError("max_backtracks must be a non-negative integer.")
    return value


def _validate_max_iter(max_iter):
    """Validate and return a non-negative iteration limit."""
    try:
        raw = np.asarray(max_iter)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_iter must be a non-negative integer.") from exc
    if raw.ndim != 0 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("max_iter must be a non-negative integer.")

    value = int(raw)
    if value < 0:
        raise ValueError("max_iter must be a non-negative integer.")
    return value


def objective(c, points, eps=1e-6):
    """Return the smoothed sum of distances from depot ``c`` to delivery ``points``.

    Args:
        c: Finite depot coordinates with shape ``(2,)``.
        points: Finite delivery coordinates with shape ``(m, 2)`` for ``m >= 1``.
        eps: Positive finite smoothing scalar.

    Returns:
        The objective value as a Python ``float``.
    """
    center, delivery_points, smoothing = _validated_inputs(c, points, eps)
    residuals = center - delivery_points
    distances = np.sqrt(np.sum(residuals * residuals, axis=1) + smoothing)
    return float(np.sum(distances))


def gradient(c, points, eps=1e-6):
    """Return the gradient of the smoothed depot-distance objective.

    Args:
        c: Finite depot coordinates with shape ``(2,)``.
        points: Finite delivery coordinates with shape ``(m, 2)`` for ``m >= 1``.
        eps: Positive finite smoothing scalar.

    Returns:
        A NumPy array with shape ``(2,)``.
    """
    center, delivery_points, smoothing = _validated_inputs(c, points, eps)
    residuals = center - delivery_points
    distances = np.sqrt(np.sum(residuals * residuals, axis=1) + smoothing)
    return np.sum(residuals / distances[:, np.newaxis], axis=0)


def hessian(c, points, eps=1e-6):
    """Return the Hessian of the smoothed depot-distance objective.

    Args:
        c: Finite depot coordinates with shape ``(2,)``.
        points: Finite delivery coordinates with shape ``(m, 2)`` for ``m >= 1``.
        eps: Positive finite smoothing scalar.

    Returns:
        A NumPy array with shape ``(2, 2)``.
    """
    center, delivery_points, smoothing = _validated_inputs(c, points, eps)
    residuals = center - delivery_points
    distances = np.sqrt(np.sum(residuals * residuals, axis=1) + smoothing)
    inverse_distances = 1.0 / distances
    outer_sum = residuals.T @ (residuals * inverse_distances[:, np.newaxis] ** 3)
    return np.eye(2) * np.sum(inverse_distances) - outer_sum


def backtracking_line_search(
    c,
    direction,
    points,
    eps=1e-6,
    alpha0=1.0,
    beta=0.5,
    c1=1e-4,
    max_backtracks=100,
):
    """Return an Armijo-acceptable step along a finite descent ``direction``.

    Args:
        c: Finite depot coordinates with shape ``(2,)``.
        direction: Finite search direction with shape ``(2,)``.
        points: Finite delivery coordinates with shape ``(m, 2)`` for ``m >= 1``.
        eps: Positive finite smoothing scalar.
        alpha0: Positive initial step size.
        beta: Step reduction factor strictly between zero and one.
        c1: Armijo parameter strictly between zero and one.
        max_backtracks: Maximum number of step-size reductions.

    Returns:
        An Armijo-acceptable step size as a Python ``float``.

    Raises:
        ValueError: If inputs are invalid or ``direction`` is not a descent direction.
        RuntimeError: If no acceptable step is found within the backtracking limit.
    """
    center, delivery_points, smoothing = _validated_inputs(c, points, eps)
    search_direction = _finite_real_array(direction, "direction", (2,))
    initial_step = _finite_real_scalar(alpha0, "alpha0")
    reduction_factor = _finite_real_scalar(beta, "beta")
    armijo_constant = _finite_real_scalar(c1, "c1")
    reductions = _validate_max_backtracks(max_backtracks)

    if initial_step <= 0.0:
        raise ValueError("alpha0 must be greater than zero.")
    if not 0.0 < reduction_factor < 1.0:
        raise ValueError("beta must satisfy 0 < beta < 1.")
    if not 0.0 < armijo_constant < 1.0:
        raise ValueError("c1 must satisfy 0 < c1 < 1.")

    current_gradient = gradient(center, delivery_points, smoothing)
    gradient_dot_direction = float(np.dot(current_gradient, search_direction))
    if not np.isfinite(gradient_dot_direction) or not gradient_dot_direction < 0.0:
        raise ValueError("direction is not a descent direction at c.")

    current_objective = objective(center, delivery_points, smoothing)
    alpha = initial_step
    for backtracks in range(reductions + 1):
        if alpha <= 0.0:
            break
        trial_objective = objective(
            center + alpha * search_direction, delivery_points, smoothing
        )
        armijo_bound = current_objective + armijo_constant * alpha * gradient_dot_direction
        if np.isfinite(trial_objective) and trial_objective <= armijo_bound:
            return float(alpha)
        if backtracks < reductions:
            alpha *= reduction_factor

    raise RuntimeError("No Armijo-acceptable step was found within max_backtracks.")


def gradient_descent(points, start=None, eps=1e-6, tol=1e-8, max_iter=1000):
    """Minimize the smoothed depot objective using gradient descent and Armijo steps.

    Args:
        points: Finite delivery coordinates with shape ``(m, 2)`` for ``m >= 1``.
        start: Optional finite initial depot coordinates with shape ``(2,)``.
        eps: Positive finite smoothing scalar.
        tol: Positive finite gradient-norm convergence tolerance.
        max_iter: Maximum number of gradient-descent updates.

    Returns:
        A dictionary containing the final depot, diagnostics, and iteration history.
    """
    started_at = time.perf_counter()
    tolerance = _finite_real_scalar(tol, "tol")
    if tolerance <= 0.0:
        raise ValueError("tol must be a finite positive scalar.")
    iteration_limit = _validate_max_iter(max_iter)

    if start is None:
        _, delivery_points, smoothing = _validated_inputs(np.zeros(2), points, eps)
        center = _finite_real_array(
            np.mean(delivery_points, axis=0), "initial depot", (2,)
        )
    else:
        center, delivery_points, smoothing = _validated_inputs(start, points, eps)

    current_objective = objective(center, delivery_points, smoothing)
    current_gradient = gradient(center, delivery_points, smoothing)
    current_grad_norm = float(np.linalg.norm(current_gradient))
    history = [
        {
            "iteration": 0,
            "x": float(center[0]),
            "y": float(center[1]),
            "objective": float(current_objective),
            "grad_norm": float(current_grad_norm),
            "step_size": 0.0,
        }
    ]

    converged = current_grad_norm < tolerance
    iteration = 0
    if not converged:
        for iteration in range(1, iteration_limit + 1):
            direction = -current_gradient
            step_size = backtracking_line_search(
                center, direction, delivery_points, smoothing
            )
            center = center + step_size * direction
            current_objective = objective(center, delivery_points, smoothing)
            current_gradient = gradient(center, delivery_points, smoothing)
            current_grad_norm = float(np.linalg.norm(current_gradient))
            history.append(
                {
                    "iteration": iteration,
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "objective": float(current_objective),
                    "grad_norm": float(current_grad_norm),
                    "step_size": float(step_size),
                }
            )
            if current_grad_norm < tolerance:
                converged = True
                break

    if converged:
        message = "Converged: gradient norm is below tol."
    else:
        message = "Maximum iterations reached without convergence."

    return {
        "depot_xy": center.copy(),
        "objective": float(current_objective),
        "iterations": int(iteration),
        "grad_norm": float(current_grad_norm),
        "runtime_sec": float(time.perf_counter() - started_at),
        "converged": bool(converged),
        "message": message,
        "history": history,
    }


def newton_method(points, start=None, eps=1e-6, tol=1e-8, max_iter=1000):
    """Minimize the smoothed depot objective with Newton steps and Armijo search.

    Args:
        points: Finite delivery coordinates with shape ``(m, 2)`` for ``m >= 1``.
        start: Optional finite initial depot coordinates with shape ``(2,)``.
        eps: Positive finite smoothing scalar.
        tol: Positive finite gradient-norm convergence tolerance.
        max_iter: Maximum number of Newton updates.

    Returns:
        A dictionary containing the final depot, diagnostics, and iteration history.
    """
    started_at = time.perf_counter()
    tolerance = _finite_real_scalar(tol, "tol")
    if tolerance <= 0.0:
        raise ValueError("tol must be a finite positive scalar.")
    iteration_limit = _validate_max_iter(max_iter)

    if start is None:
        _, delivery_points, smoothing = _validated_inputs(np.zeros(2), points, eps)
        center = _finite_real_array(
            np.mean(delivery_points, axis=0), "initial depot", (2,)
        )
    else:
        center, delivery_points, smoothing = _validated_inputs(start, points, eps)

    current_objective = objective(center, delivery_points, smoothing)
    current_gradient = gradient(center, delivery_points, smoothing)
    current_grad_norm = float(np.linalg.norm(current_gradient))
    history = [
        {
            "iteration": 0,
            "x": float(center[0]),
            "y": float(center[1]),
            "objective": float(current_objective),
            "grad_norm": float(current_grad_norm),
            "step_size": 0.0,
        }
    ]

    converged = current_grad_norm < tolerance
    iteration = 0
    if not converged:
        for iteration in range(1, iteration_limit + 1):
            current_gradient = gradient(center, delivery_points, smoothing)
            current_hessian = hessian(center, delivery_points, smoothing)
            try:
                direction = np.linalg.solve(current_hessian, -current_gradient)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    "Newton step failed: the Hessian system could not be solved."
                ) from exc

            step_size = backtracking_line_search(
                center, direction, delivery_points, smoothing
            )
            center = center + step_size * direction
            current_objective = objective(center, delivery_points, smoothing)
            current_gradient = gradient(center, delivery_points, smoothing)
            current_grad_norm = float(np.linalg.norm(current_gradient))
            history.append(
                {
                    "iteration": iteration,
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "objective": float(current_objective),
                    "grad_norm": float(current_grad_norm),
                    "step_size": float(step_size),
                }
            )
            if current_grad_norm < tolerance:
                converged = True
                break

    if converged:
        message = "Converged: gradient norm is below tol."
    else:
        message = "Maximum iterations reached without convergence."

    return {
        "depot_xy": center.copy(),
        "objective": float(current_objective),
        "iterations": int(iteration),
        "grad_norm": float(current_grad_norm),
        "runtime_sec": float(time.perf_counter() - started_at),
        "converged": bool(converged),
        "message": message,
        "history": history,
    }


def bfgs_method(points, start=None, eps=1e-6, tol=1e-8, max_iter=1000):
    """Minimize the smoothed depot objective using inverse-BFGS and Armijo search.

    Args:
        points: Finite delivery coordinates with shape ``(m, 2)`` for ``m >= 1``.
        start: Optional finite initial depot coordinates with shape ``(2,)``.
        eps: Positive finite smoothing scalar.
        tol: Positive finite gradient-norm convergence tolerance.
        max_iter: Maximum number of BFGS updates.

    Returns:
        A dictionary containing the final depot, diagnostics, and iteration history.
    """
    started_at = time.perf_counter()
    tolerance = _finite_real_scalar(tol, "tol")
    if tolerance <= 0.0:
        raise ValueError("tol must be a finite positive scalar.")
    iteration_limit = _validate_max_iter(max_iter)

    if start is None:
        _, delivery_points, smoothing = _validated_inputs(np.zeros(2), points, eps)
        center = _finite_real_array(
            np.mean(delivery_points, axis=0), "initial depot", (2,)
        )
    else:
        center, delivery_points, smoothing = _validated_inputs(start, points, eps)

    current_objective = objective(center, delivery_points, smoothing)
    current_gradient = gradient(center, delivery_points, smoothing)
    current_grad_norm = float(np.linalg.norm(current_gradient))
    history = [
        {
            "iteration": 0,
            "x": float(center[0]),
            "y": float(center[1]),
            "objective": float(current_objective),
            "grad_norm": float(current_grad_norm),
            "step_size": 0.0,
        }
    ]

    Hinv = np.eye(2)
    converged = current_grad_norm < tolerance
    iteration = 0
    if not converged:
        for iteration in range(1, iteration_limit + 1):
            current_gradient = gradient(center, delivery_points, smoothing)
            direction = -(Hinv @ current_gradient)
            if not float(np.dot(current_gradient, direction)) < 0.0:
                Hinv = np.eye(2)
                direction = -current_gradient

            step_size = backtracking_line_search(
                center, direction, delivery_points, smoothing
            )
            previous_center = center
            previous_gradient = current_gradient
            center = center + step_size * direction
            current_objective = objective(center, delivery_points, smoothing)
            current_gradient = gradient(center, delivery_points, smoothing)
            current_grad_norm = float(np.linalg.norm(current_gradient))

            s = center - previous_center
            y = current_gradient - previous_gradient
            y_dot_s = float(np.dot(y, s))
            if y_dot_s > 1e-10:
                rho = 1.0 / y_dot_s
                V = np.eye(2) - rho * np.outer(s, y)
                Hinv = V @ Hinv @ V.T + rho * np.outer(s, s)

            history.append(
                {
                    "iteration": iteration,
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "objective": float(current_objective),
                    "grad_norm": float(current_grad_norm),
                    "step_size": float(step_size),
                }
            )
            if current_grad_norm < tolerance:
                converged = True
                break

    if converged:
        message = "Converged: gradient norm is below tol."
    else:
        message = "Maximum iterations reached without convergence."

    return {
        "depot_xy": center.copy(),
        "objective": float(current_objective),
        "iterations": int(iteration),
        "grad_norm": float(current_grad_norm),
        "runtime_sec": float(time.perf_counter() - started_at),
        "converged": bool(converged),
        "message": message,
        "history": history,
    }
