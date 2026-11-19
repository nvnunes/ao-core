"""Unit validation for natural guide-star photometry."""

from __future__ import annotations

import numpy as np
from astropy import units as u


def quantity_value(value: u.Quantity, unit: u.UnitBase, *, label: str) -> np.ndarray:
    """Return real, finite values in ``unit`` from an Astropy quantity."""
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{label} must be an Astropy Quantity.")
    try:
        converted = np.asarray(value.to_value(unit))
    except u.UnitConversionError as exc:
        raise ValueError(f"{label} must have units equivalent to {unit}.") from exc
    if converted.dtype.kind not in "iuf":
        raise ValueError(f"{label} must be real.")
    values = np.asarray(converted, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must be finite.")
    return values


def positive_scalar_quantity(value: u.Quantity, unit: u.UnitBase, *, label: str) -> float:
    """Return one finite, strictly positive quantity value in ``unit``."""
    values = quantity_value(value, unit, label=label)
    if values.ndim != 0:
        raise ValueError(f"{label} must be scalar.")
    result = float(values.item())
    if result <= 0.0:
        raise ValueError(f"{label} must be > 0.")
    return result


def positive_scalar(value: float, *, label: str) -> float:
    """Return one finite, strictly positive dimensionless scalar."""
    if isinstance(value, u.Quantity):
        raise TypeError(f"{label} must be a dimensionless real scalar, not a Quantity.")
    try:
        values = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a real scalar.") from exc
    if values.ndim != 0:
        raise ValueError(f"{label} must be scalar.")
    if values.dtype.kind not in "iuf":
        raise TypeError(f"{label} must be a real scalar.")
    result = float(values.item())
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and > 0.")
    return result


def nonnegative_quantity_value(value: u.Quantity, unit: u.UnitBase, *, label: str) -> np.ndarray:
    """Return finite, nonnegative quantity values in ``unit``."""
    values = quantity_value(value, unit, label=label)
    if np.any(values < 0.0):
        raise ValueError(f"{label} must be >= 0.")
    return values
