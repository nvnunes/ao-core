"""Finite real precision and owned physical arrays."""

from __future__ import annotations

import numpy as np
from astropy import units as u
from numpy.typing import DTypeLike


def validate_float_dtype(dtype: DTypeLike) -> np.dtype:
    """Return one supported calculation dtype without silently changing precision."""
    if dtype is None:
        raise ValueError("dtype must be float32 or float64.")
    try:
        dtype = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise ValueError("dtype must be float32 or float64.") from error
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64.")
    return dtype


def _quantity(value, unit, name, ndim, *, dtype=np.float64, order="C"):
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be an Astropy Quantity")
    dtype = validate_float_dtype(dtype)
    if value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be finite real {ndim}D values")
    array = np.asarray(value.astype(dtype).to_value(unit))
    if (
        array.dtype.kind not in "fiu"
        or array.ndim != ndim
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be finite real {ndim}D values")
    if any(n == 0 for n in array.shape):
        raise ValueError(f"{name} must not be empty")
    dtype = validate_float_dtype(dtype)
    array = np.asarray(array, dtype=dtype, order=order)
    order = "F" if np.isfortran(array) else "C"
    immutable = np.frombuffer(array.tobytes(order=order), dtype=dtype).reshape(
        array.shape, order=order
    )
    return u.Quantity(immutable, unit, copy=False)


def _count(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


class _PhysicalRecord:
    """Give callers quantity views, protecting owned unit metadata and values.

    Stored arrays have immutable byte backing. Each public quantity access
    returns a lightweight view: changing that view's unit cannot alter the
    record, and values cannot become writable through a NumPy flag change.
    """

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        return (
            u.Quantity(np.asarray(value), value.unit, dtype=value.dtype, copy=False)
            if isinstance(value, u.Quantity)
            else value
        )


def _input_dtype(value, dtype):
    """Infer internal precision from prepared arrays; integers use float32."""
    if dtype is None:
        source = np.asarray(value).dtype
        dtype = source if source.kind == "f" else np.float32
    return validate_float_dtype(dtype)
