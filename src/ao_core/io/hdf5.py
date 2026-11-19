"""HDF5 dataset access for callers that apply their own file contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import h5py
import numpy as np
from astropy import units as u

from . import schema
from .schema import unit_string

UNITS_ATTRIBUTE = "units"


def require_quantity(
    value: Any,
    unit: u.UnitBase,
    *,
    label: str,
) -> u.Quantity:
    """Require a quantity and convert it to one canonical equivalent unit."""
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{label} must be an Astropy Quantity.")
    try:
        return value.to(unit, copy=False)
    except u.UnitConversionError as exc:
        raise ValueError(
            f"{label} must have units equivalent to {unit_string(unit)!r}."
        ) from exc


def quantity_value(
    value: Any,
    unit: u.UnitBase,
    *,
    label: str,
    dtype: Any | None = None,
) -> np.ndarray:
    """Return the unit-normalized numerical view of one required quantity."""
    if dtype is not None and isinstance(value, u.Quantity):
        value = value.astype(dtype)
    quantity = require_quantity(value, unit, label=label)
    return np.asarray(quantity.value, dtype=dtype)


def quantity_from_value(value: Any, unit: u.UnitBase) -> u.Quantity:
    """Wrap numerical values in one unit without copying when NumPy permits."""
    return u.Quantity(np.asarray(value), unit=unit, copy=False)


def write_value(group: h5py.Group, key: str, value: Any) -> None:
    """Write a Python value recursively into an HDF5 group key."""
    if isinstance(value, Mapping):
        sub = group[key] if key in group else group.create_group(key, track_order=True)
        if not isinstance(sub, h5py.Group):
            raise TypeError(f"Cannot write mapping over non-group HDF5 key {key!r}.")
        for sub_key, sub_value in value.items():
            write_value(sub, str(sub_key), sub_value)
        return

    if isinstance(value, str):
        dtype = h5py.string_dtype(encoding="utf-8")
        if key in group:
            del group[key]
        group.create_dataset(key, data=value, dtype=dtype)
        return

    unit: u.UnitBase | None = None
    if isinstance(value, u.Quantity):
        unit = value.unit
        arr = np.asarray(value.value)
    else:
        arr = np.asarray(value)
    if arr.dtype.kind in {"U", "O"}:
        dtype = h5py.string_dtype(encoding="utf-8")
        if key in group:
            del group[key]
        dataset = group.create_dataset(
            key, data=np.asarray(arr, dtype=object), dtype=dtype
        )
    else:
        if key in group:
            del group[key]
        dataset = group.create_dataset(key, data=arr)
    if unit is not None:
        dataset.attrs[UNITS_ATTRIBUTE] = unit_string(unit)


def _read_value(ds: h5py.Dataset, selection: int | tuple[()]) -> Any:
    """Decode UTF-8 text and reconstruct quantities for a dataset selection."""
    if h5py.check_string_dtype(ds.dtype) is not None:
        value = ds.asstr(encoding="utf-8")[selection]
        if isinstance(value, np.ndarray):
            value = value.astype(str)
    else:
        value = ds[selection]
    if UNITS_ATTRIBUTE not in ds.attrs:
        return value
    return quantity_from_value(value, u.Unit(ds.attrs[UNITS_ATTRIBUTE]))


def set_dataset_unit(ds: h5py.Dataset, unit: u.UnitBase) -> h5py.Dataset:
    """Attach one canonical unit to a newly allocated HDF5 dataset."""
    ds.attrs[UNITS_ATTRIBUTE] = unit_string(unit)
    return ds


def read_dataset_value(ds: h5py.Dataset) -> Any:
    """Read one dataset, reconstructing a quantity when units are declared."""
    return _read_value(ds, ())


def read_node(node: h5py.Group | h5py.Dataset) -> Any:
    """Read an HDF5 node into Python values, reconstructing quantities."""
    if isinstance(node, h5py.Group):
        return {k: read_node(node[k]) for k in node}

    return read_dataset_value(node)


def collect_dataset_unit_issue(
    f: h5py.File,
    path: str,
    unit: u.UnitBase,
    issues: list[str],
) -> None:
    """Require one existing quantity dataset to declare its canonical unit."""
    if path not in f or not isinstance(f[path], h5py.Dataset):
        return
    expected = unit_string(unit)
    if f[path].attrs.get(UNITS_ATTRIBUTE) != expected:
        issues.append(f"/{path.strip('/')} must declare units={expected!r}.")


def index_value(sim_idx: int) -> int:
    """Validate and normalize a zero-based simulation index."""
    idx = int(sim_idx)
    if idx < 0:
        raise IndexError(f"simulation index must be >= 0, got {idx}.")
    return idx


def require_dataset(f: h5py.File | h5py.Group, path: str) -> h5py.Dataset:
    """Return a dataset by path or raise a clear contract error."""
    if path not in f:
        raise ValueError(f"Missing required dataset '{path}'.")
    ds = f[path]
    if not isinstance(ds, h5py.Dataset):
        raise ValueError(f"{path} must be a dataset.")  # noqa: TRY004
    return ds


def require_dataset_ndim(ds: h5py.Dataset, *, path: str, ndim: int) -> h5py.Dataset:
    """Validate dataset dimensionality for a specific persisted contract."""
    if ds.ndim != ndim:
        raise ValueError(f"{path} must be {ndim}D.")
    return ds


def require_diagnostic_dataset(root: h5py.Group, path: str) -> h5py.Dataset:
    """Return one diagnostic dataset by slash-delimited path."""
    if path not in root:
        raise ValueError(f"Missing declared diagnostics dataset '/diagnostics/{path}'.")
    node = root[path]
    if not isinstance(node, h5py.Dataset):
        raise ValueError(f"/diagnostics/{path} must be a dataset.")  # noqa: TRY004
    return node


def read_row(ds: h5py.Dataset, sim_idx: int, *, path: str) -> Any:
    """Read one per-simulation dataset row with shared index and shape checks."""
    sim_idx = index_value(sim_idx)
    if ds.ndim == 0:
        raise ValueError(f"{path} must be per-simulation with first dim N.")
    if sim_idx >= ds.shape[0]:
        raise IndexError(f"sim_idx {sim_idx} out of range for {path} shape {ds.shape}")
    return _read_value(ds, sim_idx)


def read_diagnostics(
    node: h5py.Group | h5py.Dataset, sim_idx: int | None = None, path: str = ""
) -> Any:
    """Read full-analysis or one-row diagnostics recursively.

    HDF5 groups are returned as nested dictionaries. Dataset values follow the
    same conventions as other store readers: scalar UTF-8 strings become
    ``str``, string arrays become NumPy string arrays, and numerical datasets
    with unit metadata become quantities. When ``sim_idx`` is provided, only
    that row or slice is read from each diagnostic dataset.

    Args:
        node: ``/diagnostics`` group or one dataset within it.
        sim_idx: Optional simulation row. ``None`` reads full analysis arrays.
        path: Diagnostic path used for error messages.

    Returns:
        Nested diagnostics mapping or decoded dataset value.

    Raises:
        IndexError: If ``sim_idx`` is out of range.
        ValueError: If a diagnostic dataset is malformed.
    """
    if isinstance(node, h5py.Group):
        return {
            key: read_diagnostics(node[key], sim_idx=sim_idx, path=f"{path}/{key}")
            for key in node
        }
    if sim_idx is None:
        return read_dataset_value(node)
    return read_row(node, sim_idx, path=path)


def output_dtype(file: h5py.File) -> np.dtype:
    """Return declared result precision or infer it from an output buffer."""
    precision = file.get("simulation/precision")
    strehl = file.get("stats/sr")
    if strehl is not None:
        dtype = schema.real_dtype(strehl.dtype)
        if (
            precision is not None
            and schema.real_dtype(read_dataset_value(precision)) != dtype
        ):
            raise ValueError(
                "simulation.precision disagrees with stored result precision"
            )
        return dtype
    if precision is not None:
        return schema.real_dtype(read_dataset_value(precision))
    for path in ("meta/pixel_scale", "psfs/data"):
        if path in file:
            return schema.real_dtype(file[path].dtype)
    return np.dtype(np.float32)
