"""Read and write the HDF5 result groups of an open file."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import h5py
import numpy as np
from astropy import units as u

from . import hdf5, schema

_RESULT_GROUPS = (
    "simulation",
    "setup",
    "options",
    "meta",
    "psfs",
    "stats",
    "diagnostics",
)


def _require_real_values(values: u.Quantity | h5py.Dataset, path: str) -> None:
    """Require real numerical values without loading a stored dataset."""
    if values.dtype.kind not in "fiu":
        raise ValueError(f"{path} must contain real numbers.")


def _observation_count(wavelength: Any) -> int | None:
    shape = np.shape(wavelength)
    if shape == ():
        return None
    if shape is not None and len(shape) == 1 and shape[0] > 0:
        return shape[0]
    raise ValueError("/options/wavelength must be scalar or a nonempty 1D array.")


def observation_count(file: h5py.File) -> int | None:
    """Return the batch row count, or None for a standalone observation."""
    wavelength = hdf5.require_dataset(file, "/options/wavelength")
    _require_real_values(wavelength, "/options/wavelength")
    return _observation_count(wavelength)


def _observation_index(
    count: int | None, index: int | None, *, writing: bool = False
) -> int | None:
    if index is None:
        if writing and count is not None:
            raise ValueError("A batch write requires an observation index.")
        return None
    if count is None:
        raise ValueError("A standalone result has no observation index.")
    if not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_)):
        raise TypeError("Observation indices must be integers.")
    if not 0 <= index < count:
        raise IndexError(f"Observation index {index} is out of range for {count} rows.")
    return int(index)


def _num_science_points(science_radius: Any) -> int:
    shape = np.shape(science_radius)
    return shape[0] if shape else 0


def diagnostic_shape(
    spec: Mapping[str, Any],
    setup: Mapping[str, Any],
    options: Mapping[str, Any],
) -> tuple[int, ...]:
    """Resolve the per-observation dimensions of a diagnostic declaration."""
    count = _observation_count(options["wavelength"])
    dimensions: list[int] = []
    for value in spec["shape"]:
        if value == "num_sci":
            dimensions.append(_num_science_points(setup.get("sci_r")))
        elif value == "num_ngs":
            ngs = options.get("ngs_r")
            shape = np.shape(ngs)
            if ngs is None or not shape:
                dimensions.append(0)
            elif count is not None and len(shape) == 1:
                dimensions.append(1)
            else:
                dimensions.append(shape[-1])
        else:
            dimensions.append(value)
    if any(size <= 0 for size in dimensions):
        raise ValueError(
            f"diagnostic field shape resolved to invalid dimensions {dimensions}."
        )
    return tuple(dimensions)


def _diagnostic_dtype(name: str) -> np.dtype:
    return h5py.string_dtype(encoding="utf-8") if name == "str" else np.dtype(name)


def _diagnostic_fill(dtype: np.dtype) -> str | float | bool | int:
    if h5py.check_string_dtype(dtype) is not None:
        return ""
    if dtype.kind == "f":
        return np.nan
    if dtype.kind == "b":
        return False
    return 0


def _float_dataset(group, name, shape, dtype, unit):
    dataset = group.create_dataset(name, data=np.full(shape, np.nan, dtype=dtype))
    if unit is not None:
        hdf5.set_dataset_unit(dataset, unit)
    return dataset


def _stat_shapes(
    setup: Mapping[str, Any], names: Sequence[str]
) -> dict[str, tuple[int, ...]]:
    sci = _num_science_points(setup.get("sci_r"))
    apertures = np.size(setup.get("ee_apertures", ()))
    return {name: (sci, apertures) if name == "ee" else (sci,) for name in names}


def _check_fields(
    values: Mapping[str, Any], expected: Sequence[str], *, label: str
) -> None:
    missing = sorted(set(expected) - values.keys())
    extra = sorted(values.keys() - set(expected))
    if missing:
        raise ValueError(f"{label} is missing declared fields: {', '.join(missing)}")
    if extra:
        raise ValueError(f"{label} contains undeclared fields: {', '.join(extra)}")


def initialize_layout(
    file: h5py.File,
    *,
    simulation: Mapping[str, Any],
    setup: Mapping[str, Any],
    options: Mapping[str, Any],
    save_psfs: bool,
    standard_stats: Sequence[str],
) -> None:
    """Allocate result groups for a caller's prepared settings and options.

    Values in the prepared input groups are encoded as supplied. Callers that
    accept unprepared input should use :func:`initialize_result`.
    """
    count = _observation_count(options["wavelength"])
    prefix = () if count is None else (count,)
    dtype = schema.real_dtype(simulation.get("precision", "float32"))
    extra_stats = schema.extra_stat_fields(simulation)
    extra_meta = schema.normalize_meta_fields(simulation.get("meta_fields", {}))
    diagnostic_specs = schema.diagnostic_field_specs(simulation)
    collisions = set(extra_stats) & schema.STATS_FIELD_UNITS.keys()
    if collisions:
        raise ValueError(
            f"Extra statistics collide with standard names: {sorted(collisions)}."
        )
    if set(standard_stats) - schema.STATS_FIELD_UNITS.keys():
        raise ValueError("standard_stats must contain only ee, fwhm and sr.")

    stat_shapes = _stat_shapes(setup, (*standard_stats, *extra_stats))
    if any(0 in shape for shape in stat_shapes.values()):
        raise ValueError("Statistics require a science grid and EE requires apertures.")
    diagnostic_shapes = {
        name: diagnostic_shape(spec, setup, options)
        for name, spec in diagnostic_specs.items()
    }

    for name, values in (
        ("simulation", simulation),
        ("setup", setup),
        ("options", options),
    ):
        group = file.require_group(name)
        for key, value in values.items():
            hdf5.write_value(group, str(key), value)

    meta = file.require_group("meta")
    _float_dataset(
        meta, "pixel_scale", prefix, dtype, schema.META_FIELD_UNITS["pixel_scale"]
    )
    _float_dataset(
        meta, "tel_diameter", (), dtype, schema.META_FIELD_UNITS["tel_diameter"]
    )
    pupil = meta.create_dataset(
        "tel_pupil", shape=(0, 0), maxshape=(None, None), chunks=True, dtype=dtype
    )
    hdf5.set_dataset_unit(pupil, schema.META_FIELD_UNITS["tel_pupil"])
    for name, unit in extra_meta.items():
        _float_dataset(
            meta, name, prefix, dtype, None if unit is None else u.Unit(unit)
        )

    stats = file.require_group("stats")
    stat_units = {**schema.STATS_FIELD_UNITS, **extra_stats}
    for name, shape in stat_shapes.items():
        _float_dataset(stats, name, prefix + shape, dtype, stat_units[name])

    if save_psfs:
        file.require_group("psfs")
    if diagnostic_specs:
        diagnostics = file.require_group("diagnostics")
        for name, spec in diagnostic_specs.items():
            parent, _, leaf = name.rpartition("/")
            group = diagnostics.require_group(parent) if parent else diagnostics
            diag_dtype = _diagnostic_dtype(spec["dtype"])
            shape = prefix + diagnostic_shapes[name]
            if spec["dtype"] == "str":
                dataset = group.create_dataset(
                    leaf, shape=shape, dtype=diag_dtype, fillvalue=""
                )
            else:
                dataset = group.create_dataset(
                    leaf,
                    data=np.full(shape, _diagnostic_fill(diag_dtype), dtype=diag_dtype),
                )
            if "unit" in spec:
                hdf5.set_dataset_unit(dataset, u.Unit(spec["unit"]))


def _canonical_quantities(values, units, section):
    values = dict(values)
    for name in values.keys() & units.keys():
        path = f"/{section}/{name}"
        values[name] = hdf5.require_quantity(values[name], units[name], label=path)
        _require_real_values(values[name], path)
    return values


def _validate_setup_shapes(setup):
    fields = {"sci_r", "sci_theta"} & setup.keys()
    if fields:
        shapes = [np.shape(setup[name]) for name in sorted(fields)]
        if (
            len(shapes) != 2
            or shapes[0] != shapes[1]
            or len(shapes[0]) != 1
            or shapes[0][0] == 0
        ):
            raise ValueError(
                "Science coordinates must be paired nonempty vectors of the same length."
            )
    if "ee_apertures" in setup:
        shape = np.shape(setup["ee_apertures"])
        if len(shape) != 1 or not shape[0]:
            raise ValueError("EE apertures must be a nonempty vector.")
    if "ngs_magnitude_zeropoint" in setup and np.shape(
        setup["ngs_magnitude_zeropoint"]
    ):
        raise ValueError("The NGS magnitude zeropoint must be scalar.")


def _validate_option_shapes(options, count):
    prefix = () if count is None else (count,)
    for name, value in options.items():
        if (
            not isinstance(name, str)
            or not name
            or name == "."
            or "/" in name
            or "\x00" in name
            or isinstance(value, Mapping)
        ):
            raise ValueError(f"/options/{name} must be a direct dataset.")
        if prefix and np.shape(value)[:1] != prefix:
            raise ValueError(f"/options/{name} must have one value per observation.")
        if name in ("wavelength", "zenith_angle") and np.shape(value) != prefix:
            raise ValueError(f"/options/{name} must have one scalar per observation.")
    ngs_names = ("ngs_r", "ngs_theta", "ngs_magnitude")
    if any(name in options for name in ngs_names):
        shapes = [np.shape(options[name]) for name in ngs_names if name in options]
        if (
            len(shapes) != 3
            or len(set(shapes)) != 1
            or len(shapes[0]) != len(prefix) + 1
            or not shapes[0][-1]
        ):
            raise ValueError(
                "NGS radius, angle and magnitude must have matching nonempty slot axes."
            )


def initialize_result(
    file: h5py.File,
    *,
    simulation: Mapping[str, Any],
    setup: Mapping[str, Any],
    options: Mapping[str, Any],
    save_psfs: bool = False,
    standard_stats: Sequence[str] = (),
) -> None:
    """Initialize a writable, empty HDF5 file using the shared result format.

    Args:
        file: Caller-owned HDF5 file.
        simulation: Shared run settings and extension declarations.
        setup: Base science grid and other shared setup.
        options: Observation inputs with scalar or one-dimensional wavelength.
        save_psfs: Allocate a PSF group for subsequent writes.
        standard_stats: Names of standard statistics to store.

    Raises:
        ValueError: If the file is populated or input groups violate the schema.
        TypeError: If an argument has an invalid type.
    """
    if len(file):
        raise ValueError("Result initialization requires an empty file.")
    if "wavelength" not in options:
        raise ValueError("Missing required /options/wavelength.")
    setup = _canonical_quantities(setup, schema.SETUP_FIELD_UNITS, "setup")
    options = _canonical_quantities(options, schema.OPTION_FIELD_UNITS, "options")
    simulation = dict(simulation)
    count = _observation_count(options["wavelength"])
    _validate_setup_shapes(setup)
    _validate_option_shapes(options, count)
    if save_psfs and "sci_r" not in setup:
        raise ValueError("PSFs require a base science grid.")
    if "extra_stat_fields" in simulation:
        simulation["extra_stat_fields"] = schema.normalize_unit_field_mapping(
            simulation["extra_stat_fields"], label="extra_stat_fields"
        )
    if "meta_fields" in simulation:
        simulation["meta_fields"] = {
            name: unit or ""
            for name, unit in schema.normalize_meta_fields(
                simulation["meta_fields"]
            ).items()
        }
    if "diagnostic_fields" in simulation:
        simulation["diagnostic_fields"] = schema.diagnostic_field_specs(
            simulation, strict_shape=True
        )
    initialize_layout(
        file,
        simulation=simulation,
        setup=setup,
        options=options,
        save_psfs=save_psfs,
        standard_stats=standard_stats,
    )


def _dataset_value(dataset, value, label, unit, dtype=None):
    if dtype is not None and unit is None and isinstance(value, u.Quantity):
        raise TypeError(f"{label} must be an ordinary numeric scalar.")
    dtype = dataset.dtype if dtype is None else dtype
    if unit is not None:
        return hdf5.quantity_value(value, unit, label=label, dtype=dtype)
    if h5py.check_string_dtype(dtype) is not None:
        return _text_value(value, label)
    return np.asarray(value, dtype=dtype)


def _text_value(value, label):
    text = np.asarray(value, dtype=object)
    if text.ndim == 0:
        text = np.asarray(str(text.item()), dtype=object)
    for entry in text.flat:
        if not isinstance(entry, (str, bytes)):
            raise TypeError(f"{label} must contain UTF-8 strings.")
        entry = entry.decode("utf-8") if isinstance(entry, bytes) else entry
        if "\x00" in entry:
            raise ValueError(f"{label} must not contain null characters.")
        entry.encode("utf-8")
    return text.item() if text.ndim == 0 else text


def _prepare_value(dataset, value, index, *, label, unit, dtype=None):
    shape = dataset.shape[int(index is not None) :]
    values = _dataset_value(dataset, value, label, unit, dtype)
    if np.shape(values) != shape:
        raise ValueError(f"{label} must have shape {shape}, got {np.shape(values)}.")
    return dataset, values


def write_telescope_metadata(
    group: h5py.Group,
    meta: Mapping[str, Any],
    dtype: np.dtype | None = None,
) -> None:
    """Set shared telescope values and check consistency on later writes."""
    pending = []
    for name in ("tel_diameter", "tel_pupil"):
        if name not in meta:
            continue
        dataset = group[name]
        values = _dataset_value(
            dataset,
            meta[name],
            f"result.meta.{name}",
            schema.META_FIELD_UNITS[name],
            dtype,
        )
        if name == "tel_pupil" and values.ndim != 2:
            raise ValueError("result.meta.tel_pupil must be 2D.")
        if name == "tel_diameter" and values.ndim != 0:
            raise ValueError("result.meta.tel_diameter must be a scalar.")
        if name == "tel_pupil" and dataset.shape == (0, 0):
            pending.append((dataset, values))
            continue
        current = np.asarray(dataset[()], dtype=values.dtype)
        if current.shape == values.shape and np.all(np.isnan(current)):
            pending.append((dataset, values))
        elif current.shape != values.shape or not np.allclose(
            current,
            values,
            rtol=4 * np.finfo(values.dtype).eps,
            atol=0.0,
            equal_nan=True,
        ):
            raise ValueError(
                f"result.meta.{name} does not match dataset-level /meta/{name}."
            )
    for dataset, values in pending:
        if dataset.shape != values.shape:
            dataset.resize(values.shape)
        dataset[...] = values


def _flatten_mapping(mapping, prefix=""):
    flat = {}
    for key, value in mapping.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        fields = (
            _flatten_mapping(value, name)
            if isinstance(value, Mapping)
            else {name.strip("/"): value}
        )
        if flat.keys() & fields.keys():
            raise ValueError(f"Duplicate diagnostic path '{name}'.")
        flat.update(fields)
    return flat


def _prepare_psfs(file, psfs, count, dtype):
    """Validate the PSF cube before writing any observation outputs."""
    psfs = np.asarray(psfs, dtype=dtype)
    science = _num_science_points(file.get("setup/sci_r"))
    if psfs.ndim != 3 or psfs.shape[0] != science:
        raise ValueError(f"PSFs must have shape ({science}, image_y, image_x).")
    shape = psfs.shape if count is None else (count, *psfs.shape)
    if "psfs/data" in file:
        dataset = hdf5.require_dataset(file, "/psfs/data")
        if dataset.shape != shape:
            raise ValueError(
                f"/psfs/data shape mismatch: expected {shape}, got {dataset.shape}"
            )
    return psfs, shape


def _write_psfs(group, psfs, shape, selection):
    if "data" not in group:
        group.create_dataset("data", data=np.full(shape, np.nan, dtype=psfs.dtype))
    group["data"][selection] = psfs


def write_observation(
    file: h5py.File,
    *,
    meta: Mapping[str, Any],
    psfs: np.ndarray | None = None,
    stats: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    index: int | None = None,
) -> None:
    """Save supplied outputs to a standalone result or one batch row.

    Args:
        file: Initialized writable HDF5 file.
        meta: Physical and declared extra metadata.
        psfs: PSF cube in science-grid order when saving PSFs.
        stats: Selected standard statistics and declared extra statistics.
        diagnostics: Values for all declared diagnostic paths.
        index: Zero-based batch row. Omit for a standalone file.

    Raises:
        ValueError: If outputs are missing, undeclared or incompatible.
        TypeError: If an argument has an invalid type.
        IndexError: If the batch index is out of range.
    """
    count = observation_count(file)
    index = _observation_index(count, index, writing=True)
    selection = () if index is None else index
    dtype = hdf5.output_dtype(file)
    simulation = hdf5.read_node(file["simulation"])
    extra_meta = schema.normalize_meta_fields(simulation.get("meta_fields", {}))
    extra_stats = schema.extra_stat_fields(simulation)
    diagnostic_specs = schema.diagnostic_field_specs(simulation)
    meta_group = file["meta"]
    stats_group = file["stats"]

    if "psfs" in file:
        if psfs is None:
            raise ValueError("PSFs must be provided when PSF storage is enabled.")
        _check_fields(meta, (*schema.CORE_META_KEYS, *extra_meta), label="result.meta")
        psfs, psf_shape = _prepare_psfs(file, psfs, count, dtype)
    else:
        _check_fields(
            {
                name: value
                for name, value in meta.items()
                if name not in schema.META_FIELD_UNITS
            },
            tuple(extra_meta),
            label="result.meta",
        )
    stats = {} if stats is None else stats
    stat_units = _stored_stat_units(stats_group, extra_stats)
    _check_fields(stats, tuple(stat_units), label="result.stats")
    diagnostics = _flatten_mapping({} if diagnostics is None else diagnostics)
    missing = sorted(diagnostic_specs.keys() - diagnostics.keys())
    unexpected = sorted(diagnostics.keys() - diagnostic_specs.keys())
    if missing:
        raise ValueError(
            f"result.diagnostics is missing declared diagnostics: {', '.join(missing)}"
        )
    if unexpected:
        raise ValueError(
            f"result.diagnostics contains undeclared diagnostics: {', '.join(unexpected)}"
        )

    statistics = [
        _prepare_value(
            stats_group[name],
            value,
            index,
            label=f"result.{name}",
            unit=stat_units[name],
            dtype=dtype,
        )
        for name, value in stats.items()
    ]
    metadata = [
        _prepare_value(
            meta_group[name],
            meta[name],
            index,
            label=f"result.meta.{name}",
            unit=None if unit is None else u.Unit(unit),
            dtype=dtype,
        )
        for name, unit in extra_meta.items()
    ]
    diagnostic_values = [
        _prepare_value(
            hdf5.require_diagnostic_dataset(file["diagnostics"], name),
            diagnostics[name],
            index,
            label=f"result.diagnostics['{name}']",
            unit=u.Unit(spec["unit"]) if "unit" in spec else None,
        )
        for name, spec in diagnostic_specs.items()
    ]
    pixel_scale = meta_group["pixel_scale"]
    pixel_value = (
        _dataset_value(
            pixel_scale, meta["pixel_scale"], "result.meta.pixel_scale", u.mas, dtype
        )
        if "pixel_scale" in meta
        else np.nan
    )
    if np.ndim(pixel_value) != 0:
        raise ValueError("result.meta.pixel_scale must be a scalar.")
    write_telescope_metadata(meta_group, meta, dtype)
    pixel_scale[selection] = pixel_value
    for dataset, values in metadata + statistics:
        dataset[selection] = values
    if "psfs" in file:
        _write_psfs(file["psfs"], psfs, psf_shape, selection)
    for dataset, values in diagnostic_values:
        dataset[selection] = values


def clear_observation(file: h5py.File, *, index: int) -> None:
    """Write fill values to one batch row's allocated output datasets.

    Args:
        file: Initialized writable HDF5 batch file.
        index: Zero-based observation index.

    Raises:
        ValueError: If the result is standalone or a batch index is missing.
        TypeError: If the batch index is not an integer.
        IndexError: If the batch index is out of range.
    """
    index = _observation_index(observation_count(file), index, writing=True)
    if index is None:
        raise ValueError("Clearing outputs requires a batch index.")
    for name, dataset in file["meta"].items():
        if name not in ("tel_diameter", "tel_pupil"):
            dataset[index, ...] = np.nan
    for dataset in file["stats"].values():
        dataset[index, ...] = np.nan
    if "psfs/data" in file:
        file["psfs/data"][index, ...] = np.nan
    if "diagnostics" in file:

        def clear(path, node):
            if isinstance(node, h5py.Dataset):
                node[index, ...] = _diagnostic_fill(node.dtype)

        file["diagnostics"].visititems(clear)


def _stored_stat_units(stats, extras):
    collisions = extras.keys() & schema.STATS_FIELD_UNITS.keys()
    if collisions:
        raise ValueError(
            f"Extra statistics collide with standard names: {sorted(collisions)}."
        )
    standard = {
        name: unit for name, unit in schema.STATS_FIELD_UNITS.items() if name in stats
    }
    return {**standard, **extras}


def _inspect_dataset(file, path, shape, unit, dtype, issues, *, result_precision=False):
    try:
        dataset = hdf5.require_dataset(file, path)
        if shape is not None and dataset.shape != shape:
            issues.append(
                f"{path} shape mismatch: expected {shape}, got {dataset.shape}."
            )
        if dtype is not None and dataset.dtype != dtype:
            issues.append(f"{path} must have dtype {dtype}.")
        elif result_precision and dataset.dtype not in ("float32", "float64"):
            issues.append(f"{path} must have dtype float32 or float64.")
        expected_unit = schema.unit_string(unit) if unit is not None else None
        if dataset.attrs.get(hdf5.UNITS_ATTRIBUTE) != expected_unit:
            issues.append(
                f"{path} must declare units={expected_unit!r}."
                if unit is not None
                else f"{path} must not declare units."
            )
    except ValueError as error:
        issues.append(str(error))


def _inspect_inputs(file, section, units, count, issues):
    """Check standard physical inputs and the shared axis for option extensions."""
    expected_axis = () if count is None else (count,)
    fields = {}
    for name, node in file[section].items():
        path = f"/{section}/{name}"
        try:
            if name in units:
                dataset = hdf5.require_dataset(file, path)
                _require_real_values(dataset, path)
                if dataset.shape is None:
                    raise ValueError(f"{path} must have a scalar or array shape.")
                fields[name] = dataset
                hdf5.collect_dataset_unit_issue(file, path, units[name], issues)
            elif section == "options" and (
                not isinstance(node, h5py.Dataset)
                or node.shape is None
                or (expected_axis and node.shape[:1] != expected_axis)
            ):
                raise ValueError(
                    f"{path} must be a dataset with one value per observation."
                )
        except ValueError as error:
            issues.append(str(error))
    try:
        if section == "setup":
            _validate_setup_shapes(fields)
        else:
            _validate_option_shapes(fields, count)
    except ValueError as error:
        issues.append(str(error))


def _section_issues(file, count, sections):
    issues: list[str] = []
    prefix = () if count is None else (count,)
    wavelength = hdf5.require_dataset(file, "/options/wavelength")
    if "options" not in sections and wavelength.attrs.get(hdf5.UNITS_ATTRIBUTE) != "um":
        issues.append("/options/wavelength must declare units='um'.")
    invalid_groups = [
        name
        for name in sections
        if name in file and not isinstance(file[name], h5py.Group)
    ]
    if invalid_groups:
        return [*issues, *(f"/{name} must be a group." for name in invalid_groups)]
    declarations = {
        "meta": "meta_fields",
        "stats": "extra_stat_fields",
        "diagnostics": "diagnostic_fields",
    }
    simulation = {
        name: hdf5.read_node(file[f"simulation/{name}"])
        for section, name in declarations.items()
        if section in sections and f"simulation/{name}" in file
    }
    dtype = None
    use_psfs = "psfs" in sections and "psfs/data" in file
    if "simulation/precision" in file and set(sections) & {"meta", "psfs", "stats"}:
        precision = hdf5.require_dataset(file, "/simulation/precision")
        dtype = schema.real_dtype(hdf5.read_dataset_value(precision))
    for section, units in (
        ("setup", schema.SETUP_FIELD_UNITS),
        ("options", schema.OPTION_FIELD_UNITS),
    ):
        if section in sections and section in file:
            _inspect_inputs(file, section, units, count, issues)
    check_meta = "meta" in sections and (
        "meta" in file or simulation.get("meta_fields")
    )
    if check_meta or use_psfs:
        shapes = {"pixel_scale": prefix, "tel_diameter": (), "tel_pupil": None}
        meta_units = {**schema.META_FIELD_UNITS}
        if "meta" in sections:
            meta_units.update(
                schema.normalize_meta_fields(simulation.get("meta_fields", {}))
            )
        for name, unit in meta_units.items():
            _inspect_dataset(
                file,
                f"/meta/{name}",
                shapes.get(name, prefix),
                None if unit is None else u.Unit(unit),
                dtype,
                issues,
                result_precision=True,
            )
        pupil = file.get("meta/tel_pupil")
        if isinstance(pupil, h5py.Dataset) and pupil.ndim != 2:
            issues.append("/meta/tel_pupil must be 2D.")
        if (
            check_meta
            and "meta" in file
            and (undeclared := set(file["meta"]) - meta_units.keys())
        ):
            issues.append(f"Undeclared /meta fields: {sorted(undeclared)}.")
    if "stats" in sections and ("stats" in file or simulation.get("extra_stat_fields")):
        stat_units = _stored_stat_units(
            file.get("stats", {}), schema.extra_stat_fields(simulation)
        )
        setup = {
            name: file[f"setup/{name}"]
            for name in ("sci_r", "ee_apertures")
            if f"setup/{name}" in file
        }
        for name, shape in _stat_shapes(setup, tuple(stat_units)).items():
            if 0 in shape:
                issues.append(
                    f"/stats/{name} requires a science grid and EE requires apertures."
                )
            _inspect_dataset(
                file,
                f"/stats/{name}",
                prefix + shape,
                stat_units[name],
                dtype,
                issues,
                result_precision=True,
            )
        if "stats" in file and (undeclared := set(file["stats"]) - stat_units.keys()):
            issues.append(f"Undeclared /stats fields: {sorted(undeclared)}.")
    if use_psfs:
        dataset = hdf5.require_dataset(file, "/psfs/data")
        science_grid = file.get("setup/sci_r")
        science = (
            _num_science_points(science_grid)
            if isinstance(science_grid, h5py.Dataset)
            else 0
        )
        if (
            science == 0
            or dataset.ndim != len(prefix) + 3
            or dataset.shape[:-2] != prefix + (science,)
        ):
            issues.append("/psfs/data has an invalid observation or science shape.")
        _inspect_dataset(
            file, "/psfs/data", None, None, dtype, issues, result_precision=True
        )
    if "diagnostics" in sections:
        specs = schema.diagnostic_field_specs(simulation, strict_shape=True)
        if specs:
            setup = {"sci_r": file["setup/sci_r"]} if "setup/sci_r" in file else {}
            options = {"wavelength": wavelength, "ngs_r": file.get("options/ngs_r")}
            for name, spec in specs.items():
                diag_dtype = _diagnostic_dtype(spec["dtype"])
                unit = u.Unit(spec["unit"]) if "unit" in spec else None
                _inspect_dataset(
                    file,
                    f"/diagnostics/{name}",
                    prefix + diagnostic_shape(spec, setup, options),
                    unit,
                    diag_dtype,
                    issues,
                )
        if "diagnostics" in file:
            paths = set()

            def collect_path(path, node):
                if isinstance(node, h5py.Dataset):
                    paths.add(path)

            file["diagnostics"].visititems(collect_path)
            if undeclared := paths - specs.keys():
                issues.append(f"Undeclared /diagnostics fields: {sorted(undeclared)}.")
    return issues


def collect_schema_issues(file: h5py.File) -> list[str]:
    """Return shared result schema errors without reading PSF pixels."""
    try:
        return _section_issues(file, observation_count(file), _RESULT_GROUPS)
    except (KeyError, TypeError, ValueError) as error:
        return [str(error)]


def read_result(
    file: h5py.File,
    *,
    index: int | None = None,
    sections: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Read selected result groups and recover stored quantities.

    Args:
        file: Caller-owned open file.
        index: Zero-based batch row, or None to keep stored array dimensions.
        sections: Result group names, or None to read all result groups.

    Returns:
        A mapping of the available result groups to their stored data.

    Raises:
        ValueError: If a section, layout or requested dataset is invalid.
        TypeError: If an argument or a required declaration has an invalid type.
        IndexError: If the batch index is out of range.
    """
    count = observation_count(file)
    index = _observation_index(count, index)
    sections = _RESULT_GROUPS if sections is None else tuple(sections)
    unknown = set(sections) - set(_RESULT_GROUPS)
    if unknown:
        raise ValueError(f"Unknown result sections: {sorted(unknown)}.")
    issues = _section_issues(file, count, sections)
    if issues:
        raise ValueError("\n".join(issues))

    result: dict[str, Any] = {}
    for name in sections:
        if name not in file:
            continue
        group = file[name]
        if index is None or name in ("setup", "simulation"):
            result[name] = hdf5.read_node(group)
        elif name == "meta":
            result[name] = {
                key: hdf5.read_dataset_value(dataset)
                if key in ("tel_diameter", "tel_pupil")
                else hdf5.read_row(dataset, index, path=f"/meta/{key}")
                for key, dataset in group.items()
            }
        else:
            result[name] = hdf5.read_diagnostics(group, sim_idx=index, path=f"/{name}")
    return result
