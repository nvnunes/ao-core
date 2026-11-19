"""Field names, units and producer declarations for result files."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np
from astropy import units as u
from numpy.typing import DTypeLike

KEY_SIMULATION_SECTION = "simulation"
KEY_SETUP_SECTION = "setup"
KEY_OPTION_SECTION = "options"
KEY_META_SECTION = "meta"
KEY_STATS_SECTION = "stats"
KEY_PSFS_SECTION = "psfs"
KEY_DIAGNOSTICS_SECTION = "diagnostics"
KEY_SIMULATION_NAME = "name"
KEY_SIMULATION_VERSION = "version"
KEY_SIMULATION_PRECISION = "precision"
KEY_SIMULATION_EXTRA_STAT_FIELDS = "extra_stat_fields"
KEY_SIMULATION_NGS_MAG_STANDARD = "ngs_mag_standard"
KEY_SIMULATION_DIAGNOSTICS_LEVEL = "diagnostics_level"
KEY_SIMULATION_DIAGNOSTIC_FIELDS = "diagnostic_fields"
KEY_SIMULATION_META_FIELDS = "meta_fields"
KEY_SETUP_EE_APERTURES = "ee_apertures"
KEY_SETUP_PEAK_METHOD = "peak_method"
KEY_SETUP_FWHM_SUMMARY = "fwhm_summary"
KEY_SETUP_EE_GEOMETRY = "ee_geometry"
KEY_SETUP_EE_CENTER = "ee_center"
KEY_SETUP_NGS_MAGNITUDE_ZEROPOINT = "ngs_magnitude_zeropoint"
KEY_SETUP_SCI_R = "sci_r"
KEY_SETUP_SCI_THETA = "sci_theta"
KEY_OPTION_WAVELENGTH = "wavelength"
KEY_OPTION_ZENITH_ANGLE = "zenith_angle"
KEY_OPTION_NGS_R = "ngs_r"
KEY_OPTION_NGS_THETA = "ngs_theta"
KEY_OPTION_NGS_MAGNITUDE = "ngs_magnitude"
KEY_META_PIXEL_SCALE = "pixel_scale"
KEY_META_TEL_DIAMETER = "tel_diameter"
KEY_META_TEL_PUPIL = "tel_pupil"
KEY_STATS_SR = "sr"
KEY_STATS_EE = "ee"
KEY_STATS_FWHM = "fwhm"
KEY_PSFS_DATA = "data"

DIAGNOSTICS_LEVELS = ("none", "validation", "debug")
CORE_META_KEYS = (KEY_META_PIXEL_SCALE, KEY_META_TEL_DIAMETER, KEY_META_TEL_PUPIL)

CORE_STATS_KEYS = (
    KEY_STATS_SR,
    KEY_STATS_EE,
    KEY_STATS_FWHM,
)

META_FIELD_UNITS = {
    KEY_META_PIXEL_SCALE: u.mas,
    KEY_META_TEL_DIAMETER: u.m,
    KEY_META_TEL_PUPIL: u.dimensionless_unscaled,
}

STATS_FIELD_UNITS = {
    KEY_STATS_SR: u.dimensionless_unscaled,
    KEY_STATS_EE: u.dimensionless_unscaled,
    KEY_STATS_FWHM: u.mas,
}

SETUP_FIELD_UNITS = {
    KEY_SETUP_EE_APERTURES: u.mas,
    KEY_SETUP_NGS_MAGNITUDE_ZEROPOINT: u.photon / (u.m**2 * u.s),
    KEY_SETUP_SCI_R: u.arcsec,
    KEY_SETUP_SCI_THETA: u.deg,
}

OPTION_FIELD_UNITS = {
    KEY_OPTION_WAVELENGTH: u.um,
    KEY_OPTION_ZENITH_ANGLE: u.deg,
    KEY_OPTION_NGS_R: u.arcsec,
    KEY_OPTION_NGS_THETA: u.deg,
    KEY_OPTION_NGS_MAGNITUDE: u.mag,
}

_DIRECT_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def real_dtype(value: DTypeLike) -> np.dtype:
    """Validate the float32/float64 simulation output contract."""
    if value is None:
        raise ValueError("precision must be float32 or float64")
    try:
        dtype = np.dtype(value)
    except (ValueError, TypeError) as error:
        raise ValueError("precision must be float32 or float64") from error
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("precision must be float32 or float64")
    return dtype


def parse_unit(value: Any, *, label: str) -> u.UnitBase:
    """Return one validated Astropy unit from a persisted or declared value."""
    try:
        return u.Unit(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a valid Astropy unit.") from exc


def unit_string(unit: u.UnitBase) -> str:
    """Return the canonical generic string persisted for one Astropy unit."""
    unit = u.Unit(unit)
    if unit == u.dimensionless_unscaled:
        return "1"
    return unit.to_string("generic")


def validate_direct_dataset_field_name(name: Any, *, label: str) -> str:
    """Return one validated direct HDF5 dataset field name."""
    if not isinstance(name, str):
        raise TypeError(f"{label} names must be strings.")
    normalized = name.strip()
    if not normalized:
        raise ValueError(f"{label} names must be non-empty.")
    if "/" in normalized:
        raise ValueError(
            f"{label} names must be direct dataset names, got {normalized!r}."
        )
    if not _DIRECT_FIELD_NAME_RE.match(normalized):
        raise ValueError(
            f"{label} name {normalized!r} must start with a letter or underscore "
            "and contain only letters, digits, and underscores."
        )
    return normalized


def validate_meta_field_name(name: Any, *, label: str = "meta field") -> str:
    """Return one validated simulation-owned scalar ``/meta`` field name.

    Extra meta fields are direct children of the persisted ``/meta`` group and
    carry one scalar value per simulation. Names therefore use the same simple
    key grammar as core metadata and may not collide with standard
    fields such as ``pixel_scale``.
    """

    normalized = validate_direct_dataset_field_name(name, label=label)
    if normalized in CORE_META_KEYS:
        raise ValueError(
            f"{label} name {normalized!r} collides with a core /meta field."
        )
    return normalized


def normalize_unit_field_mapping(value: Any, *, label: str) -> dict[str, str]:
    """Return an insertion-ordered name-to-generic-unit mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping from field names to units.")
    normalized: dict[str, str] = {}
    for raw_name, raw_unit in value.items():
        name = validate_direct_dataset_field_name(raw_name, label=label)
        if name in normalized:
            raise ValueError(
                f"{label} contains duplicate field name {name!r} after normalization."
            )
        normalized[name] = unit_string(parse_unit(raw_unit, label=f"{label}[{name!r}]"))
    return normalized


def normalize_meta_fields(
    value: Any,
    *,
    label: str = "simulation['meta_fields']",
) -> dict[str, str | None]:
    """Return validated scalar metadata field names and optional units.

    A unit string declares a physical or scientifically dimensionless scalar.
    ``None`` declares an ordinary numeric scalar, such as a count, whose
    persisted dataset must not carry a ``units`` attribute.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping from field names to units or None.")
    fields: dict[str, str | None] = {}
    for raw_name, raw_unit in value.items():
        name = validate_meta_field_name(raw_name, label=label)
        fields[name] = (
            None
            if raw_unit is None or (isinstance(raw_unit, str) and not raw_unit.strip())
            else unit_string(parse_unit(raw_unit, label=f"{label}[{name!r}]"))
        )
    return fields


def extra_stat_fields(simulation: Mapping[str, Any]) -> dict[str, u.UnitBase]:
    """Read declared extra-stat units from a validated simulation payload."""
    fields = simulation.get(KEY_SIMULATION_EXTRA_STAT_FIELDS, {})
    if not isinstance(fields, Mapping):
        raise TypeError(
            f"simulation['{KEY_SIMULATION_EXTRA_STAT_FIELDS}'] must be a mapping."
        )
    return {
        str(name): parse_unit(unit, label=f"extra stat {name!r}")
        for name, unit in fields.items()
    }


def diagnostic_field_specs(
    simulation: Mapping[str, Any], *, strict_shape: bool = False
) -> dict[str, dict[str, Any]]:
    """Read and validate declared diagnostic specs from ``/simulation``.

    The persisted ``diagnostic_fields`` mapping may use flat slash-delimited
    keys or nested mappings that mirror the ``/diagnostics`` group tree. Each
    leaf spec declares a storage ``dtype`` and a shape excluding the leading
    simulation dimension ``N``. Supported dtypes are ``float32``, ``float64``,
    ``int32``, ``int64``, ``bool``, and ``str``. Supported symbolic shape
    tokens are ``num_sci`` and ``num_ngs``; numeric shape values must be
    positive integers.

    Args:
        simulation: Persisted ``/simulation`` payload mapping.
        strict_shape: Require a vector of integer dimensions or shape tokens.

    Returns:
        Flat mapping from slash-delimited ``/diagnostics`` field path to
        normalized dtype/shape spec.

    Raises:
        TypeError: If the diagnostic spec tree is not mapping-shaped.
        ValueError: If field names, dtypes, or shape tokens are invalid.
    """
    raw_specs = simulation.get(KEY_SIMULATION_DIAGNOSTIC_FIELDS, {})
    if raw_specs is None:
        return {}
    if not isinstance(raw_specs, Mapping):
        raise TypeError(
            f"simulation['{KEY_SIMULATION_DIAGNOSTIC_FIELDS}'] must be a mapping."
        )
    specs: dict[str, dict[str, Any]] = {}

    def iter_specs(node: Mapping[str, Any], prefix: str = ""):
        for raw_name, raw_spec in node.items():
            name = f"{prefix}/{raw_name}" if prefix else str(raw_name)
            if not isinstance(raw_spec, Mapping):
                raise TypeError(
                    f"diagnostic field spec for '{name}' must be a mapping."
                )
            if "dtype" in raw_spec or "shape" in raw_spec:
                yield name, raw_spec
            else:
                yield from iter_specs(raw_spec, prefix=name)

    for raw_name, raw_spec in iter_specs(raw_specs):
        name = str(raw_name).strip("/")
        if any(not part or part == "." or "\x00" in part for part in name.split("/")):
            raise ValueError(f"Invalid diagnostic field path '{name}'.")
        if name in specs:
            raise ValueError(f"Duplicate diagnostic path '{name}'.")
        dtype = str(raw_spec.get("dtype", "float32")).strip()
        if dtype not in {"float32", "float64", "int32", "int64", "bool", "str"}:
            raise ValueError(
                f"diagnostic field '{name}' has unsupported dtype {dtype!r}."
            )
        shape = raw_spec.get("shape", ())
        shape_values = np.asarray(shape, dtype=object)
        if strict_shape and shape_values.ndim != 1:
            raise ValueError(f"diagnostic field '{name}' shape must be a vector.")
        normalized_shape: list[str | int] = []
        for value in shape_values.reshape(-1).tolist():
            if isinstance(value, str):
                if value not in {"num_sci", "num_ngs"}:
                    try:
                        value = int(value)
                    except ValueError as exc:
                        raise ValueError(
                            f"diagnostic field '{name}' has unsupported shape token {value!r}."
                        ) from exc
                    if value <= 0:
                        raise ValueError(
                            f"diagnostic field '{name}' shape values must be positive."
                        )
                    normalized_shape.append(value)
                else:
                    normalized_shape.append(value)
            else:
                if strict_shape and (
                    not isinstance(value, (int, np.integer))
                    or isinstance(value, (bool, np.bool_))
                ):
                    raise ValueError(
                        f"diagnostic field '{name}' shape values must be integers."
                    )
                if int(value) <= 0:
                    raise ValueError(
                        f"diagnostic field '{name}' shape values must be positive."
                    )
                normalized_shape.append(int(value))
        spec = {"dtype": dtype, "shape": tuple(normalized_shape)}
        if dtype in {"float32", "float64"}:
            if "unit" not in raw_spec:
                raise ValueError(f"diagnostic field '{name}' must declare a unit.")
            spec["unit"] = unit_string(
                parse_unit(raw_spec["unit"], label=f"diagnostic field {name!r} unit")
            )
        elif "unit" in raw_spec:
            raise ValueError(
                f"non-floating diagnostic field '{name}' must not declare a unit."
            )
        specs[name] = spec
    for name in specs:
        parent = name.rpartition("/")[0]
        while parent:
            if parent in specs:
                raise ValueError(f"Diagnostic dataset '{parent}' also names a group.")
            parent = parent.rpartition("/")[0]
    return specs
