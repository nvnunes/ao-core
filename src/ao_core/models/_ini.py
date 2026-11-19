"""Read INI configuration and decode values with caller-supplied conventions."""

import ast
from collections.abc import Callable, Iterable, Mapping
from configparser import ConfigParser
from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _ini_parser() -> ConfigParser:
    parser = ConfigParser(interpolation=None)
    parser.optionxform = str
    return parser


def read_ini(path: str | Path) -> tuple[Path, ConfigParser]:
    """Read a UTF-8 INI file and return its absolute path and parsed contents.

    Expand the user directory and resolve the path before opening the file.
    Preserve option case and literal percent signs. File access and INI syntax
    errors propagate to the caller.
    """
    source = Path(path).expanduser().resolve()
    parser = _ini_parser()
    with source.open(encoding="utf-8") as stream:
        parser.read_file(stream)
    return source, parser


def parse_ini(text: str) -> ConfigParser:
    """Parse supplied INI text, preserving option case and literal percent signs."""
    parser = _ini_parser()
    parser.read_string(text)
    return parser


def read_ini_array(
    section: Mapping[str, str], option: str, *, label: str
) -> NDArray[np.float64]:
    """Read a numeric literal as a finite, nonempty float64 vector.

    A scalar becomes a one-element vector. `label` identifies the field in
    errors. Missing options, invalid literal syntax, non-finite entries and
    arrays with other shapes raise `ValueError`. Errors from NumPy's float
    conversion propagate for literals it cannot convert.
    """
    if option not in section:
        raise ValueError(f"Configuration requires {label}.")
    try:
        parsed = ast.literal_eval(section[option])
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric scalar or list.") from exc
    values = np.asarray(parsed, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional list.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite values.")
    return values


def read_ini_scalar(section: Mapping[str, str], option: str, *, label: str) -> float:
    """Read one finite float, with `label` identifying the field in errors."""
    if option not in section:
        raise ValueError(f"Configuration requires {label}.")
    try:
        value = float(section[option])
    except ValueError as exc:
        raise ValueError(f"{label} must be a finite scalar.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{label} must be a finite scalar.")
    return value


def resolve_ini_path(
    value: str, *, source: Path, label: str, must_exist: bool = True
) -> Path:
    """Resolve a path value relative to the resolved source INI file.

    Empty values raise `ValueError`. By default, the result must be an existing
    file or the function raises `FileNotFoundError`. Use `must_exist=False`
    for paths such as output files that will be created later.
    """
    raw = value.strip()
    if not raw:
        raise ValueError(f"{label} must not be empty.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    path = path.resolve()
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"File not found for {label}: {path}")
    return path


def polar_to_cartesian(
    radius: NDArray, azimuth_degrees: NDArray
) -> tuple[NDArray, NDArray]:
    """Convert equal-shaped radius and azimuth arrays to Cartesian x and y.

    Azimuth is in degrees, increasing from positive x toward positive y.
    Both outputs have the same shape and physical unit as the input radius.
    """
    if radius.shape != azimuth_degrees.shape:
        raise ValueError(
            "Polar coordinate radius and azimuth vectors must have identical shapes."
        )
    radians = np.deg2rad(azimuth_degrees)
    return radius * np.cos(radians), radius * np.sin(radians)


def ini_to_mapping[T](
    parser: ConfigParser,
    decode: Callable[[str, str, str], T],
    *,
    sections: Iterable[str] | None = None,
) -> dict[str, dict[str, T]]:
    """Build an owned mapping for an in-memory configuration handoff.

    Call `decode(section, option, raw_value)` for each option of each selected
    section. With `sections=None`, use all named sections. Options inherited
    from `DEFAULT` are included in each section. The mapping deep-copies each
    decoded value before calling the decoder again. Decoded values must
    support deep copying.
    """
    if sections is None:
        sections = parser.sections()
    return {
        section: {
            option: deepcopy(decode(section, option, raw_value))
            for option, raw_value in parser.items(section, raw=True)
        }
        for section in sections
    }
