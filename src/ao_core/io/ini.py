"""Read INI files and resolve their values and paths."""

import ast
from collections.abc import Callable, Iterable, Mapping
from configparser import ConfigParser
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True)
class IniSource:
    """Parsed configuration and the context for decoding it.

    Attributes:
        parser: Parsed INI. The owner can apply overrides before decoding.
        path: Absolute source path, when available, for relative file references.
        text: Original INI text, when available, for provenance.
    """

    parser: ConfigParser
    path: Path | None = None
    text: str | None = None


def _ini_parser() -> ConfigParser:
    parser = ConfigParser(interpolation=None)
    parser.optionxform = str
    return parser


def read_ini_source(path: str | Path, *, parser: ConfigParser | None = None) -> IniSource:
    """Read and parse a UTF-8 INI once, retaining the original text.

    Resolve the path before reading. The default parser preserves option case
    and literal percent signs. A supplied parser is populated using its own
    option and interpolation rules. File access and INI syntax errors
    propagate to the caller.
    """
    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8")
    if parser is None:
        parser = _ini_parser()
    parser.read_string(text, source=str(source))
    return IniSource(parser=parser, path=source, text=text)


def read_ini(path: str | Path) -> tuple[Path, ConfigParser]:
    """Read a UTF-8 INI file and return its absolute path and parsed contents.

    Expand the user directory and resolve the path before opening the file.
    Preserve option case and literal percent signs. File access and INI syntax
    errors propagate to the caller.
    """
    source = read_ini_source(path)
    assert source.path is not None
    return source.path, source.parser


def parse_ini(text: str) -> ConfigParser:
    """Parse supplied INI text, preserving option case and literal percent signs."""
    parser = _ini_parser()
    parser.read_string(text)
    return parser


def read_ini_array(
    section: Mapping[str, str], option: str, *, label: str, allow_empty: bool = False
) -> NDArray[np.float64]:
    """Read a numeric literal as a finite float64 vector.

    A scalar becomes a one-element vector. `label` identifies the field in
    errors. Set `allow_empty` for optional vector fields that accept an empty
    list. Missing options, invalid literal syntax, non-finite entries and
    arrays with other shapes raise `ValueError`.
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
    if values.ndim != 1 or (values.size == 0 and not allow_empty):
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


__all__ = [
    "IniSource",
    "ini_to_mapping",
    "parse_ini",
    "polar_to_cartesian",
    "read_ini",
    "read_ini_array",
    "read_ini_scalar",
    "read_ini_source",
    "resolve_ini_path",
]
