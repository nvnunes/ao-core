"""Unit-aware physical inputs and their existing AO INI field names."""

from ast import literal_eval
from collections.abc import Callable, Iterable
from configparser import ConfigParser
from dataclasses import dataclass, field, fields, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Literal, Self, overload

import numpy as np
from astropy import units as u
from astropy.io import fits
from scipy.ndimage import rotate, zoom

from ..io.ini import (
    IniSource,
    polar_to_cartesian,
    read_ini_array,
    read_ini_scalar,
    read_ini_source,
    resolve_ini_path,
)
from ._sampling_grid import SamplingSettings
from ._units import _PhysicalRecord, _quantity


def _quantities(record, **fields):
    """Own immutable float64 snapshots in the calculation units."""
    for name, (unit, ndim) in fields.items():
        quantity = getattr(record, name)
        if quantity is None:
            continue
        if not isinstance(quantity, u.Quantity):
            raise TypeError(f"{name} must be an Astropy Quantity")
        values = np.asarray(quantity.to_value(unit))
        if (
            values.dtype.kind not in "fiu"
            or values.ndim != ndim
            or values.size == 0
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(f"{name} must contain finite real values with {ndim} axes")
        values = np.ascontiguousarray(values, dtype=np.float64).reshape(values.shape)
        owned = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
        object.__setattr__(record, name, u.Quantity(owned, unit, copy=False))


def _positive(record, *names):
    for name in names:
        value = getattr(record, name)
        if value is not None and np.any(value.value <= 0):
            raise ValueError(f"{name} must be positive")


def _nonnegative(record, *names):
    for name in names:
        value = getattr(record, name)
        if value is not None and np.any(value.value < 0):
            raise ValueError(f"{name} must be nonnegative")


def _count(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _xy_pairs(value, name):
    if value is not None and value.shape[1] != 2:
        raise ValueError(f"{name} must contain [x, y] pairs")


@dataclass(frozen=True, kw_only=True, eq=False)
class TelescopeConfig:
    """Telescope diameter, central obscuration and pointing.

    Diameter is in metres, obscuration is a diameter ratio, and zenith angle
    is in degrees in [0, 90). Pupil resolution is the number of samples per
    side of the configured pupil grid. A missing value is represented by None.
    """

    diameter: u.Quantity | None = None
    obscuration_ratio: u.Quantity | None = None
    zenith_angle: u.Quantity | None = None
    pupil_resolution: int | None = None

    def __post_init__(self):
        _quantities(
            self,
            diameter=(u.m, 0),
            obscuration_ratio=(u.one, 0),
            zenith_angle=(u.deg, 0),
        )
        _positive(self, "diameter")
        _nonnegative(self, "obscuration_ratio", "zenith_angle")
        if self.obscuration_ratio is not None and self.obscuration_ratio.value >= 1:
            raise ValueError("obscuration_ratio must be less than one")
        if self.zenith_angle is not None and self.zenith_angle.value >= 90:
            raise ValueError("zenith_angle must be less than 90 degrees")
        if self.pupil_resolution is not None:
            _count(self.pupil_resolution, "pupil_resolution", 1)

    @classmethod
    def from_ini(cls, parser: ConfigParser) -> "TelescopeConfig | None":
        """Decode telescope metadata without opening pupil or aberration maps."""
        return _read_telescope(parser)


@dataclass(frozen=True, kw_only=True, eq=False)
class AtmosphereConfig:
    """Turbulence strengths and wind by layer, at the supplied reference wavelength.

    Wavelength, Fried parameter and outer scale are in metres. Seeing is in
    arcseconds. The layer vectors have matching lengths and use dimensionless
    weights, heights in metres, speeds in m/s and directions in degrees.
    Values retain the caller's observing geometry and layer order. Either or
    both turbulence strength inputs can be present. The consuming model owns
    their precedence and any seeing-to-Fried-parameter conversion.
    """

    reference_wavelength: u.Quantity | None = None
    fried_parameter: u.Quantity | None = None
    seeing: u.Quantity | None = None
    outer_scale: u.Quantity | None = None
    layer_weights: u.Quantity | None = None
    layer_heights: u.Quantity | None = None
    wind_speeds: u.Quantity | None = None
    wind_directions: u.Quantity | None = None

    def __post_init__(self):
        _quantities(
            self,
            reference_wavelength=(u.m, 0),
            fried_parameter=(u.m, 0),
            seeing=(u.arcsec, 0),
            outer_scale=(u.m, 0),
            layer_weights=(u.one, 1),
            layer_heights=(u.m, 1),
            wind_speeds=(u.m / u.s, 1),
            wind_directions=(u.deg, 1),
        )
        _positive(
            self, "reference_wavelength", "fried_parameter", "seeing", "outer_scale"
        )
        _nonnegative(self, "layer_weights", "layer_heights", "wind_speeds")
        layers = (
            self.layer_weights,
            self.layer_heights,
            self.wind_speeds,
            self.wind_directions,
        )
        if len({len(values) for values in layers if values is not None}) > 1:
            raise ValueError("Atmospheric layer vectors must have matching lengths")

    @classmethod
    def from_ini(cls, parser: ConfigParser) -> "AtmosphereConfig | None":
        """Decode the shared atmosphere fields, retaining the supplied geometry."""
        return _read_atmosphere(parser)


@dataclass(frozen=True, kw_only=True, eq=False)
class SourceConfig:
    """One group of optical sources.

    Directions contain sky-plane [x, y] pairs in arcseconds. Wavelengths are a
    vector in metres whose interpretation belongs to the selected model.
    Height is the supplied beacon height in metres, with zero representing a
    source at infinity in the existing INI convention. The optional zenith
    offsets (arcsec) and azimuths (degrees) retain the supplied polar
    coordinates and determine the sky-plane directions. Missing values are None.
    """

    directions: u.Quantity | None = None
    wavelengths: u.Quantity | None = None
    height: u.Quantity | None = None
    zenith: u.Quantity | None = None
    azimuth: u.Quantity | None = None

    def __post_init__(self):
        _quantities(
            self,
            zenith=(u.arcsec, 1),
            azimuth=(u.deg, 1),
            wavelengths=(u.m, 1),
            height=(u.m, 0),
        )
        directions = self.directions
        if (self.zenith is None) != (self.azimuth is None):
            raise ValueError("Supply both zenith and azimuth for polar directions")
        if self.zenith is not None:
            x, y = polar_to_cartesian(self.zenith.value, self.azimuth.value)
            polar_directions = np.column_stack((x, y)) * u.arcsec
            if directions is None:
                object.__setattr__(self, "directions", polar_directions)
        _quantities(self, directions=(u.arcsec, 2))
        _xy_pairs(self.directions, "directions")
        if (
            self.zenith is not None
            and directions is not None
            and not np.array_equal(self.directions.value, polar_directions.value)
        ):
            raise ValueError("Cartesian directions must agree with zenith and azimuth")
        _positive(self, "wavelengths")
        _nonnegative(self, "height")

    @classmethod
    def from_ini(cls, parser: ConfigParser, section: str) -> "SourceConfig | None":
        """Decode one source section and preserve its original polar coordinates."""
        return _read_sources(parser, section)


@dataclass(frozen=True, kw_only=True, eq=False)
class DeformableMirrorConfig:
    """Nominal parameters for one deformable mirror.

    Attributes:
        actuator_count: Number of actuator positions across the nominal grid.
        pitch: Nominal actuator spacing in metres.
        conjugate_height: Conjugation altitude in metres.
        influence_model: Name of the influence function model.
        influence_coupling: Dimensionless coupling between adjacent actuators.
    """

    actuator_count: int | None = None
    pitch: u.Quantity | None = None
    conjugate_height: u.Quantity | None = None
    influence_model: str | None = None
    influence_coupling: u.Quantity | None = None

    def __post_init__(self):
        _quantities(
            self,
            pitch=(u.m, 0),
            conjugate_height=(u.m, 0),
            influence_coupling=(u.one, 0),
        )
        _positive(self, "pitch")
        _nonnegative(self, "influence_coupling")
        if self.actuator_count is not None:
            _count(self.actuator_count, "actuator_count", 1)
        if self.influence_model is not None and (
            not isinstance(self.influence_model, str)
            or not self.influence_model.strip()
        ):
            raise ValueError("influence_model must be a nonempty string")


@dataclass(frozen=True, kw_only=True, eq=False)
class FittingConfig:
    """Mirror fitting directions as [x, y] arcseconds and dimensionless weights.

    Supplied directions and weights retain their order and original weights.
    Their lengths must match when both are provided.
    """

    directions: u.Quantity | None = None
    weights: u.Quantity | None = None

    def __post_init__(self):
        _quantities(self, directions=(u.arcsec, 2), weights=(u.one, 1))
        _xy_pairs(self.directions, "directions")
        if (
            self.directions is not None
            and self.weights is not None
            and len(self.directions) != len(self.weights)
        ):
            raise ValueError("Fitting weights must match fitting directions")


@dataclass(frozen=True, kw_only=True, eq=False)
class LensletConfig:
    """High-order wavefront sensor sampling.

    Pitches are in metres, lenslet counts are per pupil width, and grid
    offsets contain physical [x, y] displacements in metres, added to
    the nominal lenslet positions. Pitches and
    counts can be scalar or one per sensor. The INI uses lenslet-pitch units
    for its offsets. The loader converts them to metres using the configured
    pitch or telescope diameter divided by lenslet count.
    """

    pitches: u.Quantity | None = None
    counts: tuple[int, ...] | None = None
    grid_offsets: u.Quantity | None = None

    def __post_init__(self):
        _quantities(self, pitches=(u.m, 1), grid_offsets=(u.m, 2))
        _positive(self, "pitches")
        _xy_pairs(self.grid_offsets, "grid_offsets")
        if self.counts is not None:
            counts = tuple(self.counts)
            if not counts:
                raise ValueError("counts must not be empty")
            for count in counts:
                _count(count, "lenslet count", 1)
            object.__setattr__(self, "counts", counts)
        sizes = {
            len(values)
            for values in (self.counts, self.pitches)
            if values is not None and len(values) > 1
        }
        if self.grid_offsets is not None:
            sizes.add(len(self.grid_offsets))
        if len(sizes) > 1:
            raise ValueError(
                "Lenslet pitches, counts and offsets must match the sensors"
            )


@dataclass(frozen=True, kw_only=True, eq=False)
class DetectorConfig:
    """Detector sampling and noise inputs used by the shared models.

    Pixel scale is in mas/pixel. Field of view is the side length in pixels
    of the detector window or science image. Read noise and threshold are in
    electrons. Photons per frame is a vector of configured counts. The consuming model
    interprets the counts in the context of its optical sampling. Low-order
    lenslet counts are the configured number per pupil width for each sensor.
    """

    pixel_scale: u.Quantity | None = None
    field_of_view: u.Quantity | None = None
    read_noise: u.Quantity | None = None
    threshold: u.Quantity | None = None
    photons_per_frame: u.Quantity | None = None
    lenslet_counts: u.Quantity | None = None

    def __post_init__(self):
        _quantities(
            self,
            pixel_scale=(u.mas / u.pix, 0),
            field_of_view=(u.pix, 0),
            read_noise=(u.electron, 0),
            threshold=(u.electron, 0),
            photons_per_frame=(u.photon, 1),
            lenslet_counts=(u.one, 1),
        )
        _positive(self, "pixel_scale", "field_of_view", "lenslet_counts")
        _nonnegative(self, "read_noise", "photons_per_frame")
        if self.field_of_view is not None and self.field_of_view.value % 1 != 0:
            raise ValueError("field_of_view must contain a whole number of pixels")

    @classmethod
    def low_order_from_ini(cls, parser: ConfigParser) -> "DetectorConfig | None":
        """Decode shared detector and sampling fields from `[sensor_LO]`."""
        return _read_low_order_detector(parser)

    @classmethod
    def science_from_ini(cls, parser: ConfigParser) -> "DetectorConfig | None":
        """Decode shared detector sampling fields from `[sensor_science]`."""
        return _read_science_detector(parser)


@dataclass(frozen=True, kw_only=True, eq=False)
class LoopConfig:
    """Controller gain, frame rate in Hz and delay in frames.

    The frame interval in seconds is available when a frame rate is supplied.
    Controller laws and supported combinations are selected by the caller.
    """

    gain: u.Quantity | None = None
    frame_rate: u.Quantity | None = None
    delay_frames: float | None = None

    def __post_init__(self):
        _quantities(self, gain=(u.one, 0), frame_rate=(u.Hz, 0))
        _positive(self, "frame_rate")
        delay = self.delay_frames
        if delay is not None and (
            isinstance(delay, bool)
            or not isinstance(delay, Real)
            or not np.isfinite(delay)
            or delay < 0
        ):
            raise ValueError("delay_frames must be finite and nonnegative")

    @property
    def frame_interval(self) -> u.Quantity | None:
        """Return the interval between controller frames in seconds, if specified."""
        return None if self.frame_rate is None else (1 / self.frame_rate).to(u.s)

    @classmethod
    def from_ini(
        cls, parser: ConfigParser, *, stage: Literal["HO", "LO"]
    ) -> "LoopConfig | None":
        """Decode shared controller settings for the selected loop from `[RTC]`."""
        return _read_loop(parser, stage)


@dataclass(frozen=True, kw_only=True, eq=False)
class Pupil(_PhysicalRecord):
    """Resolved pupil amplitude, spatial sampling, and static optical path.

    Attributes:
        amplitude: Dimensionless field amplitude on a real, nonnegative square grid.
        pitch: Physical distance between adjacent sample centers.
        static_opd: Optical path difference on the same grid, stored in nm.
            Defaults to a zero map.
    """

    amplitude: np.ndarray
    pitch: u.Quantity
    static_opd: u.Quantity | None = None

    def __post_init__(self):
        amplitude = np.asarray(self.amplitude)
        if (
            amplitude.ndim != 2
            or amplitude.shape[0] != amplitude.shape[1]
            or amplitude.shape[0] < 3
            or amplitude.dtype.kind not in "bfiu"
            or not np.isfinite(amplitude).all()
            or np.any(amplitude < 0)
            or not np.any(amplitude > 0)
        ):
            raise ValueError("Require a finite nonnegative square pupil with light")
        amplitude = np.asarray(amplitude, dtype=np.float64)
        owned = np.frombuffer(amplitude.tobytes(), dtype=np.float64)
        object.__setattr__(self, "amplitude", owned.reshape(amplitude.shape))
        object.__setattr__(self, "pitch", _quantity(self.pitch, u.m, "pitch", 0))
        _positive(self, "pitch")
        opd = self.static_opd
        if opd is None:
            opd = np.zeros(amplitude.shape) * u.nm
        opd = _quantity(opd, u.nm, "static_opd", 2)
        if opd.shape != amplitude.shape:
            raise ValueError("static_opd must have the pupil amplitude shape")
        object.__setattr__(self, "static_opd", opd)

    @property
    def span(self) -> u.Quantity:
        """Distance between the first and last sample centers along a side."""
        return (self.amplitude.shape[0] - 1) * self.pitch


@dataclass(frozen=True, kw_only=True, eq=False)
class NaturalGuideStar(_PhysicalRecord):
    """One low-order guide star in the AO configuration.

    Attributes:
        position: Cartesian (x, y) angular offsets in the shared field frame.
        flux: Positive effective incident photon rate at the NGS sensor.
    """

    position: u.Quantity
    flux: u.Quantity

    def __post_init__(self):
        position = _quantity(self.position, u.arcsec, "position", 1)
        if position.shape != (2,):
            raise ValueError("position must contain one angular [x, y] pair")
        object.__setattr__(self, "position", position)
        object.__setattr__(
            self, "flux", _quantity(self.flux, u.photon / u.s, "flux", 0)
        )
        _positive(self, "flux")


@dataclass(frozen=True, kw_only=True, eq=False)
class Configuration:
    """Shared AO system values and values supplied for a calculation.

    The optical and control records are optional until a model needs them.
    Use an existing AO INI or construct the same records in memory. Supply
    resolved pupil data and the ordered NGS list as needed. The sampling
    uncertainty settings are shared by Fast, Balanced and Full.

    ``mirrors`` holds nominal deformable mirror records in the order supplied
    by the ``[DM]`` section. An empty tuple represents absent mirror inputs.
    """

    telescope: TelescopeConfig | None = None
    atmosphere: AtmosphereConfig | None = None
    high_order_sources: SourceConfig | None = None
    low_order_sources: SourceConfig | None = None
    science_sources: SourceConfig | None = None
    fitting: FittingConfig | None = None
    mirrors: tuple[DeformableMirrorConfig, ...] = ()
    high_order_lenslets: LensletConfig | None = None
    low_order_detector: DetectorConfig | None = None
    science_detector: DetectorConfig | None = None
    high_order_loop: LoopConfig | None = None
    low_order_loop: LoopConfig | None = None
    pupil: Pupil | None = None
    ngs: tuple[NaturalGuideStar, ...] = ()
    sampling_uncertainty_model: SamplingSettings = field(
        default_factory=SamplingSettings
    )

    def __post_init__(self):
        if self.pupil is not None and not isinstance(self.pupil, Pupil):
            raise TypeError("pupil must be a Pupil")
        mirrors = tuple(self.mirrors)
        if not all(isinstance(mirror, DeformableMirrorConfig) for mirror in mirrors):
            raise TypeError("mirrors must contain DeformableMirrorConfig records")
        object.__setattr__(self, "mirrors", mirrors)
        ngs = tuple(self.ngs)
        if not all(isinstance(star, NaturalGuideStar) for star in ngs):
            raise TypeError("ngs must contain NaturalGuideStar records")
        object.__setattr__(self, "ngs", ngs)
        if not isinstance(self.sampling_uncertainty_model, SamplingSettings):
            raise TypeError("sampling_uncertainty_model must be SamplingSettings")

    @classmethod
    def from_ini(
        cls,
        ini_parser: ConfigParser,
        *,
        ini_path: str | Path | None = None,
        load_pupil: bool = True,
    ) -> "Configuration":
        """Decode shared sections and sampling settings from a parsed INI.

        Pass the INI path returned by read_ini for relative telescope map
        paths. File-backed pupils use the configured diameter and resolution
        to determine pixel pitch. Atmospheric and beacon inputs retain their
        supplied geometry. Set `load_pupil=False` when the caller supplies
        the pupil separately or only needs INI metadata. Complete the NGS
        data with with_updates().
        """
        telescope_config = TelescopeConfig.from_ini(ini_parser)
        return cls(
            telescope=telescope_config,
            atmosphere=AtmosphereConfig.from_ini(ini_parser),
            high_order_sources=SourceConfig.from_ini(ini_parser, "sources_HO"),
            low_order_sources=SourceConfig.from_ini(ini_parser, "sources_LO"),
            science_sources=SourceConfig.from_ini(ini_parser, "sources_science"),
            fitting=_read_fitting(ini_parser),
            mirrors=_read_mirrors(ini_parser),
            high_order_lenslets=_read_lenslets(ini_parser, telescope_config),
            low_order_detector=DetectorConfig.low_order_from_ini(ini_parser),
            science_detector=DetectorConfig.science_from_ini(ini_parser),
            high_order_loop=LoopConfig.from_ini(ini_parser, stage="HO"),
            low_order_loop=LoopConfig.from_ini(ini_parser, stage="LO"),
            pupil=(
                _read_pupil(ini_parser, ini_path, telescope_config)
                if load_pupil
                else None
            ),
            sampling_uncertainty_model=SamplingSettings._from_ini(ini_parser),
        )

    @classmethod
    def from_config(
        cls, config: "Configuration", /, **extension_fields: object
    ) -> Self:
        """Construct a config subtype with the existing fields and new values.

        Pass an instance of a base class of ``cls`` and the added fields for
        the target type. Constructor fields retain their existing resolved
        values. The target constructor validates the result and rejects
        repeated or unknown field names.
        """
        if not isinstance(config, Configuration) or not issubclass(cls, type(config)):
            raise TypeError(
                "The target configuration type must extend the supplied configuration"
            )
        inherited = {
            item.name: getattr(config, item.name)
            for item in fields(config)
            if item.init
        }
        return cls(**inherited, **extension_fields)

    def with_updates(
        self,
        *,
        ngs: Iterable[NaturalGuideStar] | None = None,
        zenith_angle: u.Quantity | None = None,
        pupil: Pupil | None = None,
    ) -> Self:
        """Return a validated configuration with the supplied calculation inputs.

        Omitted inputs and None retain their values. An empty NGS sequence
        clears the guide-star collection. Subtypes retain their added fields.
        """
        telescope = self.telescope
        if zenith_angle is not None:
            telescope = replace(
                telescope or TelescopeConfig(), zenith_angle=zenith_angle
            )
        return replace(
            self,
            telescope=telescope,
            ngs=self.ngs if ngs is None else tuple(ngs),
            pupil=self.pupil if pupil is None else pupil,
        )


@overload
def load_ao_configuration(
    path: str | Path,
    *,
    prepare: Callable[[IniSource], None] | None = None,
    load_pupil: bool = True,
    decode: None = None,
) -> Configuration: ...


@overload
def load_ao_configuration[T: Configuration](
    path: str | Path,
    *,
    prepare: Callable[[IniSource], None] | None = None,
    load_pupil: bool = True,
    decode: Callable[[IniSource, Configuration], T],
) -> T: ...


def load_ao_configuration(
    path: str | Path,
    *,
    prepare: Callable[[IniSource], None] | None = None,
    load_pupil: bool = True,
    decode: Callable[[IniSource, Configuration], Configuration] | None = None,
) -> Configuration:
    """Read one AO INI and decode its shared configuration once.

    An optional ``prepare(ini_source)`` can update the parsed INI before
    shared decoding, for example to rebind calibration paths. The original
    source text remains available. After shared decoding succeeds, the
    optional ``decode(ini_source, base_config)`` returns an extended configuration.
    The shared configuration is returned when ``decode`` is omitted. Set
    ``load_pupil=False`` to leave pupil construction or map loading to the
    caller.
    """
    ini_source = read_ini_source(path)
    if prepare is not None:
        prepare(ini_source)
    base_config = Configuration.from_ini(
        ini_source.parser, ini_path=ini_source.path, load_pupil=load_pupil
    )
    if decode is None:
        return base_config
    extended_config = decode(ini_source, base_config)
    if not isinstance(extended_config, Configuration):
        raise TypeError("decode must return a Configuration or a subtype")
    return extended_config


def _telescope_map_path(section, key, ini_path):
    value = section.get(key, "").strip()
    if value.startswith(("'", '"')) and value.endswith(value[0]):
        value = value[1:-1].strip()
    if not value:
        return None
    if ini_path is not None:
        return resolve_ini_path(
            value,
            source=Path(ini_path).expanduser().resolve(),
            label=f"telescope.{key}",
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"ini_path is required for relative telescope.{key}")
    if not path.is_file():
        raise FileNotFoundError(f"File not found for telescope.{key}: {path}")
    return path


def _read_pupil(ini_parser, ini_path, telescope):
    if not ini_parser.has_section("telescope"):
        return None
    section = ini_parser["telescope"]
    pupil_path, apodizer_path, opd_path = (
        _telescope_map_path(section, key, ini_path)
        for key in ("PathPupil", "PathApodizer", "PathStaticOn")
    )
    has_maps = any((pupil_path, apodizer_path, opd_path))
    if telescope is None or telescope.diameter is None:
        if has_maps:
            raise ValueError("TelescopeDiameter is required to resolve pupil sampling")
        return None
    if telescope.pupil_resolution is None and not has_maps:
        return None
    side = telescope.pupil_resolution or 256
    pitch = telescope.diameter / side
    angle = section.getfloat("PupilAngle", fallback=0.0)

    def load_map(path):
        data = np.array(fits.getdata(path), dtype=np.float64)
        if data.ndim != 2 or min(data.shape) < 2 or np.isinf(data).any():
            raise ValueError(f"Expected a finite two-dimensional telescope map: {path}")
        data[np.isnan(data)] = 0
        if angle:
            data = rotate(data, angle, reshape=False)
        if data.shape != (side, side):
            data = zoom(data, [side / length for length in data.shape], order=1)
        return data

    if pupil_path is None:
        xy = (np.arange(side) - (side - 1) / 2) * (2.0 / side)
        radius = np.hypot(xy[:, None], xy)
        obscuration = telescope.obscuration_ratio
        obscuration = 0.0 if obscuration is None else obscuration.value
        amplitude = ((radius <= 1) & (radius > obscuration)).astype(float)
    else:
        amplitude = load_map(pupil_path)
    if apodizer_path is not None:
        amplitude *= load_map(apodizer_path)
    opd = None if opd_path is None else load_map(opd_path) * u.nm
    return Pupil(amplitude=amplitude, pitch=pitch, static_opd=opd)


def _scalar(ini, section, key, unit):
    if section in ini and key in ini[section]:
        values = ini[section]
        return read_ini_scalar(values, key, label=f"{section}.{key}") * unit
    return None


def _vector(ini, section, key, unit, *, allow_empty=False):
    if section in ini and key in ini[section]:
        values = read_ini_array(
            ini[section], key, label=f"{section}.{key}", allow_empty=allow_empty
        )
        if values.size:
            return values * unit
    return None


def _integer(value, label):
    if value % 1 != 0:
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _optional_integer(ini, section, key):
    value = _scalar(ini, section, key, u.one)
    return None if value is None else _integer(value.value, f"{section}.{key}")


def _polar_directions(ini, section, zenith, azimuth):
    radius = _vector(ini, section, zenith, u.arcsec, allow_empty=True)
    angle = _vector(ini, section, azimuth, u.deg, allow_empty=True)
    if (radius is None) != (angle is None):
        raise ValueError(f"[{section}] requires both {zenith} and {azimuth}")
    return radius, angle


def _directions(ini, section, zenith, azimuth):
    radius, angle = _polar_directions(ini, section, zenith, azimuth)
    if radius is None:
        return None
    x, y = polar_to_cartesian(radius.value, angle.value)
    return np.column_stack((x, y)) * u.arcsec


def _present[T](kind: type[T], **values) -> T | None:
    if all(value is None for value in values.values()):
        return None
    return kind(**values)


def _read_telescope(ini):
    section = "telescope"
    return _present(
        TelescopeConfig,
        diameter=_scalar(ini, section, "TelescopeDiameter", u.m),
        obscuration_ratio=_scalar(ini, section, "ObscurationRatio", u.one),
        zenith_angle=_scalar(ini, section, "ZenithAngle", u.deg),
        pupil_resolution=_optional_integer(ini, section, "Resolution"),
    )


def _read_atmosphere(ini):
    section = "atmosphere"
    use_lowercase = section in ini and "r0_value" in ini[section]
    r0_key = "r0_value" if use_lowercase else "r0_Value"
    seeing = _scalar(ini, section, "Seeing", u.arcsec)
    fried_parameter = _scalar(ini, section, r0_key, u.m)
    # Existing INIs use zero for an unused Fried parameter when seeing is set.
    if seeing is not None and fried_parameter == 0 * u.m:
        fried_parameter = None
    return _present(
        AtmosphereConfig,
        reference_wavelength=_scalar(ini, section, "Wavelength", u.m),
        fried_parameter=fried_parameter,
        seeing=seeing,
        outer_scale=_scalar(ini, section, "L0", u.m),
        layer_weights=_vector(ini, section, "Cn2Weights", u.one),
        layer_heights=_vector(ini, section, "Cn2Heights", u.m),
        wind_speeds=_vector(ini, section, "WindSpeed", u.m / u.s),
        wind_directions=_vector(ini, section, "WindDirection", u.deg),
    )


def _read_sources(ini, section):
    radius, angle = _polar_directions(ini, section, "Zenith", "Azimuth")
    return _present(
        SourceConfig,
        zenith=radius,
        azimuth=angle,
        wavelengths=_vector(ini, section, "Wavelength", u.m),
        height=_scalar(ini, section, "Height", u.m),
    )


def _read_fitting(ini):
    return _present(
        FittingConfig,
        directions=_directions(ini, "DM", "OptimizationZenith", "OptimizationAzimuth"),
        weights=_vector(ini, "DM", "OptimizationWeight", u.one),
    )


def _read_mirrors(ini, section="DM"):
    counts = _vector(ini, section, "NumberActuators", u.one)
    pitches = _vector(ini, section, "DmPitchs", u.m)
    heights = _vector(ini, section, "DmHeights", u.m)
    couplings = _vector(ini, section, "InfCoupling", u.one)
    model = ini.get(section, "InfModel", fallback=None)
    if model and model[0] in ("'", '"') and model[-1] == model[0]:
        model = model[1:-1]

    lengths = {
        len(values)
        for values in (counts, pitches, heights, couplings)
        if values is not None
    }
    if len(lengths) > 1:
        raise ValueError(
            f"[{section}] actuator counts, pitches, heights and couplings must match"
        )
    if not lengths and model is None:
        return ()

    return tuple(
        DeformableMirrorConfig(
            actuator_count=(
                None
                if counts is None
                else _integer(counts[index].value, f"{section}.NumberActuators")
            ),
            pitch=None if pitches is None else pitches[index],
            conjugate_height=None if heights is None else heights[index],
            influence_model=model,
            influence_coupling=None if couplings is None else couplings[index],
        )
        for index in range(next(iter(lengths), 1))
    )


def _read_lenslets(ini, telescope):
    section = "sensor_HO"
    counts = _vector(ini, section, "NumberLenslets", u.one)
    if counts is not None:
        counts = tuple(
            _integer(count, "sensor_HO.NumberLenslets") for count in counts.value
        )
        for count in counts:
            _count(count, "sensor_HO.NumberLenslets", 1)
    pitches = _vector(ini, section, "SizeLenslets", u.m)
    if (
        pitches is None
        and counts is not None
        and telescope is not None
        and telescope.diameter is not None
    ):
        pitches = telescope.diameter / np.array(counts)
    offsets = None
    if section in ini and "GridOffsets" in ini[section]:
        # The existing literal reader validates vectors. Decode the pair array
        # here so that offsets retain their explicit two-axis contract.
        try:
            pairs = np.asarray(literal_eval(ini[section]["GridOffsets"]), dtype=float)
        except (TypeError, ValueError, OverflowError, SyntaxError) as exc:
            raise ValueError(
                "sensor_HO.GridOffsets requires numeric [x, y] pairs"
            ) from exc
        if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
            raise ValueError("sensor_HO.GridOffsets requires numeric [x, y] pairs")
        if pitches is None or len(pitches) not in (1, len(pairs)):
            raise ValueError(
                "sensor_HO.GridOffsets requires a shared pitch or one pitch per sensor"
            )
        offsets = pairs * pitches[:, None]
    return _present(LensletConfig, pitches=pitches, counts=counts, grid_offsets=offsets)


def _read_low_order_detector(ini):
    section = "sensor_LO"
    return _present(
        DetectorConfig,
        pixel_scale=_scalar(ini, section, "PixelScale", u.mas / u.pix),
        field_of_view=_scalar(ini, section, "FieldOfView", u.pix),
        read_noise=_scalar(ini, section, "SigmaRON", u.electron),
        threshold=_scalar(ini, section, "ThresholdWCoG", u.electron),
        photons_per_frame=_vector(ini, section, "NumberPhotons", u.photon),
        lenslet_counts=_vector(ini, section, "NumberLenslets", u.one),
    )


def _read_science_detector(ini):
    return _present(
        DetectorConfig,
        pixel_scale=_scalar(ini, "sensor_science", "PixelScale", u.mas / u.pix),
        field_of_view=_scalar(ini, "sensor_science", "FieldOfView", u.pix),
    )


def _read_loop(ini, stage, section="RTC"):
    gain_key = f"LoopGain_{stage}"
    raw_gain = ini.get(section, gain_key, fallback="").strip()
    # A native gain policy leaves the shared fixed gain unspecified.
    if raw_gain.startswith(('"', "'")) and raw_gain.endswith(raw_gain[0]):
        raw_gain = raw_gain[1:-1]
    if stage == "LO" and raw_gain in {"optimize", "test"}:
        gain = None
    else:
        gain = _scalar(ini, section, gain_key, u.one)
    delay = _scalar(ini, section, f"LoopDelaySteps_{stage}", u.one)
    return _present(
        LoopConfig,
        gain=gain,
        frame_rate=_scalar(ini, section, f"SensorFrameRate_{stage}", u.Hz),
        delay_frames=None if delay is None else float(delay.value),
    )
