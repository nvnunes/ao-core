"""Unit-aware physical inputs and their existing AO INI field names."""

from ast import literal_eval
from configparser import ConfigParser
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from astropy import units as u

from ._ini import polar_to_cartesian, read_ini, read_ini_array, read_ini_scalar


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


@dataclass(frozen=True, kw_only=True, eq=False)
class SourceConfig:
    """One group of optical sources.

    Directions contain sky-plane [x, y] pairs in arcseconds. Wavelengths are a
    vector in metres whose interpretation belongs to the selected model.
    Height is the supplied beacon height in metres, with zero representing a
    source at infinity in the existing INI convention. Missing values are None.
    """

    directions: u.Quantity | None = None
    wavelengths: u.Quantity | None = None
    height: u.Quantity | None = None

    def __post_init__(self):
        _quantities(
            self, directions=(u.arcsec, 2), wavelengths=(u.m, 1), height=(u.m, 0)
        )
        _xy_pairs(self.directions, "directions")
        _positive(self, "wavelengths")
        _nonnegative(self, "height")


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
    of the low-order detector window. Read noise and threshold are in
    electrons. Photons per frame is a vector of configured counts. The consuming model
    interprets the counts in the context of its optical sampling.
    """

    pixel_scale: u.Quantity | None = None
    field_of_view: u.Quantity | None = None
    read_noise: u.Quantity | None = None
    threshold: u.Quantity | None = None
    photons_per_frame: u.Quantity | None = None

    def __post_init__(self):
        _quantities(
            self,
            pixel_scale=(u.mas / u.pix, 0),
            field_of_view=(u.pix, 0),
            read_noise=(u.electron, 0),
            threshold=(u.electron, 0),
            photons_per_frame=(u.photon, 1),
        )
        _positive(self, "pixel_scale", "field_of_view")
        _nonnegative(self, "read_noise", "photons_per_frame")
        if self.field_of_view is not None and self.field_of_view.value % 1 != 0:
            raise ValueError("field_of_view must contain a whole number of pixels")


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


@dataclass(frozen=True, kw_only=True, eq=False)
class AOConfiguration:
    """Unit-aware physical inputs grouped by optical or control component.

    Construct the records directly from quantities or use `from_ini` with the
    existing AO sections. Absent groups and fields are None. Model consumers
    supply their own defaults and check which inputs they require.
    """

    telescope: TelescopeConfig | None = None
    atmosphere: AtmosphereConfig | None = None
    high_order_sources: SourceConfig | None = None
    low_order_sources: SourceConfig | None = None
    science_sources: SourceConfig | None = None
    fitting: FittingConfig | None = None
    high_order_lenslets: LensletConfig | None = None
    low_order_detector: DetectorConfig | None = None
    science_detector: DetectorConfig | None = None
    high_order_loop: LoopConfig | None = None
    low_order_loop: LoopConfig | None = None

    @classmethod
    def from_ini(cls, ini: ConfigParser) -> "AOConfiguration":
        """Decode the shared physical values from an already parsed AO INI.

        Use the case-sensitive parser returned by `read_ini` or `parse_ini`.
        The fields use their existing numerical INI conventions. Omitted
        fields are left unspecified. Atmospheric and beacon geometry retain
        the values in the file, with no zenith-angle projection applied.
        Application sections and format-specific file paths remain available
        in the original parser for the caller.
        """
        telescope = _read_telescope(ini)
        return cls(
            telescope=telescope,
            atmosphere=_read_atmosphere(ini),
            high_order_sources=_read_sources(ini, "sources_HO"),
            low_order_sources=_read_sources(ini, "sources_LO"),
            science_sources=_read_sources(ini, "sources_science"),
            fitting=_read_fitting(ini),
            high_order_lenslets=_read_lenslets(ini, telescope),
            low_order_detector=_read_low_order_detector(ini),
            science_detector=_read_science_detector(ini),
            high_order_loop=_read_loop(ini, "HO"),
            low_order_loop=_read_loop(ini, "LO"),
        )


def load_ao_configuration(path: str | Path) -> AOConfiguration:
    """Read physical inputs from an existing AO INI file.

    If the caller also needs application fields or native pass-through
    values, use `read_ini` once and pass that parser to `AOConfiguration.from_ini`.
    """
    _, ini = read_ini(path)
    return AOConfiguration.from_ini(ini)


def _scalar(ini, section, key, unit):
    if section in ini and key in ini[section]:
        values = ini[section]
        return read_ini_scalar(values, key, label=f"{section}.{key}") * unit
    return None


def _vector(ini, section, key, unit):
    if section in ini and key in ini[section]:
        values = ini[section]
        return read_ini_array(values, key, label=f"{section}.{key}") * unit
    return None


def _integer(value, label):
    if value % 1 != 0:
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _optional_integer(ini, section, key):
    value = _scalar(ini, section, key, u.one)
    return None if value is None else _integer(value.value, f"{section}.{key}")


def _directions(ini, section, zenith, azimuth):
    radius = _vector(ini, section, zenith, u.arcsec)
    angle = _vector(ini, section, azimuth, u.deg)
    if radius is None and angle is None:
        return None
    if radius is None or angle is None:
        raise ValueError(f"[{section}] requires both {zenith} and {azimuth}")
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
    return _present(
        SourceConfig,
        directions=_directions(ini, section, "Zenith", "Azimuth"),
        wavelengths=_vector(ini, section, "Wavelength", u.m),
        height=_scalar(ini, section, "Height", u.m),
    )


def _read_fitting(ini):
    return _present(
        FittingConfig,
        directions=_directions(ini, "DM", "OptimizationZenith", "OptimizationAzimuth"),
        weights=_vector(ini, "DM", "OptimizationWeight", u.one),
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
    )


def _read_science_detector(ini):
    return _present(
        DetectorConfig,
        pixel_scale=_scalar(ini, "sensor_science", "PixelScale", u.mas / u.pix),
    )


def _read_loop(ini, stage):
    section = "RTC"
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
