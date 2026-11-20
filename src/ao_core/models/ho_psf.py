"""High-order PSF prediction on a configured science detector grid."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Literal

import numpy as np
from astropy import units as u
from scipy.ndimage import affine_transform

from ao_core.stats import PsfMetadata

from ._ho_common import predict_common
from ._ho_optics import diffraction_image
from ._ho_static import static_diffraction
from ._physical_config import _read_loop, _read_mirrors
from ._units import _PhysicalRecord, _quantity
from .config import Configuration, IniSource, LoopConfig, read_ini_array

__all__ = [
    "HoPsfConfiguration",
    "HoPsfModel",
    "HoPsfPrediction",
    "HoPsfRequest",
    "MirrorGeometry",
    "decode_ho_psf",
]


@dataclass(frozen=True, kw_only=True, eq=False)
class MirrorGeometry(_PhysicalRecord):
    """Resolved coordinates for a calibrated deformable mirror.

    Attributes:
        actuator_axis: Shared one-dimensional x/y actuator axis in metres.
        valid_mask: Boolean actuator mask in [y, x] order.
        pupil_x: One-dimensional x coordinates for the tilt-fit pupil in metres.
        pupil_y: One-dimensional y coordinates for the tilt-fit pupil in metres.
        pupil_mask: Pupil weights for calibrating command tilt on the supplied
            axes. If omitted, the model samples the configured pupil there.
        fit_cutoff: Absolute singular-value cutoff for the phase-to-command fit.
    """

    actuator_axis: u.Quantity
    valid_mask: np.ndarray
    pupil_x: u.Quantity
    pupil_y: u.Quantity
    pupil_mask: np.ndarray | None = None
    fit_cutoff: float = 5e-4

    def __post_init__(self) -> None:
        for name in ("actuator_axis", "pupil_x", "pupil_y"):
            axis = _quantity(getattr(self, name), u.m, name, 1)
            if np.any(np.diff(axis.value) <= 0):
                raise ValueError(f"{name} must increase strictly")
            object.__setattr__(self, name, axis)
        mask = np.asarray(self.valid_mask)
        side = len(self.actuator_axis)
        if (
            side < 2
            or mask.dtype != np.bool_
            or mask.shape != (side, side)
            or not mask.any()
        ):
            raise ValueError(
                "valid_mask must mark valid actuators on the square [y, x] grid"
            )
        mask = np.frombuffer(mask.tobytes(), dtype=bool).reshape(mask.shape)
        object.__setattr__(self, "valid_mask", mask)
        if self.pupil_mask is not None:
            weights = np.asarray(self.pupil_mask, dtype=np.float64)
            if (
                weights.shape != (len(self.pupil_y), len(self.pupil_x))
                or not np.isfinite(weights).all()
                or np.any(weights < 0)
                or not np.any(weights > 0)
            ):
                raise ValueError(
                    "pupil_mask must contain finite nonnegative [y, x] weights with light"
                )
            weights = np.frombuffer(weights.tobytes(), dtype=np.float64).reshape(
                weights.shape
            )
            object.__setattr__(self, "pupil_mask", weights)
        if not np.isfinite(self.fit_cutoff) or self.fit_cutoff < 0:
            raise ValueError("fit_cutoff must be finite and nonnegative")


@dataclass(frozen=True, kw_only=True, eq=False)
class HoPsfConfiguration(Configuration):
    """Shared physical configuration extended with high-order PSF settings.

    Attributes:
        inversion_floor: Relative numerical floor for the reconstruction solve.
        estimator_regularization: Diagonal regularization as a fraction of
            mean prior slope variance.
        tilt_removal: Policy for removing tilt from the residual optical phase.
        detector_origin: On an even grid, ``fft`` puts zero angle on the
            central pixel with the larger indices. ``geometric`` puts it
            between the central pixels.
        mirror_geometries: Optional calibration records in configured mirror
            order. A missing calibration uses nominal geometry.
        residual_high_order_loop: Gain and command delay for the science DM.
            Delay is measured in frames of the common high-order loop.
        static_opd_sampling: Grid selection for OPD that varies over the illuminated pupil.
        air_temperature: Air temperature for refractivity. Defaults to 17.5 °C.
        air_pressure: Air pressure for refractivity. Defaults to 75,000 Pa.
        relative_humidity: Relative humidity for refractivity. Defaults to 10%.
        ciddor_compressibility: Include real-gas density corrections in the
            Ciddor branch. Defaults to true.
    """

    inversion_floor: float
    estimator_regularization: float
    tilt_removal: Literal["none", "pupil_fit"]
    detector_origin: Literal["fft", "geometric"] = "fft"
    mirror_geometries: tuple[MirrorGeometry | None, ...] = ()
    residual_high_order_loop: LoopConfig | None = None
    static_opd_sampling: Literal["resampled", "original_map"] | None = None
    air_temperature: u.Quantity = field(default_factory=lambda: 290.65 * u.K)
    air_pressure: u.Quantity = field(default_factory=lambda: 75_000 * u.Pa)
    relative_humidity: u.Quantity = field(default_factory=lambda: 10 * u.percent)
    ciddor_compressibility: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("inversion_floor", "estimator_regularization"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.tilt_removal not in ("none", "pupil_fit"):
            raise ValueError("tilt_removal must be 'none' or 'pupil_fit'")
        if self.detector_origin not in ("fft", "geometric"):
            raise ValueError("detector_origin must be 'fft' or 'geometric'")
        if self.static_opd_sampling not in (None, "resampled", "original_map"):
            raise ValueError(
                "static_opd_sampling must be None, 'resampled' or 'original_map'"
            )
        geometries = tuple(self.mirror_geometries)
        if not all(
            item is None or isinstance(item, MirrorGeometry) for item in geometries
        ):
            raise TypeError(
                "mirror_geometries must contain MirrorGeometry records or None"
            )
        if geometries and len(geometries) != len(self.mirrors):
            raise ValueError("mirror_geometries must match the configured mirrors")
        object.__setattr__(self, "mirror_geometries", geometries)
        if self.residual_high_order_loop is not None and not isinstance(
            self.residual_high_order_loop, LoopConfig
        ):
            raise TypeError("residual_high_order_loop must be a LoopConfig")

        temperature = self.air_temperature
        if isinstance(temperature, u.Quantity):
            temperature = temperature.to(u.K, equivalencies=u.temperature())
        temperature = _quantity(temperature, u.K, "air_temperature", 0)
        pressure = _quantity(self.air_pressure, u.Pa, "air_pressure", 0)
        humidity = _quantity(self.relative_humidity, u.percent, "relative_humidity", 0)
        if temperature.value <= 0 or pressure.value <= 0:
            raise ValueError("air_temperature and air_pressure must be positive")
        if not 0 <= humidity.value <= 100:
            raise ValueError("relative_humidity must be between 0% and 100%")
        if not isinstance(self.ciddor_compressibility, bool):
            raise TypeError("ciddor_compressibility must be a boolean")
        object.__setattr__(self, "air_temperature", temperature)
        object.__setattr__(self, "air_pressure", pressure)
        object.__setattr__(self, "relative_humidity", humidity)


@dataclass(frozen=True, kw_only=True)
class HoPsfRequest:
    """A resolved HO configuration for one science wavelength and field."""

    config: HoPsfConfiguration

    def __post_init__(self) -> None:
        if not isinstance(self.config, HoPsfConfiguration):
            raise TypeError("config must be a HoPsfConfiguration")


@dataclass(frozen=True, kw_only=True, eq=False)
class HoPsfPrediction(_PhysicalRecord):
    """PSFs in request order at the configured detector sampling.

    Attributes:
        positions: Read-only Cartesian field coordinates as angular x/y pairs.
        psfs: Read-only image cube in [field, y, x] order. The initial model
            returns float64 pixels. Columns increase toward +x and rows toward +y.
        metadata: Wavelength, pixel scale, telescope diameter and pupil.
        full_flux: Shared positive integral of the uncropped sampled plane
            in native pixel units. Divide `psfs` by this value for unit
            full-plane normalization.
    """

    positions: u.Quantity
    psfs: np.ndarray
    metadata: PsfMetadata
    full_flux: float


def decode_ho_psf(ini_source: IniSource, config: Configuration) -> HoPsfConfiguration:
    """Extend the shared INI configuration with the HO model settings.

    Read the inversion floor, estimator regularization and optical tilt
    policy from [sensor_HO]. The archived ReconstructionNoiseFraction
    spelling is accepted for the estimator setting. The science channel DM
    comes from [DM_science] and the residual high-order loop from [RTC_science].
    When present, [sensor_HO] WfsType must be Shack-Hartmann,
    ReconstructionModel must be finite_aperture, RemoveMeanSlopes must be
    true, and NoiseVariance must be zero. [telescope] extraErrorNm must
    also be zero. Select the detector origin and supply resolved
    calibration on the returned record.
    """
    parser = ini_source.parser
    section = "sensor_HO"
    for key, supported in (
        ("WfsType", "shack-hartmann"),
        ("ReconstructionModel", "finite_aperture"),
    ):
        raw = parser.get(section, key, fallback=supported).strip()
        value = raw.strip("'\"").casefold()
        if value != supported:
            raise NotImplementedError(
                f"HO PSF prediction does not support {section}.{key}={raw}"
            )
    if not parser.getboolean(section, "RemoveMeanSlopes", fallback=True):
        raise NotImplementedError(
            "HO PSF prediction requires sensor_HO.RemoveMeanSlopes"
        )
    if parser.getfloat("telescope", "extraErrorNm", fallback=0) != 0:
        raise NotImplementedError("HO PSF prediction requires telescope.extraErrorNm=0")
    if parser.has_option(section, "NoiseVariance") and np.any(
        read_ini_array(
            parser[section], "NoiseVariance", label="sensor_HO.NoiseVariance"
        )
    ):
        raise NotImplementedError(
            "HO PSF prediction currently requires zero injected HO noise"
        )
    science_dms = _read_mirrors(parser, "DM_science")
    if science_dms:
        if len(science_dms) != 1 or len(config.mirrors) != 1:
            raise ValueError(
                "The HO model requires one common DM and one science DM when [DM_science] is supplied"
            )
        config = replace(config, mirrors=config.mirrors + science_dms)
    tilt_removal = parser.get(section, "TipTiltRemoval", fallback="none").strip()
    if tilt_removal.startswith(("'", '"')) and tilt_removal.endswith(tilt_removal[0]):
        tilt_removal = tilt_removal[1:-1]
    regularization_key = "ReconstructionRegularization"
    if not parser.has_option(section, regularization_key):
        regularization_key = "ReconstructionNoiseFraction"
    return HoPsfConfiguration.from_config(
        config,
        inversion_floor=parser.getfloat(
            section, "TomographicRegularizationFloor", fallback=1e-12
        ),
        estimator_regularization=parser.getfloat(
            section, regularization_key, fallback=0.0
        ),
        tilt_removal=tilt_removal,
        residual_high_order_loop=_read_loop(parser, "HO", "RTC_science"),
    )


def _diffraction_image(amplitude, pupil_pitch, period, image_side, detector_origin):
    """Sample the squared pupil transform, then crop at the selected origin."""
    pupil_side = len(amplitude)
    if pupil_side * pupil_pitch > period:
        raise ValueError(
            "science detector pixel scale gives a Fourier period smaller than the pupil grid"
        )

    fft_side = max(image_side, 2 * round(period / (2 * pupil_pitch)))
    pitch_ratio = period / (fft_side * pupil_pitch)
    if not np.isclose(pitch_ratio, 1.0, rtol=0.0, atol=8 * np.finfo(float).eps):
        optical_side = min(fft_side, round(pupil_side / pitch_ratio))
        offset = ((pupil_side - 1) - pitch_ratio * (optical_side - 1)) / 2
        amplitude = affine_transform(
            amplitude,
            np.eye(2) * pitch_ratio,
            offset=offset,
            output_shape=(optical_side, optical_side),
            order=1,
            mode="grid-constant",
        )

    full_flux = float(fft_side**2 * np.sum(amplitude**2))
    if not np.isfinite(full_flux) or full_flux <= 0:
        raise ValueError("The sampled pupil must have finite positive full-plane flux")

    return diffraction_image(
        amplitude, fft_side, image_side, detector_origin
    ), full_flux


class HoPsfModel:
    """Predict high-order PSFs from a resolved optical configuration."""

    def predict(self, request: HoPsfRequest) -> HoPsfPrediction:
        """Evaluate the field at one wavelength on the configured detector.

        The diffraction calculation requires a resolved pupil and telescope
        diameter, science directions and wavelength, and a pixel scale and
        positive even FieldOfView. With atmosphere present, the calculation
        also requires a common DM, guiding and fitting inputs, and the common
        high-order loop. The nominal square lenslet grid must span the
        telescope diameter. The Fourier period, wavelength divided by pixel
        angle in radians, must span a positive even number of lenslet pitches.
        A map with varying static OPD across the illuminated pupil requires
        a static sampling policy. The residual high-order loop supplies the
        science DM gain and its delay in common-loop frames. Zero residual-loop
        gain disables the science DM correction.
        """
        if not isinstance(request, HoPsfRequest):
            raise TypeError("request must be a HoPsfRequest")
        config = request.config
        if config.pupil is None:
            raise ValueError("HO PSF prediction requires a resolved pupil")
        if config.telescope is None or config.telescope.diameter is None:
            raise ValueError("HO PSF prediction requires a telescope diameter")
        sources = config.science_sources
        if sources is None or sources.directions is None:
            raise ValueError("HO PSF prediction requires science field directions")
        if sources.wavelengths is None or len(sources.wavelengths) != 1:
            raise ValueError("HO PSF prediction requires one science wavelength")
        detector = config.science_detector
        if (
            detector is None
            or detector.pixel_scale is None
            or detector.field_of_view is None
        ):
            raise ValueError(
                "HO PSF prediction requires science detector PixelScale and FieldOfView"
            )
        image_side = detector.field_of_view.to_value(u.pix)
        if image_side % 2 != 0:
            raise ValueError(
                "Science detector FieldOfView must be a positive even pixel count"
            )

        if config.atmosphere is None and (
            config.mirrors or config.residual_high_order_loop is not None
        ):
            raise NotImplementedError(
                "Diffraction-only prediction requires no mirror correction"
            )
        wavelength = sources.wavelengths[0]
        pixel_scale = detector.pixel_scale * u.pix
        period = wavelength.to_value(u.m) / pixel_scale.to_value(u.rad)
        opd = config.pupil.static_opd.to_value(u.m)
        amplitude = config.pupil.amplitude
        illuminated = amplitude > 0
        illuminated_opd = opd[illuminated]
        phase = None
        if np.any(illuminated_opd != illuminated_opd[0]):
            if config.static_opd_sampling is None:
                raise ValueError("Varying static OPD requires static_opd_sampling")
            illuminated_opd -= illuminated_opd[0]
            illuminated_opd -= np.average(
                illuminated_opd, weights=amplitude[illuminated] ** 2
            )
            phase = np.zeros_like(opd)
            phase[illuminated] = 2 * np.pi * illuminated_opd / wavelength.to_value(u.m)
        if config.atmosphere is not None:
            images, full_flux = predict_common(
                config,
                period,
                wavelength.to_value(u.m),
                int(image_side),
                phase,
            )
        elif phase is None:
            images, full_flux = _diffraction_image(
                config.pupil.amplitude,
                config.pupil.pitch.to_value(u.m),
                period,
                int(image_side),
                config.detector_origin,
            )
        else:
            images, full_flux = static_diffraction(
                config, phase, period, int(image_side)
            )
        psfs = np.frombuffer(images.tobytes(), dtype=images.dtype).reshape(images.shape)
        if psfs.ndim == 2:
            psfs = np.broadcast_to(psfs, (len(sources.directions), *psfs.shape))
        return HoPsfPrediction(
            positions=sources.directions,
            psfs=psfs,
            metadata=PsfMetadata(
                wavelength=wavelength,
                pixel_scale=pixel_scale,
                tel_diameter=config.telescope.diameter,
                tel_pupil=u.Quantity(config.pupil.amplitude, u.one, copy=False),
            ),
            full_flux=full_flux,
        )
