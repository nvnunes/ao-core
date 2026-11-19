"""Sampling uncertainty of mean PSF metrics on a configured adaptive-optics field."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from copy import copy
from dataclasses import dataclass, replace
from functools import partial
from math import ceil
from types import MappingProxyType

import numpy as np
from astropy import units as u
from scipy.linalg import block_diag

from ._compact_optics import CompactScienceOptics
from ._exposure import atmospheric_lags, integrate_metric_lags
from ._field_covariance import _required
from ._metric_covariance import (
    LEVELS,
    CPUCompactFastBatch,
    metric_lag_indices,
    predict_metric_lags,
)
from ._sampling_grid import SamplingSettings
from ._units import _PhysicalRecord, _quantity, validate_float_dtype
from .config import Configuration
from .control import integrator_noise_lags
from .pupil import ARCSEC, make_pupil, optical_pupil
from .residual import FieldCovariance
from .sensing import (
    centroid_noise_prior,
    diffraction_calibration,
    optical_mean,
    sensor_response,
)

__all__ = [
    "FastSettings",
    "SamplingSettings",
    "SamplingUncertaintyModel",
    "UncertaintyDiagnostics",
    "UncertaintyMetricPrediction",
    "UncertaintyModelSettings",
    "UncertaintyPrediction",
    "UncertaintyRequest",
    "gaussian_ngs_noise_covariance",
]

_METRIC_UNITS = {"sr": u.one, "ee100": u.one, "fwhm_mas": u.mas}
_MAS_PER_RADIAN = 180 / np.pi * 3600000
_LAG_BATCH_BYTES = 256 * 1024**2
_MAX_FAST_OPTICS = 8
MAX_MPS_EXPOSURE = 10.0
_SCIENCE_ANGULAR_OFFSET = 0.5


class UncertaintyModelSettings:
    """Base type for options specific to an uncertainty model."""

    __slots__ = ()


@dataclass(frozen=True, kw_only=True)
class FastSettings(UncertaintyModelSettings):
    """Optional Fast settings.

    Attributes:
        remove_science_tilt: Remove science tilt from the mean PSF and its
            fluctuations.
    """

    remove_science_tilt: bool = False

    def __post_init__(self):
        if not isinstance(self.remove_science_tilt, bool):
            raise TypeError("remove_science_tilt must be a boolean")


@dataclass(frozen=True, kw_only=True, eq=False)
class UncertaintyRequest(_PhysicalRecord):
    """Predict the requested exposures in the supplied field and window order.

    Positions are (point, 2) angular x/y pairs. Exposures are a vector of
    whole-frame durations. The science wavelength and pupil geometry
    determine the PSF pixel scale.

    Fast uses finite-order covariance, Balanced retains all orders of the
    modal phase, and Full includes covariance of the restored pupil phase.
    All three use automatic numerical grids, the restored mean PSF, and
    source-centered EE100.

    Attributes:
        config: Shared AO system, resolved pupil, NGSs and sampling settings.
        ngs_noise_covariance: Pre-filter angular measurement-noise covariance
            used for the LO reconstructor weights. Axes are interleaved x/y
            in configured NGS order. Required for multiple guide stars.
        positions: Ordered field coordinates as angular x/y pairs.
        exposures: Ordered positive whole-frame exposure durations.
        wavelength: Science wavelength.
        model_type: Fast, Balanced, or Full.
        model_settings: Optional FastSettings for Fast. Balanced and Full
            use None.
    """

    config: Configuration
    ngs_noise_covariance: u.Quantity | None = None
    wavelength: u.Quantity
    positions: u.Quantity
    exposures: u.Quantity
    model_type: str = "fast"
    model_settings: UncertaintyModelSettings | None = None

    def __post_init__(self):
        if not isinstance(self.config, Configuration):
            raise TypeError("config must be a Configuration")
        if not self.config.ngs:
            raise ValueError("Sampling requires at least one configured NGS")
        if self.config.pupil is None:
            raise ValueError("Sampling requires a resolved pupil")
        _required(self.config.telescope, "Telescope", "diameter")
        noise = self.ngs_noise_covariance
        if noise is None:
            if len(self.config.ngs) > 1:
                raise ValueError("Multiple guide stars require ngs_noise_covariance")
        else:
            noise = _quantity(noise, u.mas**2, "ngs_noise_covariance", 2)
            channels = 2 * len(self.config.ngs)
            if noise.shape != (channels, channels) or not np.allclose(
                noise.value, noise.value.T
            ):
                raise ValueError(
                    "ngs_noise_covariance must be symmetric with two axes per NGS"
                )
            eigenvalues = np.linalg.eigvalsh(noise.value)
            if eigenvalues[0] < -1e-12 * max(1, eigenvalues[-1]):
                raise ValueError("ngs_noise_covariance must be positive semidefinite")
            object.__setattr__(self, "ngs_noise_covariance", noise)
        for key, unit, ndim in (
            ("positions", u.arcsec, 2),
            ("exposures", u.s, 1),
            ("wavelength", u.m, 0),
        ):
            object.__setattr__(
                self, key, _quantity(getattr(self, key), unit, key, ndim)
            )
        if self.positions.shape[1] != 2:
            raise ValueError("Require x/y pairs for positions")
        if not isinstance(self.model_type, str):
            raise TypeError("model_type must be a string")
        model_type = self.model_type.casefold()
        object.__setattr__(self, "model_type", model_type)
        if model_type not in LEVELS:
            raise ValueError(f"Unknown sampling model_type: {model_type}")
        if self.model_settings is not None:
            if model_type != "fast":
                raise TypeError(f"{model_type} requires model_settings=None")
            if type(self.model_settings) is not FastSettings:
                raise TypeError("fast requires FastSettings")
        dt = _sampling_frame_interval(self.config)
        counts = np.rint(self.exposures.value / dt)
        if np.any(counts < 1) or not np.allclose(
            counts * dt, self.exposures.value, rtol=0, atol=1e-12
        ):
            raise ValueError(
                "Exposures must contain a positive number of whole native frames"
            )
        self.config.sampling_uncertainty_model._science_nodes(
            float(self.exposures.value.max())
        )
        if self.wavelength.value <= 0:
            raise ValueError("Wavelength must be positive")


@dataclass(frozen=True, eq=False)
class UncertaintyMetricPrediction(_PhysicalRecord):
    """Mean, variance and sampling uncertainty coefficient for one PSF metric.

    The reference mean has one value per point. Variance and coefficient
    have shape (exposure, point) and use the same mean for every exposure.
    """

    mean: u.Quantity
    variance: u.Quantity
    coefficient: u.Quantity


@dataclass(frozen=True, eq=False)
class UncertaintyDiagnostics:
    """Temporal quadrature grids and grid-specific means in exposure order.

    Each grid contains the radial node count, angular node count and period
    in seconds. Each metric's diagnostic mean has shape (exposure, point).
    """

    grids: tuple[tuple[int, int, float], ...]
    mean_by_exposure: Mapping[str, u.Quantity]


@dataclass(frozen=True, eq=False)
class UncertaintyPrediction:
    """Predicted metric means, variances and uncertainty coefficients."""

    request: UncertaintyRequest
    metrics: Mapping[str, UncertaintyMetricPrediction]
    diagnostics: UncertaintyDiagnostics | None = None


def _sampling_frame_interval(config):
    ho = _required(config.high_order_loop, "High-order loop", "frame_rate")
    lo = _required(config.low_order_loop, "Low-order loop", "frame_rate")
    high = float(ho.frame_interval.to_value(u.s))
    low = float(lo.frame_interval.to_value(u.s))
    if high != low:
        raise ValueError(
            "The sampling uncertainty model requires equal HO and LO loop rates"
        )
    return high


def _project_sampling_configuration(config):
    """Resolve the observing geometry used by the sampling calculations."""
    atmosphere = _required(config.atmosphere, "Atmosphere", "reference_wavelength")
    fried = atmosphere.fried_parameter
    if atmosphere.seeing is not None:
        fried = (
            0.976 * atmosphere.reference_wavelength / atmosphere.seeing.to_value(u.rad)
        )
    if fried is None:
        raise ValueError("Sampling requires Seeing or a Fried parameter")
    _required(atmosphere, "Atmosphere", "layer_heights")
    high_order = _required(config.high_order_sources, "Beacon geometry", "height")
    angle = config.telescope.zenith_angle
    airmass = 1.0 if angle is None else 1 / np.cos(angle.to_value(u.rad))
    low_order = _required(config.low_order_sources, "Guide stars", "wavelengths")
    return replace(
        config,
        atmosphere=replace(
            atmosphere,
            fried_parameter=fried * airmass ** (-3 / 5),
            layer_heights=atmosphere.layer_heights * airmass,
        ),
        high_order_sources=replace(high_order, height=high_order.height * airmass),
        low_order_sources=replace(
            low_order, directions=u.Quantity([star.position for star in config.ngs])
        ),
    )


def _ngs_optics(config, side, modal_order, reference_side):
    if config.pupil is None:
        raise ValueError("NGS sensing requires a resolved pupil")
    telescope = _required(config.telescope, "Telescope", "diameter")
    low_order = _required(config.low_order_sources, "Guide stars", "wavelengths")
    if len(low_order.wavelengths) != 1:
        raise ValueError("Require one shared guide-star sensing wavelength")
    detector = _required(
        config.low_order_detector,
        "Guide-star detector",
        "pixel_scale",
        "read_noise",
        "threshold",
    )
    wave = float(low_order.wavelengths.to_value(u.m)[0])
    amplitude = config.pupil.amplitude
    optical = optical_pupil(
        amplitude,
        np.zeros_like(amplitude),
        float(config.pupil.span.to_value(u.m)),
        float(telescope.diameter.to_value(u.m)),
        wave,
        side,
        modal_order,
        reference_side=reference_side,
    )
    pixel = wave / (optical.optical_size * optical.dx) * _MAS_PER_RADIAN
    pitch = float(detector.pixel_scale.to_value(u.mas / u.pix))
    if optical.optical_size * pixel / 2 < 2.5 * pitch + 70:
        raise ValueError(
            "Optical field does not cover the displaced five-pixel detector"
        )
    return optical, pixel, pitch, wave


def _ngs_photons_per_frame(config):
    dt = _sampling_frame_interval(config)
    return np.array([star.flux.to_value(u.photon / u.s) * dt for star in config.ngs])


def gaussian_ngs_noise_covariance(
    config: Configuration, *, ee: u.Quantity, fwhm: u.Quantity
) -> u.Quantity:
    """Calculate angular design noise for the configured NGSs.

    Supply one encircled-energy fraction and angular FWHM per configured
    star. The caller chooses the measurement conventions. The Gaussian
    spot calculation uses each star's flux, the configured detector and
    the diffraction centroid calibration. Returns a covariance in mas²,
    with interleaved x/y axes in the configured star order.
    """
    fraction = _quantity(ee, u.one, "ee", 1).value
    width = _quantity(fwhm, u.mas, "fwhm", 1).value
    if not len(config.ngs) == len(fraction) == len(width):
        raise ValueError("EE and FWHM must match the configured NGSs")
    settings = config.sampling_uncertainty_model
    optical, pixel, pitch, _ = _ngs_optics(
        config,
        settings.sensing_pupil_samples,
        settings.zernike_radial_order,
        settings.modal_pupil_samples,
    )
    conversion, _ = diffraction_calibration(optical, pixel, pitch)
    detector = config.low_order_detector
    noise = []
    for photons, e, width_mas in zip(
        _ngs_photons_per_frame(config), fraction, width, strict=True
    ):
        prior = centroid_noise_prior(
            float(photons),
            float(e),
            float(width_mas),
            pitch,
            5,
            float(detector.threshold.to_value(u.electron)),
            float(detector.read_noise.to_value(u.electron)),
        )
        noise.append(conversion @ prior["covariance_pixel2"] @ conversion.T)
    return block_diag(*noise) * u.mas**2


def _prepare_sensing(
    configuration,
    wavelength,
    device,
    *,
    ngs_noise_covariance=None,
    grid_settings=None,
    sensor_workers=1,
):
    settings = configuration.sampling_uncertainty_model
    if grid_settings is None:
        physical_side = reference_side = settings.modal_pupil_samples
        modal_order = settings.zernike_radial_order
        sensor_side = settings.sensing_pupil_samples
        sensing_grid = (settings.sensing_radial_nodes, settings.sensing_angular_nodes)
    else:
        physical_side = grid_settings.physical_side
        reference_side = grid_settings._reference_side
        modal_order = grid_settings.modal_order
        sensor_side = grid_settings.sensor_side
        sensing_grid = grid_settings._sensing_grid
    pupil = make_pupil(
        configuration.pupil.amplitude,
        configuration.pupil.static_opd.to_value(u.nm),
        float(configuration.pupil.span.to_value(u.m)),
        float(configuration.telescope.diameter.to_value(u.m)),
        wavelength,
        physical_side,
        modal_order,
    )
    optical, pixel, pitch, sensing_wave = _ngs_optics(
        configuration, sensor_side, modal_order, reference_side
    )
    reference_wave = float(configuration.atmosphere.reference_wavelength.to_value(u.m))
    detector = configuration.low_order_detector
    propagation = None
    if device == "mps":
        from ._sensor_metal import MpsSensorPropagation

        propagation = MpsSensorPropagation(optical)
    conversion, _ = diffraction_calibration(
        optical, pixel, pitch, propagation=propagation
    )

    directions = configuration.low_order_sources.directions.to_value(u.rad)
    photons = _ngs_photons_per_frame(configuration)
    design = np.column_stack([pupil.coordinates, np.ones(len(pupil.coordinates))])
    coefficients = np.linalg.lstsq(design, pupil.basis[:2].T, rcond=None)[0]
    modal_to_angle = reference_wave / (2 * np.pi) * coefficients[:2] * _MAS_PER_RADIAN
    inverse_angle = np.linalg.inv(modal_to_angle)
    prior = None
    if ngs_noise_covariance is not None:
        transform = np.kron(np.eye(len(directions)), inverse_angle)
        prior = transform @ ngs_noise_covariance.to_value(u.mas**2) @ transform.T
    # The sensing grid uses unshifted polar rings. The science grid is staggered.
    field = FieldCovariance(
        configuration,
        pupil,
        wavelength * u.m,
        *sensing_grid,
        period=None,
        common_only=True,
    )
    gains, innovations, responses = [], [], []
    for count, theta in zip(photons, directions, strict=True):
        covariance = field.ngs_ho_covariance(theta) * (wavelength / sensing_wave) ** 2
        modes = range(2, len(pupil.basis))
        if propagation is None:
            image, derivatives = optical_mean(
                optical,
                covariance,
                modes,
                derivative_workers=sensor_workers,
            )
        else:
            image, derivatives = propagation.calculate(covariance, modes)
        gain, innovation, response, _ = sensor_response(
            image,
            derivatives,
            pixel,
            pitch,
            count,
            conversion,
            len(directions),
            read_noise_e=float(detector.read_noise.value),
            threshold_e=float(detector.threshold.value),
        )
        gains.append(inverse_angle @ gain @ np.linalg.inv(inverse_angle))
        innovations.append(inverse_angle @ innovation @ inverse_angle.T)
        responses.append(inverse_angle @ response * reference_wave / sensing_wave)
    return pupil, (
        prior,
        np.array(gains),
        block_diag(*innovations),
        np.array(responses),
    )


class _PreparedGrid:
    """Own the fixed AO grid and the reference wavelength's noise lags."""

    def __init__(
        self,
        configuration,
        wavelength,
        exposures,
        grid,
        sensing,
        *,
        compact=False,
        grid_settings=None,
        instantaneous=False,
    ):
        self.configuration = configuration
        self.pupil = configuration.pupil.amplitude
        self.physical, self.sensing = sensing
        self.radial_order, angular, period = grid
        self.grids = ((period, angular),)
        self.instantaneous = instantaneous
        if grid_settings is None:
            settings = configuration.sampling_uncertainty_model
            self.science_side = settings.science_pupil_samples
            self.modal_order = settings.zernike_radial_order
            self.reference_side = settings.modal_pupil_samples
            self.angular_offset = _SCIENCE_ANGULAR_OFFSET
            self.spectral_interval = None
        else:
            self.science_side = grid_settings.optical_side
            self.modal_order = grid_settings.modal_order
            self.reference_side = grid_settings._reference_side
            self.angular_offset = grid_settings.angular_offset
            self.spectral_interval = grid_settings.spectral_interval_s
        self.dt = _sampling_frame_interval(configuration)
        self.windows = {}
        for time in exposures:
            frames = round(time / self.dt)
            window = 1 - np.arange(frames) / frames
            window[0] = 0.5
            self.windows[time] = window
        self.max_count = max(map(len, self.windows.values()), default=1)
        self.compact = compact
        self._context = None
        self._noise_lags = None
        self._set_optics(wavelength)

    def _set_optics(self, wave):
        config = self.configuration
        support = float(config.pupil.span.to_value(u.m))
        diameter = float(config.telescope.diameter.to_value(u.m))
        self.optical = optical_pupil(
            self.pupil,
            config.pupil.static_opd.value,
            support,
            diameter,
            wave,
            self.science_side,
            self.modal_order,
            reference_side=self.reference_side,
        )

        self.metadata = {
            "support_m": support,
            "wavelength_m": wave,
            "pixel_scale_mas": wave
            / (self.optical.optical_size * self.optical.dx)
            * _MAS_PER_RADIAN,
        }

    def at_science_optics(self, wave):
        view = copy(self)
        view._set_optics(wave)
        if self._context is not None:
            ratio = self.metadata["wavelength_m"] / wave
            view._context = self._context.at_wavelength(wave * u.m)
            view._noise_lags = self._noise_lags * ratio**2
        return view

    def context(self, period, angular):
        if (period, angular) != self.grids[0]:
            raise ValueError("Fourier grid does not match the prepared request")
        if self._context is None:
            prior, gains, innovations, responses = self.sensing
            config = self.configuration
            wave = self.metadata["wavelength_m"]
            context = FieldCovariance(
                config,
                self.physical,
                wave * u.m,
                self.radial_order,
                angular,
                period,
                self.spectral_interval,
                instantaneous=self.instantaneous,
                tilt_noise_prior=prior,
                sensing_gains=gains,
                sensing_ho=responses,
                angular_offset=self.angular_offset,
                compact=self.compact,
            )
            r = context.tilt_reconstructor
            lo = config.low_order_loop
            self._noise_lags = (
                integrator_noise_lags(
                    context.tilt_feedback,
                    r @ innovations @ r.T,
                    float(lo.gain.value),
                    int(lo.delay_frames),
                    self.max_count,
                )
                * context.scale**2
            )
            self._context = context
        return self._context

    def point_lags(self, position, *, terms=None):
        period, angular = self.grids[0]
        context = self.context(period, angular)
        direction = np.asarray(position) * ARCSEC
        frames = round(period / self.dt)
        if self.max_count == 1:
            residual = context.residual_response(direction, terms=terms)
            weights = context.zero_lag_weights(frames)
            lags = ((residual * weights) @ residual.conj().T).real[None]
        else:
            spectrum = context.spectrum(direction, terms=terms)
            lags = atmospheric_lags(spectrum, frames, self.max_count)
        lags[:, :2, :2] += self._noise_lags
        return lags

    def modal_phase_lags(self, position, zero_lag, count, terms, *, remove_tilt):
        """Return native-frame modal covariance using the restored zero lag."""
        lags = zero_lag[None]
        if count > 1:
            lags = self.point_lags(position, terms=terms)[:count]
            if not np.allclose(lags[0], zero_lag, rtol=1e-8, atol=1e-11):
                raise ValueError(
                    "Temporal and restored modal covariance disagree at zero lag"
                )
            lags[0] = zero_lag
        if remove_tilt:
            lags[:, :2, :] = 0
            lags[:, :, :2] = 0
        return lags


class SamplingUncertaintyModel:
    """Predict finite-exposure sampling uncertainty on CPU or Metal.

    CPU calculations use float64. Metal uses float32 for its documented
    contractions and CPU float64 for preparation and fallback. An explicit
    dtype must match the device's supported contraction precision. Reuse one
    predictor sequentially to retain its most recent sensing and Fourier grid.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        batch_size: int = 4,
        dtype=None,
    ):
        if device not in {"cpu", "mps"}:
            raise ValueError("device must be cpu or mps")
        if type(batch_size) is not int or not 1 <= batch_size <= 8:
            raise ValueError("batch_size must be from one through eight")
        expected_dtype = np.dtype(np.float32 if device == "mps" else np.float64)
        if dtype is not None and validate_float_dtype(dtype) != expected_dtype:
            raise ValueError(f"The {device} contraction requires {expected_dtype.name}")
        self.device = device
        self.dtype = expected_dtype
        self.batch_size = batch_size
        self._sensing_key = None
        self._sensing = None
        self._grid_key = None
        self._prepared = None

    def predict_many(
        self, requests: Iterable[UncertaintyRequest], *, diagnostics: bool = False
    ) -> Iterator[UncertaintyPrediction]:
        """Yield predictions in the caller's request, point and exposure order.

        Requests sharing a configuration can use the same sensing and residual
        covariance. Request diagnostics to retain the temporal quadrature
        grids and their metric means.
        """
        requests = tuple(requests)
        configurations = defaultdict(list)
        results = {}
        for index, request in enumerate(requests):
            if not isinstance(request, UncertaintyRequest):
                raise TypeError("request must be an UncertaintyRequest")
            configurations[request.config, _noise_key(request)].append(index)
        for indices in configurations.values():
            group = [requests[index] for index in indices]
            predictions = self._predict_standard(group, diagnostics=diagnostics)
            results.update(zip(indices, predictions, strict=True))
        yield from (results[index] for index in range(len(requests)))

    def predict(
        self, request: UncertaintyRequest, *, diagnostics: bool = False
    ) -> UncertaintyPrediction:
        """Evaluate one request, optionally retaining grid diagnostics."""
        return next(self.predict_many((request,), diagnostics=diagnostics))

    def _predict_standard(self, requests, *, diagnostics):
        config = requests[0].config
        frame_interval = _sampling_frame_interval(config)
        atmosphere = _required(config.atmosphere, "Atmosphere", "wind_speeds")
        wind_speeds = atmosphere.wind_speeds.to_value(u.m / u.s)
        moving = wind_speeds[wind_speeds > 0]
        diameter = config.telescope.diameter.to_value(u.m)
        minimum_period_frames = (
            ceil(4 * (diameter / moving.min()) / frame_interval) if moving.size else 0
        )
        science_grid = partial(
            config.sampling_uncertainty_model._science_grid,
            frame_interval=frame_interval,
            minimum_period_frames=minimum_period_frames,
        )
        all_fast = all(request.model_type == "fast" for request in requests)
        dense_long_grids = {
            (_science_band(request), science_grid(exposure))
            for request in requests
            if request.model_type != "fast"
            for exposure in request.exposures.value
            if exposure > MAX_MPS_EXPOSURE
        }
        jobs = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
        buffers = []
        representatives = {}
        for index, request in enumerate(requests):
            grids = tuple(map(science_grid, request.exposures.value))
            shape = (len(grids), len(request.positions), len(_METRIC_UNITS))
            buffers.append(
                (np.empty(shape), (grids, np.empty(shape)) if diagnostics else None)
            )
            band = _science_band(request)
            representatives.setdefault(band, request)
            for exposure_index, grid in enumerate(grids):
                exposure = float(request.exposures.value[exposure_index])
                compact_fast = (
                    request.model_type == "fast"
                    and exposure > MAX_MPS_EXPOSURE
                    and (band, grid) not in dense_long_grids
                )
                accelerated = compact_fast or (all_fast and self.device == "mps")
                # Metal Fast and the largest CPU Fast grid reuse phase lags across bands.
                key = grid, accelerated, band[1] if accelerated else band
                for point_index, xy in enumerate(map(tuple, request.positions.value)):
                    jobs[key][band][xy][request.model_type, exposure].append(
                        (index, exposure_index, point_index)
                    )

        for (grid, accelerated, _), bands in jobs.items():
            if accelerated:
                batches = self._fast_band_batches(grid, bands, representatives)
            else:
                band = next(iter(bands))
                windows = _band_windows(bands)
                prepared, _, optics = self._prepare_grid(
                    representatives[band],
                    "cpu",
                    grid,
                    windows,
                    compact=False,
                )
                batches = (
                    (
                        prepared,
                        self._dense_points(
                            band,
                            bands[band],
                            optics,
                            _covariance_device(self.device, windows[-1]),
                        ),
                    ),
                )

            for prepared, points in batches:
                for point_band, xy, means, lags in points:
                    for (model, duration), indices in bands[point_band][xy].items():
                        integrated = integrate_metric_lags(
                            model, lags[model], prepared.windows[duration], prepared.dt
                        )
                        for index, exposure_index, point_index in indices:
                            field_integrated, detail = buffers[index]
                            field_integrated[exposure_index, point_index] = integrated
                            if detail is not None:
                                detail[1][exposure_index, point_index] = means
        reference = self._reference_means(requests)
        return [
            self._result(request, reference[_science_band(request)], *buffer)
            for request, buffer in zip(requests, buffers, strict=True)
        ]

    def _reference_means(self, requests):
        """Compute the zero-lag mean once per science band and field point."""
        bands = defaultdict(dict)
        for request in requests:
            points = bands[_science_band(request)]
            for position in map(tuple, request.positions.value):
                points.setdefault(position, request)

        first = requests[0]
        config, sensing = self._prepare_detector(first, self.device)
        settings = config.sampling_uncertainty_model
        grid = (
            settings.science_min_radial_nodes,
            settings.science_min_angular_nodes,
            None,
        )
        prepared = _PreparedGrid(
            config,
            float(first.wavelength.value),
            (),
            grid,
            sensing,
            instantaneous=True,
            compact=True,
        )
        means = {}
        for band, points in bands.items():
            wave, _ = band
            view = (
                prepared
                if wave == first.wavelength.value
                else prepared.at_science_optics(wave)
            )
            optics = self._optics(view, next(iter(points.values())))
            means[band] = {
                position: optics.mean_metrics(position) for position in points
            }
        return means

    @staticmethod
    def _dense_points(band, points, optics, covariance_device):
        prepared = optics.predictor
        for xy, users in points.items():
            frame_counts = defaultdict(list)
            for model, duration in users:
                frame_counts[model].append(len(prepared.windows[duration]))
            indices = metric_lag_indices(frame_counts)
            structure, modal_zero_lag = optics.restore_phase(xy)
            means, kernels, coherence = optics.metric_response(structure)

            terms = None
            if any(lags[-1] > 0 for lags in indices.values()):
                terms = optics.science.context.direction_terms(np.asarray(xy) * ARCSEC)
            modal_frames = max(
                (
                    indices[model][-1] + 1
                    for model in ("fast", "balanced")
                    if model in indices
                ),
                default=1,
            )
            modal_lags = prepared.modal_phase_lags(
                xy,
                modal_zero_lag,
                modal_frames,
                terms,
                remove_tilt=optics.tilt is not None,
            )
            pupil_lags = (
                optics.pupil_phase_lags(structure, terms, indices["full"])
                if "full" in indices
                else None
            )
            lags = predict_metric_lags(
                indices,
                modal_lags,
                pupil_lags,
                optics.pupil.basis,
                kernels,
                coherence,
                device=covariance_device,
            )
            yield band, xy, means, lags

    @staticmethod
    def _result(request, reference, integrated, detail):
        means = np.array(
            [reference[position] for position in map(tuple, request.positions.value)]
        )
        if (
            np.any(means <= 0)
            or np.any(integrated < 0)
            or not np.isfinite(means).all()
            or not np.isfinite(integrated).all()
        ):
            raise ValueError("Invalid predicted metric covariance or mean")

        def quantities(index, unit):
            mean = means[:, index]
            variance = integrated[..., index] / request.exposures.value[:, None]
            relative = np.sqrt(integrated[..., index]) / mean
            return UncertaintyMetricPrediction(
                _quantity(mean * unit, unit, "mean", 1),
                _quantity(variance * unit**2, unit**2, "variance", 2),
                _quantity(relative * u.s**0.5, u.s**0.5, "coefficient", 2),
            )

        metrics = MappingProxyType(
            {
                name: quantities(i, unit)
                for i, (name, unit) in enumerate(_METRIC_UNITS.items())
            }
        )
        diagnostics = None
        if detail is not None:
            grids, grid_means = detail
            diagnostics = UncertaintyDiagnostics(
                grids,
                MappingProxyType(
                    {
                        name: _quantity(
                            grid_means[..., i] * unit, unit, "mean_by_exposure", 2
                        )
                        for i, (name, unit) in enumerate(_METRIC_UNITS.items())
                    }
                ),
            )
        return UncertaintyPrediction(request, metrics, diagnostics)

    def _prepare_detector(self, request, device):
        configuration = request.config
        sensing_key = (configuration, device, _noise_key(request))
        if self._sensing_key != sensing_key:
            self._prepared = self._grid_key = None
            projected = _project_sampling_configuration(configuration)
            sensing = _prepare_sensing(
                projected,
                float(request.wavelength.value),
                device,
                ngs_noise_covariance=request.ngs_noise_covariance,
            )
            self._sensing = projected, sensing
            self._sensing_key = sensing_key
        return self._sensing

    def _prepare_grid(self, request, device, grid, windows, *, compact):
        projected, sensing = self._prepare_detector(request, device)
        wavelength = float(request.wavelength.value)
        sensing_key = self._sensing_key
        key = (sensing_key, grid, windows, compact)
        if key != self._grid_key:
            self._prepared = self._grid_key = None
            if device == "mps":
                import torch

                torch.mps.empty_cache()
            prepared = _PreparedGrid(
                projected, wavelength, windows, grid, sensing, compact=compact
            )
            optics = self._optics(prepared, request)
            engine = self._engine(device, compact, prepared, optics)
            band = _science_band(request)
            self._prepared = (prepared, engine, band, optics)
            self._grid_key = key
        prepared, engine, band, optics = self._prepared
        wave, tilt = _science_band(request)
        if (wave, tilt) != band:
            view = prepared.at_science_optics(wave)
            optics = self._optics(view, request)
        return prepared, engine, optics

    @staticmethod
    def _optics(prepared, request):
        return CompactScienceOptics(
            prepared,
            remove_science_tilt=_remove_science_tilt(request.model_settings),
        )

    @staticmethod
    def _engine(device, compact, prepared, optics):
        if device == "cpu":
            return CPUCompactFastBatch(optics) if compact else None
        from ._fast_metal import MPSFastBatch

        return MPSFastBatch(optics.science, prepared)

    def _fast_band_batches(self, grid, bands, representatives):
        keys = tuple(bands)
        for start in range(0, len(keys), _MAX_FAST_OPTICS):
            chunk = {key: bands[key] for key in keys[start : start + _MAX_FAST_OPTICS]}
            windows = _band_windows(chunk)
            duration = windows[-1]
            yield self._fast_points(
                grid,
                windows,
                chunk,
                representatives,
                _covariance_device(self.device, duration),
                compact=duration > MAX_MPS_EXPOSURE,
            )

    def _fast_points(self, grid, windows, bands, representatives, device, *, compact):
        optics = {}
        users = defaultdict(list)
        for band, points in bands.items():
            prepared, engine, optics[band] = self._prepare_grid(
                representatives[band], device, grid, windows, compact=compact
            )
            for xy in points:
                users[xy].append(band)
        positions = tuple(users)
        size = min(2, self.batch_size)
        pipeline = False
        if device == "mps":
            backend = engine.backend
            lag_bytes = 4 * backend.nmodes**2 * backend.retained
            response_bytes = 8 * backend.nmodes * len(backend.spatial)
            size = min(
                self.batch_size,
                max(1, _LAG_BATCH_BYTES // max(lag_bytes, response_bytes)),
            )
            pipeline = (
                len(positions) > size and size * 2 * lag_bytes <= _LAG_BATCH_BYTES
            )
        batches = [positions[i : i + size] for i in range(0, len(positions), size)]
        results = (
            (band, xy, means, {"fast": lags})
            for rows, _, _ in engine.evaluate_many(
                batches, optics, users, remove_tilt=band[1], pipeline=pipeline
            )
            for band, xy, means, lags in rows
        )
        return prepared, results


def _covariance_device(device, exposure):
    """Keep the largest Fast grid on the bounded CPU path."""
    return "cpu" if exposure > MAX_MPS_EXPOSURE else device


def _noise_key(request):
    noise = request.ngs_noise_covariance
    return None if noise is None else noise.value.tobytes()


def _remove_science_tilt(settings):
    return isinstance(settings, FastSettings) and settings.remove_science_tilt


def _science_band(request):
    return float(request.wavelength.value), _remove_science_tilt(request.model_settings)


def _band_windows(bands):
    return tuple(
        sorted(
            {
                time
                for points in bands.values()
                for users in points.values()
                for _, time in users
            }
        )
    )
