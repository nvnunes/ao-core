"""Sampling uncertainty of mean PSF metrics on a configured adaptive-optics field."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from copy import copy
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from astropy import units as u
from scipy.linalg import block_diag

from ._all_order import SOBOL_POWER, SOBOL_REPLICATES, SOBOL_SEED
from ._compact_optics import (
    LEVELS,
    MAX_EXPOSURE,
    MAX_MPS_EXPOSURE,
    CompactScienceOptics,
    CPUCompactFastBatch,
    fast_grid_device,
    science_grid,
)
from ._explicit_optics import ExplicitScienceOptics
from ._exposure import atmospheric_lags, integrate_metric_lags
from ._field_covariance import _required
from ._units import _count, _PhysicalRecord, _quantity, validate_float_dtype
from .config import AOConfiguration
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
    "SamplingConfiguration",
    "SamplingMetric",
    "SamplingNumerics",
    "SamplingPredictor",
    "SamplingRequest",
    "SamplingResult",
]

_METRIC_UNITS = {"sr": u.one, "ee100": u.one, "fwhm_mas": u.mas}
_MAS_PER_RADIAN = 180 / np.pi * 3600000
_LAG_BATCH_BYTES = 256 * 1024**2
_MAX_FAST_OPTICS = 8
_EXPLICIT_MODELS = ("limited-order", "all-order", "full-pupil")


@dataclass(frozen=True, kw_only=True, eq=False)
class SamplingConfiguration(_PhysicalRecord):
    """Physical state shared by sampling predictions across field and wavelength.

    Supply telescope diameter, layered atmosphere, fitting field, projected
    LGS beacon height, and the HO and LO controllers through `physical`.
    The low-order detector supplies pixel pitch, incident photons per frame,
    read noise and threshold for the design prior. The NGS source record
    supplies positions and the sensing wavelength. The runtime detector
    statistics use the explicit runtime read noise and threshold.
    """

    physical: AOConfiguration
    pupil_amplitude: np.ndarray
    pupil_support: u.Quantity
    static_opd: u.Quantity
    guide_star_ee: u.Quantity
    guide_star_fwhm: u.Quantity
    frame_interval: u.Quantity
    runtime_read_noise: u.Quantity = 0.5 * u.electron
    runtime_threshold: u.Quantity = 0.5 * u.electron

    def __post_init__(self):
        if not isinstance(self.physical, AOConfiguration):
            raise TypeError("physical must be an AOConfiguration")
        for key, unit, ndim in (
            ("pupil_support", u.m, 0),
            ("static_opd", u.nm, 2),
            ("guide_star_ee", u.one, 1),
            ("guide_star_fwhm", u.mas, 1),
            ("frame_interval", u.s, 0),
            ("runtime_read_noise", u.electron, 0),
            ("runtime_threshold", u.electron, 0),
        ):
            object.__setattr__(
                self, key, _quantity(getattr(self, key), unit, key, ndim)
            )
        pupil = np.asarray(self.pupil_amplitude)
        if (
            pupil.ndim != 2
            or pupil.shape != self.static_opd.shape
            or pupil.shape[0] != pupil.shape[1]
            or pupil.shape[0] < 3
            or pupil.dtype.kind not in "fiu"
            or not np.isfinite(pupil).all()
            or np.any(pupil < 0)
            or not np.any(pupil > 0)
        ):
            raise ValueError(
                "Require a finite nonnegative square pupil matching static_opd"
            )
        object.__setattr__(
            self,
            "pupil_amplitude",
            np.frombuffer(pupil.tobytes(), dtype=pupil.dtype).reshape(pupil.shape),
        )
        if (
            min(
                self.frame_interval.value,
                self.pupil_support.value,
                self.runtime_read_noise.value,
            )
            <= 0
        ):
            raise ValueError(
                "Require positive pupil support, frame interval and read noise"
            )

        physical = self.physical
        _required(physical.telescope, "Telescope", "diameter")
        ngs = _required(
            physical.low_order_sources, "Guide stars", "directions", "wavelengths"
        )
        detector = _required(
            physical.low_order_detector,
            "Guide-star detector",
            "pixel_scale",
            "read_noise",
            "threshold",
            "photons_per_frame",
        )
        if len(ngs.wavelengths) != 1:
            raise ValueError("Require one shared guide-star sensing wavelength")
        if not (
            len(ngs.directions)
            == len(detector.photons_per_frame)
            == len(self.guide_star_ee)
            == len(self.guide_star_fwhm)
        ):
            raise ValueError("Flux, EE and FWHM must match the ordered guide stars")
        if np.any(self.guide_star_ee.value <= 0) or np.any(
            self.guide_star_ee.value > 1
        ):
            raise ValueError("Guide-star EE must be a positive fraction")
        if np.any(self.guide_star_fwhm.value <= 0):
            raise ValueError("Guide-star FWHM must be positive")
        for loop in (physical.high_order_loop, physical.low_order_loop):
            _required(loop, "Controller", "gain", "delay_frames", "frame_rate")
            if not np.isclose(
                loop.frame_interval.to_value(u.s),
                self.frame_interval.value,
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError(
                    "The guide-star loops must match the native frame interval"
                )


@dataclass(frozen=True, kw_only=True)
class SamplingNumerics:
    """Explicit pupil and Fourier sampling for peak-centered metric measurements.

    The spectral period must span at least twice the longest requested
    exposure. Limited order can use the modal or restored mean. All order
    and full pupil use the restored mean for one native frame.

    Attributes:
        physical_side: Pupil samples along one side for modal projection.
        optical_side: Pupil samples along one side for science optics.
        sensor_side: Pupil samples along one side for guide-star optics.
        modal_order: Maximum radial modal order, one of 5, 7, 9 or 11.
        radial_order: Radial quadrature order for the modal grid.
        angular_order: Azimuthal quadrature order for the science grid.
        spectral_period_s: Fourier series period in seconds.
        spectral_interval_s: Temporal sampling interval in seconds that sets the spectral Nyquist limit.
        angular_offset: Science ring offset in azimuthal grid steps.
        sensor_angular_order: Azimuthal quadrature order for sensing.
        sensor_workers: Independent workers for sensor optical derivatives.
        full_radial_order: Radial order of the explicit full-pupil grid.
        tail_order: Limited-order Gaussian expansion order, either 4 or 6.
        restore_mean: Use the restored mean in the limited-order calculation.
        sobol_power: Base-2 logarithm of the number of quartets per scramble.
        sobol_replicates: Number of independent scrambles.
        sobol_seed: Reproducible scramble seed.
        sobol_workers: Independent workers for the Sobol scrambles.
    """

    physical_side: int
    optical_side: int
    sensor_side: int
    modal_order: int
    radial_order: int
    angular_order: int
    spectral_period_s: float
    spectral_interval_s: float
    angular_offset: float = 0.0
    sensor_angular_order: int = 48
    sensor_workers: int = 1
    full_radial_order: int = 160
    tail_order: int = 6
    restore_mean: bool = False
    sobol_power: int = SOBOL_POWER
    sobol_replicates: int = SOBOL_REPLICATES
    sobol_seed: int = SOBOL_SEED
    sobol_workers: int = 1

    def __post_init__(self):
        for name, minimum in (
            ("physical_side", 3),
            ("optical_side", 3),
            ("sensor_side", 3),
            ("radial_order", 1),
            ("angular_order", 1),
            ("sensor_angular_order", 1),
            ("full_radial_order", 1),
            ("sobol_power", 10),
            ("sobol_replicates", 4),
            ("sobol_seed", 0),
            ("modal_order", 1),
            ("tail_order", 1),
            ("sobol_workers", 1),
            ("sensor_workers", 1),
        ):
            _count(getattr(self, name), name, minimum)
        if (
            self.modal_order not in (5, 7, 9, 11)
            or self.tail_order not in (4, 6)
            or not 1 <= self.sobol_workers <= 4
            or not 1 <= self.sensor_workers <= 4
        ):
            raise ValueError("Invalid modal order, tail order or worker count")
        times = (self.spectral_period_s, self.spectral_interval_s)
        if not np.isfinite((*times, self.angular_offset)).all() or min(times) <= 0:
            raise ValueError(
                "Require a finite angular offset and positive spectral times"
            )


@dataclass(frozen=True, kw_only=True, eq=False)
class SamplingRequest(_PhysicalRecord):
    """Predict the requested exposures in the supplied field and window order.

    Positions are (point, 2) angular x/y pairs. Exposures are a vector of
    whole-frame durations. Science wavelength and pixel scale specify the
    PSF used for the observables. Fast uses finite-order covariance, Balanced
    retains all orders of the modal phase, and Full includes covariance of the
    restored pupil phase. All use the same restored mean PSF.
    """

    configuration: SamplingConfiguration
    positions: u.Quantity
    exposures: u.Quantity
    wavelength: u.Quantity
    pixel_scale: u.Quantity
    model: str = "fast"
    remove_science_tilt: bool = False
    numerics: SamplingNumerics | None = None

    def __post_init__(self):
        if not isinstance(self.configuration, SamplingConfiguration):
            raise TypeError("configuration must be a SamplingConfiguration")
        for key, unit, ndim in (
            ("positions", u.arcsec, 2),
            ("exposures", u.s, 1),
            ("wavelength", u.m, 0),
            ("pixel_scale", u.mas / u.pix, 0),
        ):
            object.__setattr__(
                self, key, _quantity(getattr(self, key), unit, key, ndim)
            )
        if self.positions.shape[1] != 2:
            raise ValueError("Require x/y pairs for positions")
        if not isinstance(self.model, str):
            raise TypeError("model must be a string")
        object.__setattr__(self, "model", self.model.casefold())
        if self.numerics is None:
            if self.model not in LEVELS:
                raise ValueError(
                    "Use Fast, Balanced or Full for automatic numerical grids"
                )
        elif not isinstance(self.numerics, SamplingNumerics):
            raise TypeError("numerics must be a SamplingNumerics")
        elif self.model not in _EXPLICIT_MODELS:
            raise ValueError(
                "Use limited-order, all-order or full-pupil with an explicit grid"
            )
        if self.remove_science_tilt and self.model != "fast":
            raise ValueError("Science tilt removal requires the Fast model")
        dt = self.configuration.frame_interval.value
        counts = np.rint(self.exposures.value / dt)
        if np.any(counts < 1) or not np.allclose(
            counts * dt, self.exposures.value, rtol=0, atol=1e-12
        ):
            raise ValueError(
                "Exposures must contain a positive number of whole native frames"
            )
        if self.numerics is None and max(self.exposures.value) > MAX_EXPOSURE:
            raise ValueError(
                "Automatic numerical grids cover exposures through 20 seconds"
            )
        if self.numerics is not None:
            if 2 * max(self.exposures.value) > self.numerics.spectral_period_s:
                raise ValueError(
                    "Spectral period must cover at least two full exposures"
                )
            if self.model != "limited-order" and np.any(counts != 1):
                raise ValueError(
                    "Explicit all-order and full-pupil models require one native frame"
                )
        if min(self.wavelength.value, self.pixel_scale.value) <= 0:
            raise ValueError("Wavelength and pixel scale must be positive")


@dataclass(frozen=True, eq=False)
class SamplingMetric(_PhysicalRecord):
    """Mean, one-realization variance and relative coefficient for one metric.

    The mean has one value per point for the first requested exposure.
    The per-exposure mean, variance and coefficient have shape (exposure,
    point). The per-exposure mean retains differences when exposure windows
    select different Fourier grids.
    """

    mean: u.Quantity
    variance: u.Quantity
    coefficient: u.Quantity
    mean_by_exposure: u.Quantity


@dataclass(frozen=True, eq=False)
class SamplingResult:
    """Sampling metrics and the grids used for the associated request."""

    request: SamplingRequest
    metrics: Mapping[str, SamplingMetric]
    grids: tuple[tuple[int, int, float], ...]


def _prepare_sensing(configuration, wavelength, device, numerics=None):
    physical = configuration.physical
    telescope = physical.telescope
    amplitude = configuration.pupil_amplitude
    opd = configuration.static_opd.value
    support = float(configuration.pupil_support.value)
    diameter = float(telescope.diameter.to_value(u.m))
    physical_side, sensor_side, modal_order = (
        (64, 96, 11)
        if numerics is None
        else (numerics.physical_side, numerics.sensor_side, numerics.modal_order)
    )
    pupil = make_pupil(
        amplitude, opd, support, diameter, wavelength, physical_side, modal_order
    )
    ngs = physical.low_order_sources
    sensing_wave = float(ngs.wavelengths.to_value(u.m)[0])
    reference_wave = float(physical.atmosphere.reference_wavelength.to_value(u.m))
    optical = optical_pupil(
        amplitude,
        np.zeros_like(opd),
        support,
        diameter,
        sensing_wave,
        sensor_side,
        modal_order,
    )
    pixel = sensing_wave / (optical.optical_size * optical.dx) * _MAS_PER_RADIAN
    detector = physical.low_order_detector
    pitch = float(detector.pixel_scale.to_value(u.mas / u.pix))
    if optical.optical_size * pixel / 2 < 2.5 * pitch + 70:
        raise ValueError(
            "Optical field does not cover the displaced five-pixel detector"
        )
    propagation = None
    if device == "mps":
        from ._sensor_metal import MpsSensorPropagation

        propagation = MpsSensorPropagation(optical)
    conversion, _ = diffraction_calibration(
        optical, pixel, pitch, propagation=propagation
    )

    directions = ngs.directions.to_value(u.rad)
    photons = detector.photons_per_frame.to_value(u.photon)
    ee = configuration.guide_star_ee.value
    fwhm = configuration.guide_star_fwhm.value
    angular = []
    for count, fraction, width in zip(photons, ee, fwhm, strict=True):
        prior = centroid_noise_prior(
            float(count),
            float(fraction),
            float(width),
            pitch,
            5,
            float(detector.threshold.to_value(u.electron)),
            float(detector.read_noise.to_value(u.electron)),
        )
        angular.append(prior["covariance_pixel2"] * pitch**2)
    design = np.column_stack([pupil.coordinates, np.ones(len(pupil.coordinates))])
    coefficients = np.linalg.lstsq(design, pupil.basis[:2].T, rcond=None)[0]
    modal_to_angle = reference_wave / (2 * np.pi) * coefficients[:2] * _MAS_PER_RADIAN
    inverse_angle = np.linalg.inv(modal_to_angle)
    prior = block_diag(
        *(inverse_angle @ covariance @ inverse_angle.T for covariance in angular)
    )
    prior *= float(conversion[0, 0] / pitch) ** 2
    # The sensing grid uses unshifted polar rings. The science grid is staggered.
    grid = (
        (40, 48, 20.0, 0.01)
        if numerics is None
        else (
            numerics.radial_order,
            numerics.sensor_angular_order,
            numerics.spectral_period_s,
            numerics.spectral_interval_s,
        )
    )
    field = FieldCovariance(
        physical,
        pupil,
        wavelength * u.m,
        *grid,
        tilt_noise_prior=prior,
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
                derivative_workers=1 if numerics is None else numerics.sensor_workers,
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
            read_noise_e=float(configuration.runtime_read_noise.value),
            threshold_e=float(configuration.runtime_threshold.value),
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

    def __init__(self, configuration, request, exposures, grid, sensing, *, compact):
        self.configuration = configuration
        self.pupil = configuration.pupil_amplitude
        self.physical, self.sensing = sensing
        self.radial_order, angular, period = grid
        self.grids = ((period, angular),)
        self.numerics = request.numerics
        self.instantaneous = request.model == "full-pupil"
        self.angular_offset = (
            0.5 if self.numerics is None else self.numerics.angular_offset
        )
        self.dt = float(configuration.frame_interval.value)
        self.windows = {}
        for time in exposures:
            frames = round(time / self.dt)
            window = 1 - np.arange(frames) / frames
            window[0] = 0.5
            self.windows[time] = window
        self.max_count = max(map(len, self.windows.values()))
        self.compact = compact
        self._context = None
        self._noise_lags = None
        self._set_optics(
            float(request.wavelength.value), float(request.pixel_scale.value)
        )

    def _set_optics(self, wave, pixel):
        config = self.configuration
        support = float(config.pupil_support.value)
        self.metadata = {
            "support_m": support,
            "wavelength_m": wave,
            "pixel_scale_mas": pixel,
        }
        diameter = float(config.physical.telescope.diameter.to_value(u.m))
        side, order = (
            (64, 11)
            if self.numerics is None
            else (self.numerics.optical_side, self.numerics.modal_order)
        )
        self.optical = optical_pupil(
            self.pupil, config.static_opd.value, support, diameter, wave, side, order
        )

    def at_science_optics(self, wave, pixel):
        view = copy(self)
        view._set_optics(wave, pixel)
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
            config = self.configuration.physical
            wave = self.metadata["wavelength_m"]
            context = FieldCovariance(
                config,
                self.physical,
                wave * u.m,
                self.radial_order,
                angular,
                period,
                0.01 if self.numerics is None else self.numerics.spectral_interval_s,
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


class SamplingPredictor:
    """Predict sampling error using CPU or Metal with bounded field batches.

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
        pipeline: bool = True,
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
        self.pipeline = pipeline
        self._sensing_key = None
        self._sensing = None
        self._grid_key = None
        self._prepared = None

    def predict_many(
        self, requests: Iterable[SamplingRequest]
    ) -> Iterator[SamplingResult]:
        """Yield predictions in the caller's request, point and exposure order.

        Requests sharing a configuration can use the same sensing and residual
        covariance. Balanced and Full share a Sobol importance distribution
        when configuration, wavelength, pixel scale, point and grid match.
        """
        requests = tuple(requests)
        configurations = defaultdict(list)
        results = {}
        for index, request in enumerate(requests):
            if not isinstance(request, SamplingRequest):
                raise TypeError("request must be a SamplingRequest")
            if request.numerics is None:
                configurations[request.configuration].append(index)
            else:
                results[index] = self._predict_explicit(request)
        for indices in configurations.values():
            group = [requests[index] for index in indices]
            predictions = self._predict_standard(group)
            results.update(zip(indices, predictions, strict=True))
        yield from (results[index] for index in range(len(requests)))

    def predict(self, request: SamplingRequest) -> SamplingResult:
        """Evaluate one request using the prepared detector and Fourier grid."""
        return next(self.predict_many((request,)))

    def predict_fast_means(self, request: SamplingRequest) -> Mapping[str, u.Quantity]:
        """Evaluate the restored Fast metric means without temporal contractions.

        A single exposure selects the same Fourier grid as a Fast prediction.
        The returned unit-aware arrays follow the requested point order.
        """
        if not isinstance(request, SamplingRequest):
            raise TypeError("request must be a SamplingRequest")
        if request.model != "fast" or len(request.exposures) != 1:
            raise ValueError("Mean prediction requires one exposure on the Fast model")
        duration = float(request.exposures.value[0])
        device = fast_grid_device(self.device, duration)
        prepared, engine, optics = self._prepare_grid(
            request,
            device,
            science_grid(duration),
            (duration,),
            compact=duration > MAX_MPS_EXPOSURE,
        )
        points = [tuple(point) for point in request.positions.value]
        if device == "mps":
            backend = engine.backend
            response_bytes = 8 * backend.nmodes * len(backend.spatial)
            size = min(self.batch_size, max(1, _LAG_BATCH_BYTES // response_bytes))
            scale = prepared.metadata["wavelength_m"] / request.wavelength.value
            means = np.array(
                [
                    mean
                    for start in range(0, len(points), size)
                    for mean in engine.means(
                        points[start : start + size], optics, phase_scale=scale
                    )
                ]
            )
        else:
            means = np.array(
                [
                    optics.metric_response(
                        optics.science.tiled_phase_structures([xy])[0][0],
                        means_only=True,
                    )[0]
                    for xy in points
                ]
            )
        return MappingProxyType(
            {
                name: _quantity(means[:, i] * unit, unit, name, 1)
                for i, (name, unit) in enumerate(_METRIC_UNITS.items())
            }
        )

    def _predict_explicit(self, request):
        numerics = request.numerics
        radial = (
            numerics.full_radial_order
            if request.model == "full-pupil"
            else numerics.radial_order
        )
        grid = (radial, numerics.angular_order, numerics.spectral_period_s)
        windows = tuple(dict.fromkeys(map(float, request.exposures.value)))
        if self.device == "mps" and (
            numerics.restore_mean or request.model != "limited-order"
        ):
            raise ValueError(
                "Explicit grids with restored science coherence require CPU"
            )
        _prepared, engine, optics = self._prepare_grid(
            request, self.device, grid, windows, compact=False
        )
        points = list(map(tuple, request.positions.value))
        unique = tuple(dict.fromkeys(points))
        if engine is None:
            points_by_id = {xy: optics.point(xy) for xy in unique}
        else:
            points_by_id = {}
            for start in range(0, len(unique), self.batch_size):
                batch = unique[start : start + self.batch_size]
                lags = engine.point_lags(np.asarray(batch) * ARCSEC)
                responses = [
                    optics.modal_response(value) for value in engine.covariances(lags)
                ]
                estimates = engine.integrate(lags, responses)
                points_by_id.update(
                    (
                        xy,
                        (
                            response[0],
                            {exposure: values[i] for exposure, values in estimates},
                        ),
                    )
                    for i, (xy, response) in enumerate(
                        zip(batch, responses, strict=True)
                    )
                )
        shape = (len(request.exposures), len(points), len(_METRIC_UNITS))
        means = np.empty(shape)
        means[:] = [points_by_id[xy][0] for xy in points]
        integrated = np.array(
            [
                [points_by_id[xy][1][float(exposure)] for xy in points]
                for exposure in request.exposures.value
            ]
        )
        return self._result(
            request, (grid,) * len(request.exposures), means, integrated
        )

    def _predict_standard(self, requests):
        all_fast = all(request.model == "fast" for request in requests)
        dense_long_grids = {
            (_science_band(request), science_grid(exposure))
            for request in requests
            if request.model != "fast"
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
            buffers.append((grids, np.empty(shape), np.empty(shape)))
            band = _science_band(request)
            representatives.setdefault(band, request)
            for exposure_index, grid in enumerate(grids):
                exposure = float(request.exposures.value[exposure_index])
                compact_fast = (
                    request.model == "fast"
                    and exposure > MAX_MPS_EXPOSURE
                    and (band, grid) not in dense_long_grids
                )
                accelerated = compact_fast or (all_fast and self.device == "mps")
                # Metal Fast and the largest CPU Fast grid reuse phase lags across bands.
                key = grid, accelerated, band[2] if accelerated else band
                for point_index, xy in enumerate(map(tuple, request.positions.value)):
                    jobs[key][band][xy][request.model, exposure].append(
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
                    covariance_device=fast_grid_device(self.device, windows[-1]),
                )
                batches = (
                    (prepared, self._dense_points(band, bands[band], optics, prepared)),
                )

            for prepared, points in batches:
                for point_band, xy, means, lags in points:
                    for (model, duration), indices in bands[point_band][xy].items():
                        integrated = integrate_metric_lags(
                            model, lags[model], prepared.windows[duration], prepared.dt
                        )
                        for index, exposure_index, point_index in indices:
                            _, field_means, field_integrated = buffers[index]
                            field_means[exposure_index, point_index] = means
                            field_integrated[exposure_index, point_index] = integrated
        return [
            self._result(request, *buffer)
            for request, buffer in zip(requests, buffers, strict=True)
        ]

    @staticmethod
    def _dense_points(band, points, optics, prepared):
        for xy, users in points.items():
            counts = defaultdict(list)
            for model, duration in users:
                counts[model].append(len(prepared.windows[duration]))
            means, lags = optics.point(
                xy, {model: tuple(frames) for model, frames in counts.items()}
            )
            yield band, xy, means, lags

    @staticmethod
    def _result(request, grids, means, integrated):
        if (
            np.any(means <= 0)
            or np.any(integrated < 0)
            or not np.isfinite(means + integrated).all()
        ):
            raise ValueError("Invalid predicted metric covariance or mean")

        def quantities(index, unit):
            axes = means[..., index]
            variance = integrated[..., index] / request.exposures.value[:, None]
            relative = np.sqrt(integrated[..., index]) / axes
            return SamplingMetric(
                _quantity(axes[0] * unit, unit, "mean", 1),
                _quantity(variance * unit**2, unit**2, "variance", 2),
                _quantity(relative * u.s**0.5, u.s**0.5, "coefficient", 2),
                _quantity(axes * unit, unit, "mean_by_exposure", 2),
            )

        metrics = MappingProxyType(
            {
                name: quantities(i, unit)
                for i, (name, unit) in enumerate(_METRIC_UNITS.items())
            }
        )
        return SamplingResult(request, metrics, grids)

    def _prepare_grid(
        self, request, device, grid, windows, *, compact, covariance_device="cpu"
    ):
        configuration = request.configuration
        sensing_key = (configuration, device, request.numerics)
        if self._sensing_key != sensing_key:
            self._prepared = self._grid_key = None
            self._sensing = _prepare_sensing(
                configuration, float(request.wavelength.value), device, request.numerics
            )
            self._sensing_key = sensing_key
        explicit_model = request.model if request.numerics is not None else None
        key = (sensing_key, grid, windows, compact, covariance_device, explicit_model)
        if key != self._grid_key:
            self._prepared = self._grid_key = None
            if device == "mps":
                import torch

                torch.mps.empty_cache()
            prepared = _PreparedGrid(
                configuration, request, windows, grid, self._sensing, compact=compact
            )
            optics = self._optics(prepared, request, covariance_device)
            engine = self._engine(request, device, compact, prepared, optics)
            band = _science_band(request)
            self._prepared = (prepared, engine, band, optics)
            self._grid_key = key
        prepared, engine, band, optics = self._prepared
        wave, pixel, tilt = _science_band(request)
        if (wave, pixel, tilt) != band:
            view = prepared.at_science_optics(wave, pixel)
            optics = self._optics(view, request, covariance_device)
            if explicit_model is not None:
                return (
                    view,
                    self._engine(request, device, compact, view, optics),
                    optics,
                )
        return prepared, engine, optics

    @staticmethod
    def _optics(prepared, request, covariance_device):
        if request.numerics is not None:
            return ExplicitScienceOptics(prepared, request.numerics, request.model)
        return CompactScienceOptics(
            prepared,
            remove_science_tilt=request.remove_science_tilt,
            covariance_device=covariance_device,
        )

    @staticmethod
    def _engine(request, device, compact, prepared, optics):
        if device == "cpu":
            return CPUCompactFastBatch(optics) if compact else None
        if request.numerics is not None:
            from ._tensor_batch import TensorBatch

            context = prepared.context(*prepared.grids[0])
            windows = [
                (time, len(window), window) for time, window in prepared.windows.items()
            ]
            return TensorBatch(
                context,
                prepared.dt,
                prepared.max_count,
                prepared._noise_lags,
                windows,
                device,
            )
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
                fast_grid_device(self.device, duration),
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
                self.pipeline
                and len(positions) > size
                and size * 2 * lag_bytes <= _LAG_BATCH_BYTES
            )
        batches = [positions[i : i + size] for i in range(0, len(positions), size)]
        results = (
            (band, xy, means, {"fast": lags})
            for rows, _, _ in engine.evaluate_many(
                batches, optics, users, remove_tilt=band[2], pipeline=pipeline
            )
            for band, xy, means, lags in rows
        )
        return prepared, results


def _science_band(request):
    return (
        float(request.wavelength.value),
        float(request.pixel_scale.value),
        request.remove_science_tilt,
    )


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
