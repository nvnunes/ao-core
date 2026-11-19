"""Sampling uncertainty calculations on explicit numerical grids."""

from copy import copy
from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np
from astropy import units as u

from ._all_order import (
    SOBOL_POWER,
    SOBOL_REPLICATES,
    SOBOL_SEED,
    integrate_all_orders,
    quadratic_variance,
    restored_covariance,
)
from ._compact_optics import CompactScienceOptics
from ._exposure import integrate_metric_lags
from ._image_response import image_metric_kernels
from ._metric_covariance import predict_metric_lags
from ._restored_pupil import FullScienceMean
from ._tensor_batch import TensorBatch
from ._units import _count
from .metrics import metric_covariance, metric_kernels, phase_responses
from .optics import PupilPairGeometry
from .pupil import ARCSEC, make_pupil
from .uncertainty import (
    _prepare_sensing,
    _PreparedGrid,
    _project_sampling_configuration,
    _sampling_frame_interval,
)


@dataclass(frozen=True, kw_only=True)
class ExplicitGridSettings:
    """Pupil and Fourier grid settings for the explicit sampling models.

    The spectral period must span at least twice the longest requested
    exposure.

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

    def __post_init__(self):
        for name, minimum in (
            ("physical_side", 3),
            ("optical_side", 3),
            ("sensor_side", 3),
            ("radial_order", 1),
            ("angular_order", 1),
            ("sensor_angular_order", 1),
            ("modal_order", 1),
        ):
            _count(getattr(self, name), name, minimum)
        if self.modal_order not in (5, 7, 9, 11):
            raise ValueError("modal_order must be 5, 7, 9 or 11")
        times = (self.spectral_period_s, self.spectral_interval_s)
        if not np.isfinite((*times, self.angular_offset)).all() or min(times) <= 0:
            raise ValueError(
                "Require a finite angular offset and positive spectral times"
            )

    @property
    def _sensing_grid(self):
        return self.radial_order, self.sensor_angular_order

    _reference_side = 64


@dataclass(frozen=True, kw_only=True)
class _ExplicitModelSettings:
    grid_settings: ExplicitGridSettings
    sensor_workers: int = 1

    def __post_init__(self):
        if not isinstance(self.grid_settings, ExplicitGridSettings):
            raise TypeError("grid_settings must be an ExplicitGridSettings")
        _count(self.sensor_workers, "sensor_workers")
        if self.sensor_workers > 4:
            raise ValueError("sensor_workers must be at most 4")


@dataclass(frozen=True, kw_only=True)
class LimitedOrderSettings(_ExplicitModelSettings):
    """Settings for finite-order covariance on an explicit grid.

    Attributes:
        grid_settings: Pupil and Fourier grids.
        sensor_workers: Independent workers for sensor optical derivatives.
        tail_order: Gaussian expansion order, either 4 or 6.
    """

    tail_order: int = 6

    def __post_init__(self):
        super().__post_init__()
        _count(self.tail_order, "tail_order")
        if self.tail_order not in (4, 6):
            raise ValueError("tail_order must be 4 or 6")


@dataclass(frozen=True, kw_only=True)
class _SobolModelSettings(_ExplicitModelSettings):
    sobol_power: int = SOBOL_POWER
    sobol_replicates: int = SOBOL_REPLICATES
    sobol_seed: int = SOBOL_SEED
    sobol_workers: int = 1

    def __post_init__(self):
        super().__post_init__()
        _count(self.sobol_power, "sobol_power", 10)
        _count(self.sobol_replicates, "sobol_replicates", 4)
        _count(self.sobol_seed, "sobol_seed", 0)
        _count(self.sobol_workers, "sobol_workers")
        if self.sobol_workers > 4:
            raise ValueError("sobol_workers must be at most 4")


@dataclass(frozen=True, kw_only=True)
class AllOrderSettings(_SobolModelSettings):
    """Settings for all-orders modal covariance on an explicit grid.

    Attributes:
        grid_settings: Pupil and Fourier grids.
        sensor_workers: Independent workers for sensor optical derivatives.
        sobol_power: Base-2 logarithm of the number of quartets per scramble.
        sobol_replicates: Number of independent scrambles.
        sobol_seed: Reproducible scramble seed.
        sobol_workers: Independent workers for the Sobol scrambles.
    """


@dataclass(frozen=True, kw_only=True)
class FullPupilSettings(_SobolModelSettings):
    """Settings for all-orders full-pupil covariance on an explicit grid.

    Attributes:
        grid_settings: Pupil and Fourier grids. Its radial order is used for
            sensing.
        sensor_workers: Independent workers for sensor optical derivatives.
        sobol_power: Base-2 logarithm of the number of quartets per scramble.
        sobol_replicates: Number of independent scrambles.
        sobol_seed: Reproducible scramble seed.
        sobol_workers: Independent workers for the Sobol scrambles.
        full_radial_order: Radial quadrature order for science covariance.
    """

    full_radial_order: int = 160

    def __post_init__(self):
        super().__post_init__()
        _count(self.full_radial_order, "full_radial_order")


class ExplicitScienceOptics:
    """Use the native peak-centered EE and Fourier contour on a sampled pupil."""

    def __init__(self, prepared, model_settings, model_type, *, restore_mean=True):
        self.prepared = prepared
        self.model_settings = model_settings
        self.model_type = model_type
        self.restore_mean = restore_mean
        self.pupil = prepared.optical
        self.pixel_scale = prepared.metadata["pixel_scale_mas"]
        self.geometry = PupilPairGeometry(self.pupil, self.pixel_scale)

    @cached_property
    def science(self):
        grid = self.prepared
        context = grid.context(*grid.grids[0])
        return FullScienceMean(
            context,
            self.geometry,
            grid.configuration.atmosphere,
            grid._noise_lags[0],
            radial_order=grid.radial_order,
            angular_order=grid.grids[0][1],
            angular_offset=grid.angular_offset,
        )

    def metric_response(self, covariance, coherence=None):
        """Evaluate the peak-centered metric convention and its matched kernels."""
        if coherence is None:
            coherence = self.geometry.coherence(covariance)
        means, kernels, _, _ = metric_kernels(
            self.pupil,
            covariance,
            self.pixel_scale,
            geometry=self.geometry,
            coherence=coherence,
            include_width=False,
        )
        width, _, _, contour = image_metric_kernels(self.geometry, coherence, (2,))
        return (
            np.concatenate((means, width)),
            np.concatenate((kernels, contour)),
            coherence,
        )

    def modal_response(self, covariance, coherence=None):
        means, kernels, coherence = self.metric_response(covariance, coherence)
        return (
            means,
            *phase_responses(
                self.pupil.basis,
                covariance,
                kernels,
                coherence,
                tail_order=self.model_settings.tail_order,
            ),
        )

    def point_inputs(self, positions):
        """Prepare the restored coherence and modal lags for a bounded tile."""
        grid = self.prepared
        if not self.restore_mean:
            return [(grid.point_lags(xy), None) for xy in positions]
        science = self.science
        if grid.max_count == 1:
            period, _ = grid.grids[0]
            weights = (
                None
                if self.model_type == "full-pupil"
                else science.context.zero_lag_weights(round(period / grid.dt))
            )
            coherences, lags = science.tiled_restoration(positions, lag_weights=weights)
            if lags is None:
                lags = [None] * len(positions)
        else:
            components, lags = [], []
            for xy in positions:
                terms = science.context.direction_terms(np.asarray(xy) * ARCSEC)
                components.append(science.components(xy, terms=terms))
                lags.append(grid.point_lags(xy, terms=terms))
            coherences = science.coherences(components)
        return list(zip(lags, coherences, strict=True))

    def point(self, xy):
        """Evaluate a single point with the scalar covariance calculation."""
        (inputs,) = self.point_inputs([xy])
        return self.integrate_point(*inputs)

    def integrate_point(self, lags, coherence):
        """Evaluate one position from prepared lags and coherence."""
        grid = self.prepared
        model = self.model_type
        covariance = None if lags is None else lags[0]
        if model == "limited-order":
            means, first, second, tails = self.modal_response(covariance, coherence)
            series = np.arange(len(lags)), metric_covariance(lags, first, second, tails)
            return means, {
                exposure: integrate_metric_lags("fast", series, window, grid.dt)
                for exposure, window in grid.windows.items()
            }

        response = self.all_order_response(covariance, coherence)
        variance, _, _, _ = integrate_all_orders(
            (response,),
            self.model_settings.sobol_power,
            self.model_settings.sobol_replicates,
            self.model_settings.sobol_seed,
            workers=self.model_settings.sobol_workers,
        )
        return response[0], {
            exposure: grid.dt * variance[0] for exposure in grid.windows
        }

    def all_order_response(self, covariance, coherence):
        """Build the weighted pupil response and exact quadratic variance."""
        means, kernels, coherence = self.metric_response(covariance, coherence)
        weights = coherence[None] * kernels
        if self.model_type == "full-pupil":
            phase = restored_covariance(coherence, self.geometry.amplitude_product)
            base = quadratic_variance(phase, weights)
        else:
            basis = self.pupil.basis
            phase = basis.T @ covariance @ basis
            base = quadratic_variance(covariance, weights, basis=basis)
        return means, phase, weights, base

    def at_wavelength(self, wavelength_m):
        """Reuse the fixed sensing solution at another science wavelength."""
        return ExplicitScienceOptics(
            self.prepared.at_science_optics(wavelength_m),
            self.model_settings,
            self.model_type,
            restore_mean=self.restore_mean,
        )

    def without_science_tilt(self):
        """Remove static tilt from the science pupil and retain the AO solution."""
        prepared = copy(self.prepared)
        pupil = self.pupil
        tilt = pupil.basis[:2].T
        static_tilt = tilt @ np.linalg.lstsq(tilt, pupil.ncpa, rcond=None)[0]
        prepared.optical = replace(pupil, ncpa=pupil.ncpa - static_tilt)
        return ExplicitScienceOptics(
            prepared,
            self.model_settings,
            self.model_type,
            restore_mean=self.restore_mean,
        )

    def tensor_batch(self, device, *, remove_science_tilt=False):
        """Prepare batched finite-modal covariance for this Fourier grid."""
        if self.restore_mean:
            raise ValueError("Tensor batches require finite-modal science coherence")
        grid = self.prepared
        context = grid.context(*grid.grids[0])
        noise = grid._noise_lags
        if remove_science_tilt:
            noise = np.zeros_like(noise)
        return TensorBatch(
            context,
            grid.dt,
            grid.max_count,
            noise,
            [
                (exposure, len(weights), weights)
                for exposure, weights in grid.windows.items()
            ],
            device,
        )

    @staticmethod
    def tensor_inputs(backend, positions, *, remove_science_tilt=False):
        """Transform one science tile and project out tilt when requested."""
        directions = np.asarray(positions) * ARCSEC
        if remove_science_tilt:
            lags = backend.lags(backend.spectrum(directions))
            lags[:, :, :2, :] = 0
            lags[:, :, :, :2] = 0
            return lags
        return backend.point_lags(directions)

    def integrate_batch(self, backend, lags):
        """Compute each science response on CPU and integrate its lags on device."""
        covariances = backend.covariances(lags)
        responses = [self.modal_response(covariance) for covariance in covariances]
        integrated = backend.integrate(lags, responses)
        return covariances, [
            (response[0], {exposure: values[i] for exposure, values in integrated})
            for i, response in enumerate(responses)
        ]


def prepare_explicit_optics(
    config,
    wavelength_m,
    exposures_s,
    model_settings,
    model_type,
    *,
    ngs_noise_covariance,
    restore_mean=True,
    device="cpu",
    sensing=None,
):
    """Prepare the historical explicit-grid optics and the matched lag calculation."""
    config = _project_sampling_configuration(config)
    grid_settings = model_settings.grid_settings
    dt = _sampling_frame_interval(config)
    exposures = tuple(exposures_s)
    if (
        not exposures
        or not np.isfinite(exposures).all()
        or any(
            t <= 0 or not np.isclose(t, round(t / dt) * dt, rtol=0, atol=1e-12)
            for t in exposures
        )
    ):
        raise ValueError("Require positive whole-frame exposure windows")
    instantaneous = model_type == "full-pupil"
    if model_type != "limited-order" and any(t > dt + 1e-12 for t in exposures):
        raise ValueError("Explicit all-order calculations require one native frame")
    if not restore_mean and model_type != "limited-order":
        raise ValueError("Explicit all-order calculations require pupil restoration")
    if not instantaneous and 2 * max(exposures) > grid_settings.spectral_period_s:
        raise ValueError("The Fourier period must span at least twice the exposure")
    if sensing is None:
        sensing = _prepare_sensing(
            config,
            wavelength_m,
            device,
            ngs_noise_covariance=ngs_noise_covariance,
            grid_settings=grid_settings,
            sensor_workers=model_settings.sensor_workers,
        )
    grid = (
        model_settings.full_radial_order
        if instantaneous
        else grid_settings.radial_order,
        grid_settings.angular_order,
        grid_settings.spectral_period_s,
    )
    prepared = _PreparedGrid(
        config,
        wavelength_m,
        exposures,
        grid,
        sensing,
        grid_settings=grid_settings,
        instantaneous=instantaneous,
    )
    return ExplicitScienceOptics(
        prepared,
        model_settings,
        model_type,
        restore_mean=restore_mean,
    )


def full_single_frame_on_grid(
    config,
    wavelength_m,
    positions_arcsec,
    grid_settings,
    *,
    device,
    common_only=False,
    remove_science_tilt=False,
    detector_side=None,
):
    """Yield Full metric means and integrated covariance with ideal sensing.

    Use an explicit quadrature grid and the configured native frame interval.
    Common-only correction uses the common HO mirror. The field-dependent
    path also includes the science mirror and the configured ideal LO loop.
    Both paths omit detector noise. The detector extent limits tolerated
    interpolation ringing to the wings outside the camera.
    """
    if device not in ("cpu", "mps"):
        raise ValueError("device must be cpu or mps")
    settings = grid_settings
    frame_rate = config.high_order_loop.frame_rate.to_value(u.Hz)
    dt = float(1 / frame_rate)
    physical = make_pupil(
        config.pupil.amplitude,
        config.pupil.static_opd.to_value(u.nm),
        float(config.pupil.span.to_value(u.m)),
        float(config.telescope.diameter.to_value(u.m)),
        wavelength_m,
        settings.physical_side,
        settings.modal_order,
    )
    grid = settings.radial_order, settings.angular_order, settings.spectral_period_s
    prepared = _PreparedGrid(
        config,
        wavelength_m,
        (dt,),
        grid,
        (physical, (None, None, None, None)),
        grid_settings=settings,
        common_only=common_only,
    )
    optics = CompactScienceOptics(
        prepared, remove_science_tilt=remove_science_tilt, detector_side=detector_side
    )
    indices = np.array([0])
    for position in positions_arcsec:
        structure, _ = optics.restore_phase(position)
        means, kernels, coherence = optics.metric_response(structure)
        pupil_lags = optics.pupil_phase_lags(structure, None, indices)
        lags = predict_metric_lags(
            {"full": indices},
            None,
            pupil_lags,
            optics.pupil.basis,
            kernels,
            coherence,
            device=device,
        )
        yield (
            means,
            integrate_metric_lags("full", lags["full"], prepared.windows[dt], dt),
        )
