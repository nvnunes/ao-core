"""Peak-centered metric conventions for explicit numerical grids."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ._all_order import (
    SOBOL_POWER,
    SOBOL_REPLICATES,
    SOBOL_SEED,
    quadratic_variance,
    remainder_draws,
    restored_covariance,
)
from ._exposure import integrate_metric_lags
from ._image_response import image_metric_kernels
from ._restored_pupil import FullScienceMean
from ._units import _count
from .metrics import metric_covariance, metric_kernels, phase_responses
from .optics import PupilPairGeometry
from .pupil import ARCSEC


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

    def __init__(self, prepared, model_settings, model_type):
        self.prepared = prepared
        self.model_settings = model_settings
        self.model_type = model_type
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

    def point(self, xy):
        grid = self.prepared
        model = self.model_type
        context = self.science.context
        if grid.max_count == 1:
            period, _ = grid.grids[0]
            weights = (
                None
                if model == "full-pupil"
                else context.zero_lag_weights(round(period / grid.dt))
            )
            coherences, all_lags = self.science.tiled_restoration(
                [xy], lag_weights=weights
            )
            coherence = coherences[0]
            lags = None if all_lags is None else all_lags[0]
        else:
            terms = context.direction_terms(np.asarray(xy) * ARCSEC)
            coherence = self.science.coherence(xy, terms=terms)
            lags = grid.point_lags(xy, terms=terms)

        covariance = None if lags is None else lags[0]
        if model == "limited-order":
            means, first, second, tails = self.modal_response(covariance, coherence)
            series = np.arange(len(lags)), metric_covariance(lags, first, second, tails)
            return means, {
                exposure: integrate_metric_lags("fast", series, window, grid.dt)
                for exposure, window in grid.windows.items()
            }

        means, kernels, coherence = self.metric_response(covariance, coherence)
        weights = coherence[None] * kernels
        if model == "full-pupil":
            phase = restored_covariance(coherence, self.geometry.amplitude_product)
            base = quadratic_variance(phase, weights)
        else:
            basis = self.pupil.basis
            phase = basis.T @ covariance @ basis
            base = quadratic_variance(covariance, weights, basis=basis)
        fine, coarse = remainder_draws(
            phase[None],
            weights[None],
            self.model_settings.sobol_power,
            self.model_settings.sobol_replicates,
            self.model_settings.sobol_seed,
            workers=self.model_settings.sobol_workers,
        )
        variance = base + fine.mean(axis=0)[0]
        if np.any(variance <= 0) or np.any(base + coarse.mean(axis=0)[0] <= 0):
            raise ValueError("Nonpositive instantaneous metric variance")
        return means, {exposure: grid.dt * variance for exposure in grid.windows}
