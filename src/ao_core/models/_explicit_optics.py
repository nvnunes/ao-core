"""Peak-centered metric conventions for explicit numerical grids."""

from functools import cached_property

import numpy as np

from ._all_order import (
    quadratic_variance,
    remainder_draws,
    restored_covariance,
)
from ._exposure import integrate_metric_lags
from ._image_response import image_metric_kernels
from ._restored_pupil import FullScienceMean
from .metrics import metric_covariance, metric_kernels, phase_responses
from .optics import PupilPairGeometry
from .pupil import ARCSEC


class ExplicitScienceOptics:
    """Use the native peak-centered EE and Fourier contour on a sampled pupil."""

    def __init__(self, prepared, numerics, model):
        self.prepared = prepared
        self.numerics = numerics
        self.model = model
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
            grid.configuration.physical.atmosphere,
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
                tail_order=self.numerics.tail_order,
            ),
        )

    def point(self, xy):
        grid = self.prepared
        model = self.model
        coherence = None
        if model != "limited-order" or self.numerics.restore_mean:
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
        else:
            lags = grid.point_lags(xy)

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
            self.numerics.sobol_power,
            self.numerics.sobol_replicates,
            self.numerics.sobol_seed,
            workers=self.numerics.sobol_workers,
        )
        variance = base + fine.mean(axis=0)[0]
        if np.any(variance <= 0) or np.any(base + coarse.mean(axis=0)[0] <= 0):
            raise ValueError("Nonpositive instantaneous metric variance")
        return means, {exposure: grid.dt * variance for exposure in grid.windows}
