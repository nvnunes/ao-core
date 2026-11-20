"""Shared science optics with a native static aberration map."""

from dataclasses import replace
from functools import cached_property
from types import SimpleNamespace

import numpy as np
from astropy import units as u
from numpy.polynomial.legendre import leggauss
from scipy.fft import hfft2
from scipy.special import j0

from ._all_order import restored_covariance
from ._exposure import atmospheric_lags
from ._image_response import ContourOutsideCrop, contour_fwhm
from ._lag_transfer import CubicLagTransfer
from ._restored_pupil import FullScienceMean
from ._temporal_pupil import atmospheric_offset_covariance, full_pupil_lags
from .atmosphere import VK_COEFFICIENT, von_karman_correlation
from .optics import (
    _coherence_from_variance_buffer,
    _pupil_autocorrelation,
    _sum_pairs_by_lag,
)
from .pupil import CORRECTION_CUTOFF, raw_modes

CONTOUR_REFINEMENT = 16


def _project_out_tilt(matrix, tilt):
    projected = matrix - tilt @ (tilt.T @ matrix)
    return projected - (projected @ tilt) @ tilt.T


def _without_science_tilt(structure, tilt):
    """Remove the two orthonormal, piston-free slopes from phase differences."""
    projected = _project_out_tilt(structure, tilt)
    variance = projected.diagonal().copy()
    return projected - 0.5 * (variance[:, None] + variance[None, :])


def _cell_weights(native_side, side):
    """Map native sample areas to the coarser cells with unit column sums."""
    native = np.linspace(-1, 1, native_side)
    coarse = np.linspace(-1, 1, side)
    half = 1 / (native_side - 1)
    edges = np.r_[-1 - half, (coarse[1:] + coarse[:-1]) / 2, 1 + half]
    return np.maximum(
        0,
        np.minimum(edges[1:, None], native + half)
        - np.maximum(edges[:-1, None], native - half),
    ) / (2 * half)


def _high_structure(side, spacing, atmosphere, scale):
    """Return stationary phase structure above the correction cutoff."""
    offsets = np.arange(side) ** 2
    radii2, inverse = np.unique(offsets[:, None] + offsets, return_inverse=True)
    radius = spacing * np.sqrt(radii2)
    outer = 1 / float(atmosphere.outer_scale.to_value(u.m))
    nu = 5 / 6
    correlation = von_karman_correlation(radius, outer)
    strength = VK_COEFFICIENT * float(atmosphere.fried_parameter.to_value(u.m)) ** (
        -5 / 3
    )
    full = 2 * np.pi * strength * outer ** (-2 * nu) / nu * (1 - correlation)
    nodes, weights = leggauss(256)
    lo, hi = np.log(1e-5), np.log(CORRECTION_CUTOFF)
    f = np.exp((lo + hi) / 2 + (hi - lo) / 2 * nodes)
    weights *= (
        (hi - lo)
        / 2
        * f
        * f
        * 4
        * np.pi
        * strength
        * (f * f + outer * outer) ** (-11 / 6)
    )
    low = (1 - j0(2 * np.pi * radius[:, None] * f)) @ weights
    result = (full - low) * np.sum(atmosphere.layer_weights.value) * scale**2
    return result[inverse].reshape(side, side)


def _hermitian_lags(lags):
    return 0.5 * (lags + lags[..., ::-1, ::-1].conj())


class StaticTransfer:
    """Transfer atmosphere-weighted optical lags onto the original map lattice."""

    def __init__(self, field, coarse_field, high):
        coarse_edge = len(coarse_field) - 1
        self.side = 4 * coarse_edge
        center = self.side // 2
        self.window = slice(center - coarse_edge, center + coarse_edge + 1)
        # Pad the signed lags so the periodic spline spans the unsupported margin.
        coarse_static = np.zeros((self.side, self.side), dtype=complex)
        coarse_static[self.window, self.window] = _pupil_autocorrelation(coarse_field)
        native_edge = len(field) - 1
        lags = np.arange(-native_edge, native_edge + 1)
        self.transfer = CubicLagTransfer(
            coarse_static, lags * coarse_edge / native_edge
        )
        lags = abs(lags)
        self.native_factor = _pupil_autocorrelation(field) * np.exp(
            -0.5 * high[lags[:, None], lags]
        )

    def mean(self, coarse):
        padded = np.zeros((self.side, self.side), dtype=coarse.dtype)
        padded[self.window, self.window] = coarse
        return _hermitian_lags(self.transfer.forward(padded, self.native_factor))

    def responses(self, native):
        """Apply the static transfer adjoint to a batch of metric kernels."""
        coarse = np.array(
            [
                self.transfer.adjoint(_hermitian_lags(kernel), self.native_factor)[
                    self.window, self.window
                ]
                for kernel in native
            ]
        )
        return _hermitian_lags(coarse)


class CoreTransform:
    """Fourier evaluation and transpose on the existing refined FWHM grid."""

    def __init__(self, native_size, fft_size, radius):
        self.radius = radius
        self.size = fft_size
        edge = native_size // 2
        self.lags = np.arange(-edge, edge + 1)
        count = 2 * radius * CONTOUR_REFINEMENT + 1
        shifts = np.linspace(-radius, radius, count)
        self.phase = -2j * np.pi * self.lags / fft_size
        self.waves = np.exp(shifts[:, None] * self.phase)

    def _waves(self, centre):
        return self.waves * np.exp(np.asarray(centre)[..., None, None] * self.phase)

    def image(self, otf, y, x):
        return (self._waves(y) @ otf @ self._waves(x).swapaxes(-1, -2)).real

    def kernel(self, gradient, y, x):
        return self._waves(y).swapaxes(-1, -2) @ gradient @ self._waves(x)


class CompactScienceOptics:
    """Reuse one wavelength's static optics for field predictions.

    The supplied point predictor owns the AO covariance and modal basis. All
    levels share the restored mean and science metric kernels. Only the phase
    covariance response differs between levels.
    """

    def __init__(
        self,
        predictor,
        *,
        remove_science_tilt=False,
        atmosphere_at_frame=None,
        detector_side=None,
    ):
        self.predictor = predictor
        self._atmosphere_at_frame = atmosphere_at_frame
        self._detector_side = detector_side
        self.pixel = predictor.metadata["pixel_scale_mas"]
        wave = predictor.metadata["wavelength_m"]
        support = predictor.metadata["support_m"]
        amplitude = predictor.pupil
        ncpa = predictor.configuration.pupil.static_opd.to_value(u.nm)
        if remove_science_tilt:
            mask = amplitude > 0
            y_native, x_native = np.nonzero(mask)
            slopes = np.column_stack((x_native, y_native)).astype(float)
            slopes -= slopes.mean(axis=0)
            ncpa = np.broadcast_to(ncpa, amplitude.shape).astype(float, copy=True)
            ncpa[mask] -= slopes @ np.linalg.lstsq(slopes, ncpa[mask], rcond=None)[0]
        field = amplitude * np.exp(2j * np.pi * ncpa * 1e-9 / wave)
        self.native_side = len(field)
        self.science_side = predictor.science_side
        self.size = 4 * (self.native_side - 1)
        self.flux = np.sum(amplitude**2)
        self.diffraction_norm = np.sum(amplitude) ** 2
        weights = _cell_weights(self.native_side, self.science_side)
        coarse_field = weights @ field @ weights.T
        occupancy = weights @ amplitude**2 @ weights.T > 0
        y, x = np.nonzero(occupancy)
        coordinates = (
            np.column_stack((x, y)) * support / (self.science_side - 1) - support / 2
        )
        base = predictor.optical
        diameter = float(predictor.configuration.telescope.diameter.to_value(u.m))
        raw = raw_modes(base, base.labels, diameter)
        transform = np.linalg.lstsq(
            np.vstack((np.ones(raw.shape[1]), raw)).T, base.basis.T, rcond=None
        )[0].T
        raw = raw_modes(SimpleNamespace(coordinates=coordinates), base.labels, diameter)
        basis = transform @ np.vstack((np.ones(len(x)), raw))
        self.pupil = replace(base, coordinates=coordinates, basis=basis)
        self.tilt = None
        if remove_science_tilt:
            slopes = basis[:2].T
            self.tilt, _ = np.linalg.qr(slopes - slopes.mean(axis=0))
        y_lag = y[:, None] - y + self.science_side - 1
        x_lag = x[:, None] - x + self.science_side - 1
        self.coarse_lags = 2 * self.science_side - 1
        self.flat_lag = (y_lag * self.coarse_lags + x_lag).ravel()
        sampled = coarse_field[y, x]
        self.static_pairs = sampled[:, None] * sampled.conj()
        atmosphere = predictor.configuration.atmosphere
        scale = float(atmosphere.reference_wavelength.to_value(u.m)) / wave
        high = _high_structure(
            self.native_side, support / (self.native_side - 1), atmosphere, scale
        )
        self.transfer = StaticTransfer(field, coarse_field, high)
        coarse_high = _high_structure(
            self.science_side, support / (self.science_side - 1), atmosphere, scale
        )
        self.high_pairs = coarse_high[abs(y[:, None] - y), abs(x[:, None] - x)]
        lags = np.arange(1 - self.native_side, self.native_side)
        self.frequency = 2 * np.pi * lags / self.size
        self._core_transforms = {}
        self.fft_index = np.ix_(lags % self.size, np.arange(self.native_side))
        fy, fx = np.meshgrid(self.frequency, self.frequency, indexing="ij")
        self.derivatives = -1j * np.array([fx, fy])
        self.curvature = -self.derivatives[:, None] * self.derivatives[None, :]
        half = 50 / self.pixel
        aperture = np.sinc(self.frequency * half / np.pi)
        self.aperture = (2 * half) ** 2 * aperture[:, None] * aperture

    @cached_property
    def science(self):
        """Prepare the atmospheric projection only when this optics owns it."""
        grid = self.predictor.grids[0]
        context = self.predictor.context(*grid)
        return FullScienceMean(
            context,
            SimpleNamespace(pupil=self.pupil),
            self.predictor.configuration.atmosphere,
            self.predictor._noise_lags[0],
            radial_order=self.predictor.radial_order,
            angular_order=grid[1],
            angular_offset=self.predictor.angular_offset,
            pupil_offsets=self.pupil_offsets,
        )

    @cached_property
    def lag_weights(self):
        period = self.predictor.grids[0][0]
        return self.science.context.zero_lag_weights(round(period / self.predictor.dt))

    @cached_property
    def pupil_offsets(self):
        spacing = self.predictor.metadata["support_m"] / (self.science_side - 1)
        offset = np.arange(1 - self.science_side, self.science_side) * spacing
        x, y = np.meshgrid(offset, offset)
        offsets = np.column_stack((x.ravel(), y.ravel()))
        pairs = self.flat_lag.reshape(len(self.pupil.coordinates), -1)
        return offsets, pairs

    @cached_property
    def full_atmospheric_offsets(self):
        if self._atmosphere_at_frame is not None:
            return self._atmosphere_at_frame
        offsets, _ = self.pupil_offsets
        atmosphere = self.predictor.configuration.atmosphere
        return atmospheric_offset_covariance(atmosphere, self.predictor.dt, offsets)

    def _peak_kernel(self, centre, norm):
        waves = np.exp(-1j * centre[:, None] * self.frequency)
        return waves[1, :, None] * waves[0] / norm

    def _valid_mean_psf(self, image):
        if not np.isfinite(image).all() or abs(image.sum() - 1) > 1e-9:
            return False
        if image.min() >= -1e-12:
            return True
        if self._detector_side is None:
            return False
        # Compact-to-native OTF interpolation can ring in the far wings.
        half = self._detector_side // 2
        detector = np.arange(-half, self._detector_side - half) % self.size
        return (
            image[np.ix_(detector, detector)].min() >= -1e-12
            and -image[image < 0].sum() <= 1e-8
        )

    def _kernels(self, mean_otf, *, means_only=False):
        norm = self.size**2 * self.flux
        centre = np.zeros(2)
        derivative = self.derivatives
        for _ in range(12):
            peak = self._peak_kernel(centre, norm)
            weighted = mean_otf * peak
            gradient = np.einsum("ab,iab->i", weighted, derivative).real
            curvature = np.einsum("ab,ijab->ij", weighted, self.curvature).real
            if np.min(np.linalg.eigvalsh(curvature)) <= 0:
                raise ValueError("Mean central PSF peak is not locally concave")
            step = np.linalg.solve(curvature, gradient)
            centre += step
            if np.linalg.norm(step) < 1e-9:
                break
        else:
            raise ValueError("Mean central PSF peak did not converge")
        peak = self._peak_kernel(centre, norm)
        # EE100 integrates the same square at the nominal source for the mean and response.
        ee_kernel = self.aperture / norm
        sr_kernel = peak * (norm / self.diffraction_norm)

        plane = np.zeros((self.size, self.size // 2 + 1), complex)
        plane[self.fft_index] = mean_otf[:, self.native_side - 1 :]
        image = hfft2(plane, s=(self.size, self.size)) / norm
        if not self._valid_mean_psf(image):
            raise ValueError(
                "Compact static transfer did not produce a nonnegative, unit-flux mean PSF"
            )
        # Apply the centred ordering to peak ties without shifting the image.
        peak_pixels = np.argwhere(image == image.max())
        offsets = (peak_pixels + self.size // 2) % self.size - self.size // 2
        y, x = min(map(tuple, offsets))
        for radius in (8, 16, 32, 64):
            key = radius, CONTOUR_REFINEMENT
            if key not in self._core_transforms:
                self._core_transforms[key] = CoreTransform(
                    len(self.frequency), self.size, radius
                )
            transform = self._core_transforms[key]
            refined = transform.image(mean_otf, y, x) / norm
            try:
                width, gradient = contour_fwhm(
                    refined, self.pixel / CONTOUR_REFINEMENT, require_closed=True
                )
            except ContourOutsideCrop:
                if radius == 64:
                    raise
            else:
                break
        means = np.array(
            [
                np.sum(mean_otf * sr_kernel).real,
                np.sum(mean_otf * ee_kernel).real,
                width,
            ]
        )
        if means_only:
            return means, ()
        width_kernel = transform.kernel(gradient, y, x) / norm
        return means, (sr_kernel, ee_kernel, width_kernel)

    def metric_response(self, structure, *, means_only=False, lag_grid=False):
        """Build the metric kernels from the restored structure on the compact pupil."""
        if self.tilt is not None:
            structure = _without_science_tilt(structure, self.tilt)
        coherence = self.static_pairs * _coherence_from_variance_buffer(
            structure - self.high_pairs
        )
        coarse = _sum_pairs_by_lag(coherence, self.flat_lag, self.coarse_lags**2)
        mean_otf = self.transfer.mean(
            coarse.reshape(self.coarse_lags, self.coarse_lags)
        )
        means, native_kernels = self._kernels(mean_otf, means_only=means_only)
        if means_only:
            return means, None, None
        responses = self.transfer.responses(np.asarray(native_kernels))
        if lag_grid:
            return means, responses, coherence
        kernels = responses.reshape(len(responses), -1).take(self.flat_lag, axis=1)
        return means, kernels.reshape(len(responses), *coherence.shape), coherence

    def mean_metrics(self, position):
        """Evaluate the metric means without building temporal covariance."""
        (structure,), _ = self.science.tiled_phase_structures([position])
        return self.metric_response(structure, means_only=True)[0]

    def compact_phases(self, positions):
        context = self.science.context
        spectra = np.zeros(
            (len(positions), context.bin_count, context.nmodes, context.nmodes), complex
        )
        structures, lags = self.science.tiled_phase_structures(
            positions, lag_weights=self.lag_weights, spectrum=spectra
        )
        return zip(structures, lags, spectra, strict=True)

    def compact_lags(self, spectrum, covariance, count):
        frames = round(self.science.context.period / self.predictor.dt)
        lags = atmospheric_lags(spectrum, frames, count)
        lags[:, :2, :2] += self.predictor._noise_lags[:count]
        if not np.allclose(lags[0], covariance, rtol=1e-8, atol=1e-11):
            raise ValueError(
                "Temporal and restored modal covariance disagree at zero lag"
            )
        lags[0] = covariance
        return lags

    def restore_phase(self, position):
        """Return the pupil phase structure function and zero-lag modal covariance."""
        if self.science.context.common_only:
            correction, coefficients = self.science.components(position)
            projection = (self.science.waves @ coefficients.T).real
            return self.science._phase_structure(correction, projection), None
        (structure,), lags = self.science.tiled_phase_structures(
            [position], lag_weights=self.lag_weights
        )
        return structure, lags[0, 0]

    def pupil_phase_lags(self, structure, terms, indices):
        """Yield the piston-free pupil covariance at the requested frame lags."""
        if self.tilt is not None:
            structure = _without_science_tilt(structure, self.tilt)
        coherence = np.exp(-0.5 * structure)
        zero_lag = restored_covariance(coherence, np.ones_like(coherence))
        temporal = ()
        if len(indices) > 1:
            offsets, pairs = self.pupil_offsets
            temporal = full_pupil_lags(
                self.science,
                terms,
                indices,
                offsets,
                pairs,
                atmosphere=self.predictor.configuration.atmosphere,
                frame_interval_s=self.predictor.dt,
                noise_lags=self.predictor._noise_lags,
                atmosphere_at_frame=self.full_atmospheric_offsets,
            )
            if self.tilt is not None:
                temporal = (_project_out_tilt(lag, self.tilt) for lag in temporal)
            if not np.allclose(next(temporal), zero_lag, rtol=1e-8, atol=1e-10):
                raise ValueError(
                    "Temporal and restored pupil covariance disagree at zero lag"
                )
        yield zero_lag
        yield from temporal
