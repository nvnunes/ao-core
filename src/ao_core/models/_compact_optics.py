"""Shared science optics with a native static aberration map."""

from dataclasses import replace
from functools import cached_property, partial
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from astropy import units as u
from numpy.polynomial.legendre import leggauss
from scipy.fft import hfft2
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve
from scipy.special import j0

from ._all_order import (
    SOBOL_POWER,
    SOBOL_REPLICATES,
    SOBOL_SEED,
    RemainderQuadrature,
    quadratic_covariance,
    quadratic_response,
    restored_covariance,
)
from ._exposure import atmospheric_lags, sampled_lag_indices
from ._image_response import ContourOutsideCrop, contour_fwhm
from ._restored_pupil import FullScienceMean
from ._temporal_pupil import atmospheric_offset_covariance, full_pupil_lags
from .atmosphere import VK_COEFFICIENT, von_karman_correlation
from .metrics import metric_covariance, phase_responses
from .pupil import ARCSEC, CORRECTION_CUTOFF, raw_modes

LEVELS = ("fast", "balanced", "full")
SCIENCE_SIDE = 64
CONTOUR_REFINEMENT = 16
MAX_EXPOSURE = 20.0
MAX_MPS_EXPOSURE = 10.0


def science_grid(exposure):
    """Resolve radial nodes, angular nodes and spectral period from exposure."""
    if exposure <= 1.0:
        return 80, 192, 20.0
    if exposure <= 5.0:
        return 160, 192, 20.0
    if exposure <= 10.0:
        return 160, 384, 20.0
    return 320, 768, 40.0


def fast_grid_device(device, exposure):
    """Keep the largest Fast grid on the bounded CPU path."""
    return "cpu" if exposure > MAX_MPS_EXPOSURE else device


class CPUCompactFastBatch:
    """Reuse one large CPU projection and lag cube across matched bands."""

    def __init__(self, reference):
        self.reference = reference
        self.wavelength_m = reference.predictor.metadata["wavelength_m"]

    def evaluate_many(self, batches, optics, users, *, remove_tilt, pipeline):
        for positions in batches:
            started = perf_counter()
            results = []
            restored = self.reference.compact_phases(positions)
            for xy, (structure, zero, spectrum) in zip(
                positions, restored, strict=True
            ):
                lags = self.reference.compact_lags(
                    spectrum, zero[0], self.reference.predictor.max_count
                )
                if remove_tilt:
                    lags[:, :2, :] = 0
                    lags[:, :, :2] = 0
                for key in users[xy]:
                    means, series = optics[key].fast_metric_lags(
                        structure, lags, phase_scale=self.wavelength_m / key[0]
                    )
                    results.append((key, xy, means, series))
                del lags
            yield results, (perf_counter() - started) / len(positions), len(positions)


def _autocorrelation(field):
    return fftconvolve(field, field[::-1, ::-1].conj(), mode="full")


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


class StaticTransfer:
    """Transfer atmosphere-weighted optical lags onto the original map lattice."""

    def __init__(self, field, coarse_field, high):
        self.coarse_static = _autocorrelation(coarse_field)
        self.supported = abs(self.coarse_static) > 1e-8 * abs(self.coarse_static).max()
        self.nearest = tuple(
            distance_transform_edt(
                ~self.supported, return_distances=False, return_indices=True
            )
        )
        size = len(self.coarse_static)
        target = np.linspace(0, size - 1, 2 * len(field) - 1)
        left = np.minimum(target.astype(int), size - 2)
        fraction = target - left
        self.interpolation = np.zeros((len(target), size))
        self.interpolation[np.arange(len(target)), left] = 1 - fraction
        self.interpolation[np.arange(len(target)), left + 1] = fraction
        lags = abs(np.arange(1 - len(field), len(field)))
        self.native_factor = _autocorrelation(field) * np.exp(
            -0.5 * high[lags[:, None], lags]
        )

    def mean(self, coarse):
        ratio = np.zeros_like(self.coarse_static)
        ratio[self.supported] = (
            coarse[self.supported] / self.coarse_static[self.supported]
        )
        matrix = self.interpolation
        native = (matrix @ ratio[self.nearest] @ matrix.T) * self.native_factor
        return 0.5 * (native + native[::-1, ::-1].conj())

    def responses(self, native):
        """Apply the static transfer adjoint to a batch of metric kernels."""
        matrix = self.interpolation
        pulled = matrix.T @ (native * self.native_factor) @ matrix
        coarse = np.zeros(
            (len(native), *self.coarse_static.shape), dtype=self.coarse_static.dtype
        )
        np.add.at(coarse, (slice(None), *self.nearest), pulled)
        coarse[:, self.supported] /= self.coarse_static[self.supported]
        return 0.5 * (coarse + coarse[:, ::-1, ::-1].conj())


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
        covariance_device="cpu",
        atmosphere_at_frame=None,
    ):
        self.predictor = predictor
        self.covariance_device = covariance_device
        self._atmosphere_at_frame = atmosphere_at_frame
        self.pixel = predictor.metadata["pixel_scale_mas"]
        wave = predictor.metadata["wavelength_m"]
        support = predictor.metadata["support_m"]
        amplitude = predictor.pupil
        ncpa = predictor.configuration.static_opd.to_value(u.nm)
        if remove_science_tilt:
            mask = amplitude > 0
            y_native, x_native = np.nonzero(mask)
            slopes = np.column_stack((x_native, y_native)).astype(float)
            slopes -= slopes.mean(axis=0)
            ncpa = np.broadcast_to(ncpa, amplitude.shape).astype(float, copy=True)
            ncpa[mask] -= slopes @ np.linalg.lstsq(slopes, ncpa[mask], rcond=None)[0]
        field = amplitude * np.exp(2j * np.pi * ncpa * 1e-9 / wave)
        self.native_side = len(field)
        self.size = 4 * (self.native_side - 1)
        self.flux = np.sum(amplitude**2)
        self.diffraction_norm = np.sum(amplitude) ** 2
        weights = _cell_weights(self.native_side, SCIENCE_SIDE)
        coarse_field = weights @ field @ weights.T
        occupancy = weights @ amplitude**2 @ weights.T > 0
        y, x = np.nonzero(occupancy)
        coordinates = (
            np.column_stack((x, y)) * support / (SCIENCE_SIDE - 1) - support / 2
        )
        base = predictor.optical
        diameter = float(
            predictor.configuration.physical.telescope.diameter.to_value(u.m)
        )
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
        y_lag = y[:, None] - y + SCIENCE_SIDE - 1
        x_lag = x[:, None] - x + SCIENCE_SIDE - 1
        self.coarse_lags = 2 * SCIENCE_SIDE - 1
        self.flat_lag = (y_lag * self.coarse_lags + x_lag).ravel()
        sampled = coarse_field[y, x]
        self.static_pairs = sampled[:, None] * sampled.conj()
        atmosphere = predictor.configuration.physical.atmosphere
        scale = float(atmosphere.reference_wavelength.to_value(u.m)) / wave
        high = _high_structure(
            self.native_side, support / (self.native_side - 1), atmosphere, scale
        )
        self.transfer = StaticTransfer(field, coarse_field, high)
        coarse_high = _high_structure(
            SCIENCE_SIDE, support / (SCIENCE_SIDE - 1), atmosphere, scale
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
            self.predictor.configuration.physical.atmosphere,
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
        spacing = self.predictor.metadata["support_m"] / (SCIENCE_SIDE - 1)
        offset = np.arange(1 - SCIENCE_SIDE, SCIENCE_SIDE) * spacing
        x, y = np.meshgrid(offset, offset)
        offsets = np.column_stack((x.ravel(), y.ravel()))
        pairs = self.flat_lag.reshape(len(self.pupil.coordinates), -1)
        return offsets, pairs

    @cached_property
    def full_atmospheric_offsets(self):
        if self._atmosphere_at_frame is not None:
            return self._atmosphere_at_frame
        offsets, _ = self.pupil_offsets
        atmosphere = self.predictor.configuration.physical.atmosphere
        return atmospheric_offset_covariance(atmosphere, self.predictor.dt, offsets)

    def _peak_kernel(self, centre, norm):
        waves = np.exp(-1j * centre[:, None] * self.frequency)
        return waves[1, :, None] * waves[0] / norm

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
        if abs(image.sum() - 1) > 1e-9 or image.min() < -1e-12:
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
        coherence = self.static_pairs * np.exp(-0.5 * (structure - self.high_pairs))
        bins = self.coarse_lags**2
        coarse = np.bincount(
            self.flat_lag, coherence.real.ravel(), minlength=bins
        ) + 1j * np.bincount(self.flat_lag, coherence.imag.ravel(), minlength=bins)
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

    def fast_metric_lags(self, structure, lags, *, phase_scale=1.0):
        if phase_scale != 1.0:
            structure = structure * phase_scale**2
        means, kernels, coherence = self.metric_response(structure, lag_grid=True)
        first, second, (projection, tails) = phase_responses(
            self.pupil.basis,
            lags[0],
            kernels,
            coherence,
            tail_order=6,
            lag_indices=self.flat_lag.reshape(coherence.shape),
        )
        if phase_scale != 1.0:
            first *= phase_scale
            second *= phase_scale**2
            tails = [
                [tensor * phase_scale**order for order, tensor in enumerate(terms, 3)]
                for terms in tails
            ]
        return means, (
            np.arange(len(lags)),
            metric_covariance(lags, first, second, (projection, tails)),
        )

    def point(self, xy, frame_counts):
        """Return metric means and sampled covariance for each level and its frame counts."""
        context = self.science.context
        structures, lags = self.science.tiled_phase_structures(
            [xy], lag_weights=self.lag_weights
        )
        structure, covariance = structures[0], lags[0, 0]
        means, kernels, coherence = self.metric_response(structure)
        basis = self.pupil.basis
        metric_lags = {}
        modal_frames = max(
            frame_counts.get("fast", ()) + frame_counts.get("balanced", ()), default=1
        )
        full_frames = max(frame_counts.get("full", ()), default=1)
        terms = None
        if max(modal_frames, full_frames) > 1:
            terms = context.direction_terms(np.asarray(xy) * ARCSEC)
        phase_lags = covariance[None]
        if modal_frames > 1:
            phase_lags = self.predictor.point_lags(xy, terms=terms)[:modal_frames]
            if not np.allclose(phase_lags[0], covariance, rtol=1e-8, atol=1e-11):
                raise ValueError(
                    "Temporal and restored modal covariance disagree at zero lag"
                )
            phase_lags[0] = covariance
        if "fast" in frame_counts:
            if self.tilt is not None:
                phase_lags[:, :2, :] = 0
                phase_lags[:, :, :2] = 0
                covariance = phase_lags[0]
            first, second, tails = phase_responses(
                basis, covariance, kernels, coherence, tail_order=6
            )
            count = max(frame_counts["fast"])
            metric_lags["fast"] = (
                np.arange(count),
                metric_covariance(phase_lags[:count], first, second, tails),
            )
        all_order = {
            level: np.unique(
                np.concatenate(
                    [sampled_lag_indices(count) for count in frame_counts[level]]
                )
            )
            for level in ("balanced", "full")
            if level in frame_counts
        }
        if all_order:
            metric_lags.update(
                self._all_order_lags(
                    all_order, terms, phase_lags, structure, coherence[None] * kernels
                )
            )
        return means, metric_lags

    def _all_order_lags(self, indices, terms, modal, structure, weights):
        basis = self.pupil.basis
        zero, kernels = {}, {}
        if "balanced" in indices:
            zero["balanced"] = basis.T @ modal[0] @ basis
            kernels["balanced"] = quadratic_response(weights, basis=basis)
        full = None
        if "full" in indices:
            if self.tilt is not None:
                structure = _without_science_tilt(structure, self.tilt)
            coherence = np.exp(-0.5 * structure)
            zero["full"] = restored_covariance(coherence, np.ones_like(coherence))
            first, second = quadratic_response(weights)
            if self.covariance_device == "mps":
                from ._metal_covariance import MpsQuadraticCovariance

                full_quadratic = MpsQuadraticCovariance(first, second)
            else:
                full_quadratic = partial(
                    quadratic_covariance, first=first, second=second
                )
            del first, second
            if len(indices["full"]) > 1:
                offsets, pairs = self.pupil_offsets
                full = full_pupil_lags(
                    self.science,
                    terms,
                    indices["full"],
                    offsets,
                    pairs,
                    atmosphere=self.predictor.configuration.physical.atmosphere,
                    frame_interval_s=self.predictor.dt,
                    noise_lags=self.predictor._noise_lags,
                    atmosphere_at_frame=self.full_atmospheric_offsets,
                )
                if self.tilt is not None:
                    full = (_project_out_tilt(lag, self.tilt) for lag in full)
                if not np.allclose(next(full), zero["full"], rtol=1e-8, atol=1e-10):
                    raise ValueError(
                        "Temporal and restored pupil covariance disagree at zero lag"
                    )

        frames = np.unique(np.concatenate(tuple(indices.values())))
        model_weights = np.broadcast_to(weights, (len(zero), *weights.shape))
        metal_sobol = None
        try_metal = self.covariance_device == "mps" and len(frames) > 1
        quadrature = RemainderQuadrature(np.stack(tuple(zero.values())), model_weights)
        values = {level: [] for level in indices}
        frames_by_level = {level: set(lags) for level, lags in indices.items()}
        for frame in frames:
            covariances, bases, levels = [], [], []
            if frame in frames_by_level.get("balanced", ()):
                levels.append("balanced")
                covariances.append(
                    zero["balanced"] if frame == 0 else basis.T @ modal[frame] @ basis
                )
                bases.append(quadratic_covariance(modal[frame], *kernels["balanced"]))
            if frame in frames_by_level.get("full", ()):
                levels.append("full")
                covariance = zero["full"] if frame == 0 else next(full)
                covariances.append(covariance)
                bases.append(full_quadratic(covariance))
            lag_covariances = np.stack(covariances)
            if metal_sobol is not None:
                remainders, coarse = metal_sobol(lag_covariances)
                if not (np.isfinite(remainders).all() and np.isfinite(coarse).all()):
                    metal_sobol = None
            if metal_sobol is None:
                remainders, coarse = quadrature.evaluate(
                    lag_covariances,
                    model_weights[: len(levels)],
                    SOBOL_POWER,
                    SOBOL_REPLICATES,
                    SOBOL_SEED,
                    workers=4,
                    reuse_draws=len(frames) > 1,
                )
            if (
                frame == 0
                and try_metal
                and (draws := quadrature.metal_indices) is not None
            ):
                from ._metal_covariance import MpsRemainderQuadrature

                metal_sobol = MpsRemainderQuadrature(
                    draws, weights, quadrature.proposal, len(zero)
                )
            estimate = np.asarray(bases) + remainders.mean(axis=0)
            if frame == 0 and (
                np.any(estimate <= 0) or np.any(bases + coarse.mean(axis=0) <= 0)
            ):
                raise ValueError("Nonpositive instantaneous metric variance")
            for level, value in zip(levels, estimate, strict=True):
                values[level].append(value)
        return {
            level: (lags, np.asarray(values[level]).T)
            for level, lags in indices.items()
        }
