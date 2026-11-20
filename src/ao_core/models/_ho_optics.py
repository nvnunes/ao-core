"""Pupil-pair phase coherence on the high-order detector grid.

Original-pupil tilt moments follow the project-authored MASTSEL pupil fit
(245a6af). Pair-averaged optics follows the project-authored TIPTOP finite
pupil calculation (0396d12, 2e3ff53, a89635c, b0d6faf).
"""

import numpy as np
from scipy import fft, ndimage

from ._ho_reconstruction import covariance_kernel, gaussian_basis
from .optics import (
    _coherence_from_variance_buffer,
    _image_from_otf,
    _phase_difference_variance_inplace,
    _pupil_autocorrelation,
    _sum_pairs_by_lag,
)
from .pupil import _pupil_tilt_fit

NATIVE_OPTICAL_SIDE = 512
_PROBE_PAIRS_PER_GROUP = 32
_FLOAT32_ROUNDOFF_MARGIN = 8 * (np.finfo(np.float32).eps / 2)


class InvalidCovarianceOptics(ValueError):
    """The residual covariance failed a physical check during image formation."""


def sample_pupil_map(values, width):
    """Sample the original map on the even native pupil grid."""
    coordinates = np.arange(width) * len(values) / width
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    return ndimage.map_coordinates(values, (y, x), order=1, mode="grid-constant")


def diffraction_image(field, side, image_side, origin):
    """Sample the squared Fourier transform of a real or complex pupil field.

    See §4.3 for the Fraunhofer pupil-to-field transform.
    [Goodman, Introduction to Fourier Optics, 4th ed.](https://www.macmillanlearning.com/ed/uk/product/Introduction-to-Fourier-Optics-4th-edition/p/1319119166)
    """
    if origin == "geometric":
        phase = np.exp(-1j * np.pi * np.arange(len(field)) / side)
        field = field * phase[:, None] * phase[None, :]
    transform = fft.fftshift(fft.fft2(field, s=(side, side)))
    window = (side - image_side) // 2
    return (
        np.abs(transform[window : window + image_side, window : window + image_side])
        ** 2
    )


class OpticalPupil:
    """Finite optical samples and their fit on the original pupil."""

    def __init__(self, amplitude, pupil_width, period, side, tilt_removal):
        width = 2 * round(pupil_width * side / (2 * period))
        if width < 2 or width > side:
            raise ValueError("The science sampling cannot resolve the optical pupil")
        self.side = side
        self.period = period
        self.width = width
        size = len(amplitude)
        sampled = sample_pupil_map(amplitude, width)
        self.y, self.x = np.nonzero(sampled > 0)
        self.amplitude = sampled[self.y, self.x]
        if not len(self.amplitude):
            raise ValueError("No light on the sampled optical pupil")
        self.pupil_power = float(np.sum(self.amplitude**2))
        self.full_flux = side**2 * self.pupil_power
        self.x = self.x - width // 2
        self.y = self.y - width // 2
        self.x_index, self.y_index = self.x % side, self.y % side
        self.points = np.column_stack((self.x, self.y)) * (period / side)
        self.fit_axis = (np.arange(size) / size - 0.5) * (width * period / side)
        if tilt_removal == "pupil_fit":
            self.tilt_fit = _pupil_tilt_fit(amplitude)
            self.ramps = (2 / width) * np.array([self.x, self.y]) + 1 / size
        else:
            self.tilt_fit = np.empty((0, size, size))
            self.ramps = np.empty((0, len(self.amplitude)))

    def fit_modes(self, side):
        modes = (np.arange(side) - side // 2) / self.period
        waves = np.exp(2j * np.pi * modes[:, None] * self.fit_axis)
        return np.array([waves @ fit.T @ waves.T for fit in self.tilt_fit]).reshape(
            -1, side, side
        )

    def fitted_mirror(self, mirror):
        basis = gaussian_basis(self.points, mirror.centres, mirror.width)
        if not len(self.tilt_fit):
            return basis
        xyz = np.exp(
            -((mirror.width * (self.fit_axis[:, None, None] - mirror.centres)) ** 2)
        )
        moments = np.einsum(
            "ya,iyx,xa->ia", xyz[..., 1], self.tilt_fit, xyz[..., 0], optimize=True
        )
        return basis - self.ramps.T @ moments

    def stationary_moments(self, power, kernels):
        lags = covariance_kernel(power, self.side)
        cross = np.asarray(
            [
                covariance_kernel(power * kernel.conj(), self.side)[
                    self.x_index, self.y_index
                ]
                for kernel in kernels
            ]
        ).reshape(len(kernels), len(self.x))
        tilt = np.einsum(
            "xy,ixy,jxy->ij", power, kernels, kernels.conj(), optimize=True
        ).real
        return lags, cross, tilt


def precision_probe_pairs(pupil):
    """Sample neighbouring and wider pupil pairs once per optical grid."""
    points = list(zip(pupil.x, pupil.y, strict=True))
    if len(points) < 2:
        return np.empty((0, 2), dtype=int)

    by_coordinate = {point: index for index, point in enumerate(points)}
    neighbours = [
        (index, by_coordinate[neighbour])
        for index, (x, y) in enumerate(points)
        for neighbour in ((x + 1, y), (x, y + 1))
        if neighbour in by_coordinate
    ]
    near = np.asarray(neighbours, dtype=int).reshape(-1, 2)
    if len(near) > _PROBE_PAIRS_PER_GROUP:
        near = near[np.linspace(0, len(near) - 1, _PROBE_PAIRS_PER_GROUP, dtype=int)]

    rng = np.random.default_rng(1729)
    first = rng.integers(len(points), size=_PROBE_PAIRS_PER_GROUP)
    second = (first + rng.integers(1, len(points), size=_PROBE_PAIRS_PER_GROUP)) % len(
        points
    )
    return np.vstack((near, np.column_stack((first, second))))


def phase_pairs_need_float64(pupil, pairs, source, basis, command, cross):
    """Screen sampled phase differences against a float32 roundoff estimate.

    Evaluate only the selected pairs with the prepared float64 arrays. Use
    the magnitudes of the separate covariance contributions so cancellation
    between atmospheric and command terms remains visible in the estimate.
    """
    lags, sample_tilt, tilt_covariance = source
    ramps = pupil.ramps.T

    def covariance_terms(left, right):
        x = (pupil.x[left] - pupil.x[right]) % pupil.side
        y = (pupil.y[left] - pupil.y[right]) % pupil.side
        return np.stack(
            (
                lags[x, y],
                np.sum((ramps[left] @ tilt_covariance) * ramps[right], axis=1),
                -np.sum(sample_tilt[:, left].T * ramps[right], axis=1),
                -np.sum(ramps[left] * sample_tilt[:, right].T, axis=1),
                np.sum((basis[left] @ command) * basis[right], axis=1),
                -np.sum(cross[left] * basis[right], axis=1),
                -np.sum(basis[left] * cross[right], axis=1),
            )
        )

    left, right = pairs.T
    left_diag = covariance_terms(left, left)
    right_diag = covariance_terms(right, right)
    off_diagonal = covariance_terms(left, right)
    difference = (left_diag + right_diag - 2 * off_diagonal).sum(axis=0)
    scale = (abs(left_diag) + abs(right_diag) + 2 * abs(off_diagonal)).sum(axis=0)
    return bool(
        np.any(~np.isfinite(difference) | ~np.isfinite(scale))
        or np.any((scale > 0) & (difference <= _FLOAT32_ROUNDOFF_MARGIN * scale))
    )


class ResidualCovariance:
    """Return bounded upper strips of a complete science phase covariance."""

    def __init__(
        self,
        pupil,
        source,
        basis,
        command_covariance,
        source_command,
        *,
        dtype=np.float64,
    ):
        self.pupil = pupil
        self.lags, self.sample_tilt, tilt_covariance = (
            values.astype(dtype, copy=False) for values in source
        )
        self.ramps = pupil.ramps.astype(dtype, copy=False)
        self.basis = basis.astype(dtype, copy=False)
        self.command = self.basis @ command_covariance.astype(dtype, copy=False)
        self.cross = source_command.astype(dtype, copy=False)
        self.source_weight = self.ramps.T @ tilt_covariance
        self.diagonal = np.full(len(basis), self.lags[0, 0])
        self.diagonal += np.einsum("ij,ij->i", self.source_weight, self.ramps.T)
        self.diagonal -= 2 * np.einsum("ij,ij->j", self.sample_tilt, self.ramps)
        self.diagonal += np.einsum("ij,ij->i", self.command, self.basis)
        self.diagonal -= 2 * np.einsum("ij,ij->i", self.cross, self.basis)

    def blocks(self, rows=128):
        pupil = self.pupil
        for first in range(0, len(self.diagonal), rows):
            left = slice(first, first + rows)
            right = slice(first, None)
            x = (pupil.x[left, None] - pupil.x[right]) % pupil.side
            y = (pupil.y[left, None] - pupil.y[right]) % pupil.side
            values = self.lags[x, y]
            values += self.source_weight[left] @ self.ramps[:, right]
            values -= self.sample_tilt[:, left].T @ self.ramps[:, right]
            values -= self.ramps[:, left].T @ self.sample_tilt[:, right]
            values += self.command[left] @ self.basis[right].T
            values -= self.cross[left] @ self.basis[right].T
            values -= self.basis[left] @ self.cross[right].T
            index = np.arange(len(values))
            values[index, index] = self.diagonal[left]
            yield first, values


def coherence_lags(pupil, residual, field=None):
    """Integrate Gaussian pairs on signed pupil lags with optional complex weights.

    See Eqs. (3)–(4) for the pupil integral with position-dependent
    residual phase differences. [Fétick et al. (2018)](https://arxiv.org/pdf/1809.06785)
    """
    lag_origin = np.array(
        [pupil.y.min() - pupil.y.max(), pupil.x.min() - pupil.x.max()]
    )
    lag_shape = tuple(1 - 2 * lag_origin)
    dtype = residual.diagonal.dtype
    single = dtype == np.float32
    static_weights = pupil.amplitude if field is None else field
    weights_dtype = dtype if field is None else np.result_type(dtype, np.complex64)
    weights = static_weights.astype(weights_dtype, copy=False)
    coherence = np.zeros(np.prod(lag_shape), dtype=weights.dtype)
    for first, pair in residual.blocks():
        end = first + len(pair)
        rows = slice(first, end)
        _phase_difference_variance_inplace(
            pair, residual.diagonal[rows], residual.diagonal[first:]
        )
        upper = np.arange(first, len(pupil.x)) >= np.arange(first, end)[:, None]
        if np.any(pair[upper] < -1e-8):
            raise InvalidCovarianceOptics(
                "A residual phase pair has negative difference variance"
            )
        np.maximum(pair, 0, out=pair)
        if single:
            # Integrate the atmospheric change from exact static diffraction.
            # This preserves diffraction zeros in the weak-turbulence limit.
            pair *= -0.5
            np.expm1(pair, out=pair)
        else:
            pair = _coherence_from_variance_buffer(pair)
        if field is None:
            pair *= weights[rows, None]
            pair *= weights[first:]
        else:
            pair = pair * weights[rows, None] * weights[first:].conj()
        pair[~upper] = 0
        y = pupil.y[rows, None] - pupil.y[first:] - lag_origin[0]
        x = pupil.x[rows, None] - pupil.x[first:] - lag_origin[1]
        indices = (y * lag_shape[1] + x).ravel()
        if single:
            values = np.zeros_like(coherence)
            np.add.at(values, indices, pair.ravel())
        else:
            values = _sum_pairs_by_lag(pair, indices, len(coherence))
        coherence += values + values[::-1].conj()
    coherence = coherence.astype(
        np.float64 if field is None else np.complex128, copy=False
    )
    power = pupil.pupil_power if field is None else np.sum(np.abs(field) ** 2)
    coherence[len(coherence) // 2] = power
    lags = coherence.reshape(lag_shape)
    if single:
        static_pupil = np.zeros(tuple(1 - lag_origin), dtype=static_weights.dtype)
        static_pupil[pupil.y - pupil.y.min(), pupil.x - pupil.x.min()] = static_weights
        static_lags = _pupil_autocorrelation(static_pupil)
        static_lags[-lag_origin[0], -lag_origin[1]] = 0
        lags += static_lags
    return lags


def fold_lags(lags, side):
    """Fold a centred, signed-lag plane onto an unshifted FFT grid."""
    y, x = np.indices(lags.shape) - (np.asarray(lags.shape) // 2)[:, None, None]
    folded = np.zeros((side, side), dtype=lags.dtype)
    np.add.at(folded, (y.ravel() % side, x.ravel() % side), lags.ravel())
    return folded


def validate_detector_intensity(image, full_flux):
    """Enforce the same numerical intensity limit for both detector transforms."""
    if not np.isfinite(image).all():
        raise InvalidCovarianceOptics(
            "The residual optics produced nonfinite PSF intensity"
        )
    if image.min() < -1e-11 * full_flux:
        raise InvalidCovarianceOptics(
            "The residual optics produced negative PSF intensity"
        )


def image_from_signed_lags(lags, side, image_side, origin):
    """Evaluate signed-lag coherence at either detector origin and crop."""
    if origin == "geometric":
        y, x = np.indices(lags.shape) - (np.asarray(lags.shape) // 2)[:, None, None]
        lags = lags * np.exp(-1j * np.pi * (x + y) / side)
    image = _image_from_otf(fold_lags(lags, side))
    validate_detector_intensity(
        image, side**2 * lags[lags.shape[0] // 2, lags.shape[1] // 2].real
    )
    window = (side - image_side) // 2
    return image[window : window + image_side, window : window + image_side]


def coherence_image(pupil, residual, image_side, origin, field=None):
    """Integrate the coherence of each pupil pair before Fourier sampling."""
    return image_from_signed_lags(
        coherence_lags(pupil, residual, field), pupil.side, image_side, origin
    )
