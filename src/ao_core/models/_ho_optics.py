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
    _sum_pairs_by_lag,
)

NATIVE_OPTICAL_SIDE = 512


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


def pupil_tilt_fit(amplitude):
    """Return the amplitude-squared weighted ramp fit on the original pupil."""
    size = len(amplitude)
    axis = 2 * (np.arange(size) - (size - 1) / 2) / size
    ramps = np.array(np.meshgrid(axis, axis))
    weights = amplitude**2
    centroid = np.einsum("iyx,yx->i", ramps, weights) / weights.sum()
    # Fitting centred ramps is equivalent to including a piston intercept.
    ramps -= centroid[:, None, None]
    gram = np.einsum("iyx,jyx,yx->ij", ramps, ramps, weights)
    return np.linalg.solve(gram, ramps.reshape(2, -1)).reshape(ramps.shape) * weights


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
            self.tilt_fit = pupil_tilt_fit(amplitude)
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


class ResidualCovariance:
    """Return bounded upper strips of a complete science phase covariance."""

    def __init__(self, pupil, source, basis, command_covariance, source_command):
        self.pupil = pupil
        self.lags, self.sample_tilt, self.tilt_covariance = source
        self.basis = basis
        self.command = basis @ command_covariance
        self.cross = source_command
        self.source_weight = pupil.ramps.T @ self.tilt_covariance
        self.diagonal = np.full(len(basis), self.lags[0, 0])
        self.diagonal += np.einsum("ij,ij->i", self.source_weight, pupil.ramps.T)
        self.diagonal -= 2 * np.einsum("ij,ij->j", self.sample_tilt, pupil.ramps)
        self.diagonal += np.einsum("ij,ij->i", self.command, basis)
        self.diagonal -= 2 * np.einsum("ij,ij->i", self.cross, basis)

    def blocks(self, rows=128):
        pupil = self.pupil
        for first in range(0, len(self.diagonal), rows):
            left = slice(first, first + rows)
            right = slice(first, None)
            x = (pupil.x[left, None] - pupil.x[right]) % pupil.side
            y = (pupil.y[left, None] - pupil.y[right]) % pupil.side
            values = self.lags[x, y]
            values += self.source_weight[left] @ pupil.ramps[:, right]
            values -= self.sample_tilt[:, left].T @ pupil.ramps[:, right]
            values -= pupil.ramps[:, left].T @ self.sample_tilt[:, right]
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
    coherence = np.zeros(
        np.prod(lag_shape), dtype=np.float64 if field is None else np.complex128
    )
    for first, pair in residual.blocks():
        end = first + len(pair)
        rows = slice(first, end)
        _phase_difference_variance_inplace(
            pair, residual.diagonal[rows], residual.diagonal[first:]
        )
        upper = np.arange(first, len(pupil.x)) >= np.arange(first, end)[:, None]
        if np.any(pair[upper] < -1e-8):
            raise ValueError("A residual phase pair has negative difference variance")
        np.maximum(pair, 0, out=pair)
        pair = _coherence_from_variance_buffer(pair)
        if field is None:
            pair *= pupil.amplitude[rows, None]
            pair *= pupil.amplitude[first:]
        else:
            pair = pair * field[rows, None] * field[first:].conj()
        pair[~upper] = 0
        y = pupil.y[rows, None] - pupil.y[first:] - lag_origin[0]
        x = pupil.x[rows, None] - pupil.x[first:] - lag_origin[1]
        values = _sum_pairs_by_lag(pair, (y * lag_shape[1] + x).ravel(), len(coherence))
        coherence += values + values[::-1].conj()
    power = pupil.pupil_power if field is None else np.sum(np.abs(field) ** 2)
    coherence[len(coherence) // 2] = power
    return coherence.reshape(lag_shape)


def fold_lags(lags, side):
    """Fold a centred, signed-lag plane onto an unshifted FFT grid."""
    y, x = np.indices(lags.shape) - (np.asarray(lags.shape) // 2)[:, None, None]
    folded = np.zeros((side, side), dtype=lags.dtype)
    np.add.at(folded, (y.ravel() % side, x.ravel() % side), lags.ravel())
    return folded


def image_from_signed_lags(lags, side, image_side, origin):
    """Evaluate signed-lag coherence at either detector origin and crop."""
    if origin == "geometric":
        y, x = np.indices(lags.shape) - (np.asarray(lags.shape) // 2)[:, None, None]
        lags = lags * np.exp(-1j * np.pi * (x + y) / side)
    image = _image_from_otf(fold_lags(lags, side))
    if (
        image.min()
        < -1e-11 * side**2 * lags[lags.shape[0] // 2, lags.shape[1] // 2].real
    ):
        raise ValueError("The finite-pupil covariance produced negative PSF intensity")
    window = (side - image_side) // 2
    return image[window : window + image_side, window : window + image_side]


def coherence_image(pupil, residual, image_side, origin, field=None):
    """Integrate the coherence of each pupil pair before Fourier sampling."""
    return image_from_signed_lags(
        coherence_lags(pupil, residual, field),
        pupil.side,
        image_side,
        origin,
    )
