"""Gaussian pupil-pair optics and static phase."""

from __future__ import annotations

import numpy as np
from scipy.fft import fft2, fftshift
from scipy.signal import fftconvolve

from ._displacement import covariance_matched_displacement_otfs
from ._supplied_optics import SuppliedImageModalResponse

__all__ = [
    "PupilPairGeometry",
    "SuppliedImageModalResponse",
    "covariance_matched_displacement_otfs",
]


def _phase_difference_variance_inplace(
    covariance, left_variance, right_variance
) -> None:
    """Overwrite a covariance block with the variance of each phase difference."""
    covariance *= -2
    covariance += left_variance[:, None] + right_variance[None, :]


def _coherence_from_variance_buffer(variance, *, phase_exponent=None):
    """Consume writable phase-difference variances to compute Gaussian coherence.

    The real path overwrites the input with coherence. With a complex phase
    exponent, the input holds the real part of the exponent and the function
    returns a separate complex array.
    """
    variance *= -0.5
    if phase_exponent is not None:
        return np.exp(variance + phase_exponent)
    return np.exp(variance, out=variance)


def _sum_pairs_by_lag(pairs, indices, count):
    """Accumulate real or complex pupil-pair weights into caller-defined lags."""
    weights = np.bincount(indices, weights=pairs.real.ravel(), minlength=count)
    if np.iscomplexobj(pairs):
        weights = weights + 1j * np.bincount(
            indices, weights=pairs.imag.ravel(), minlength=count
        )
    return weights


def _image_from_otf(otf):
    """Transform FFT-ordered lags on the last two axes to native intensities."""
    return fftshift(fft2(otf), axes=(-2, -1)).real


def _pupil_autocorrelation(field):
    """Return the complex pupil's linear autocorrelation on signed lags."""
    return fftconvolve(field, field[::-1, ::-1].conj(), mode="full")


class PupilPairGeometry:
    """Fixed pupil-pair geometry for one sampled pupil and metric pixel scale.

    Reuse this only while its pupil basis, amplitude and NCPA remain unchanged.
    Covariance-dependent coherence is computed separately for each point and
    can be supplied to both analytical and image-based metric responses.
    """

    def __init__(self, pupil, pixel_scale):
        self.pupil, self.pixel_scale = pupil, pixel_scale
        mask = pupil.amplitude > 0
        y, x = np.nonzero(mask)
        amplitude = pupil.amplitude[mask]
        size = pupil.optical_size
        self.dy, self.dx = (y[:, None] - y) % size, (x[:, None] - x) % size
        self.flat_lag = (self.dy * size + self.dx).ravel()
        self.amplitude_product = amplitude[:, None] * amplitude[None]
        self.static_phase = 1j * (pupil.ncpa[:, None] - pupil.ncpa[None])
        self.normalization = size**2 * np.sum(amplitude**2)
        self.frequency = 2 * np.pi / size * np.array([x[:, None] - x, y[:, None] - y])
        half_width = 50 / pixel_scale
        self.aperture = (2 * half_width) ** 2 * np.prod(
            np.sinc(half_width * self.frequency / np.pi), axis=0
        )
        self.diffraction_peak = amplitude.sum() ** 2 / self.normalization

    def require(self, pupil, pixel_scale):
        """Reject geometry supplied for a different pupil or angular scale."""
        if self.pupil is not pupil or self.pixel_scale != pixel_scale:
            raise ValueError(
                "Pupil-pair geometry belongs to a different pupil or pixel scale"
            )

    def coherence(self, covariance):
        """Return the point's Gaussian pupil coherence, without modifying inputs."""
        coherence = self.pupil.basis.T @ covariance @ self.pupil.basis
        coherence = coherence.astype(np.result_type(coherence, 0.5), copy=False)
        variance = coherence.diagonal().copy()
        _phase_difference_variance_inplace(coherence, variance, variance)
        return self.amplitude_product * _coherence_from_variance_buffer(
            coherence, phase_exponent=self.static_phase
        )
