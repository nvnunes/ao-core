"""Gaussian pupil-pair optics and static phase."""

from __future__ import annotations

import numpy as np

__all__ = [
    "PupilPairGeometry",
]


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
        phase = self.pupil.basis.T @ covariance @ self.pupil.basis
        variance = np.diag(phase)
        structure = variance[:, None] + variance[None] - 2 * phase
        return self.amplitude_product * np.exp(-0.5 * structure + self.static_phase)
