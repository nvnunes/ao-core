"""Resampled static OPD and original-map propagation for high-order optics.

These paths follow the project-authored TIPTOP static phase and finite
static optics in 0396d12, a89635c and b0d6faf.
"""

import numpy as np
from scipy import fft

from ._ho_optics import (
    NATIVE_OPTICAL_SIDE,
    OpticalPupil,
    coherence_lags,
    diffraction_image,
    fold_lags,
    image_from_signed_lags,
    pupil_tilt_fit,
    sample_pupil_map,
)
from ._ho_reconstruction import fourier_sum
from ._lag_transfer import CubicLagTransfer
from .optics import _pupil_autocorrelation


def _phase_variance(phase, power):
    mean = np.average(phase, weights=power)
    return np.average((phase - mean) ** 2, weights=power)


def resampled_field(pupil, amplitude, phase):
    """Return the fitted static field at the native samples, with RMS compensation."""
    phase = np.where(amplitude > 0, phase, 0)
    sampled = sample_pupil_map(phase, pupil.width)
    y = pupil.y + pupil.width // 2
    x = pupil.x + pupil.width // 2
    sampled = sampled[y, x]
    original = _phase_variance(phase, amplitude**2)
    scale = 1.0
    if original > np.finfo(float).eps ** 2 * max(1.0, np.max(abs(phase)) ** 2):
        variance = _phase_variance(sampled, pupil.amplitude**2)
        if variance <= 0:
            raise ValueError(
                "Static OPD loses all phase variance on the resampled pupil"
            )
        scale = np.sqrt(original / variance)
    coefficients = np.einsum("iyx,yx->i", pupil.tilt_fit, phase)
    fitted = scale * (sampled - coefficients @ pupil.ramps)
    return pupil.amplitude * np.exp(1j * fitted)


def _original_field(amplitude, phase, tilt_fit):
    if len(tilt_fit):
        size = len(amplitude)
        axis = 2 * (np.arange(size) - (size - 1) / 2) / size
        x, y = np.meshgrid(axis, axis)
        tilt_x, tilt_y = np.einsum("iyx,yx->i", tilt_fit, phase)
        phase = phase - tilt_x * x - tilt_y * y
    return amplitude * np.exp(1j * phase)


def original_map_side(period, pitch, pupil_side, image_side):
    """Check that original pupil pitch is commensurate with the Fourier period."""
    samples = period / pitch
    side = round(samples)
    if (
        side % 2
        or side < max(pupil_side, image_side)
        or not np.isclose(samples, side, rtol=1e-12, atol=0)
    ):
        raise ValueError(
            "Original-map pupil sampling requires an even integral Fourier period covering the detector and pupil"
        )
    return side


def _original_detector_image(lags, side, image_side, origin):
    if image_side == side:
        return image_from_signed_lags(lags, side, image_side, origin)
    # The centred CZT evaluates the same signed-lag sum as the complete FFT.
    first = image_side / 2 - (0.5 if origin == "geometric" else 0.0)
    image = fourier_sum(lags, 1 / side, np.full(2, first), -1, image_side).real
    if image.min() < -1e-11 * side**2 * lags[len(lags) // 2, len(lags) // 2].real:
        raise ValueError("Original-map static transfer produced negative PSF intensity")
    return image


class OriginalMapOptics:
    """Transfer coarse atmospheric coherence onto the original static lags.

    The transfer approximates the effect of nonstationary atmospheric
    coherence. It is exact in the static-only limit. Coarse dynamic weights
    use area averages of the same fitted original complex field.
    """

    def __init__(self, pupil, amplitude, phase, pitch, fine_side):
        self.pupil = pupil
        self.side = fine_side
        self.lags = np.arange(1 - len(amplitude), len(amplitude))
        original = _original_field(amplitude, phase, pupil.tilt_fit)
        self.static_lags = _pupil_autocorrelation(original)
        self.power = float(np.sum(amplitude**2))
        self.static_lags[len(amplitude) - 1, len(amplitude) - 1] = self.power
        self.full_flux = fine_side**2 * self.power

        step = pupil.period / pupil.side
        fine = (np.arange(len(amplitude)) - len(amplitude) / 2) * pitch
        coarse = (np.arange(pupil.side) - pupil.side / 2) * step
        overlap = (
            np.maximum(
                0.0,
                np.minimum(coarse[:, None] + step / 2, fine + pitch / 2)
                - np.maximum(coarse[:, None] - step / 2, fine - pitch / 2),
            )
            / step
        )
        averaged = overlap @ original @ overlap.T
        self.field = averaged[pupil.y + pupil.side // 2, pupil.x + pupil.side // 2]
        self.coarse_power = float(np.sum(abs(self.field) ** 2))
        if self.coarse_power <= 0:
            raise ValueError("Static OPD averaged to a dark coarse pupil")
        coarse_field = np.zeros_like(averaged)
        coarse_field[pupil.y_index, pupil.x_index] = self.field
        denominator = fft.ifft2(abs(fft.fft2(coarse_field)) ** 2) / self.coarse_power
        self.transfer = CubicLagTransfer(
            fft.fftshift(denominator), self.lags * pupil.side / fine_side
        )

    def image(self, residual, image_side, origin, high_lags=None):
        coarse = coherence_lags(self.pupil, residual, self.field) / self.coarse_power
        coarse = fft.fftshift(fold_lags(coarse, self.pupil.side))
        coarse_factor = None
        native_factor = self.static_lags
        if high_lags is not None:
            coarse_high, fine_high = high_lags
            coarse_factor = fft.fftshift(np.exp(coarse_high[0, 0] - coarse_high))
            indices = self.lags % self.side
            native_factor = native_factor * np.exp(
                fine_high[np.ix_(indices, indices)] - fine_high[0, 0]
            )
        lags = self.transfer.forward(coarse, native_factor, coarse_factor)
        return _original_detector_image(lags, self.side, image_side, origin)


def static_diffraction(config, phase, period, image_side):
    """Produce the turbulence-free limit on the configured static pupil grid."""
    amplitude = config.pupil.amplitude
    pitch = config.pupil.pitch.value
    origin = config.detector_origin
    if config.static_opd_sampling == "original_map":
        side = original_map_side(period, pitch, len(amplitude), image_side)
        tilt_fit = (
            pupil_tilt_fit(amplitude) if config.tilt_removal == "pupil_fit" else ()
        )
        field = _original_field(amplitude, phase, tilt_fit)
        lags = _pupil_autocorrelation(field)
        return _original_detector_image(
            lags, side, image_side, origin
        ), side**2 * float(np.sum(amplitude**2))

    side = max(NATIVE_OPTICAL_SIDE, image_side)
    pupil = OpticalPupil(
        amplitude, len(amplitude) * pitch, period, side, config.tilt_removal
    )
    field = np.zeros((pupil.width, pupil.width), dtype=complex)
    field[pupil.y + pupil.width // 2, pupil.x + pupil.width // 2] = resampled_field(
        pupil, amplitude, phase
    )
    return diffraction_image(field, side, image_side, origin), pupil.full_flux
