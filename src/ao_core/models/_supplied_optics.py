"""Linear modal responses transferred from diffraction optics to supplied spots."""

import numpy as np
from scipy.fft import fft2, ifft2


class SuppliedImageModalResponse:
    """Prepare a pupil and selected modes for an incident-normalized spot.

    Amplitude has axes (y, x) and modes has axes (y, x, mode). The supplied
    mode shapes use the caller's normalization. phase_per_unit_rad gives the
    phase in radians per unit of mode amplitude at the sensing wavelength.
    Pupil and FFT sampling follow the even-grid, centred optical convention.
    Detector bins return flux in row-major pixel order.
    """

    def __init__(
        self,
        amplitude,
        modes,
        *,
        fft_size,
        field_crop,
        detector_side,
        phase_per_unit_rad,
    ):
        amplitude, modes = np.asarray(amplitude), np.asarray(modes)
        if (
            amplitude.dtype.kind not in "fiu"
            or modes.dtype.kind not in "fiu"
            or amplitude.ndim != 2
            or amplitude.shape[0] != amplitude.shape[1]
            or amplitude.shape[0] % 2
            or not np.isfinite(amplitude).all()
            or np.any(amplitude < 0)
            or not np.any(amplitude > 0)
            or modes.ndim != 3
            or modes.shape[:2] != amplitude.shape
            or modes.shape[-1] < 1
            or not np.isfinite(modes).all()
        ):
            raise ValueError("Require an even square pupil and matching finite modes")
        if (
            any(
                type(value) is not int
                for value in (fft_size, field_crop, detector_side)
            )
            or fft_size < len(amplitude)
            or fft_size % 2
            or field_crop <= 0
            or field_crop > fft_size
            or field_crop % 2
            or detector_side < 1
            or field_crop % detector_side
            or not np.isfinite(phase_per_unit_rad)
            or phase_per_unit_rad <= 0
        ):
            raise ValueError("Invalid optical sampling or phase conversion")
        amplitude, modes = (
            amplitude.astype(float, copy=False),
            modes.astype(float, copy=False),
        )
        self.nfft = fft_size
        self.crop = field_crop
        self.detector_side = detector_side
        self.start = (fft_size - field_crop) // 2
        self.block = field_crop // detector_side
        self.incident_power = np.sum(amplitude**2)
        self.modes = modes.copy()
        self.phase_per_unit_rad = phase_per_unit_rad
        coordinate = np.arange(len(amplitude)) * (1 - fft_size) / fft_size
        self.field = amplitude * np.exp(
            -1j * np.pi * (coordinate[:, None] + coordinate)
        )

    def bin(self, image):
        """Sum the final two image axes on the optical FFT grid into detector pixels."""
        image = np.asarray(image)
        if image.shape[-2:] != (self.nfft, self.nfft):
            raise ValueError("Supplied image must match the optical Fourier grid")
        start, stop = self.start, self.start + self.crop
        part = image[..., start:stop, start:stop]
        side, block = self.detector_side, self.block
        return (
            part.reshape(*part.shape[:-2], side, block, side, block)
            .sum(axis=(-3, -1))
            .reshape(*part.shape[:-2], side**2)
        )

    def kernel_otfs(self):
        """Return one linear transfer kernel per mode, with zero DC response."""
        size = (self.nfft, self.nfft)
        fourier = fft2(self.field, s=size) / self.nfft
        otf = fft2(abs(fourier) ** 2 / self.incident_power)
        valid = abs(otf) > 1e-10 * abs(otf[0, 0])
        kernels = []
        for mode in np.moveaxis(self.modes, -1, 0):
            derivative = (
                fft2(self.field * (1j * self.phase_per_unit_rad * mode), s=size)
                / self.nfft
            )
            image_derivative = (
                2 * (fourier.conj() * derivative).real / self.incident_power
            )
            response = np.zeros_like(otf)
            response[valid] = fft2(image_derivative)[valid] / otf[valid]
            response[0, 0] = 0
            if abs(response).max() > 4 * self.phase_per_unit_rad * abs(mode).max():
                raise ValueError("Ill-conditioned modal optical transfer")
            kernels.append(response)
        return np.stack(kernels)

    def psf_derivatives(self, image, kernels):
        """Return incident flux and signed flux derivatives integrated by pixel.

        Supply a nonnegative incident-normalized image on the optical FFT grid
        and one transfer kernel per selected mode, as from `kernel_otfs`.
        The result has shapes (detector pixels,) and (detector pixels, modes),
        with row-major detector order.
        """
        image, kernels = np.asarray(image), np.asarray(kernels)
        if (
            image.shape != (self.nfft, self.nfft)
            or image.dtype.kind not in "fiu"
            or not np.isfinite(image).all()
            or np.any(image < 0)
        ):
            raise ValueError(
                "Supplied image must be finite and nonnegative on the optical Fourier grid"
            )
        if (
            kernels.shape != (self.modes.shape[-1], self.nfft, self.nfft)
            or not np.isfinite(kernels).all()
        ):
            raise ValueError("Supply one finite optical Fourier kernel per mode")
        otf = fft2(image)
        values = []
        for kernel in kernels:
            derivative = ifft2(otf * kernel)
            if abs(derivative.imag).max() >= 1e-10:
                raise ValueError("Optical derivative must be real")
            values.append(self.bin(derivative.real))
        return self.bin(image), np.stack(values, axis=-1)
