"""Device-resident propagation of the native zero-static-phase NGS optics."""

from __future__ import annotations

import numpy as np


class MpsSensorPropagation:
    """Reuse pupil geometry across NGS mean images and modal derivatives.

    Optical covariance preparation and detector statistics remain double
    precision on CPU. Dense pupil coherence, lag accumulation and image FFTs
    use MPS float32 in bounded modal chunks. Returned images use the same
    incident-flux normalization and half-pixel origin as optical_mean().
    """

    def __init__(self, pupil, derivative_batch=4):
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS sensing was requested but no Metal device is available"
            )
        if np.any(pupil.ncpa) or derivative_batch < 1:
            raise ValueError(
                "Require zero-static-phase NGS optics and a positive modal batch"
            )
        self.torch, self.pupil = torch, pupil
        self.derivative_batch = derivative_batch
        self.size = pupil.optical_size
        mask = pupil.amplitude > 0
        y, x = np.nonzero(mask)
        ids = (
            ((y[:, None] - y) % self.size) * self.size + (x[:, None] - x) % self.size
        ).ravel()
        self.ids = torch.as_tensor(ids, device="mps")
        self.basis = torch.as_tensor(pupil.basis, dtype=torch.float32, device="mps")
        amplitude = pupil.amplitude[mask]
        self.amplitude = torch.as_tensor(amplitude, dtype=torch.float32, device="mps")
        self.normalization = self.size**2 * np.sum(amplitude**2)
        frequency = np.fft.fftfreq(self.size)
        self.centering = torch.as_tensor(
            np.exp(-1j * np.pi * (frequency[:, None] + frequency[None])),
            dtype=torch.complex64,
            device="mps",
        )

    def _propagate(self, values, factor=1):
        torch = self.torch
        flat = values.reshape(len(values), -1)
        otf = torch.zeros(
            (len(values), self.size**2), dtype=torch.float32, device="mps"
        )
        otf.scatter_add_(1, self.ids[None].expand(len(values), -1), flat)
        image = torch.fft.fftshift(
            torch.fft.fft2(
                otf.reshape(-1, self.size, self.size) * self.centering * factor
            ),
            dim=(-2, -1),
        ).real
        return image / self.normalization

    def calculate(self, covariance, derivative_modes=()):
        """Return the same mean image and requested derivatives as CPU optics."""
        torch = self.torch
        covariance = torch.as_tensor(covariance, dtype=torch.float32, device="mps")
        phase = self.basis.T @ covariance @ self.basis
        diagonal = phase.diag()
        structure = diagonal[:, None] + diagonal[None] - 2 * phase
        coherence = (
            torch.exp(-0.5 * structure) * self.amplitude[:, None] * self.amplitude[None]
        )
        image = self._propagate(coherence[None])[0].cpu().numpy().astype(float)
        if image.min() < -1e-7 or abs(image.sum() - 1) > 1e-5:
            raise ValueError("MPS optical mean lost positivity or incident flux")
        modes = list(derivative_modes)
        derivatives = []
        for start in range(0, len(modes), self.derivative_batch):
            basis = self.basis[modes[start : start + self.derivative_batch]]
            values = (basis[:, :, None] - basis[:, None]) * coherence
            derivatives.append(self._propagate(values, 1j))
        arrays = (
            torch.cat(derivatives).cpu().numpy().astype(float) if derivatives else []
        )
        return np.maximum(image, 0), list(arrays)
