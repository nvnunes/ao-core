"""Modal pupil preparation and spatial Fourier quadrature."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from astropy import units as u
from numpy.polynomial.legendre import leggauss
from scipy.fft import fft2, ifftshift
from scipy.linalg import solve_triangular
from scipy.ndimage import map_coordinates, spline_filter
from scipy.special import eval_jacobi

from ._units import _quantity

__all__ = [
    "ModalPupil",
    "prepare_modal_pupil",
]

ARCSEC = np.pi / (180 * 3600)
CORRECTION_CUTOFF = 1.6


@dataclass(frozen=True, eq=False)
class ModalPupil:
    """Actual sampled pupil, orthonormal modes and their Fourier projections."""

    amplitude: np.ndarray
    basis: np.ndarray
    weights: np.ndarray
    coordinates: np.ndarray
    dx: float
    ncpa: np.ndarray
    spectrum: np.ndarray | None
    fft_size: int
    optical_size: int
    labels: tuple[tuple[int, int, str], ...]

    def response(self, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        """Project spatial Fourier waves onto the sampled, piston-free basis."""
        if self.spectrum is None:
            raise ValueError("This optical pupil has no modal Fourier response")
        if np.max(np.hypot(fx, fy)) >= 0.48 / self.dx:
            raise ValueError("Fourier query exceeds sampled-pupil Nyquist range")
        indices = np.array(
            [
                fy * self.fft_size * self.dx % self.fft_size,
                fx * self.fft_size * self.dx % self.fft_size,
            ]
        )
        values = np.array(
            [
                map_coordinates(
                    v.real, indices, order=3, mode="grid-wrap", prefilter=False
                )
                + 1j
                * map_coordinates(
                    v.imag, indices, order=3, mode="grid-wrap", prefilter=False
                )
                for v in self.spectrum
            ]
        )
        return values * np.exp(1j * np.pi * self.dx * (fx + fy))


def make_pupil(
    pupil: np.ndarray,
    ncpa: np.ndarray,
    support: float,
    diameter: float,
    wavelength: float,
    side: int,
    order: int,
    *,
    fourier_response=True,
) -> ModalPupil:
    """Keep native pupil/NCPA sampling; optionally omit unused Fourier projections."""
    coordinates = np.linspace(0, len(pupil) - 1, side)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    amplitude = map_coordinates(
        np.asarray(pupil, dtype=float), [yy, xx], order=0, mode="nearest"
    )
    static = (
        map_coordinates(ncpa, [yy, xx], order=1, mode="nearest")
        * 1e-9
        * 2
        * np.pi
        / wavelength
    )
    q = np.linspace(-support / 2, support / 2, side)
    py, px = np.meshgrid(q, q, indexing="ij")
    mask = amplitude > 0
    x, y = px[mask], py[mask]
    rho, angle = np.hypot(x, y) / (diameter / 2), np.arctan2(y, x)
    raw, labels = [], []
    for n in range(1, order + 1):
        for m in range(n % 2, n + 1, 2):
            radial = rho**m * eval_jacobi((n - m) // 2, 0, m, 2 * rho**2 - 1)
            for kind in ("cos",) if m == 0 else ("cos", "sin"):
                angular = np.cos(m * angle) if kind == "cos" else np.sin(m * angle)
                raw.append(np.sqrt(n + 1) * (np.sqrt(2) if m else 1) * radial * angular)
                labels.append((n, m, kind))
    raw = np.asarray(raw)
    intensity = amplitude[mask] ** 2
    intensity /= intensity.sum()
    raw -= raw @ intensity[:, None]
    gram = (raw * intensity) @ raw.T
    basis = solve_triangular(np.linalg.cholesky(gram), raw, lower=True)
    weights = basis * intensity
    np.testing.assert_allclose(weights @ basis.T, np.eye(len(basis)), atol=1e-10)
    size = 512
    offset = (size - side) // 2
    transform = None
    if fourier_response:
        transform = []
        for row in weights:
            image = np.zeros((side, side))
            image[mask] = row
            field = np.pad(image, ((offset, size - side - offset),) * 2)
            values = fft2(ifftshift(field)).conj()
            transform.append(
                spline_filter(values.real, order=3, mode="grid-wrap")
                + 1j * spline_filter(values.imag, order=3, mode="grid-wrap")
            )
        transform = np.asarray(transform)
    return ModalPupil(
        amplitude,
        basis,
        weights,
        np.column_stack([x, y]),
        support / (side - 1),
        static[mask],
        transform,
        size,
        4 * (side - 1),
        tuple(labels),
    )


def quadrature(
    radial_order: int, angular_order: int, limit: float, angular_offset: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, weights = leggauss(radial_order)
    lo, hi = np.log(1e-5), np.log(limit)
    radius = np.exp((lo + hi) / 2 + (hi - lo) / 2 * nodes)
    radial_weights = (hi - lo) / 2 * weights * radius**2
    angle = 2 * np.pi * np.arange(angular_order) / angular_order
    if angular_offset:
        angle += angular_offset * 2 * np.pi / angular_order
    return (
        (radius[:, None] * np.cos(angle)).ravel(),
        (radius[:, None] * np.sin(angle)).ravel(),
        np.repeat(radial_weights * 2 * np.pi / angular_order, angular_order),
    )


def prepare_modal_pupil(
    amplitude: np.ndarray,
    static_opd: u.Quantity,
    support: u.Quantity,
    diameter: u.Quantity,
    reference_wavelength: u.Quantity,
    side: int,
    order: int,
    *,
    fourier_response: bool = True,
) -> ModalPupil:
    """Prepare piston-free, intensity-orthonormal modes in radial order.

    The supplied amplitude and static OPD share the square y/x pupil grid.
    Support is the physical span between first and last samples along each
    axis. Static OPD is converted to radians at the supplied phase wavelength.
    Mode labels are tuples of radial order, azimuthal order and sine/cosine.
    """
    amplitude = np.asarray(amplitude)
    opd = _quantity(static_opd, u.nm, "static_opd", 2).value
    support_m = float(_quantity(support, u.m, "support", 0).value)
    diameter_m = float(_quantity(diameter, u.m, "diameter", 0).value)
    wavelength_m = float(
        _quantity(reference_wavelength, u.m, "reference_wavelength", 0).value
    )
    if (
        amplitude.dtype.kind not in "fiu"
        or amplitude.ndim != 2
        or amplitude.shape != opd.shape
        or amplitude.shape[0] != amplitude.shape[1]
        or not np.isfinite(amplitude).all()
        or np.any(amplitude < 0)
        or not np.any(amplitude > 0)
    ):
        raise ValueError(
            "Require a finite nonnegative square amplitude grid matching the OPD"
        )
    if min(support_m, diameter_m, wavelength_m) <= 0:
        raise ValueError("Pupil lengths and phase wavelength must be positive")
    if (
        type(side) is not int
        or not 2 <= side <= 512
        or type(order) is not int
        or order < 1
    ):
        raise ValueError("Require size from 2 to 512 and a positive modal order")
    return make_pupil(
        amplitude,
        opd,
        support_m,
        diameter_m,
        wavelength_m,
        side,
        order,
        fourier_response=fourier_response,
    )


def raw_modes(pupil, labels, diameter):
    x, y = pupil.coordinates.T
    rho, angle = 2 * np.hypot(x, y) / diameter, np.arctan2(y, x)
    rows = []
    for n, m, kind in labels:
        radial = rho**m * eval_jacobi((n - m) // 2, 0, m, 2 * rho**2 - 1)
        angular = np.cos(m * angle) if kind == "cos" else np.sin(m * angle)
        rows.append(np.sqrt(n + 1) * (np.sqrt(2) if m else 1) * radial * angular)
    return np.asarray(rows)


def optical_pupil(
    amplitude,
    opd_nm,
    support_m,
    diameter_m,
    wavelength_m,
    side,
    order,
    *,
    reference_side=64,
):
    """Sample the optical pupil on a separate grid and retain the physical basis."""
    inputs = (amplitude, opd_nm, support_m, diameter_m, wavelength_m)
    reference = make_pupil(*inputs, reference_side, order, fourier_response=False)
    coarse = make_pupil(*inputs, side, order, fourier_response=False)
    raw = raw_modes(reference, reference.labels, diameter_m)
    # Include piston when recovering the reference pupil's mean-subtracted modes.
    augmented = np.vstack([np.ones(raw.shape[1]), raw])
    transform = np.linalg.lstsq(augmented.T, reference.basis.T, rcond=None)[0].T
    basis = transform @ np.vstack(
        [
            np.ones(coarse.basis.shape[1]),
            raw_modes(coarse, reference.labels, diameter_m),
        ]
    )
    return replace(coarse, basis=basis)
