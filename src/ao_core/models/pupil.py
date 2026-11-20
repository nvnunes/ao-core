"""Modal pupil preparation and spatial Fourier quadrature."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from astropy import units as u
from numpy.polynomial.legendre import leggauss
from scipy.fft import fft2, fftfreq, ifftshift
from scipy.linalg import solve_triangular
from scipy.ndimage import map_coordinates, spline_filter
from scipy.special import eval_jacobi

from ._units import _quantity

__all__ = [
    "LocalCubicPupilResponse",
    "ModalPupil",
    "prepare_modal_pupil",
    "pupil_tilt_weights",
    "sampled_pupil_fourier",
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
        return _cubic_pupil_response(self.spectrum, self.dx, self.fft_size, fx, fy)


def _pupil_tilt_fit(amplitude):
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


def pupil_tilt_weights(
    amplitude: np.ndarray,
    *,
    sample_pitch: u.Quantity,
    reference_wavelength: u.Quantity,
) -> u.Quantity:
    """Return angular tilt weights for a sampled phase in x/y order.

    Supply the square pupil amplitude and centre-to-centre sample pitch.
    The fit uses amplitude-squared weights and includes a piston intercept.
    A boolean aperture mask gives equal weight to illuminated samples.
    Contract the resulting (2, y, x) weights, in mas per phase radian, with
    a phase grid at the reference wavelength to obtain the angular tilt.
    """
    amplitude = np.asarray(amplitude)
    if (
        amplitude.ndim != 2
        or amplitude.shape[0] != amplitude.shape[1]
        or len(amplitude) < 2
        or amplitude.dtype.kind not in "bfiu"
        or not np.isfinite(amplitude).all()
        or np.any(amplitude < 0)
        or not np.any(amplitude > 0)
    ):
        raise ValueError("Expected a finite nonnegative square pupil amplitude")
    pitch = float(_quantity(sample_pitch, u.m, "sample_pitch", 0).value)
    wavelength = float(
        _quantity(reference_wavelength, u.m, "reference_wavelength", 0).value
    )
    if pitch <= 0 or wavelength <= 0:
        raise ValueError("Sample pitch and reference wavelength must be positive")
    # The normalized ramp coordinates span a width of two over n pixel cells.
    gradient = _pupil_tilt_fit(amplitude.astype(float, copy=False)) * (
        2 / (len(amplitude) * pitch)
    )
    radians_to_mas = 1000 / ARCSEC
    return gradient * (wavelength / (2 * np.pi) * radians_to_mas) * (u.mas / u.rad)


def _cubic_pupil_response(spectrum, pitch, size, fx, fy):
    if np.max(np.hypot(fx, fy)) >= 0.48 / pitch:
        raise ValueError("Fourier query exceeds sampled-pupil Nyquist range")
    indices = np.array(
        [
            np.atleast_1d(fy) * size * pitch % size,
            np.atleast_1d(fx) * size * pitch % size,
        ]
    )
    values = np.array(
        [
            map_coordinates(v.real, indices, order=3, mode="grid-wrap", prefilter=False)
            + 1j
            * map_coordinates(
                v.imag, indices, order=3, mode="grid-wrap", prefilter=False
            )
            for v in spectrum
        ]
    )
    values = values.reshape(len(spectrum), *np.shape(fx))
    return values * np.exp(1j * np.pi * pitch * (fx + fy))


def _pupil_spline(image, size):
    side = len(image)
    offset = (size - side) // 2
    field = np.pad(image, ((offset, size - side - offset),) * 2)
    values = fft2(ifftshift(field)).conj()
    return spline_filter(values.real, order=3, mode="grid-wrap") + 1j * spline_filter(
        values.imag, order=3, mode="grid-wrap"
    )


def _frequency_arrays(fx, fy):
    fx, fy = np.asarray(fx), np.asarray(fy)
    if (
        fx.shape != fy.shape
        or not fx.size
        or any(f.dtype.kind not in "fiu" or not np.isfinite(f).all() for f in (fx, fy))
    ):
        raise ValueError("Require matching nonempty arrays of finite real frequencies")
    return fx, fy


def sampled_pupil_fourier(weights, support, fx, fy, *, fft_size):
    """Evaluate positive-exponent Fourier projections for real pupil weights.

    Weights have shape (mode, y, x) and include their pupil normalization.
    Support is the span between first and last pupil samples in metres. The
    padded grid is even, and native weights are embedded on its centred pixels.
    Query arrays match in shape and give spatial frequencies in cycles/m.
    Returns shape (mode, *frequency_shape). Work proceeds one mode at a time
    to avoid storing a padded transform for every mode.
    """
    weights = np.asarray(weights)
    fx, fy = _frequency_arrays(fx, fy)
    if (
        weights.ndim != 3
        or weights.shape[1] != weights.shape[2]
        or weights.shape[1] < 2
        or weights.dtype.kind not in "fiu"
        or not len(weights)
        or not np.isfinite(weights).all()
        or type(fft_size) is not int
        or fft_size < weights.shape[1]
        or fft_size % 2
        or (fft_size - weights.shape[1]) % 2
    ):
        raise ValueError("Require real sampled weights on an even square padded grid")
    if not np.isfinite(support) or support <= 0:
        raise ValueError("Require positive finite pupil support")
    weights = weights.astype(float, copy=False)
    pitch = support / (weights.shape[1] - 1)
    result = np.empty((len(weights), *fx.shape), dtype=complex)
    for index, image in enumerate(weights):
        result[index] = _cubic_pupil_response(
            _pupil_spline(image, fft_size)[None], pitch, fft_size, fx, fy
        )[0]
    return result


class LocalCubicPupilResponse:
    """Positive-exponent Fourier response of real pupil weights.

    The weights have shape (channel, y, x) with the caller's normalization.
    Supply the sample pitch as `pitch_m` or the first-to-last sample-centre
    span as `support`, both in metres. Zero padding retains that pitch. The FFT
    is evaluated with a local four-node Lagrange stencil per axis. Query
    coordinates are cycles/m. For an even pupil, the full stencil must lie
    inside the signed FFT bins.
    """

    def __init__(self, weights, support=None, *, fft_size, pitch_m=None):
        weights = np.asarray(weights)
        if (
            weights.ndim != 3
            or weights.shape[1] != weights.shape[2]
            or weights.shape[1] < 2
            or weights.dtype.kind not in "fiu"
            or not len(weights)
            or not np.isfinite(weights).all()
            or type(fft_size) is not int
            or fft_size < weights.shape[1]
            or fft_size % 2
        ):
            raise ValueError("Invalid pupil weights, support or Fourier grid")
        weights = weights.astype(float, copy=False)
        side, size = weights.shape[1], fft_size
        if (support is None) == (pitch_m is None):
            raise ValueError("Supply either the sample-centre span or the sample pitch")
        dx = support / (side - 1) if pitch_m is None else pitch_m
        if not np.isfinite(dx) or dx <= 0:
            raise ValueError("Sample pitch must be positive and finite")
        offset = (size - side) // 2
        fields = np.pad(
            weights,
            ((0, 0), (offset, size - side - offset), (offset, size - side - offset)),
        )
        transform = fft2(np.fft.ifftshift(fields, axes=(-2, -1))).conj()
        del fields
        frequency = fftfreq(size, dx)
        displacement = offset + (side - 1) / 2 - size / 2
        # Embedded pixel coordinates differ from centred physical coordinates.
        transform *= np.exp(
            -2j * np.pi * displacement * dx * (frequency[:, None] + frequency[None])
        )
        self.transform, self.size, self.dx = transform, size, dx
        self._half_sample_centre = side % 2 == 0

    def __call__(self, fx, fy):
        """Evaluate the response in caller channel order at the supplied nodes."""
        fx, fy = _frequency_arrays(fx, fy)
        if max(np.max(abs(fx)), np.max(abs(fy))) >= 0.49 / self.dx:
            raise ValueError("Sampled-pupil query exceeds the validated Nyquist range")
        x, y = fx * self.size * self.dx, fy * self.size * self.dx
        ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
        if self._half_sample_centre and any(
            index.min() - 1 < -self.size // 2 or index.max() + 2 >= self.size // 2
            for index in (ix, iy)
        ):
            raise ValueError("Cubic pupil interpolation stencil crosses Nyquist")

        def weights(t):
            return [
                -t * (t - 1) * (t - 2) / 6,
                (t + 1) * (t - 1) * (t - 2) / 2,
                -(t + 1) * t * (t - 2) / 2,
                (t + 1) * t * (t - 1) / 6,
            ]

        wx, wy = weights(x - ix), weights(y - iy)
        result = np.zeros((len(self.transform), *fx.shape), dtype=complex)
        for i in range(4):
            for j in range(4):
                result += (
                    wx[i]
                    * wy[j]
                    * self.transform[
                        :, (iy + j - 1) % self.size, (ix + i - 1) % self.size
                    ]
                )
        return result


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
    """Keep native pupil/NCPA sampling; optionally omit unused Fourier projections.

    See the arbitrary-aperture Zernike orthogonalization for the basis
    construction. Here the pupil intensity defines the inner product.
    [Upton & Ellerbroek (2004)](https://doi.org/10.1364/OL.29.002840)
    """
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
    transform = None
    if fourier_response:
        transform = []
        for row in weights:
            image = np.zeros((side, side))
            image[mask] = row
            transform.append(_pupil_spline(image, size))
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
