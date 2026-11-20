"""NGS intensity, calibrated centroid gain and independent runtime innovations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from astropy import units as u
from scipy.fft import fft2, fftshift
from scipy.stats import poisson

from ._centroid import (
    DirectCentroid,
    PixelSpot,
    _centroid_jacobian,
    _legendre_nodes,
    clipped_normal_laplace,
)


class _CentroidMoments:
    """Deterministic thresholded centroid moments in calibrated angular units.

    Reuse Hybrid's exact Poisson/read-noise denominator for faint reads. For
    bright reads, evaluate Poisson generating functions in pixels with >=50
    expected photons, where clipping probability is negligible; evaluate the
    remaining pixels with the same exact clipped-normal Poisson sums. The
    finite bright integration interval is the Hybrid validation prescription.
    """

    def __init__(self, conversion, read_noise=0.5, threshold=0.5, nodes=128):
        self.conversion = np.asarray(conversion, float)
        self.read_noise, self.threshold, self.nodes = read_noise, threshold, nodes
        x, y = np.meshgrid(np.arange(5) - 2, np.arange(5) - 2)
        self.coordinates = np.column_stack([x.ravel(), y.ravel()])

    def calculate(self, flux, *, jacobian=False):
        flux = np.asarray(flux, float)
        if flux.ndim != 2 or flux.shape[1] != 25 or np.any(flux < 0):
            raise ValueError("Require nonnegative batch-by-25 photon expectations")
        answer = {key: [] for key in ("mean", "covariance", "mass_error")}
        if jacobian:
            answer["jacobian"] = []
        for row in flux:
            if row.sum() < 1000 or (self.read_noise, self.threshold) != (0.5, 0.5):
                # Bound the photon sum by the brightest pixel, not total light.
                exact = DirectCentroid(
                    float(row.max()) * u.photon,
                    self.conversion * u.mas / u.pix,
                    self.read_noise * u.electron,
                    self.threshold * u.electron,
                    self.nodes,
                )
                result = exact._calculate_counts(row[None], jacobian=jacobian)
            else:
                result = self._bright(row, jacobian=jacobian)
            for key, values in answer.items():
                values.append(result[key][0])
        return {key: np.asarray(values) for key, values in answer.items()}

    def _bright(self, flux, *, jacobian=False):
        points, weights = _legendre_nodes(self.nodes)
        upper = 64 / flux.sum()
        times, weights = (points + 1) * upper / 2, weights * upper / 2
        m0, m1, m2 = (np.zeros((25, self.nodes)) for _ in range(3))
        faint = flux < 50
        bound = float(flux[faint].max()) if faint.any() else 0.0
        counts = np.arange(int(np.ceil(bound + 12 * np.sqrt(bound + 1) + 30)) + 1)
        conditional = clipped_normal_laplace(
            np.append(counts, counts[-1] + 1) if jacobian else counts,
            times,
            self.read_noise,
            self.threshold,
        )
        probability = poisson.pmf(counts[None], flux[faint, None])
        for target, value in zip((m0, m1, m2), conditional, strict=True):
            target[faint] = probability @ (value[:-1] if jacobian else value)
        count = flux[~faint, None]
        decay = np.exp(-times)[None]
        zero = np.exp(
            self.threshold * times
            + 0.5 * self.read_noise**2 * times**2
            + count * np.expm1(-times)
        )
        first = count * decay - self.threshold - self.read_noise**2 * times
        m0[~faint] = zero
        m1[~faint] = zero * first
        m2[~faint] = zero * (first**2 + count * decay + self.read_noise**2)
        product = np.exp(np.log(m0).sum(0))
        ratio = m1 / m0
        numerator = ratio.T @ self.coordinates
        mean = np.einsum("t,ti,t->i", product, numerator, weights)
        extra = np.einsum(
            "pt,pi,pj->tij", m2 / m0 - ratio**2, self.coordinates, self.coordinates
        )
        raw = np.einsum(
            "t,tij,t->ij",
            product,
            extra + np.einsum("ti,tj->tij", numerator, numerator),
            times * weights,
        )
        covariance = raw - np.outer(mean, mean)
        c = self.conversion
        # Empty-read probability is negligible when total detected flux >=1000.
        mass = np.sum(product * ratio.sum(0) * weights)
        answer = {
            "mean": (c @ mean)[None],
            "covariance": (c @ covariance @ c.T)[None],
            "mass_error": np.array([mass - 1]),
        }
        if jacobian:
            d0, d1 = np.empty_like(m0), np.empty_like(m1)
            d0[faint] = probability @ np.diff(conditional[0], axis=0)
            d1[faint] = probability @ np.diff(conditional[1], axis=0)
            laplace_delta = np.expm1(-times)
            d0[~faint] = zero * laplace_delta
            d1[~faint] = zero * (first * laplace_delta + decay)
            derivative = _centroid_jacobian(
                product[None],
                ratio[None],
                numerator[None],
                m0[None],
                d0[None],
                d1[None],
                self.coordinates,
                weights,
            )
            # The finite integration interval scales as 64 / sum(flux).
            integrand_slope = product[:, None] * (
                -ratio.sum(axis=0)[:, None] * numerator
                - (m2 / m0 - ratio**2).T @ self.coordinates
            )
            rescaling = (
                -(mean + np.einsum("ti,t,t->i", integrand_slope, times, weights))
                / flux.sum()
            )
            answer["jacobian"] = (derivative + rescaling) @ c.T
        return answer


def _derivative_lags(basis, coherence, y, x, size):
    """Accumulate one modal block over unordered pupil pairs."""
    output = np.zeros((size * size, basis.shape[1]))
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dy, dx = y[i] - y[j], x[i] - x[j]
            if dy < 0:
                dy += size
            if dx < 0:
                dx += size
            lag = dy * size + dx
            weight = coherence[i, j]
            for mode in range(basis.shape[1]):
                output[lag, mode] += weight * (basis[i, mode] - basis[j, mode])
    return output


@lru_cache(maxsize=1)
def _derivative_accumulator():
    try:
        from numba import njit
    except ImportError:
        return None
    return njit(nogil=True)(_derivative_lags)


def optical_mean(pupil, covariance, derivative_modes=(), *, derivative_workers=1):
    """Gaussian pupil coherence -> incident-normalized image and phase responses.

    Derivatives are with respect to added phase in the supplied mode basis.
    Tilt is absent from the shape covariance. No saved OOMAO PSF is consumed.
    """
    mask = pupil.amplitude > 0
    y, x = np.nonzero(mask)
    size = pupil.optical_size
    modes = tuple(derivative_modes)
    amplitude = pupil.amplitude[mask]
    normalization = size**2 * np.sum(amplitude**2)
    if not modes and not np.any(covariance):
        # With no random phase the OTF is a pupil autocorrelation. The
        # half-pixel centering is the same phase ramp on the pupil field.
        field = np.zeros((size, size), dtype=complex)
        field[y, x] = amplitude * np.exp(1j * (pupil.ncpa - np.pi * (x + y) / size))
        return np.abs(fftshift(fft2(field))) ** 2 / normalization, []

    phase = pupil.basis.T @ covariance @ pupil.basis
    diagonal = np.diag(phase).copy()
    phase *= -2
    phase += diagonal[:, None] + diagonal[None]
    phase *= -0.5
    # NGS optics have zero static phase. Keep their pair weights real until
    # after lag accumulation, avoiding dense complex arrays for every mode.
    static_phase = np.any(pupil.ncpa)
    if static_phase:
        coherence = np.exp(phase + 1j * (pupil.ncpa[:, None] - pupil.ncpa[None]))
    else:
        np.exp(phase, out=phase)
        coherence = phase
    coherence *= amplitude[:, None]
    coherence *= amplitude[None]
    del phase
    lag_ids = (y[:, None] - y) % size
    lag_ids *= size
    lag_ids += (x[:, None] - x) % size
    lag_ids = lag_ids.ravel()
    # PixelSpot treats samples as cell integrals, with centres at j+1/2.
    # Put the optical origin at the detector's central cell centre, matching
    # the native even-grid diffraction phasor instead of introducing half-pixel tilt.
    frequency = np.fft.fftfreq(size)
    centering = np.exp(-1j * np.pi * (frequency[:, None] + frequency[None]))

    def propagate(values, factor=1):
        otf = np.bincount(lag_ids, weights=values.real.ravel(), minlength=size**2)
        if np.iscomplexobj(values):
            otf = otf + 1j * np.bincount(
                lag_ids, weights=values.imag.ravel(), minlength=size**2
            )
        return (
            fftshift(fft2(otf.reshape(size, size) * centering * factor)).real
            / normalization
        )

    image = propagate(coherence)
    if image.min() < -1e-12 or abs(image.sum() - 1) > 1e-10:
        raise ValueError("Optical mean lost positivity or incident flux")
    accumulate = (
        _derivative_accumulator()
        if modes
        and not static_phase
        and np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14)
        else None
    )
    if accumulate is None:
        derivatives = [
            propagate((pupil.basis[m, :, None] - pupil.basis[m, None]) * coherence, 1j)
            for m in modes
        ]
    else:
        del lag_ids
        reverse = -np.arange(size) % size
        opposite = (size * reverse[:, None] + reverse[None]).ravel()
        blocks = [modes[start : start + 4] for start in range(0, len(modes), 4)]

        def derivative_images(block):
            basis = np.ascontiguousarray(pupil.basis[list(block)].T)
            otf = accumulate(basis, coherence, y, x, size)
            # Reversing each pair flips its lag and the phase derivative.
            otf -= otf[opposite]
            images = fft2(otf.T.reshape(-1, size, size) * centering * 1j)
            return fftshift(images, axes=(-2, -1)).real / normalization

        if derivative_workers == 1:
            derivatives = [
                image for block in blocks for image in derivative_images(block)
            ]
        else:
            with ThreadPoolExecutor(max_workers=derivative_workers) as pool:
                derivatives = [
                    image
                    for result in pool.map(derivative_images, blocks)
                    for image in result
                ]
    return np.maximum(image, 0), derivatives


def diffraction_calibration(optical, pixel, pitch, *, propagation=None):
    """Reproduce the native noise-free three-position diffraction slope fit.

    Native gainCalibration uses nStep=floor(5/3)*2 and tipStep=pixelScale/4.
    The inferred conversion is angular mas per raw centroid pixel; no OOMAO
    gain measurement or e010 sampling strength is supplied.
    """
    covariance = np.zeros((len(optical.basis),) * 2)
    image, _ = (
        optical_mean(optical, covariance)
        if propagation is None
        else propagation.calculate(covariance)
    )
    offsets = np.column_stack([np.arange(3) * pitch / 4, np.zeros(3)])
    spot = PixelSpot(image, pixel * u.mas / u.pix, 1 * u.photon, pitch * u.mas / u.pix)
    flux = spot.flux(offsets * u.mas).value
    x, y = np.meshgrid(np.arange(5) - 2, np.arange(5) - 2)
    coords = np.column_stack([x.ravel(), y.ravel()])
    means = flux @ coords / flux.sum(1)[:, None]
    slope = np.polyfit(offsets[:, 0], means[:, 0], 1)[0]
    if slope <= 0:
        raise ValueError("Invalid diffraction calibration slope")
    return np.eye(2) / slope, {
        "offsets_mas": offsets.tolist(),
        "raw_centroids": means.tolist(),
        "conversion_mas_per_pixel": 1 / slope,
        "physical_pixel_mas": pitch,
    }


def sensor_response(
    image,
    derivatives,
    pixel,
    pitch,
    photons,
    conversion,
    count,
    nodes=128,
    *,
    offsets_mas=None,
    read_noise_e=0.5,
    threshold_e=0.5,
):
    """Fixed-range Hybrid sensing gain, centred runtime noise and HO response.

    The image and derivative arrays use incident-normalized intensity.
    Pixel scales and conversion use mas per pixel. Photon numbers are incident
    expectations per frame, and the noise and threshold are in electrons.
    The gain is dimensionless, the conditional covariance is in mas squared
    and the phase responses are mas per radian of the supplied mode.
    The displacement ranges determine gain and state-averaged
    conditional covariance. Supplied offsets must span both axes. Contract
    the optical pixel derivatives with the analytic centroid Jacobian at
    zero displacement, retaining the random centroid denominator. This is a
    mean-HO-shape closure, not an instantaneous nonlinear sensor model.
    """
    radii = (10, 30, 50, 70) if count == 1 else (20, 40)
    offsets = np.array(
        ([[0.0, 0.0]] if count == 1 else [])
        + [
            [r * s if a == 0 else 0.0, r * s if a == 1 else 0.0]
            for r in radii
            for a in (0, 1)
            for s in (-1, 1)
        ]
    )
    if offsets_mas is not None:
        offsets = np.asarray(offsets_mas, dtype=np.float64)
        if (
            offsets.ndim != 2
            or offsets.shape[1] != 2
            or len(offsets) < 2
            or not np.isfinite(offsets).all()
            or np.linalg.matrix_rank(offsets) != 2
        ):
            raise ValueError("offsets_mas must be finite x/y pairs that span both axes")
    spot = PixelSpot(
        image, pixel * u.mas / u.pix, photons * u.photon, pitch * u.mas / u.pix
    )
    moments = _CentroidMoments(conversion, read_noise_e, threshold_e, nodes=nodes)
    flux = spot.flux(offsets * u.mas).value
    result = moments.calculate(flux)
    gains = np.linalg.lstsq(offsets, result["mean"], rcond=None)[0].T
    covariance = result["covariance"].mean(0)
    zero = spot.flux(np.zeros(2) * u.mas).value
    count_derivatives = [spot.derivative_flux(mode).value[0] for mode in derivatives]
    centre = moments.calculate(zero, jacobian=bool(count_derivatives))
    responses = (
        (np.asarray(count_derivatives) @ centre["jacobian"][0]).T
        if count_derivatives
        else np.empty((2, 0))
    )
    if np.linalg.eigvalsh(covariance).min() <= 0 or not np.isfinite(gains).all():
        raise ValueError("Invalid PSF-driven sensor moments")
    return (
        gains,
        covariance,
        responses,
        {
            "gain": gains.tolist(),
            "runtime_covariance_mas2": covariance.tolist(),
            "maximum_denominator_mass_error": float(abs(result["mass_error"]).max()),
            "detected_photons_range": [
                float(flux.sum(1).min()),
                float(flux.sum(1).max()),
            ],
            "mean_center_mas": centre["mean"][0].tolist(),
            "offsets_mas": offsets,
            "conditional_mean_mas": result["mean"],
            "conditional_covariance_mas2": result["covariance"],
        },
    )
