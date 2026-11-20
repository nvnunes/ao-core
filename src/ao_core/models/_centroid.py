"""Pixel integration and thresholded centroid statistics.

Images have axes (y, x) and angular offsets are ordered (x, y).
Closed Gaussian integrals and Poisson sums precede numerical integration
over the centroid denominator.
"""

from functools import lru_cache
from typing import Literal

import numpy as np
from astropy import units as u
from scipy.interpolate import RectBivariateSpline, make_interp_spline
from scipy.special import erfcx, log_ndtr, ndtr
from scipy.stats import poisson

from ._units import _count


def _physical_values(value, unit, name):
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be an Astropy Quantity")
    if value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must have real values")
    array = np.asarray(value.to_value(unit), dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite nonempty values")
    return array


def _scalar(value, unit, name):
    array = _physical_values(value, unit, name)
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    return float(array)


@lru_cache(maxsize=8)
def _legendre_nodes(nodes):
    """Reuse the same Gauss–Legendre rule before scaling its integration interval."""
    points, weights = np.polynomial.legendre.leggauss(nodes)
    points.flags.writeable = weights.flags.writeable = False
    return points, weights


@lru_cache(maxsize=8)
def _zero_offset_weights(length, pixel_mas, pitch_mas):
    """Turn the cumulative cubic spline integral into five fixed pixel weights."""
    grid = np.arange(length + 1)
    basis = make_interp_spline(grid, np.eye(len(grid)), axis=0)
    lower = (np.arange(5) - 2.5) * pitch_mas / pixel_mas + length / 2
    upper = np.clip(lower + pitch_mas / pixel_mas, 0, length)
    difference = basis(upper) - basis(np.clip(lower, 0, length))
    # Image pixel i contributes to every cumulative value above i.
    return difference[:, :0:-1].cumsum(axis=1)[:, ::-1]


class PixelSpot:
    """Prepare a cumulative-image integral over a square 5-by-5 detector.

    The optical image contains incident-normalized flux per pixel, in y/x
    order. Its sum can be below one when the optical image is cropped.
    Supply image and detector pixel scales as angles per pixel and incident
    photon expectation in photons. Detector clipping preserves flux loss.
    """

    def __init__(
        self,
        image: np.ndarray,
        pixel_scale: u.Quantity,
        photons: u.Quantity,
        detector_pitch: u.Quantity,
    ):
        image = np.asarray(image, float)
        if (
            image.ndim != 2
            or min(image.shape) < 3
            or not np.isfinite(image).all()
            or image.min() < 0
        ):
            raise ValueError(
                "Expected finite nonnegative optical image with at least 3 pixels per axis"
            )
        self.photons = _scalar(photons, u.photon, "photons")
        self.pitch_mas = pitch_mas = _scalar(
            detector_pitch, u.mas / u.pix, "detector_pitch"
        )
        self.pixel_mas = _scalar(pixel_scale, u.mas / u.pix, "pixel_scale")
        if self.photons < 0 or min(self.pixel_mas, self.pitch_mas) <= 0:
            raise ValueError(
                "Photon expectation must be nonnegative and pixel scales positive"
            )
        self.shape = image.shape
        cumulative = np.pad(image.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        self.spline = RectBivariateSpline(
            np.arange(image.shape[0] + 1), np.arange(image.shape[1] + 1), cumulative
        )
        self.left, self.top = np.meshgrid(
            (np.arange(5) - 2.5) * pitch_mas, (np.arange(5) - 2.5) * pitch_mas
        )

    def flux(self, offsets: u.Quantity) -> u.Quantity:
        """Integrate flux in photons with shape (offsets, 25), in row-major order.

        Offsets are angular x/y pairs with shape (2,) or (offsets, 2).
        """
        r = _physical_values(offsets, u.mas, "offsets")
        if r.ndim not in (1, 2) or r.shape[-1] != 2:
            raise ValueError("Expected angular x/y offset pairs")
        r = r.reshape(-1, 2)
        left = (self.left.ravel()[None] - r[:, :1]) / self.pixel_mas + self.shape[1] / 2
        top = (self.top.ravel()[None] - r[:, 1:]) / self.pixel_mas + self.shape[0] / 2
        right, bottom = (
            left + self.pitch_mas / self.pixel_mas,
            top + self.pitch_mas / self.pixel_mas,
        )
        left, right = np.clip(left, 0, self.shape[1]), np.clip(right, 0, self.shape[1])
        top, bottom = np.clip(top, 0, self.shape[0]), np.clip(bottom, 0, self.shape[0])
        flux = self.photons * (
            self.spline.ev(bottom, right)
            - self.spline.ev(top, right)
            - self.spline.ev(bottom, left)
            + self.spline.ev(top, left)
        )
        if flux.min() < -1e-6:
            raise ValueError("Negative integrated pixel flux")
        return np.maximum(flux, 0) * u.photon

    def derivative_flux(self, derivative: np.ndarray) -> u.Quantity:
        """Integrate a signed optical response at zero offset.

        The input is in incident-normalized flux per pixel per unit perturbation,
        with the same shape as the optical image. The (1, 25) result is in
        photons for that perturbation. Apply any phase or displacement units
        associated with the perturbation separately.
        """
        derivative = np.asarray(derivative, float)
        if derivative.shape != self.shape or not np.isfinite(derivative).all():
            raise ValueError("Expected a finite derivative on the optical image grid")
        y, x = (
            _zero_offset_weights(length, self.pixel_mas, self.pitch_mas)
            for length in self.shape
        )
        positive = self.photons * (y @ np.maximum(derivative, 0) @ x.T)
        negative = self.photons * (y @ np.maximum(-derivative, 0) @ x.T)
        if min(positive.min(), negative.min()) < -1e-6:
            raise ValueError("Negative integrated pixel flux")
        return (np.maximum(positive, 0) - np.maximum(negative, 0)).reshape(
            1, 25
        ) * u.photon


def clipped_normal_laplace(counts, times, sigma=0.5, threshold=0.5):
    """Return the zeroth through second conditional Laplace moments.

    For thresholded weight W = max(n + sigma Z - threshold, 0), return
    E[W**j exp(-t W) | n] for j = 0, 1, 2. Z is standard normal.
    The numerical inputs are 1D count and inverse-electron arrays, with
    positive read noise sigma and threshold expressed in electrons. One
    detected photon contributes one electron. Each output is indexed by
    count, then Laplace argument.
    """
    a = np.asarray(counts)[:, None] - threshold
    t = np.asarray(times)[None]
    shifted = a - sigma**2 * t
    u = shifted / sigma
    density = np.exp(-0.5 * (a / sigma) ** 2) / np.sqrt(2 * np.pi)
    density = np.broadcast_to(density, u.shape)
    # exp(-ta+sigma^2t^2/2) Phi(u), evaluated stably at negative u.
    A = np.empty_like(u)
    negative = u < 0
    A[negative] = (
        density[negative] * np.sqrt(np.pi / 2) * erfcx(-u[negative] / np.sqrt(2))
    )
    exponent = np.broadcast_to(-t * a + sigma**2 * t**2 / 2, u.shape)
    A[~negative] = np.exp(exponent[~negative] + log_ndtr(u[~negative]))
    first = shifted * A + sigma * density
    second = (shifted**2 + sigma**2) * A + shifted * sigma * density
    # Direct subtraction loses precision for a large negative shifted mean.
    far = u < -12
    q = -u[far]
    first_series, second_series = np.zeros_like(q), np.zeros_like(q)
    coefficient = 1.0
    for k in range(1, 12):
        if k > 1:
            coefficient *= -(2 * k - 1)
        first_series += coefficient * q ** (-2 * k)
        second_series += 2 * k * coefficient * q ** (-2 * k - 1)
    first[far] = sigma * density[far] * first_series
    second[far] = sigma**2 * density[far] * second_series
    zeroth = ndtr(-a / sigma) + A
    assert first.min() > -1e-12 and second.min() > -1e-12
    return zeroth, np.maximum(first, 0), np.maximum(second, 0)


def _centroid_jacobian(product, ratio, numerator, m0, d0, d1, coordinates, weights):
    """Apply the product rule to the integrated centroid numerator and denominator."""
    derivative = np.einsum("bt,bpt,bti,t->bpi", product, d0 / m0, numerator, weights)
    derivative += np.einsum(
        "bt,bpt,pi,t->bpi", product, d1 / m0 - ratio * d0 / m0, coordinates, weights
    )
    return derivative


class DirectCentroid:
    """Integrate moments and mean-response Jacobians for supplied pixel fluxes.

    Flux arrays have shape (batch, 25) and are in photons. The calibrated
    centroid conversion is a 2-by-2 angular-per-pixel matrix in x/y order.
    Supply a scalar photon bound at least as large as every individual
    pixel expectation, positive read noise and a threshold in electrons, and
    the number of denominator quadrature nodes.
    Rows with at least 1,000 expected detector photons use finite Laplace
    quadrature for the default detector settings or a zero threshold. The
    interval scales with incident flux for the default settings and with
    the expected clipped pixel sum for a zero threshold. Fainter rows and
    other detector settings use the positive half-line. Custom settings
    raise FloatingPointError if the quadrature cannot resolve the centroid
    denominator mass. Choose `bright_quadrature="detected_flux"` to use a
    finite interval based on the total expected detector photons for every
    bright row, including custom thresholds. This policy retains the caller's
    denominator mass diagnostics without automatic fallback.
    """

    def __init__(
        self,
        photons: u.Quantity,
        conversion: u.Quantity,
        read_noise: u.Quantity = 0.5 * u.electron,
        threshold: u.Quantity = 0.5 * u.electron,
        nodes: int = 128,
        *,
        bright_quadrature: Literal["auto", "detected_flux"] = "auto",
    ):
        if bright_quadrature not in ("auto", "detected_flux"):
            raise ValueError("Unknown bright centroid quadrature policy")
        self.bright_quadrature = bright_quadrature
        self.photons = _scalar(photons, u.photon, "photons")
        self.conversion = _physical_values(
            conversion, u.mas / u.pix, "conversion"
        ).copy()
        if self.photons < 0 or self.conversion.shape != (2, 2):
            raise ValueError(
                "Expected a nonnegative photon bound and a 2-by-2 conversion"
            )
        if type(nodes) is not int or nodes < 1:
            raise ValueError("nodes must be a positive integer")
        xx, yy = np.meshgrid(np.arange(5) - 2, np.arange(5) - 2)
        self.coordinates = np.column_stack([xx.ravel(), yy.ravel()])
        self.read_noise = _scalar(read_noise, u.electron, "read_noise")
        self.threshold = _scalar(threshold, u.electron, "threshold")
        if self.read_noise <= 0:
            raise ValueError("read_noise must be positive")
        self.nodes = nodes
        points, weights = _legendre_nodes(nodes)
        s, weights = (points + 1) / 2, weights / 2
        scale = min(max(self.photons, 1), 10.0)
        self.times = s / (1 - s) / scale
        self.weights = weights / (1 - s) ** 2 / scale
        self.counts = np.arange(
            int(np.ceil(self.photons + 12 * np.sqrt(self.photons + 1) + 30)) + 1
        )
        values = clipped_normal_laplace(
            np.append(self.counts, self.counts[-1] + 1),
            self.times,
            sigma=self.read_noise,
            threshold=self.threshold,
        )
        self.conditional = tuple(x[:-1] for x in values)
        self.differences = tuple(np.diff(x, axis=0) for x in values[:2])

    def calculate(
        self, flux: u.Quantity, *, jacobian: bool = False
    ) -> dict[str, u.Quantity]:
        """Return thresholded centroid statistics for each supplied detector state.

        The fields are `mean` in mas, `covariance` in mas squared and the
        dimensionless denominator `mass_error`. If requested, `jacobian`
        has shape (batch, 25, 2) in mas per expected photon. It differentiates
        the centroid at the selected quadrature nodes and weights. The
        finite-interval calculation holds that interval fixed.
        """
        counts = _physical_values(flux, u.photon, "flux")
        result = self._calculate_counts(counts, jacobian=jacobian)
        units = {
            "mean": u.mas,
            "covariance": u.mas**2,
            "mass_error": u.one,
            "jacobian": u.mas / u.photon,
        }
        return {name: value * units[name] for name, value in result.items()}

    def _calculate_counts(self, flux, *, jacobian=False):
        if (
            flux.ndim != 2
            or flux.shape[1] != 25
            or not np.isfinite(flux).all()
            or flux.min() < 0
        ):
            raise ValueError("Expected finite nonnegative batch-by-25 pixel counts")
        total = flux.sum(axis=1)
        default_detector = (self.read_noise, self.threshold) == (0.5, 0.5)
        fixed = self.bright_quadrature == "detected_flux"
        bright = (total >= 1000.0) & (
            fixed or default_detector or self.threshold == 0.0
        )
        if not bright.any():
            return self._integrate(flux, jacobian=jacobian)
        if self.threshold == 0.0 and not fixed:
            _, clipped_mean, _ = clipped_normal_laplace(
                self.counts, [0.0], sigma=self.read_noise, threshold=0.0
            )
            probability = poisson.pmf(self.counts, flux[bright, :, None])
            total[bright] = (probability @ clipped_mean[:, 0]).sum(axis=1)
        levels = np.full(len(flux), -1, dtype=int)
        levels[bright] = np.floor(np.log2(total[bright] / 1000.0)).astype(int)
        combined = {}
        for level in np.unique(levels):
            selected = levels == level
            upper = None if level < 0 else 64.0 / total[selected].min()
            result = self._integrate(flux[selected], jacobian=jacobian, upper=upper)
            if selected.all():
                return result
            for name, values in result.items():
                if name not in combined:
                    combined[name] = np.empty(
                        (len(flux), *values.shape[1:]), dtype=values.dtype
                    )
                combined[name][selected] = values
        return combined

    def _integrate(self, flux, *, jacobian=False, upper=None):
        """Evaluate the same moment formulas on a half-line or finite interval.

        See Eqs. (1.1)–(1.2) for the add-one difference used by the
        photon-count Jacobian. [Last (2014)](https://arxiv.org/pdf/1203.3181)
        """
        times, weights = self.times, self.weights
        conditional, differences = self.conditional, self.differences
        if upper is not None:
            points, weights = _legendre_nodes(self.nodes)
            times = (points + 1) * upper / 2
            weights = weights * upper / 2
            values = clipped_normal_laplace(
                np.append(self.counts, self.counts[-1] + 1),
                times,
                sigma=self.read_noise,
                threshold=self.threshold,
            )
            conditional = tuple(x[:-1] for x in values)
            differences = tuple(np.diff(x, axis=0) for x in values[:2])
        probability = poisson.pmf(self.counts[None], flux.reshape(-1, 1))
        np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-12)
        shape = (*flux.shape, len(times))
        m0, m1, m2 = [(probability @ x).reshape(shape) for x in conditional]
        if not np.isfinite(m0).all() or (m0 <= 0).any():
            raise FloatingPointError("Centroid denominator moments lost positivity")
        product = np.exp(np.log(m0).sum(axis=1))
        ratio = m1 / m0
        numerator = np.einsum("bpt,pi->bti", ratio, self.coordinates)
        mean = np.einsum("bt,bti,t->bi", product, numerator, weights)
        diagonal = m2 / m0 - ratio**2
        extra = np.einsum(
            "bpt,pi,pj->btij", diagonal, self.coordinates, self.coordinates
        )
        raw = np.einsum(
            "bt,btij,t->bij",
            product,
            extra + np.einsum("bti,btj->btij", numerator, numerator),
            times * weights,
        )
        covariance = raw - np.einsum("bi,bj->bij", mean, mean)
        C = self.conversion
        answer = {"mean": mean @ C.T, "covariance": C[None] @ covariance @ C.T[None]}
        nonempty = 1 - np.prod(
            (
                probability @ ndtr((self.threshold - self.counts) / self.read_noise)
            ).reshape(flux.shape),
            axis=1,
        )
        mass = np.einsum("bt,bt,t->b", product, ratio.sum(axis=1), weights)
        answer["mass_error"] = mass - nonempty
        if (
            self.bright_quadrature == "auto"
            and (self.read_noise, self.threshold) != (0.5, 0.5)
            and np.any(np.abs(answer["mass_error"]) > 1e-8)
        ):
            if upper is not None:
                return self._integrate(flux, jacobian=jacobian)
            raise FloatingPointError(
                "Centroid quadrature did not resolve the denominator mass"
            )
        if jacobian:
            d0, d1 = [(probability @ x).reshape(shape) for x in differences]
            answer["jacobian"] = (
                _centroid_jacobian(
                    product, ratio, numerator, m0, d0, d1, self.coordinates, weights
                )
                @ C.T
            )
        return answer


def sample_centroid_errors(
    rng, flux, mean_centroids, coordinates, conversion, read_noise, threshold, count
):
    """Sample errors around the conditional mean of a thresholded centroid.

    Each frame draws a uniform state from a caller's spot ensemble. Supply
    expected electron counts with shape (states, pixels), detector coordinates
    with shape (pixels, 2), and conditional means with shape (states, 2).
    The 2-by-2 conversion maps detector positions into the units of the means.
    Read noise and threshold are in electrons. The result has shape (count, 2)
    in the units of the means. The caller's random generator advances only
    after input validation.
    """
    flux, mean_centroids, coordinates, conversion = map(
        np.asarray, (flux, mean_centroids, coordinates, conversion)
    )
    if (
        flux.ndim != 2
        or not all(flux.shape)
        or mean_centroids.shape != (len(flux), 2)
        or coordinates.shape != (flux.shape[1], 2)
        or conversion.shape != (2, 2)
        or any(
            values.dtype.kind not in "fiu" or not np.isfinite(values).all()
            for values in (flux, mean_centroids, coordinates, conversion)
        )
        or np.any(flux < 0)
    ):
        raise ValueError(
            "Expected finite detector arrays with matching state and pixel axes"
        )
    if not np.isfinite(read_noise) or read_noise < 0 or not np.isfinite(threshold):
        raise ValueError(
            "Expected nonnegative read noise and finite detector threshold"
        )
    _count(count, "count", minimum=0)
    errors = np.empty((count, 2))
    for start in range(0, count, 32_768):
        stop = min(start + 32_768, count)
        ids = rng.integers(len(flux), size=stop - start)
        counts = rng.poisson(flux[ids]).astype(float)
        counts += rng.normal(0, read_noise, counts.shape)
        weights = np.maximum(counts - threshold, 0.0)
        total = weights.sum(axis=1)
        xy = weights @ coordinates
        centroid = (
            np.divide(
                xy, total[:, None], out=np.zeros_like(xy), where=total[:, None] > 0
            )
            @ conversion.T
        )
        errors[start:stop] = centroid - mean_centroids[ids]
    return errors
