"""Gaussian EE/FWHM prescriptions for reconstructor design noise."""

import numpy as np
from scipy.special import erf, gammaln, ndtr


def subtract_clip_moments(
    flux, threshold: float, ron: float, excess: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and variance after subtracting a threshold and clipping.

    Match OOMAO 14.5's Gaussian approximation above ten photons and its 30-term
    Poisson/read-noise sum below ten. Background is zero in this instrument model.
    All quantities use electrons. Read noise must be positive; excess applies to
    the Gaussian branch as in the reference implementation.

    See Eqs. (32)–(33) for the clipped-normal mean and variance.
    [Foi et al. (2008)](https://doi.org/10.1109/TIP.2008.2001399)
    """
    flux = np.asarray(flux, dtype=float)
    if (
        not np.isfinite(flux).all()
        or np.any(flux < 0)
        or not np.isfinite([threshold, ron, excess]).all()
        or ron <= 0
        or excess <= 0
    ):
        raise ValueError(
            "Finite nonnegative flux and positive read/excess noise required"
        )
    mean, second = np.zeros_like(flux), np.zeros_like(flux)
    bright = flux >= 10

    def moments(value, sigma):
        offset = value - threshold + 1e-20
        probability = 0.5 * (1 + erf(offset / (np.sqrt(2) * sigma)))
        density = np.exp(-(offset**2) / (2 * sigma**2)) / np.sqrt(2 * np.pi)
        return (
            offset * probability + sigma * density,
            (offset**2 + sigma**2) * probability + offset * sigma * density,
        )

    mean[bright], second[bright] = moments(
        flux[bright], np.sqrt(excess * flux[bright] + ron**2)
    )
    faint = flux[~bright]
    for k in range(30):
        probability = np.exp(-faint - gammaln(k + 1)) * faint**k
        m, s = moments(float(k), ron)
        mean[~bright] += probability * m
        second[~bright] += probability * s
    return mean, np.maximum(second - mean**2, 0)


def centroid_noise_prior(
    photons: float,
    ee: float,
    fwhm_mas: float,
    pixel_scale_mas: float,
    detector_pixels: int,
    threshold_e: float,
    read_noise_e: float,
) -> dict:
    """Calculate the reference Gaussian prior in detector-pixel units.

    EE scales the total modeled Gaussian flux; FWHM specifies its width in mas.
    The reference uses a 204-square fine grid, sum-bins to 51-square pixels and
    centroids within the central detector-sized square. Returned arrays retain
    the model image and threshold moments; covariance has shape (2, 2), in pixel².
    Reject invalid metrics, sampling or empty thresholded light.
    """
    if (
        not np.isfinite([photons, ee, fwhm_mas, pixel_scale_mas]).all()
        or min(photons, ee, fwhm_mas, pixel_scale_mas) <= 0
        or ee > 1
        or not 0 < detector_pixels <= 51
    ):
        raise ValueError(
            "Positive finite prior values and valid detector size required"
        )
    fine_axis = np.arange(204) - 101.5
    x, y = np.meshgrid(fine_axis, fine_axis)
    sigma = fwhm_mas / 2.355 / (pixel_scale_mas / 4)
    fine = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    fine *= ee * photons / fine.sum()
    frame = fine.reshape(51, 4, 51, 4).sum(axis=(1, 3))
    mean, variance = subtract_clip_moments(frame, threshold_e, read_noise_e)
    axis = np.arange(51) - 25
    selected = np.sort(np.argsort(np.abs(axis), kind="stable")[:detector_pixels])
    mask = np.zeros((51, 51), dtype=bool)
    mask[np.ix_(selected, selected)] = True
    mass = mean[mask].sum()
    if mass <= 0:
        raise ValueError("No prior light remains after thresholding")
    x, y = np.meshgrid(axis, axis)
    centers = np.array([(mean * mask * x).sum(), (mean * mask * y).sum()]) / mass
    centered = np.stack((x - centers[0], y - centers[1]), axis=-1)
    covariance = (
        np.einsum("ija,ijb,ij->ab", centered, centered, variance * mask) / mass**2
    )
    # The adopted Gaussian prior retains independent per-axis variances.
    covariance = np.diag(np.diag(covariance))
    return {
        "frame": frame,
        "mean": mean,
        "variance": variance,
        "covariance_pixel2": covariance,
        "centroid_mass_e": float(mass),
    }


def centered_measurement_noise(
    photons: np.ndarray,
    ee: np.ndarray,
    fwhm_mas: np.ndarray,
    *,
    pixel_mas: float,
    angular_scale_mas: float,
    read_noise: float,
    threshold: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return C_eta^ctr and detector moments, with no fitted coefficients.

    Photons are incident counts per frame. angular_scale_mas is the native
    centroid-to-tilt normalization used to construct the reconstructor,
    rather than the fitted sensor response. All NGS inputs have length N.
    """
    photons, ee, fwhm_mas = [
        np.atleast_1d(v).astype(float) for v in (photons, ee, fwhm_mas)
    ]
    if not (photons.ndim == 1 and photons.shape == ee.shape == fwhm_mas.shape):
        raise ValueError("Photons, EE and FWHM must have one value per NGS.")
    if not np.isfinite(
        np.r_[
            photons, ee, fwhm_mas, pixel_mas, angular_scale_mas, read_noise, threshold
        ]
    ).all():
        raise ValueError("Detector inputs must be finite.")
    if np.any(
        np.r_[photons, ee, fwhm_mas, pixel_mas, angular_scale_mas, read_noise] <= 0
    ):
        raise ValueError(
            "Photon counts, spot metrics and detector scales must be positive."
        )
    if threshold < 0:
        raise ValueError("Native frame threshold must be nonnegative.")
    mean, variance = _subtract_clip_moments(np.array([0.0]), threshold, read_noise)
    coordinates = np.arange(-101.5, 102.0)
    moments, noise = [], []
    x = np.broadcast_to(np.arange(-2, 3), (5, 5))
    for count, encircled, width in zip(photons, ee, fwhm_mas, strict=True):
        sigma = width / 2.355 / (pixel_mas / 4)
        profile = np.exp(-(coordinates**2) / (2 * sigma**2))
        profile /= profile.sum()
        binned = profile.reshape(51, 4).sum(axis=1)
        moments.append(np.sum(binned * np.arange(-25, 26) ** 2))
        frame = np.outer(binned[23:28], binned[23:28]) * count * encircled
        measured, pixel_variance = _subtract_clip_moments(frame, threshold, read_noise)
        mass = measured.sum()
        if mass <= 0:
            raise ValueError("No light remains after native frame thresholding.")
        centroid = (measured * x).sum() / mass
        noise.append(
            angular_scale_mas**2 * (((x - centroid) / mass) ** 2 * pixel_variance).sum()
        )
    signal = photons * ee
    details = {
        "photons": photons,
        "signal": signal,
        "spotSecondMomentPixel2": np.asarray(moments),
        "angularScaleMasPerPixel": angular_scale_mas,
        "zeroLightMean": float(mean[0]),
        "zeroLightVariance": float(variance[0]),
        "threshold": threshold,
        "readNoise": read_noise,
        "pixelMas": pixel_mas,
        "fittedParameterCount": 0,
    }
    return np.diag(np.repeat(noise, 2)), details


def _subtract_clip_moments(flux: np.ndarray, threshold: float, ron: float):
    """Moments of max(Poisson(flux)+Normal(0,ron)-threshold,0).

    Match the native detector prior: Gaussian photon approximation at >=10
    photons/pixel, otherwise sum Poisson orders 0..29. Only the central 5-by-5
    mask is evaluated; pixels outside it cannot affect the centroid.

    See Eqs. (32)–(33) for the clipped-normal mean and variance.
    [Foi et al. (2008)](https://doi.org/10.1109/TIP.2008.2001399)
    """
    mean, second = np.zeros_like(flux), np.zeros_like(flux)
    bright = flux >= 10

    def clipped(offset, sigma):
        probability = ndtr(offset / sigma)
        density = np.exp(-0.5 * (offset / sigma) ** 2) / np.sqrt(2 * np.pi)
        return (
            offset * probability + sigma * density,
            (offset**2 + sigma**2) * probability + offset * sigma * density,
        )

    mean[bright], second[bright] = clipped(
        flux[bright] - threshold, np.sqrt(flux[bright] + ron**2)
    )
    faint = flux[~bright]
    for order in range(30):
        probability = np.exp(-faint - gammaln(order + 1)) * faint**order
        m, s = clipped(order - threshold, ron)
        mean[~bright] += probability * m
        second[~bright] += probability * s
    return mean, np.maximum(second - mean**2, 0)
