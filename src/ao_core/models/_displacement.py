"""Covariance matching and empirical OTFs for angular displacement arrays."""

import numpy as np
from scipy.fft import fft2


def _covariance_root(covariance, inverse=False):
    values, vectors = np.linalg.eigh(covariance)
    if values.min() <= 0:
        raise ValueError("Noise covariance is not positive")
    return (vectors * values ** (-0.5 if inverse else 0.5)) @ vectors.T


def _empirical_otf(samples, pixel_mas, side, oversample=4):
    """Positive CIC histogram with its assignment window removed in Fourier space.

    See Eq. (18) with p = 2 for the cloud-in-cell Fourier amplitude
    window. [Jing (2005)](https://arxiv.org/pdf/astro-ph/0409240)
    """
    fine = side * oversample
    xy = samples / (pixel_mas / oversample)
    ij = np.floor(xy).astype(np.int64)
    part = xy - ij
    histogram = np.zeros(fine * fine)
    for dx in (0, 1):
        for dy in (0, 1):
            weight = (
                (part[:, 0] if dx else 1 - part[:, 0])
                * (part[:, 1] if dy else 1 - part[:, 1])
                / len(samples)
            )
            index = ((ij[:, 1] + dy) % fine) * fine + (ij[:, 0] + dx) % fine
            histogram += np.bincount(index, weights=weight, minlength=fine * fine)
    fourier = fft2(histogram.reshape(fine, fine))
    k = np.rint(np.fft.fftfreq(side) * side).astype(int)
    window = np.sinc(k / fine) ** 2
    result = fourier[np.ix_(k % fine, k % fine)] / (window[:, None] * window[None])
    np.testing.assert_allclose(result[0, 0], 1.0, atol=1e-12)
    return result


def covariance_matched_displacement_otfs(
    samples, covariance, pixel_size, side, *, oversample=4
):
    """Return OTFs for a displacement record and its two contiguous halves.

    Samples have shape (frames, x/y). The covariance is in the square of the
    displacement unit and the pixel size is in that same displacement unit.
    Each record is centred and transformed to the supplied positive-definite
    covariance before cloud-in-cell estimation of its OTF. The calculation
    uses float64. The histogram assignment window is removed in Fourier space. The halves are matched
    separately and can remain correlated across the split.
    """
    samples, covariance = np.asarray(samples), np.asarray(covariance)
    if (
        samples.ndim != 2
        or samples.shape[1] != 2
        or len(samples) < 6
        or samples.dtype.kind not in "fiu"
        or not np.isfinite(samples).all()
        or covariance.shape != (2, 2)
        or covariance.dtype.kind not in "fiu"
        or not np.isfinite(covariance).all()
        or not np.isfinite(pixel_size)
        or pixel_size <= 0
        or type(side) is not int
        or side < 2
        or type(oversample) is not int
        or oversample < 1
    ):
        raise ValueError("Invalid displacement record, covariance or image sampling")
    samples, covariance = (
        samples.astype(float, copy=False),
        covariance.astype(float, copy=False),
    )
    if not np.allclose(
        covariance, covariance.T, rtol=1e-10, atol=1e-12 * np.max(np.abs(covariance))
    ):
        raise ValueError("Expected a symmetric displacement covariance")
    target_root = _covariance_root(covariance)

    def match(record):
        record = record - record.mean(axis=0)
        return (
            record @ _covariance_root(np.cov(record.T, bias=True), True) @ target_root
        )

    matched = match(samples)
    np.testing.assert_allclose(
        np.cov(matched.T, bias=True), covariance, rtol=1e-10, atol=1e-9
    )
    whole = _empirical_otf(matched, pixel_size, side, oversample)
    halves = np.stack(
        [
            _empirical_otf(match(half), pixel_size, side, oversample)
            for half in np.array_split(samples, 2)
        ]
    )
    return whole, halves
