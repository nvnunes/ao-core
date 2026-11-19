"""Finite-window metric variance and the native modal lag transform."""

import math
from functools import lru_cache

import numpy as np
from astropy import units as u
from scipy.fft import irfft
from scipy.signal import CZT

TEMPORAL_WORK_ELEMENTS = 1 << 20
LAG_SPACING_DIVISOR = 16


@lru_cache(maxsize=1)
def _lag_transform(bins, count, retained):
    """Cache one small chirp-z transform for the worker's current lag grid."""
    return CZT(bins, m=retained, w=np.exp(2j * np.pi / count))


def _lag_block(spectrum, count, retained):
    """Use chirp-z for short windows with the irfft endpoint weights."""
    if retained >= count // 4:
        return irfft(spectrum * count, n=count, axis=0)[:retained]
    positive = spectrum[: count // 2 + 1].copy()
    positive[0] *= 0.5
    if count % 2 == 0 and len(positive) == count // 2 + 1:
        positive[-1] *= 0.5
    return 2 * _lag_transform(len(positive), count, retained)(positive, axis=0).real


def atmospheric_lags(spectrum, count, retained):
    """Evaluate native-frame lags, tiling independent mode pairs on large grids."""
    block_size = max(1, TEMPORAL_WORK_ELEMENTS // max(count, len(spectrum)))
    modes = math.prod(spectrum.shape[1:])
    if modes <= block_size:
        return _lag_block(spectrum, count, retained)
    flattened = spectrum.reshape(len(spectrum), modes)
    lags = np.empty((retained, modes), dtype=float)
    for start in range(0, modes, block_size):
        columns = slice(start, start + block_size)
        lags[:, columns] = _lag_block(flattened[:, columns], count, retained)
    return lags.reshape(retained, *spectrum.shape[1:])


def sampled_lag_indices(frames):
    """Sample native lags densely near zero and always include the window endpoint."""
    last = frames - 1
    indices = [0]
    while indices[-1] < last:
        lag = indices[-1]
        indices.append(min(last, lag + max(1, lag // LAG_SPACING_DIVISOR)))
    return np.asarray(indices)


def integrate_metric_lags(model, sampled, window, dt):
    """Integrate a native or interpolated covariance over one triangular exposure window."""
    indices, values = sampled
    count = len(window)
    if model == "fast":
        return 2 * dt * (values[:, :count] @ window)
    knots = sampled_lag_indices(count)
    selected = values[:, np.searchsorted(indices, knots)]
    lags = np.arange(count)
    return (
        2
        * dt
        * (np.array([np.interp(lags, knots, metric) for metric in selected]) @ window)
    )


def exposure_variance(
    covariance: u.Quantity,
    frame_interval: u.Quantity,
    frames: int,
    *,
    realizations: int = 1,
) -> u.Quantity:
    """Average positive-lag covariance over a whole-frame exposure.

    Supply zero through frames-minus-one positive lags along the last axis.
    Independent realizations are averaged with equal weight. The result uses
    the supplied metric covariance units. The frame interval is explicit so
    the same convention can be used to form fractional sampling coefficients.
    """
    dt = float(frame_interval.to_value(u.s))
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("Require a positive finite frame interval")
    if (
        type(frames) is not int
        or frames < 1
        or type(realizations) is not int
        or realizations < 1
    ):
        raise ValueError(
            "Frame and independent-realization counts must be positive integers"
        )
    if not isinstance(covariance, u.Quantity):
        raise TypeError("covariance must carry the squared metric unit")
    if covariance.ndim < 1 or covariance.shape[-1] < frames:
        raise ValueError("Covariance must contain each positive lag in the exposure")
    if not np.isfinite(covariance.value[..., :frames]).all():
        raise ValueError("Observed covariance lags must be finite")
    window = 1 - np.arange(frames) / frames
    window[0] = 0.5
    return 2 * (covariance[..., :frames] @ window) / (frames * realizations)
