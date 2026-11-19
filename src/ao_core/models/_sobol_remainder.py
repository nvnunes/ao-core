"""Decode and accumulate bounded blocks of Sobol pupil quartets."""

import math
from functools import lru_cache

import numpy as np

CDF_BUCKETS = 256


def pair_search_tables(marginal, conditional):
    """Index one marginal CDF and the conditional CDF of each first pupil node."""
    offsets = np.r_[0, 2 * np.arange(len(marginal))]
    cdf = np.vstack((marginal, conditional + offsets[1:, None]))
    edges = np.arange(CDF_BUCKETS + 1) / CDF_BUCKETS
    bounds = np.array(
        [
            np.searchsorted(row, edges + offset)
            for row, offset in zip(cdf, offsets, strict=True)
        ]
    )
    return cdf, offsets, bounds


def _lookup(cdf, offsets, bounds, row, draw):
    bucket = int(draw * CDF_BUCKETS)
    lower, upper = bounds[row, bucket], bounds[row, bucket + 1]
    value = draw + offsets[row]
    # Keep the same offset arithmetic and left-boundary rule as the
    # original flattened CDF, within the part of the row that can match.
    while lower < upper:
        middle = (lower + upper) // 2
        if cdf[row, middle] < value:
            lower = middle + 1
        else:
            upper = middle
    return lower


def _decode_one(cdf, offsets, bounds, draw, lookup):
    a = lookup(cdf, offsets, bounds, 0, draw[0])
    c = lookup(cdf, offsets, bounds, 0, draw[2])
    b = lookup(cdf, offsets, bounds, a + 1, draw[1])
    d = lookup(cdf, offsets, bounds, c + 1, draw[3])
    if a >= b or c >= d:
        raise ValueError("Invalid unordered pupil pair in the numerical quadrature")
    return a, b, c, d


def _add_quartet(total, covariances, weights, probability, a, b, c, d):
    multiplier = 4 / (probability[a, b] * probability[c, d])
    for variant in range(len(covariances)):
        covariance = covariances[variant]
        k = covariance[a, c] - covariance[a, d] - covariance[b, c] + covariance[b, d]
        k2 = k * k
        if abs(k) < 0.1:
            even = k2**2 / 24 * (1 + k2 / 30 * (1 + k2 / 56 * (1 + k2 / 90)))
            odd = k * k2 / 6 * (1 + k2 / 20 * (1 + k2 / 42 * (1 + k2 / 72)))
        else:
            plus, minus = math.expm1(k), math.expm1(-k)
            even = 0.5 * (plus + minus) - 0.5 * k2
            odd = 0.5 * (plus - minus) - k
        for metric in range(weights.shape[1]):
            left, right = weights[variant, metric, a, b], weights[variant, metric, c, d]
            total[variant, metric] += (
                left.real * right.real * multiplier * even
                + left.imag * right.imag * multiplier * odd
            )


def _sum_fused(
    covariances, weights, probability, cdf, offsets, bounds, draws, lookup, decode, add
):
    total = np.zeros(weights.shape[:2])
    for i in range(len(draws)):
        a, b, c, d = decode(cdf, offsets, bounds, draws[i], lookup)
        add(total, covariances, weights, probability, a, b, c, d)
    return total


def _decode_fused(cdf, offsets, bounds, draws, indices, lookup, decode):
    for i in range(len(draws)):
        a, b, c, d = decode(cdf, offsets, bounds, draws[i], lookup)
        indices[i, 0], indices[i, 1], indices[i, 2], indices[i, 3] = a, b, c, d


def _sum_decoded_fused(covariances, weights, probability, indices, add):
    total = np.zeros(weights.shape[:2])
    for a, b, c, d in indices:
        add(total, covariances, weights, probability, a, b, c, d)
    return total


def _decode_numpy(cdf, offsets, bounds, draws, indices):
    aa, cc = np.searchsorted(cdf[0], draws[:, (0, 2)]).T
    conditional = cdf[1:].ravel()
    size = cdf.shape[1]
    bb = np.searchsorted(conditional, draws[:, 1] + offsets[aa + 1]) - aa * size
    dd = np.searchsorted(conditional, draws[:, 3] + offsets[cc + 1]) - cc * size
    if np.any(aa >= bb) or np.any(cc >= dd):
        raise ValueError("Invalid unordered pupil pair in the numerical quadrature")
    indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3] = aa, bb, cc, dd


def _sum_numpy(covariances, weights, probability, cdf, offsets, bounds, draws):
    indices = np.empty((len(draws), 4), dtype=np.intp)
    _decode_numpy(cdf, offsets, bounds, draws, indices)
    return _sum_decoded_numpy(covariances, weights, probability, indices)


def _sum_decoded_numpy(covariances, weights, probability, indices):
    aa, bb, cc, dd = indices.T
    multiplier = 4 / (probability[aa, bb] * probability[cc, dd])
    left, right = weights[:, :, aa, bb], weights[:, :, cc, dd]
    k = (
        covariances[:, aa, cc]
        - covariances[:, aa, dd]
        - covariances[:, bb, cc]
        + covariances[:, bb, dd]
    )
    plus, minus = np.expm1(k), np.expm1(-k)
    even = 0.5 * (plus + minus) - 0.5 * k**2
    odd = 0.5 * (plus - minus) - k
    small = np.abs(k) < 0.1
    z, z2 = k[small], k[small] ** 2
    even[small] = z2**2 / 24 * (1 + z2 / 30 * (1 + z2 / 56 * (1 + z2 / 90)))
    odd[small] = z * z2 / 6 * (1 + z2 / 20 * (1 + z2 / 42 * (1 + z2 / 72)))
    return (
        (left.real * right.real * multiplier * even[:, None])
        + (left.imag * right.imag * multiplier * odd[:, None])
    ).sum(axis=-1)


@lru_cache(maxsize=1)
def _compiled_functions():
    try:
        from numba import njit
    except ImportError:
        return _sum_numpy, _decode_numpy, _sum_decoded_numpy
    lookup = njit(inline="always")(_lookup)
    decode = njit(inline="always")(_decode_one)
    add = njit(inline="always")(_add_quartet)
    fused = njit(nogil=True)(_sum_fused)
    decode_block = njit(nogil=True)(_decode_fused)
    sum_block = njit(nogil=True)(_sum_decoded_fused)

    def accumulate(covariances, weights, probability, cdf, offsets, bounds, draws):
        return fused(
            covariances,
            weights,
            probability,
            cdf,
            offsets,
            bounds,
            draws,
            lookup,
            decode,
            add,
        )

    def decode_indices(cdf, offsets, bounds, draws, indices):
        decode_block(cdf, offsets, bounds, draws, indices, lookup, decode)

    def accumulate_indices(covariances, weights, probability, indices):
        return sum_block(covariances, weights, probability, indices, add)

    return accumulate, decode_indices, accumulate_indices


def quartet_accumulator():
    """Return the compiled block accumulator or its NumPy equivalent."""
    return _compiled_functions()[0]


def decoded_quartet_functions():
    """Return block decoding and accumulation functions for repeated lags."""
    return _compiled_functions()[1:]
