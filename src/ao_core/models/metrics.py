"""Mathematical kernels retained from the sampling-coefficient report prototype."""

from __future__ import annotations

import itertools
import math
from functools import cache

import numpy as np

from ._exposure import TEMPORAL_WORK_ELEMENTS
from .optics import PupilPairGeometry

__all__ = [
    "metric_covariance",
    "phase_responses",
]


def metric_kernels(
    pupil, covariance, pixel_scale, *, geometry=None, coherence=None, include_width=True
):
    """Return means and pupil-pair influence kernels for SR, EE100 and FWHM.

    Intensities and angular derivatives use focal-plane pixel coordinates.
    EE100 integrates a 100-mas square exactly for the retained pupil model.
    Peak motion is included by implicit differentiation of its stationarity
    condition. FWHM uses Gaussian curvature, not a measured half-max contour.
    Supplied geometry/coherence must match this pupil, scale and covariance.
    """
    if geometry is None:
        geometry = PupilPairGeometry(pupil, pixel_scale)
    geometry.require(pupil, pixel_scale)
    if coherence is None:
        coherence = geometry.coherence(covariance)
    normalization, frequency = geometry.normalization, geometry.frequency
    peak_location = np.zeros(2)
    for iteration in range(12):
        kernel = (
            np.exp(-1j * np.einsum("iab,i->ab", frequency, peak_location))
            / normalization
        )
        weighted = coherence * kernel
        gradient = np.einsum("ab,iab->i", weighted, -1j * frequency).real
        curvature = np.einsum("ab,iab,jab->ij", weighted, frequency, frequency).real
        if np.min(np.linalg.eigvalsh(curvature)) <= 0:
            raise ValueError("Mean central PSF peak is not locally concave")
        step = np.linalg.solve(curvature, gradient)
        peak_location += step
        if np.linalg.norm(step) < 1e-9:
            break
    else:
        raise ValueError("Mean central peak did not converge")
    # Recompute at the converged peak, without constructing an image.
    kernel = (
        np.exp(-1j * np.einsum("iab,i->ab", frequency, peak_location)) / normalization
    )
    weighted = coherence * kernel
    peak = weighted.sum().real
    curvature = np.einsum("ab,iab,jab->ij", weighted, frequency, frequency).real
    inverse = np.linalg.inv(curvature)
    ee_kernel = kernel * geometry.aperture
    ee = np.sum(coherence * ee_kernel).real
    ee_location_gradient = np.einsum(
        "ab,iab->i", coherence * ee_kernel, -1j * frequency
    ).real
    ee_kernel = ee_kernel + kernel * np.einsum(
        "i,iab->ab", inverse @ ee_location_gradient, -1j * frequency
    )
    diffraction_peak = geometry.diffraction_peak
    means = [peak / diffraction_peak, ee]
    kernels = [kernel / diffraction_peak, ee_kernel]
    diagnostics = {
        "peak_location_pixels": peak_location.tolist(),
        "peak_iterations": iteration + 1,
    }
    if include_width:
        covariance_gaussian = peak * inverse
        width = (
            np.sqrt(8 * np.log(2))
            * np.linalg.det(covariance_gaussian) ** 0.25
            * pixel_scale
        )
        # dK at a moving peak also contains the mean PSF's third derivative.
        third = np.einsum(
            "ab,iab,jab,kab->ijk", weighted, frequency, frequency, 1j * frequency
        ).real
        shift_response = 0.25 * inverse @ np.einsum("ij,ijk->k", inverse, third)
        width_kernel = (
            width
            * kernel
            * (
                0.5 / peak
                - 0.25 * np.einsum("ij,iab,jab->ab", inverse, frequency, frequency)
                - 1j * np.einsum("i,iab->ab", shift_response, frequency)
            )
        )
        means.append(width)
        kernels.append(width_kernel)
        diagnostics.update(
            gaussian_covariance_pixels2=covariance_gaussian.tolist(),
            gaussian_core_integrated_flux=float(
                2 * np.pi * peak * np.sqrt(np.linalg.det(covariance_gaussian))
            ),
        )
    return np.array(means), np.array(kernels), coherence, diagnostics


@cache
def _phase_moments(tail_order):
    """Prepare the fixed binomial contractions for each retained tensor order."""
    counts = [
        c
        for c in itertools.product(range(tail_order + 1), repeat=2)
        if sum(c) <= tail_order
    ]
    lookup = {c: i for i, c in enumerate(counts)}
    contractions = []
    for order in range(3, tail_order + 1):
        powers = [tuple(np.bincount(i, minlength=2)) for i in np.ndindex((2,) * order)]
        distinct = tuple(dict.fromkeys(powers))
        terms = []
        for power in distinct:
            factors = []
            for right in itertools.product(*(range(p + 1) for p in power)):
                left = tuple(a - b for a, b in zip(power, right, strict=True))
                coefficient = (-1) ** sum(right) * math.prod(
                    math.comb(a, b) for a, b in zip(power, right, strict=True)
                )
                factors.append((lookup[left], lookup[right], coefficient))
            terms.append(factors)
        contractions.append((tuple(distinct.index(power) for power in powers), terms))
    return np.asarray(counts), contractions


def phase_responses(
    basis, covariance, kernels, coherence, tail_order=4, *, lag_indices=None
):
    """Gaussian phase derivatives with full orders 1–2 and rank-two higher orders.

    This is the same truncation as the retained direct-image calculation;
    only the metric influence kernels change. Pupil-node binomial moments
    avoid explicitly constructing high-dimensional pupil-pair tensors.
    Supply modal synthesis as (mode, pupil node), zero-lag covariance as
    (mode, mode) in radian squared, and Gaussian coherence as (pupil node,
    pupil node). Kernels are the matched metric influence weights with axes
    (metric, pupil node, pupil node). With lag_indices, supply kernels indexed
    by flattened pupil separation and one separation index per pupil pair.
    Returns the first and second modal derivatives, followed by the projection
    and per-metric tensors for the higher orders, for metric_covariance.
    """
    _, vectors = np.linalg.eigh(covariance)
    projection = vectors[:, -2:]
    reduced_basis = projection.T @ basis
    counts, contractions = _phase_moments(tail_order)
    powers = np.cumprod(
        np.vstack(
            (
                np.ones((1, 2, basis.shape[1])),
                np.broadcast_to(reduced_basis, (tail_order, *reduced_basis.shape)),
            )
        ),
        axis=0,
    )
    features = powers[counts[:, 0], 0] * powers[counts[:, 1], 1]
    combined = np.vstack((basis, features))
    first, second, mixed = [], [], []
    if lag_indices is not None:
        real, imag = np.empty(coherence.shape), np.empty(coherence.shape)
    for kernel in kernels:
        if lag_indices is None:
            weights = coherence * kernel
            real = np.ascontiguousarray(weights.real)
            imag = np.ascontiguousarray(weights.imag)
        else:
            for start in range(0, len(real), 128):
                rows = slice(start, start + 128)
                weights = kernel.take(lag_indices[rows])
                weights *= coherence[rows]
                real[rows], imag[rows] = weights.real, weights.imag
        first.append(basis @ (imag.sum(0) - imag.sum(1)))
        product = combined @ real
        quadratic = product[: len(basis)] @ basis.T
        second.append(
            quadratic + quadratic.T - (basis * (real.sum(1) + real.sum(0))) @ basis.T
        )
        mixed.append(
            product[len(basis) :] @ features.T + 1j * (features @ imag @ features.T)
        )
    mixed = np.asarray(mixed)
    tails = [[] for _ in kernels]
    for order, (indices, terms) in enumerate(contractions, start=3):
        moments = np.stack(
            [
                sum(factor * mixed[:, left, right] for left, right, factor in term)
                for term in terms
            ],
            axis=1,
        )
        tensors = (1j**order * moments[:, indices]).real.reshape(
            (len(kernels),) + (2,) * order
        )
        for result, tensor in zip(tails, tensors, strict=True):
            result.append(tensor)
    return np.array(first), np.array(second), (projection, tails)


def metric_covariance(lag, first, second, tails):
    """Contract the matched phase responses and return (metric, lag) covariance.

    Supply phase covariance in radian squared with shape (lag, mode, mode).
    Use the first, second and higher-order responses from phase_responses on
    the same pupil, modal basis and reference wavelength. The result has
    each metric's squared units.

    See Theorem 1 for the derivatives with respect to Gaussian
    cross-covariance that generate the expansion orders.
    [Voigtlaender (2021)](https://arxiv.org/pdf/1710.03576v2)
    """
    total = np.einsum("mi,tij,mj->mt", first, lag, first)
    block_size = max(1, TEMPORAL_WORK_ELEMENTS // math.prod(lag.shape[1:]))
    for start in range(0, len(lag), block_size):
        window = slice(start, start + block_size)
        covariances = lag[window]
        transpose = np.swapaxes(covariances, 1, 2)
        for index, hessian in enumerate(second):
            total[index, window] += 0.5 * np.einsum(
                "tij,tji->t", hessian @ covariances, hessian @ transpose
            )
    projection, tensors = tails
    reduced = np.einsum("ir,tij,js->trs", projection, lag, projection)
    for metric, orders in enumerate(tensors):
        for order, tensor in enumerate(orders, 3):
            transformed = np.broadcast_to(tensor, (len(lag), *tensor.shape)).copy()
            for axis in range(1, order + 1):
                transformed = np.moveaxis(transformed, axis, -1)
                transformed = np.einsum("tij,t...j->t...i", reduced, transformed)
                transformed = np.moveaxis(transformed, -1, axis)
            total[metric] += (transformed * tensor).sum(
                axis=tuple(range(1, order + 1))
            ) / math.factorial(order)
    return total
