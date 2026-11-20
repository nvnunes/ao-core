"""Mathematical kernels retained from the sampling-coefficient report prototype."""

from __future__ import annotations

import itertools
import math
from functools import cache

import contourpy
import numpy as np
from scipy.fft import fft2, ifftshift
from scipy.interpolate import interp1d
from scipy.ndimage import shift
from scipy.signal import resample

from .optics import PupilPairGeometry, _image_from_otf, _sum_pairs_by_lag


@cache
def refinement_matrix(side, factor=4):
    pad = (side + 1) // 2
    matrix = np.pad(np.eye(side), ((pad, pad), (0, 0)))
    return resample(matrix, matrix.shape[0] * factor, axis=0)[
        pad * factor : (pad + side - 1) * factor + 1
    ]


class ContourOutsideCrop(ValueError):
    """The supplied image crop does not contain a closed half-maximum contour."""


def contour_fwhm(refined, pixel_scale, *, require_closed=False):
    """Return the half-maximum contour width and its local image gradient.

    The pixel scale is in mas on the supplied image. The gradient follows
    the active peak and contour vertices of that image.
    """
    peak_index = np.unravel_index(np.argmax(refined), refined.shape)
    peak = refined[peak_index]
    contours = contourpy.contour_generator(z=refined).lines(float(peak / 2))
    if require_closed and not contours:
        raise ContourOutsideCrop(
            "The half-maximum contour must close inside the refinement crop"
        )
    contour = max(contours, key=len)
    if require_closed and (
        not np.array_equal(contour[0], contour[-1])
        or np.min(contour) <= 0
        or np.any(contour.max(0) >= np.array(refined.shape[::-1]) - 1)
    ):
        raise ContourOutsideCrop(
            "The half-maximum contour must close inside the refinement crop"
        )
    centre = (contour.max(0) + contour.min(0)) / 2
    radii = np.linalg.norm(contour - centre, axis=1)
    imin, imax = int(np.argmin(radii)), int(np.argmax(radii))
    width = 2 * np.sqrt(radii[imin] * radii[imax]) * pixel_scale
    vertex_gradient = np.zeros_like(contour)
    for index in (imin, imax):
        vertex_gradient[index] += (
            width / 2 * (contour[index] - centre) / radii[index] ** 2
        )
    centre_gradient = -vertex_gradient.sum(0)
    for axis in (0, 1):
        for index in (
            int(np.argmin(contour[:, axis])),
            int(np.argmax(contour[:, axis])),
        ):
            vertex_gradient[index, axis] += centre_gradient[axis] / 2
    gradient = np.zeros_like(refined)
    for index in np.flatnonzero(np.any(vertex_gradient, axis=1)):
        x, y = contour[index]
        if abs(x - round(x)) < 1e-7:
            x0, y0, dx, dy = round(x), int(np.floor(y)), 0, 1
        elif abs(y - round(y)) < 1e-7:
            x0, y0, dx, dy = int(np.floor(x)), round(y), 1, 0
        else:
            raise ValueError("Contour vertex is not on a grid edge")
        u, v = refined[y0, x0], refined[y0 + dy, x0 + dx]
        denominator = v - u
        influence = vertex_gradient[index] @ np.array([dx, dy])
        gradient[y0, x0] += influence * (peak / 2 - v) / denominator**2
        gradient[y0 + dy, x0 + dx] -= influence * (peak / 2 - u) / denominator**2
        gradient[peak_index] += influence * 0.5 / denominator
    return width, gradient


def metric_responses(image, pixel_scale, *, metric_indices=(0, 1, 2)):
    """Return local pixel derivatives of the e010 peak, EE and contour width.

    Differentiation holds the selected peak grid location and contour extrema
    fixed. At branch changes this derivative is not a global nonlinear model.
    SR's fixed diffraction peak cancels from its relative uncertainty.
    """
    side = len(image)
    refine = refinement_matrix(side)
    refined = refine @ image @ refine.T
    peak_index = np.unravel_index(np.argmax(refined), refined.shape)
    peak = refined[peak_index]
    means, gradients = {}, {}
    if 0 in metric_indices:
        peak_gradient = np.zeros_like(refined)
        peak_gradient[peak_index] = 1
        sr_gradient = refine.T @ peak_gradient @ refine
        means[0], gradients[0] = peak, sr_gradient
    if 1 in metric_indices:
        location = np.asarray(peak_index) / 4
        anchor = np.floor(location + 0.5 - 1e-12).astype(int)
        delta = anchor - location
        sy = shift(
            np.eye(side), (delta[0], 0), order=3, mode="constant", prefilter=True
        )
        sx = shift(
            np.eye(side), (delta[1], 0), order=3, mode="constant", prefilter=True
        )
        anchored = sy @ image @ sx.T
        radius = min(
            int(min(*anchor, *(side - 1 - anchor))),
            2 * int(np.ceil(50 / pixel_scale - 0.5)),
        )
        aperture_weights = interp1d(
            (np.arange(radius + 1) + 0.5) * pixel_scale,
            np.eye(radius + 1),
            axis=0,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )(50)
        weights = np.zeros_like(image)
        for r, weight in enumerate(aperture_weights):
            weights[
                anchor[0] - r : anchor[0] + r + 1, anchor[1] - r : anchor[1] + r + 1
            ] += weight
        ee = float(np.sum(weights * np.maximum(anchored, 0)))
        ee_gradient = sy.T @ (weights * (anchored > 0)) @ sx
        means[1], gradients[1] = ee, ee_gradient
    if 2 in metric_indices:
        width, gradient = contour_fwhm(refined, pixel_scale / 4)
        means[2], gradients[2] = width, refine.T @ gradient @ refine
    return (
        np.array([means[i] for i in metric_indices]),
        np.array([gradients[i] for i in metric_indices]),
    )


def image_metric_kernels(geometry, coherence, metric_indices):
    """Return requested image metrics, pixel gradients and pupil-pair kernels."""
    size = geometry.pupil.optical_size
    otf = _sum_pairs_by_lag(coherence, geometry.flat_lag, size**2).reshape(size, size)
    mean = _image_from_otf(otf) / geometry.normalization
    metrics, gradients = metric_responses(
        mean, geometry.pixel_scale, metric_indices=metric_indices
    )
    kernels = np.array(
        [
            fft2(ifftshift(gradient))[geometry.dy, geometry.dx] / geometry.normalization
            for gradient in gradients
        ]
    )
    return metrics, mean, gradients, kernels


def response_coefficients(
    pupil,
    covariance,
    pixel_scale,
    tail_order=2,
    tail_rank=2,
    *,
    metric_indices=(0, 1, 2),
    geometry=None,
    coherence=None,
):
    """Evaluate selected responses, optionally reusing matching geometry/coherence.

    Supplied coherence must be for this covariance and unmodified pupil.
    Defaults construct both and retain peak, EE and FWHM responses.
    """
    if tail_order < 2:
        raise ValueError(
            "The covariance expansion must retain at least quadratic terms"
        )
    if (
        not metric_indices
        or len(set(metric_indices)) != len(metric_indices)
        or any(index not in (0, 1, 2) for index in metric_indices)
    ):
        raise ValueError("Require distinct peak, EE or FWHM metric indices")
    if geometry is None:
        geometry = PupilPairGeometry(pupil, pixel_scale)
    geometry.require(pupil, pixel_scale)
    if coherence is None:
        coherence = geometry.coherence(covariance)
    metrics, mean, gradients, influence = image_metric_kernels(
        geometry, coherence, metric_indices
    )
    basis = pupil.basis
    first, second = [], []
    kernels = []
    for kernel in influence:
        weights = coherence * kernel
        first.append((1j * basis @ (weights.sum(1) - weights.sum(0))).real)
        real = weights.real
        second.append(
            -(basis * (real.sum(1) + real.sum(0))) @ basis.T
            + basis @ (real + real.T) @ basis.T
        )
        kernels.append(weights)
    if tail_order == 2:
        return metrics, np.asarray(first), np.asarray(second), mean, gradients, None
    _, vectors = np.linalg.eigh(covariance)
    projection = vectors[:, -tail_rank:]
    reduced_basis = projection.T @ basis
    # Expand (z_a-z_b)^alpha by the binomial theorem. Pupil-node monomials
    # replace pairwise powers, preserving every retained tensor coefficient.
    feature_counts = [
        counts
        for counts in itertools.product(range(tail_order + 1), repeat=tail_rank)
        if sum(counts) <= tail_order
    ]
    features = np.asarray(
        [
            np.prod(reduced_basis ** np.asarray(counts)[:, None], axis=0)
            for counts in feature_counts
        ]
    )
    lookup_features = {counts: index for index, counts in enumerate(feature_counts)}
    mixed = np.asarray([features @ kernel @ features.T for kernel in kernels])
    tails = [[] for _ in gradients]
    for order in range(3, tail_order + 1):
        indices = list(np.ndindex((tail_rank,) * order))
        counts = [tuple(np.bincount(index, minlength=tail_rank)) for index in indices]
        unique = list(dict.fromkeys(counts))
        moments = []
        for count in unique:
            moment = np.zeros(len(gradients), complex)
            for right in itertools.product(*(range(power + 1) for power in count)):
                left = tuple(a - b for a, b in zip(count, right, strict=True))
                coefficient = (-1) ** sum(right) * math.prod(
                    math.comb(a, b) for a, b in zip(count, right, strict=True)
                )
                moment += (
                    coefficient
                    * mixed[:, lookup_features[left], lookup_features[right]]
                )
            moments.append(((1j**order) * moment).real)
        moments = np.asarray(moments)
        lookup = {count: index for index, count in enumerate(unique)}
        tensors = moments[[lookup[count] for count in counts]].T.reshape(
            (len(gradients),) + (tail_rank,) * order
        )
        for metric, tensor in enumerate(tensors):
            tails[metric].append(tensor)
    return (
        metrics,
        np.asarray(first),
        np.asarray(second),
        mean,
        gradients,
        (projection, tails),
    )


def contract_covariance(lag, first, second, dt, tails=None):
    """Contract explicit equal-spaced modal lag covariances over one PH.

    The first sample is zero lag, and PH duration is len(lag)*dt. This accepts
    independently propagated detector-noise covariances without discretizing
    them onto the atmospheric frequency grid.
    """
    n = len(lag)
    linear = np.einsum("mi,tij,mj->mt", first, lag, first)
    quadratic = np.empty_like(linear)
    for index, hessian in enumerate(second):
        left = hessian @ lag
        right = hessian @ np.swapaxes(lag, 1, 2)
        quadratic[index] = 0.5 * np.einsum("tij,tji->t", left, right)
    weights = 1 - np.arange(n) / n
    weights[0] = 0.5
    total = linear + quadratic
    if tails is not None:
        projection, tensors = tails
        reduced_lag = np.einsum("ir,tij,js->trs", projection, lag, projection)
        for metric, orders in enumerate(tensors):
            for order, tensor in enumerate(orders, 3):
                transformed = np.broadcast_to(tensor, (n, *tensor.shape)).copy()
                for axis in range(1, order + 1):
                    transformed = np.moveaxis(transformed, axis, -1)
                    transformed = np.einsum(
                        "tij,t...j->t...i", reduced_lag, transformed
                    )
                    transformed = np.moveaxis(transformed, -1, axis)
                total[metric] += (transformed * tensor).sum(
                    axis=tuple(range(1, order + 1))
                ) / math.factorial(order)
    return 2 * dt * (total @ weights), linear, quadratic
