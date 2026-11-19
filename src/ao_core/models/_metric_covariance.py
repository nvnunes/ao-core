"""Propagate residual phase covariance through PSF metric responses."""

from functools import partial
from time import perf_counter

import numpy as np

from ._all_order import (
    SOBOL_POWER,
    SOBOL_REPLICATES,
    SOBOL_SEED,
    RemainderQuadrature,
    quadratic_covariance,
    quadratic_response,
)
from ._exposure import sampled_lag_indices
from .metrics import metric_covariance, phase_responses

LEVELS = ("fast", "balanced", "full")


def metric_lag_indices(frame_counts):
    """Select native-frame lags for each model and union the requested windows."""
    indices = {}
    if "fast" in frame_counts:
        indices["fast"] = np.arange(max(frame_counts["fast"]))
    for model in ("balanced", "full"):
        if model in frame_counts:
            indices[model] = np.unique(
                np.concatenate(
                    [sampled_lag_indices(count) for count in frame_counts[model]]
                )
            )
    return indices


def fast_metric_covariance(
    basis, modal_lags, kernels, coherence, *, phase_scale=1.0, lag_indices=None
):
    """Contract orders 1–2 in all modes and orders 3–6 in two covariance directions."""
    first, second, (projection, higher_orders) = phase_responses(
        basis,
        modal_lags[0],
        kernels,
        coherence,
        tail_order=6,
        lag_indices=lag_indices,
    )
    if phase_scale != 1.0:
        # The shared modal covariance is at the reference band wavelength.
        first *= phase_scale
        second *= phase_scale**2
        higher_orders = [
            [tensor * phase_scale**order for order, tensor in enumerate(terms, 3)]
            for terms in higher_orders
        ]
    return metric_covariance(modal_lags, first, second, (projection, higher_orders))


def predict_metric_lags(
    indices,
    modal_lags,
    pupil_lags,
    basis,
    kernels,
    coherence,
    *,
    device,
):
    """Evaluate metric covariance in the modal and restored pupil representations.

    Modal covariance is dense in native-frame time. Pupil covariance is
    streamed in the order of the Full lag indices, including zero lag.
    Fast and Balanced use modal covariance. Balanced maps it to pupil
    coordinates for the Sobol remainder it shares with Full, which uses
    the restored pupil covariance.
    """
    metric_lags = {}
    if "fast" in indices:
        frames = indices["fast"]
        fast = modal_lags[: len(frames)]
        metric_lags["fast"] = (
            frames,
            fast_metric_covariance(basis, fast, kernels, coherence),
        )
    all_order = {
        model: indices[model] for model in ("balanced", "full") if model in indices
    }
    if all_order:
        metric_lags.update(
            _all_order_lags(
                all_order,
                modal_lags,
                pupil_lags,
                basis,
                coherence[None] * kernels,
                device,
            )
        )
    return metric_lags


def _all_order_lags(indices, modal, pupil, basis, weights, device):
    zero, kernels = {}, {}
    if "balanced" in indices:
        zero["balanced"] = basis.T @ modal[0] @ basis
        kernels["balanced"] = quadratic_response(weights, basis=basis)
    if "full" in indices:
        zero["full"] = next(pupil)
        first, second = quadratic_response(weights)
        if device == "mps":
            from ._metal_covariance import MpsQuadraticCovariance

            full_quadratic = MpsQuadraticCovariance(first, second)
        else:
            full_quadratic = partial(quadratic_covariance, first=first, second=second)
        del first, second

    frames = np.unique(np.concatenate(tuple(indices.values())))
    model_weights = np.broadcast_to(weights, (len(zero), *weights.shape))
    metal_sobol = None
    try_metal = device == "mps" and len(frames) > 1
    quadrature = RemainderQuadrature(np.stack(tuple(zero.values())), model_weights)
    values = {level: [] for level in indices}
    frames_by_level = {level: set(lags) for level, lags in indices.items()}
    for frame in frames:
        covariances, bases, levels = [], [], []
        if frame in frames_by_level.get("balanced", ()):
            levels.append("balanced")
            covariances.append(
                zero["balanced"] if frame == 0 else basis.T @ modal[frame] @ basis
            )
            bases.append(quadratic_covariance(modal[frame], *kernels["balanced"]))
        if frame in frames_by_level.get("full", ()):
            levels.append("full")
            covariance = zero["full"] if frame == 0 else next(pupil)
            covariances.append(covariance)
            bases.append(full_quadratic(covariance))
        lag_covariances = np.stack(covariances)
        if metal_sobol is not None:
            remainders, coarse = metal_sobol(lag_covariances)
            if not (np.isfinite(remainders).all() and np.isfinite(coarse).all()):
                metal_sobol = None
        if metal_sobol is None:
            remainders, coarse = quadrature.evaluate(
                lag_covariances,
                model_weights[: len(levels)],
                SOBOL_POWER,
                SOBOL_REPLICATES,
                SOBOL_SEED,
                workers=4,
                reuse_draws=len(frames) > 1,
            )
        if frame == 0 and try_metal and (draws := quadrature.metal_indices) is not None:
            from ._metal_covariance import MpsRemainderQuadrature

            metal_sobol = MpsRemainderQuadrature(
                draws, weights, quadrature.proposal, len(zero)
            )
        estimate = np.asarray(bases) + remainders.mean(axis=0)
        if frame == 0 and (
            np.any(estimate <= 0) or np.any(bases + coarse.mean(axis=0) <= 0)
        ):
            raise ValueError("Nonpositive instantaneous metric variance")
        for level, value in zip(levels, estimate, strict=True):
            values[level].append(value)
    return {
        level: (lags, np.asarray(values[level]).T) for level, lags in indices.items()
    }


class CPUCompactFastBatch:
    """Reuse one large CPU projection and lag cube across matched bands."""

    def __init__(self, reference):
        self.reference = reference
        self.wavelength_m = reference.predictor.metadata["wavelength_m"]

    def evaluate_many(self, batches, optics, users, *, remove_tilt, pipeline):
        for positions in batches:
            started = perf_counter()
            results = []
            restored = self.reference.compact_phases(positions)
            for xy, (structure, zero, spectrum) in zip(
                positions, restored, strict=True
            ):
                lags = self.reference.compact_lags(
                    spectrum, zero[0], self.reference.predictor.max_count
                )
                if remove_tilt:
                    lags[:, :2, :] = 0
                    lags[:, :, :2] = 0
                for key in users[xy]:
                    science_optics = optics[key]
                    scale = self.wavelength_m / key[0]
                    band_structure = structure if scale == 1.0 else structure * scale**2
                    means, kernels, coherence = science_optics.metric_response(
                        band_structure, lag_grid=True
                    )
                    covariance = fast_metric_covariance(
                        science_optics.pupil.basis,
                        lags,
                        kernels,
                        coherence,
                        phase_scale=scale,
                        lag_indices=science_optics.flat_lag.reshape(coherence.shape),
                    )
                    results.append((key, xy, means, (np.arange(len(lags)), covariance)))
                del lags
            yield results, (perf_counter() - started) / len(positions), len(positions)
