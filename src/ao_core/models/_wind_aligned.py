"""Wind-aligned temporal integration of the angular pupil response."""

import numpy as np
from astropy import units as u
from numpy.polynomial.legendre import leggauss

from ._units import _count, _quantity
from .atmosphere import von_karman_spectrum
from .control import scalar_integrator_transfer


def _gauss_intervals(edges, order):
    """Piecewise Gauss-Legendre nodes, including a nonsingular zero-frequency interval."""
    z, w = leggauss(order)
    lo, hi = np.asarray(edges[:-1]), np.asarray(edges[1:])
    return ((lo[:, None] + hi[:, None]) / 2 + (hi - lo)[:, None] * z / 2).ravel(), (
        (hi - lo)[:, None] * w / 2
    ).ravel()


def _transverse_nodes(limit, order=12):
    """Resolve low-frequency structure and aperture/Bessel oscillations separately."""
    low = [0, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.04]
    edges = np.unique(np.r_[low, np.arange(0.08, limit, 0.08), limit])
    return _gauss_intervals(edges[edges <= limit], order)


def wind_aligned_tilt_covariance(
    atmosphere,
    positions,
    response,
    controller,
    strengths,
    *,
    frequency_order=16,
    transverse_order=8,
    frequency_max=100.0,
    transverse_limit=2.0,
):
    """Integrate the joint unfiltered and filtered angular tilt in mas squared.

    Supply an `AtmosphereConfig` with projected layer heights, weights and
    winds, and Fried parameter and outer scale at the pupil's phase reference.
    Layer wind speeds must be positive. Positions are angular quantities with
    shape (direction, 2). The response takes two vectors of frequencies in
    cycles/m and returns an array of shape (2, frequencies) in mas per phase
    radian. Supply a `FixedGainIntegrator` and a vector of dimensionless scalar
    feedback strengths. Each strength must give a stable loop. The integration
    limits are in Hz and cycles/m, respectively.

    The result has shape (K, K) with K = 2 * n_directions * (1 + n_strengths).
    Each axis flattens (direction, unfiltered followed by each filtered channel,
    x/y), in that order. The continuous spectrum samples the periodic filter.

    The transverse-frequency integral follows Eq. (43). Apply the
    channel transfer functions to that cross-spectrum. [Plantet et al. (2022)](https://arxiv.org/pdf/2209.00931)
    """
    _count(frequency_order, "frequency_order")
    _count(transverse_order, "transverse_order")
    if not all(
        np.isfinite(limit) and limit > 0 for limit in (frequency_max, transverse_limit)
    ):
        raise ValueError("Frequency limits must be positive and finite")
    required = (
        "fried_parameter",
        "outer_scale",
        "layer_heights",
        "layer_weights",
        "wind_speeds",
        "wind_directions",
    )
    if any(getattr(atmosphere, field) is None for field in required):
        raise ValueError(f"Wind-aligned covariance requires {', '.join(required)}")
    edges = np.array([0, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    nu, fw = _gauss_intervals(
        np.r_[edges[edges < frequency_max], frequency_max], frequency_order
    )
    q, qw = _transverse_nodes(transverse_limit, transverse_order)
    q, qw = np.r_[-q, q], np.r_[qw, qw]
    positions = _quantity(positions, u.rad, "positions", 2).value
    if positions.shape[1] != 2:
        raise ValueError("Require x/y positions by direction")
    strengths = np.asarray(strengths)
    if (
        strengths.ndim != 1
        or strengths.dtype.kind not in "fiu"
        or not np.isfinite(strengths).all()
    ):
        raise ValueError("Expected a vector of finite real feedback strengths")
    for strength in strengths:
        controller.transition(np.eye(2) * strength)
    count, channels = len(positions), 1 + len(strengths)
    size = count * channels * 2
    result = np.zeros((size, size))
    indexes = np.broadcast_to(
        np.arange(count * 2).reshape(count, 1, 2), (count, channels, 2)
    ).reshape(-1)
    r0 = atmosphere.fried_parameter.to_value(u.m)
    outer_scale = atmosphere.outer_scale.to_value(u.m)
    layers = zip(
        atmosphere.layer_heights.to_value(u.m),
        atmosphere.layer_weights.value,
        atmosphere.wind_speeds.to_value(u.m / u.s),
        atmosphere.wind_directions.to_value(u.rad),
        strict=True,
    )
    dt = 1 / controller.frame_rate.to_value(u.Hz)
    gain, delay = controller.gain, controller.delay
    for height, fraction, speed, angle in layers:
        if speed <= 0:
            raise ValueError(
                "Wind-aligned temporal integration requires positive wind speed"
            )
        ca, sa = np.cos(angle), np.sin(angle)
        for frequency, weight in zip(nu, fw, strict=True):
            parallel = -frequency / speed
            fx, fy = parallel * ca - q * sa, parallel * sa + q * ca
            power = (
                von_karman_spectrum(np.hypot(fx, fy), 0.0, r0, outer_scale)
                * fraction
                * qw
                / speed
            )
            # exp(+i f x_left) exp(-i f x_right) implements equation 43's separation.
            phase = np.exp(
                2j
                * np.pi
                * height
                * (positions[:, 0, None] * fx + positions[:, 1, None] * fy)
            )
            amplitudes = np.asarray(response(fx, fy))
            if amplitudes.shape != (2, len(fx)) or not np.isfinite(amplitudes).all():
                raise ValueError(
                    "Pupil response must supply two finite tilt amplitudes per node"
                )
            optical = (phase[:, None] * amplitudes[None]).reshape(count * 2, -1)
            spatial = (optical * power) @ optical.conj().T
            h = np.r_[
                1.0,
                scalar_integrator_transfer(
                    np.exp(-2j * np.pi * frequency * dt), strengths, gain, delay
                ),
            ]
            factors = np.broadcast_to(h[None, :, None], (count, channels, 2)).reshape(
                -1
            )
            contribution = (
                2
                * weight
                * (
                    spatial[np.ix_(indexes, indexes)]
                    * factors[:, None]
                    * factors.conj()[None]
                ).real
            )
            result += contribution
    return result * u.mas**2
