"""Layer spectra and frozen-flow translations at the supplied phase wavelength."""

from __future__ import annotations

import numpy as np
from scipy.special import gamma, kv

__all__ = [
    "frozen_flow_frequency",
    "von_karman_spectrum",
]

VK_COEFFICIENT = 0.022895587108555225
_KOLMOGOROV_INDEX = 5 / 6


def von_karman_spectrum(
    fx: np.ndarray, fy: np.ndarray, fried_parameter_m: float, outer_scale_m: float
) -> np.ndarray:
    """Return the reference phase spectrum in rad squared per Fourier area.

    Spatial frequencies are cycles per metre. The Fried parameter is specified
    at the wavelength used for the phase. Multiply by each layer's weight
    and by the area of its Fourier quadrature cell when forming covariance.
    """
    return (
        VK_COEFFICIENT
        * fried_parameter_m ** (-5 / 3)
        * (fx**2 + fy**2 + outer_scale_m**-2) ** (-11 / 6)
    )


def frozen_flow_frequency(
    fx: np.ndarray, fy: np.ndarray, speed_m_per_s: float, direction_rad: float
) -> np.ndarray:
    """Project a translating layer onto Fourier frequencies in cycles per second."""
    return -speed_m_per_s * (np.cos(direction_rad) * fx + np.sin(direction_rad) * fy)


def von_karman_correlation(
    distance_m: np.ndarray, inverse_outer_scale: float
) -> np.ndarray:
    """Evaluate the unit-variance von Kármán phase correlation."""
    nu = _KOLMOGOROV_INDEX
    z = 2 * np.pi * inverse_outer_scale * distance_m
    correlation = np.ones_like(z)
    positive = z > 0
    sample = z[positive]
    correlation[positive] = 2 ** (1 - nu) / gamma(nu) * sample**nu * kv(nu, sample)
    return correlation


def von_karman_variance(fried_parameter_m: float, inverse_outer_scale: float) -> float:
    """Return unprojected phase variance in rad squared at the reference wavelength."""
    nu = _KOLMOGOROV_INDEX
    return float(
        np.pi
        * VK_COEFFICIENT
        * fried_parameter_m ** (-5 / 3)
        * inverse_outer_scale ** (-2 * nu)
        / nu
    )
