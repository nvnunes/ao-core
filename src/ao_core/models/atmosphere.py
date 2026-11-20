"""Layer spectra and frozen-flow translations at the supplied phase wavelength."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from astropy import units as u
from scipy.special import gamma, kv

__all__ = [
    "frozen_flow_frequency",
    "von_karman_spectrum",
]

VK_COEFFICIENT = 0.022895587108555225
_KOLMOGOROV_INDEX = 5 / 6


def _project_zenith_atmosphere(atmosphere, angle):
    """Apply caller zenith angle to vertical heights and turbulence strength.

    See Eq. (1) for the conversion from Kolmogorov seeing to the
    Fried parameter. [Martinez et al. (2010)](https://arxiv.org/pdf/1003.4593)
    """
    if atmosphere.reference_wavelength is None or atmosphere.layer_heights is None:
        raise ValueError("Atmosphere requires reference_wavelength and layer_heights")
    fried = atmosphere.fried_parameter
    if atmosphere.seeing is not None:
        fried = (
            0.976 * atmosphere.reference_wavelength / atmosphere.seeing.to_value(u.rad)
        )
    if fried is None:
        raise ValueError("Atmosphere requires Seeing or a Fried parameter")
    airmass = 1.0 if angle is None else 1 / np.cos(angle.to_value(u.rad))
    return replace(
        atmosphere,
        fried_parameter=fried * airmass ** (-3 / 5),
        layer_heights=atmosphere.layer_heights * airmass,
    ), airmass


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
