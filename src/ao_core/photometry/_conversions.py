"""Convert NGS magnitudes, photon flux, and WFS photon inputs.

Pre-aperture flux uses only the caller's magnitude zeropoint. Per-frame counts
and effective rates additionally use the explicit telescope and WFS calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u

from ._units import (
    nonnegative_quantity_value,
    positive_scalar,
    positive_scalar_quantity,
    quantity_value,
)

_PHOTON_FLUX_UNIT = u.photon / (u.m**2 * u.s)


@dataclass(frozen=True)
class WFSPhotometryConfig:
    """Explicit calibration for WFS-scaled NGS photon conversions.

    Attributes:
        telescope_diameter: Positive scalar diameter, convertible to metres.
        n_channels: Positive real scalar divisor of the telescope diameter;
            the nominal subaperture area is
            ``(telescope_diameter / n_channels)**2``. Quantities, strings,
            and booleans are invalid.
        frame_rate: Positive scalar frame rate, convertible to hertz.
        zeropoint: Positive scalar zero-magnitude photon flux before area
            scaling, convertible to ``photon / (m2 s)``. Instrument throughput
            may already be included in this caller-supplied calibration.

    Raises:
        TypeError: A required quantity is absent or ``n_channels`` is not a
            real numeric scalar.
        ValueError: A quantity has incompatible units, or a calibration value
            is non-finite, non-scalar, or non-positive.
    """

    telescope_diameter: u.Quantity
    n_channels: float
    frame_rate: u.Quantity
    zeropoint: u.Quantity

    def __post_init__(self) -> None:
        """Reject invalid calibration at the public object boundary."""
        positive_scalar_quantity(self.telescope_diameter, u.m, label="telescope_diameter")
        positive_scalar(self.n_channels, label="n_channels")
        positive_scalar_quantity(self.frame_rate, u.Hz, label="frame_rate")
        positive_scalar_quantity(self.zeropoint, _PHOTON_FLUX_UNIT, label="zeropoint")


def magnitudes_to_photon_flux(
    magnitudes: u.Quantity,
    zeropoint: u.Quantity,
) -> u.Quantity:
    """Convert magnitudes to pre-aperture photon flux.

    The WFS area factor is not applied, and the input shape is retained.

    Args:
        magnitudes: Scalar or array of finite Astropy magnitudes.
        zeropoint: Positive scalar zeropoint in ``photon / (m2 s)``.

    Returns:
        Astropy ``photon / (m2 s)`` quantity with the input shape.

    Raises:
        TypeError: An input is not an Astropy quantity.
        ValueError: An input has incompatible units, is non-finite or complex,
            or the zeropoint is non-scalar or non-positive.
    """
    zeropoint_value = positive_scalar_quantity(zeropoint, _PHOTON_FLUX_UNIT, label="zeropoint")
    magnitude_values = quantity_value(magnitudes, u.mag, label="magnitudes")
    photon_flux = zeropoint_value * (10.0 ** (-0.4 * magnitude_values))
    return np.asarray(photon_flux, dtype=float) * _PHOTON_FLUX_UNIT


def photon_flux_to_magnitudes(
    photon_flux: u.Quantity,
    zeropoint: u.Quantity,
) -> u.Quantity:
    """Invert pre-aperture photon flux to magnitudes.

    Zero input uses a ``1e-30 photon / (m2 s)`` logarithm floor. The input
    shape is retained.

    Args:
        photon_flux: Scalar or array of nonnegative ``photon / (m2 s)`` quantities.
        zeropoint: Positive scalar zeropoint in ``photon / (m2 s)``.

    Returns:
        Astropy magnitude quantity with the input shape.

    Raises:
        TypeError: An input is not an Astropy quantity.
        ValueError: An input has incompatible units, is non-finite or complex,
            photon flux is negative, or the zeropoint is non-scalar or
            non-positive.
    """
    zeropoint_value = positive_scalar_quantity(zeropoint, _PHOTON_FLUX_UNIT, label="zeropoint")
    flux_values = nonnegative_quantity_value(
        photon_flux, _PHOTON_FLUX_UNIT, label="photon_flux"
    )
    magnitudes = -2.5 * np.log10(np.clip(flux_values, 1e-30, None) / zeropoint_value)
    return np.asarray(magnitudes, dtype=float) * u.mag


def photon_flux_to_photons_per_frame(
    photon_flux: u.Quantity,
    photometry: WFSPhotometryConfig,
) -> u.Quantity:
    """Convert pre-aperture flux to unrounded WFS photons per frame.

    The input shape is retained. A frame is a dimensionless sample count, so
    the output has Astropy ``photon`` units. No rounding is performed.

    Args:
        photon_flux: Scalar or array of nonnegative ``photon / (m2 s)`` quantities.
        photometry: Explicit telescope and WFS calibration.

    Returns:
        Astropy photon quantity with the input shape.

    Raises:
        TypeError: The flux is not an Astropy quantity.
        ValueError: The flux has incompatible units, is non-finite or complex,
            or is negative; also raised if calibration values are invalid.
    """
    frame_rate = positive_scalar_quantity(photometry.frame_rate, u.Hz, label="frame_rate")
    telescope_diameter = positive_scalar_quantity(
        photometry.telescope_diameter, u.m, label="telescope_diameter"
    )
    n_channels = positive_scalar(photometry.n_channels, label="n_channels")
    flux_values = nonnegative_quantity_value(
        photon_flux, _PHOTON_FLUX_UNIT, label="photon_flux"
    )
    photons_per_frame = flux_values / frame_rate * (telescope_diameter / n_channels) ** 2
    return photons_per_frame * u.photon


def photons_per_frame_to_photon_flux(
    photons_per_frame: u.Quantity,
    photometry: WFSPhotometryConfig,
) -> u.Quantity:
    """Invert unrounded WFS photons per frame to pre-aperture flux.

    The input shape is retained. The result reverses WFS area and frame-rate
    scaling to recover flux before area scaling.

    Args:
        photons_per_frame: Scalar or array of nonnegative photon quantities.
        photometry: Explicit telescope and WFS calibration.

    Returns:
        Astropy ``photon / (m2 s)`` quantity with the input shape.

    Raises:
        TypeError: The count is not an Astropy quantity.
        ValueError: The count has incompatible units, is non-finite or complex,
            or is negative; also raised if calibration values are invalid.
    """
    frame_rate = positive_scalar_quantity(photometry.frame_rate, u.Hz, label="frame_rate")
    telescope_diameter = positive_scalar_quantity(
        photometry.telescope_diameter, u.m, label="telescope_diameter"
    )
    n_channels = positive_scalar(photometry.n_channels, label="n_channels")
    photon_values = nonnegative_quantity_value(photons_per_frame, u.photon, label="photons_per_frame")
    photon_flux = photon_values * frame_rate / (telescope_diameter / n_channels) ** 2
    return np.asarray(photon_flux, dtype=float) * _PHOTON_FLUX_UNIT


def magnitudes_to_photons_per_frame(
    magnitudes: u.Quantity,
    photometry: WFSPhotometryConfig,
) -> u.Quantity:
    """Convert magnitudes to a one-dimensional unrounded WFS photon vector.

    Args:
        magnitudes: Scalar or array of finite Astropy magnitudes.
        photometry: Explicit telescope and WFS calibration.

    Returns:
        One-dimensional Astropy photon quantity, including one element for a
        scalar magnitude. No rounding is performed.

    Raises:
        TypeError: The magnitudes are not an Astropy quantity.
        ValueError: Magnitudes have incompatible units or invalid values, or
            calibration values are invalid.
    """
    magnitude_values = quantity_value(magnitudes, u.mag, label="magnitudes").reshape(-1)
    photon_flux = magnitudes_to_photon_flux(magnitude_values * u.mag, photometry.zeropoint)
    return photon_flux_to_photons_per_frame(photon_flux, photometry)


def photons_per_frame_to_magnitudes(
    photons_per_frame: u.Quantity,
    photometry: WFSPhotometryConfig,
) -> u.Quantity:
    """Invert WFS photons per frame to a one-dimensional magnitude vector.

    Zero input uses a ``1e-30 photon / (m2 s)`` logarithm floor after
    reversing the WFS scaling.

    Args:
        photons_per_frame: Scalar or array of nonnegative photon quantities.
        photometry: Explicit telescope and WFS calibration.

    Returns:
        One-dimensional Astropy magnitude quantity, including one element for
        a scalar count.

    Raises:
        TypeError: The count is not an Astropy quantity.
        ValueError: The count has incompatible units or invalid values, or
            calibration values are invalid.
    """
    photon_values = nonnegative_quantity_value(
        photons_per_frame, u.photon, label="photons_per_frame"
    ).reshape(-1)
    photon_flux = photons_per_frame_to_photon_flux(photon_values * u.photon, photometry)
    return photon_flux_to_magnitudes(photon_flux, photometry.zeropoint)


def magnitudes_to_effective_photons_per_second(
    magnitudes: u.Quantity,
    photometry: WFSPhotometryConfig,
) -> u.Quantity:
    """Return the unrounded WFS-effective photons/s vector.

    This multiplies the unrounded per-frame result by frame rate after
    conversion. It is not the pre-aperture photon flux.

    Args:
        magnitudes: Scalar or array of finite Astropy magnitudes.
        photometry: Explicit telescope and WFS calibration.

    Returns:
        One-dimensional Astropy photon/s quantity, including one element for
        a scalar magnitude. No rounding is performed.

    Raises:
        TypeError: The magnitudes are not an Astropy quantity.
        ValueError: Magnitudes have incompatible units or invalid values, or
            calibration values are invalid.
    """
    photons_per_frame = magnitudes_to_photons_per_frame(magnitudes, photometry)
    frame_rate = positive_scalar_quantity(photometry.frame_rate, u.Hz, label="frame_rate")
    values = photons_per_frame.to_value(u.photon) * frame_rate
    return values * (u.photon / u.s)
