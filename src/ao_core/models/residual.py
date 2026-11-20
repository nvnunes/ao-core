"""Response covariance contractions and residual covariance assembly."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from astropy import units as u

from ._field_covariance import FieldCovariance
from ._units import _input_dtype, _PhysicalRecord, _quantity

__all__ = [
    "FieldCovariance",
    "TiltCommandCovariances",
    "TiltResidualCovariances",
    "assemble_tilt_residual_covariances",
    "response_covariance",
    "source_minus_correction_covariance",
]


@dataclass(frozen=True, kw_only=True, eq=False)
class TiltCommandCovariances(_PhysicalRecord):
    """Science tilt and filtered command covariances in physical x/y order.

    Inputs are angular-squared quantities, stored in mas² at the precision of
    `science_tilt`. Terms involving science tilt have shape (science, 2, 2).
    Command-only terms have shape (2, 2). Cross covariances can be asymmetric.

    Attributes:
        science_tilt: Atmospheric tilt covariance at each science direction.
        atmospheric_command: Covariance of the command driven by sensed tilt.
        ho_command: Covariance of the command driven by sensed HO aberrations.
        atmospheric_ho_command: Cross covariance of the atmospheric command
            with the HO command, in that order.
        science_atmospheric_command: Cross covariance of science tilt with
            the atmospheric command, in that order.
        science_ho_command: Cross covariance of science tilt with the HO
            command, in that order.
    """

    science_tilt: u.Quantity
    atmospheric_command: u.Quantity
    ho_command: u.Quantity
    atmospheric_ho_command: u.Quantity
    science_atmospheric_command: u.Quantity
    science_ho_command: u.Quantity

    def __post_init__(self) -> None:
        _freeze_tilt_covariances(
            self, ("science_tilt", "science_atmospheric_command", "science_ho_command")
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class TiltResidualCovariances(_PhysicalRecord):
    """Assembled tip–tilt residual terms in mas² and physical x/y order.

    Attributes:
        atmospheric_residual: Residual atmospheric covariance at each science
            direction, with shape (science, 2, 2).
        ho_contribution: Signed change from HO-induced commands, including
            their cross covariances with the atmospheric residual. Same shape
            as `atmospheric_residual`.
        command_noise: Noise-driven command covariance of shape (2, 2),
            shared across the science directions.
    """

    atmospheric_residual: u.Quantity
    ho_contribution: u.Quantity
    command_noise: u.Quantity

    def __post_init__(self) -> None:
        _freeze_tilt_covariances(self, ("atmospheric_residual", "ho_contribution"))

    @property
    def noiseless(self) -> u.Quantity:
        """Return the sum of the atmospheric residual and HO contribution."""
        return self.atmospheric_residual + self.ho_contribution

    @property
    def total(self) -> u.Quantity:
        """Return the residual covariance including command noise at each direction."""
        return self.noiseless + self.command_noise


def _freeze_tilt_covariances(
    record: TiltCommandCovariances | TiltResidualCovariances,
    field_terms: tuple[str, ...],
) -> None:
    reference = getattr(record, field_terms[0])
    dtype = _input_dtype(reference, None)
    field_shape = np.shape(reference)[:1] + (2, 2)
    for item in fields(record):
        name = item.name
        shape = field_shape if name in field_terms else (2, 2)
        covariance = _quantity(
            getattr(record, name), u.mas**2, name, len(shape), dtype=dtype
        )
        if covariance.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        object.__setattr__(record, name, covariance)


def assemble_tilt_residual_covariances(
    terms: TiltCommandCovariances, *, command_noise: u.Quantity
) -> TiltResidualCovariances:
    """Combine atmospheric tilt, HO-induced commands and command noise.

    The science tilt is reduced by the two filtered commands. The orientation
    of each cross covariance determines which transpose enters the result.
    The atmospheric and HO outputs are symmetrized to remove roundoff.
    Applications apply their own positive-semidefinite acceptance thresholds.

    Args:
        terms: Science tilt, filtered commands and their cross covariances.
        command_noise: Angular covariance of the noise-driven command,
            independent of atmospheric tilt and HO aberrations, with shape (2, 2).

    Returns:
        Atmospheric residual, signed HO contribution and command noise in mas².
    """
    science_atmospheric = terms.science_atmospheric_command.value
    atmospheric = source_minus_correction_covariance(
        terms.science_tilt.value,
        terms.atmospheric_command.value,
        science_atmospheric,
        science_atmospheric.swapaxes(-1, -2),
    )
    # Cov(t_atm - u_atm, u_HO) retains the argument order.
    residual_ho_cross = (
        terms.science_ho_command.value - terms.atmospheric_ho_command.value
    )
    ho_contribution = (
        terms.ho_command.value - residual_ho_cross - residual_ho_cross.swapaxes(-1, -2)
    )
    return TiltResidualCovariances(
        atmospheric_residual=((atmospheric + atmospheric.swapaxes(-1, -2)) / 2)
        * u.mas**2,
        ho_contribution=((ho_contribution + ho_contribution.swapaxes(-1, -2)) / 2)
        * u.mas**2,
        command_noise=command_noise,
    )


def response_covariance(
    left: np.ndarray, right: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Contract responses with a common Fourier sample axis at the end.

    This is the covariance of the left response with the right response.
    The weights include the layer spectrum and Fourier cell area. Complex
    conjugation applies to the right channels, retaining cross-term orientation.
    """
    return (left * weights) @ right.conj().swapaxes(-1, -2)


def source_minus_correction_covariance(
    source: np.ndarray,
    correction: np.ndarray,
    source_correction: np.ndarray,
    correction_source: np.ndarray,
) -> np.ndarray:
    """Assemble residual covariance from source, correction and both cross terms.

    Blocks have the same units, left and right axis order and lag convention.
    The two cross terms can differ at nonzero lag or between field directions.
    """
    return source + correction - source_correction - correction_source
