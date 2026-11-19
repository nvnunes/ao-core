"""Matched Fourier response and residual covariance contractions."""

from __future__ import annotations

import numpy as np

from ._field_covariance import FieldCovariance

__all__ = [
    "FieldCovariance",
    "response_covariance",
    "source_minus_correction_covariance",
]


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
