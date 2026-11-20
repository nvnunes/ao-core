"""Field-centre low-order reconstruction in interleaved x/y order."""

from __future__ import annotations

import numpy as np
from astropy import units as u
from numpy.typing import DTypeLike

from ._units import _input_dtype, _quantity, validate_float_dtype

__all__ = [
    "factor_modal_covariance",
    "lo_reconstructor",
    "modal_covariance_blocks",
    "solve_lo_reconstructor",
]


def modal_covariance_blocks(measurement, target, weights, *, real_contraction=False):
    """Return measurement and target–measurement covariance on Fourier nodes.

    Response rows are caller-selected channels. Weights include spectral
    density, layer strength and quadrature weight. The real-contraction
    form uses a real-inner-product BLAS kernel for tiled modal calculations.
    For that form supply complex128 response arrays with a contiguous Fourier
    axis and float64 weights.
    """
    if real_contraction:
        if measurement.dtype != np.complex128 or target.dtype != np.complex128:
            raise ValueError("The real contraction requires complex128 responses")
        adjoint = measurement.view(float).T
        return (
            (measurement * weights).view(float) @ adjoint,
            (target * weights).view(float) @ adjoint,
        )
    adjoint = measurement.conj().T
    return (
        ((measurement * weights) @ adjoint).real,
        ((target * weights) @ adjoint).real,
    )


def factor_modal_covariance(covariance, *, relative_floor=1e-8):
    """Return a floored covariance inverse and its inverse-square-root factor.

    The eigensystem uses the symmetric covariance's lower triangle. Eigenvalues
    are floored relative to the largest one. The factor times its transpose
    equals the inverse. Supply a finite real symmetric matrix with at least
    one positive eigenvalue and a positive finite relative floor.
    """
    covariance = np.asarray(covariance)
    if (
        covariance.ndim != 2
        or not len(covariance)
        or covariance.shape[0] != covariance.shape[1]
        or covariance.dtype.kind not in "fiu"
        or not np.isfinite(covariance).all()
    ):
        raise ValueError("Expected a finite real symmetric modal covariance")
    if not np.allclose(
        covariance, covariance.T, rtol=1e-12, atol=1e-12 * np.max(np.abs(covariance))
    ):
        raise ValueError("Expected a symmetric modal covariance")
    if not np.isfinite(relative_floor) or relative_floor <= 0:
        raise ValueError("Relative eigenvalue floor must be positive and finite")
    values, vectors = np.linalg.eigh(covariance)
    floor = values[-1] * relative_floor
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("Modal eigenvalue floor must be positive and finite")
    floored = np.maximum(values, floor)
    return (vectors / floored) @ vectors.T, vectors / np.sqrt(floored)


def solve_lo_reconstructor(
    ngs_covariance: u.Quantity,
    centre_ngs_covariance: u.Quantity,
    measurement_noise: u.Quantity,
    *,
    dtype: DTypeLike = np.float64,
    method: str = "direct",
    rcond: float = 1e-10,
) -> u.Quantity:
    """Solve absolute field-centre LO weights from explicit angular covariances.

    NGS/NGS and noise inputs have shape (2*N,2*N); field-centre/NGS has shape
    (2,2*N), in interleaved physical x/y order. Convert all to mas squared and
    solve R=Cas(Css+C_eta)^-1 without conditioning or block normalization. One
    NGS returns I. Return an immutable dimensionless (2,2*N) quantity. No
    calibrated model or native software is required. The direct solver raises
    for a singular matrix. For truncated SVD use method='pinv'.

    Args:
        ngs_covariance: Atmospheric covariance between all NGS tilt axes.
        centre_ngs_covariance: Field-centre tilt cross-covariance with NGS axes.
        measurement_noise: Pre-filter NGS measurement-noise covariance.
        dtype: Numerical precision of the solve.
        method: Direct solve or thresholded SVD pseudoinverse.
        rcond: Relative singular value cutoff for the pseudoinverse.

    Returns:
        Dimensionless field-centre weights in interleaved NGS x/y order.

    Raises:
        TypeError: If a covariance lacks an Astropy quantity.
        ValueError: If covariance inputs are invalid.
        numpy.linalg.LinAlgError: If the required multistar matrix is singular.
    """
    dtype = validate_float_dtype(dtype)
    values = [
        _quantity(value, u.mas**2, name, 2, dtype=dtype).value
        for value, name in (
            (ngs_covariance, "ngs_covariance"),
            (centre_ngs_covariance, "centre_ngs_covariance"),
            (measurement_noise, "measurement_noise"),
        )
    ]
    return _quantity(
        lo_reconstructor(*values, dtype=dtype, method=method, rcond=rcond) * u.one,
        u.one,
        "weights",
        2,
        dtype=dtype,
    )


def lo_reconstructor(
    css_mas2: np.ndarray,
    cas_center_mas2: np.ndarray,
    centered_noise_mas2: np.ndarray,
    *,
    dtype=np.float64,
    method="direct",
    rcond=1e-10,
) -> np.ndarray:
    """Return absolute R = Cas(0,0) (Css + C_eta)^-1, or I for one NGS.

    Css and C_eta are 2*N square; Cas(0,0) is 2 by 2*N. All covariances
    are supplied in mas². This solver preserves the noise-dependent amplitude
    of R without conditioning matrices or normalizing reconstructor blocks.

    See Eq. (5) for the minimum-variance covariance solve used in
    the multistar case. [Correia et al. (2018)](https://arxiv.org/pdf/1806.07938)
    """
    dtype = _input_dtype(css_mas2, dtype)
    css, cas, noise = [
        np.asarray(v, dtype=dtype)
        for v in (css_mas2, cas_center_mas2, centered_noise_mas2)
    ]
    n = css.shape[0] if css.ndim == 2 else 0
    if (
        n < 2
        or n % 2
        or css.shape != (n, n)
        or noise.shape != css.shape
        or cas.shape != (2, n)
    ):
        raise ValueError(
            "Expected Css and noise (2*N,2*N), and field-centre Cas (2,2*N)."
        )
    if not all(np.isfinite(v).all() for v in (css, cas, noise)):
        raise ValueError("Reconstructor covariances must be finite.")
    if method not in ("direct", "pinv"):
        raise ValueError("method must be direct or pinv")
    if method == "pinv" and (not np.isfinite(rcond) or rcond < 0):
        raise ValueError("rcond must be finite and nonnegative")
    if n == 2:
        return np.eye(2, dtype=dtype)
    if method == "pinv":
        return cas @ np.linalg.pinv(css + noise, rcond=rcond)
    return np.linalg.solve((css + noise).T, cas.T).T
