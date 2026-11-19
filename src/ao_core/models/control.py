"""Fixed-gain delayed-integrator response and stationary command covariance.

Numerical inputs use two angular tilt axes. Reconstructed sensing response and
controller gain are dimensionless; noise covariance uses the caller's squared
angular unit. No calibration bundle or E2E implementation is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy.linalg import solve_discrete_lyapunov
from scipy.signal import lfilter, lfilter_zi

from ._units import (
    _count,
    _input_dtype,
    _PhysicalRecord,
    _quantity,
    validate_float_dtype,
)

__all__ = [
    "FixedGainIntegrator",
    "filter_input",
    "integrator_noise_covariance",
    "integrator_noise_lags",
    "integrator_transition",
    "polc_transfer",
    "scalar_filter",
    "settling_cut",
    "state_matrix",
]


@dataclass(frozen=True, kw_only=True, eq=False)
class FixedGainIntegrator(_PhysicalRecord):
    """Fixed delayed-integrator settings shared by calibration and prediction.

    Attributes:
        dtype: Real calculation/state precision, float64 by default or float32.
        gain: Positive dimensionless gain, fixed while filtering.
        delay: Positive integer measurement-to-command delay, in frames.
        frame_rate: Positive scalar frequency; one telemetry sample per frame.
    """

    gain: float
    delay: int
    frame_rate: u.Quantity

    dtype: DTypeLike = field(default=np.float64, kw_only=True)

    def __post_init__(self):
        object.__setattr__(self, "dtype", validate_float_dtype(self.dtype))
        if isinstance(self.gain, bool) or not np.isfinite(self.gain) or self.gain <= 0:
            raise ValueError("gain must be positive and finite")
        _count(self.delay, "delay")
        rate = _quantity(self.frame_rate, u.Hz, "frame_rate", 0, dtype=self.dtype)
        if rate.value <= 0:
            raise ValueError("frame_rate must be positive")
        object.__setattr__(self, "gain", float(self.gain))
        object.__setattr__(self, "frame_rate", rate)

    def transition(self, response: np.ndarray) -> np.ndarray:
        """Return the stable two-axis delayed-command transition matrix.

        response is the finite dimensionless 2-by-2 reconstructed sensing
        matrix. History blocks run newest to oldest. Unstable loops raise
        ValueError; the returned array is newly owned by the caller.

        Args:
            response: Reconstructed dimensionless feedback matrix of shape ``(2, 2)``.

        Returns:
            Command-history transition with shape ``(2*delay, 2*delay)``.

        Raises:
            ValueError: If the response is invalid or the loop is unstable.
        """
        return integrator_transition(response, self.gain, self.delay, dtype=self.dtype)

    def filter(
        self, history: u.Quantity, response: np.ndarray, *, equilibrium: bool = False
    ) -> u.Quantity:
        """Filter delayed inputs independently across realizations.

        history has shape (realizations,frames,2), angular or OPD units.
        response is the dimensionless 2-by-2 feedback matrix. Return commands
        in the same units and shape. The default has zero command/input prehistory;
        equilibrium=True instead
        holds each first input constant before the first frame. No centring or
        settling cut is applied. Coupled responses are retained exactly, including
        nonsymmetric terms.

        Args:
            history: Input history with shape ``(realizations, frames, 2)``.
            response: Reconstructed dimensionless feedback matrix.
            equilibrium: Start from steady state under the first input of each realization.

        Returns:
            Command history with the same shape and physical units as the input.

        Raises:
            TypeError: If history is not an Astropy quantity.
            ValueError: If history, response or feedback stability is invalid.
        """
        if not isinstance(history, u.Quantity):
            raise TypeError("history must be an Astropy quantity")
        values = np.asarray(history.value)
        if (
            values.ndim != 3
            or values.shape[-1] != 2
            or not np.isfinite(values).all()
            or not values.size
        ):
            raise ValueError("history must be finite nonempty (realizations,frames,2)")
        if not (history.unit.is_equivalent(u.rad) or history.unit.is_equivalent(u.m)):
            raise ValueError("history must have angular or OPD units")
        if values.dtype.kind not in "fiu" or type(equilibrium) is not bool:
            raise ValueError("history must be real and equilibrium must be boolean")
        response = np.asarray(response)
        if response.dtype.kind not in "fiu":
            raise ValueError("response must be a real matrix")
        response = response.astype(self.dtype)
        values = values.astype(self.dtype)
        initial = (
            np.linalg.solve(response, values[:, 0, :].T).T
            if equilibrium
            else np.zeros((len(values), 2), dtype=self.dtype)
        )
        inputs = values - values[:, :1, :] if equilibrium else values
        result = filter_input(
            inputs.transpose(2, 0, 1), response, self.gain, self.delay, dtype=self.dtype
        )
        result = result + initial.T[:, :, None]
        return result.transpose(1, 2, 0) * history.unit

    def noise_covariance(
        self, response: np.ndarray, reconstructed_noise: u.Quantity
    ) -> u.Quantity:
        """Solve the stationary Lyapunov command covariance for white input noise.

        reconstructed_noise is a finite symmetric PSD 2-by-2 angular-squared
        quantity, independent between frames. Return command covariance in
        mas squared. The supplied response is the actual feedback matrix.

        Args:
            response: Actual reconstructed dimensionless feedback matrix.
            reconstructed_noise: Per-frame stationary x/y input covariance in
                angular-squared units.

        Returns:
            A ``(2, 2)`` angular command covariance in square milliarcseconds.

        Raises:
            TypeError: If reconstructed noise is not an Astropy quantity.
            ValueError: If covariance, response or feedback stability is invalid.
        """
        if not isinstance(reconstructed_noise, u.Quantity):
            raise TypeError("reconstructed_noise must be an Astropy quantity")
        return (
            integrator_noise_covariance(
                response,
                reconstructed_noise.to_value(u.mas**2),
                self.gain,
                self.delay,
                dtype=self.dtype,
            )
            * u.mas**2
        )


def integrator_transition(
    response: ArrayLike, kappa: float, delay: int, *, dtype=np.float64
) -> NDArray:
    """Build the stable 2n-by-2n command-history transition matrix.

    ``response`` is the finite two-by-two reconstructed sensing response B=RG;
    ``kappa`` is the positive controller gain. ``delay`` is the positive integer
    measurement-to-command delay in frames. The command history runs from
    newest to oldest. Its first block obeys u[k]=u[k-1]-kappa*B*u[k-delay]
    in the absence of inputs. Invalid inputs or spectral radius >=1 raise
    ValueError. Input arrays are not mutated.
    """
    if (
        isinstance(delay, (bool, np.bool_))
        or not isinstance(delay, (int, np.integer))
        or delay < 1
    ):
        raise ValueError("Delay must be a positive integer")
    response = np.asarray(response)
    if (
        response.dtype.kind not in "fiu"
        or response.shape != (2, 2)
        or not np.isfinite(response).all()
    ):
        raise ValueError(
            "Reconstructed sensing response must be a finite two-by-two matrix"
        )
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("Controller gain must be finite and positive")
    dtype = _input_dtype(response, dtype)
    response = response.astype(dtype, copy=False)
    kappa = dtype.type(kappa)
    transition = np.zeros((2 * delay, 2 * delay), dtype=dtype)
    transition[:2, :2] = np.eye(2, dtype=dtype)
    # For n=1 the current and delayed blocks coincide: I - kappa B.
    transition[:2, -2:] -= kappa * response
    transition[2:, :-2] = np.eye(2 * (delay - 1), dtype=dtype)
    radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if radius >= 1:
        raise ValueError(
            f"Unstable LO loop: spectral radius rho(A)={radius:.12g} must be < 1 "
            f"(gain={kappa}, delay={delay})"
        )
    return transition


def integrator_noise_covariance(
    response: ArrayLike,
    reconstructed_noise: ArrayLike,
    kappa: float,
    delay: int,
    *,
    dtype=np.float64,
) -> NDArray:
    """Return the stationary two-by-two noise-driven command covariance.

    ``response``, ``kappa`` and ``delay`` follow ``integrator_transition``.
    ``reconstructed_noise`` is a finite symmetric PSD two-by-two covariance
    of reconstructed measurement noise, independent between frames. The output
    uses the same squared angular unit. Noise enters the command after ``delay``
    frames; stationarity permits indexing that white innovation at the update.

    Solve S=A*S*A.T+Q with Q's first block kappa**2*reconstructed_noise and
    all other blocks zero, then return S's first two-by-two block. Invalid
    inputs or an unstable loop raise ValueError. Inputs are not mutated.
    """
    transition, stationary = _stationary_state(
        response, reconstructed_noise, kappa, delay, dtype=dtype
    )
    covariance = np.asarray(stationary[:2, :2], dtype=transition.dtype)
    if not np.isfinite(covariance).all():
        raise ValueError("Nonfinite stationary LO noise covariance")
    return (covariance + covariance.T) / 2


def _stationary_state(response, reconstructed_noise, kappa, delay, *, dtype):
    dtype = _input_dtype(reconstructed_noise, dtype)
    transition = integrator_transition(response, kappa, delay, dtype=dtype)
    kappa = dtype.type(kappa)
    noise = np.asarray(reconstructed_noise, dtype=dtype)
    if (
        noise.shape != (2, 2)
        or not np.isfinite(noise).all()
        or not np.allclose(
            noise, noise.T, rtol=max(1e-12, 8 * np.finfo(dtype).eps), atol=1e-12
        )
    ):
        raise ValueError(
            "Reconstructed noise covariance must be finite, symmetric and two-by-two"
        )
    noise = (noise + noise.T) / 2
    if np.linalg.eigvalsh(noise).min() < -1e-12 * max(1.0, np.linalg.norm(noise)):
        raise ValueError("Reconstructed noise covariance must be positive semidefinite")
    innovation = np.zeros_like(transition)
    innovation[:2, :2] = kappa**2 * noise
    stationary = solve_discrete_lyapunov(transition, innovation)
    return transition, stationary


def integrator_noise_lags(
    response: ArrayLike,
    reconstructed_noise: ArrayLike,
    gain: float,
    delay: int,
    count: int,
    *,
    dtype: DTypeLike = np.float64,
) -> NDArray:
    """Return command covariance for lags 0 through count minus one, in frames.

    The two angular axes follow the input response. At positive lag k the
    row axis is at t+k and the column axis is at t. Reconstructed noise is
    independent between frames and the result retains its squared unit.
    """
    _count(count, "count")
    transition, stationary = _stationary_state(
        response, reconstructed_noise, gain, delay, dtype=dtype
    )
    power = np.eye(len(transition), dtype=transition.dtype)
    output = np.empty((count, 2, 2), dtype=transition.dtype)
    for lag in range(count):
        output[lag] = (power @ stationary)[:2, :2]
        power = transition @ power
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite stationary LO noise lag covariance")
    return output


def state_matrix(
    gain: float, delay: int, strength: float, *, dtype=np.float64
) -> np.ndarray:
    """Chronological command history transition for the delayed integrator."""
    dtype = validate_float_dtype(dtype)
    gain, strength = dtype.type(gain), dtype.type(strength)
    state = np.zeros((delay, delay), dtype=dtype)
    if delay > 1:
        state[:-1, 1:] = np.eye(delay - 1, dtype=dtype)
    state[-1, -1] += 1
    state[-1, 0] -= gain * strength
    return state


def settling_cut(
    gain: float,
    delay: int,
    strengths,
    *,
    tolerance: float = 1e-8,
    maximum: int = 500,
    dtype=np.float64,
) -> tuple[int, float]:
    """Find a common settling window for the specified feedback strengths.

    Return the number of initial frames to discard and the largest
    remaining state-propagation spectral norm across the grid. Raise
    ValueError for an unstable strength or if the maximum frame budget
    cannot meet the tolerance.
    """
    states = [state_matrix(gain, delay, b, dtype=dtype) for b in strengths]
    if any(max(abs(np.linalg.eigvals(a))) >= 1 for a in states):
        raise ValueError("Unstable feedback grid")
    for cut in range(delay, maximum + 1):
        envelope = max(
            np.linalg.norm(np.linalg.matrix_power(a, cut), 2) for a in states
        )
        if envelope <= tolerance:
            return cut, float(envelope)
    raise ValueError(f"No settled simulation window within {maximum} saved frames")


def scalar_filter(
    history: np.ndarray,
    strength: float,
    gain: float,
    delay: int,
    *,
    equilibrium: bool = True,
    dtype=np.float64,
) -> np.ndarray:
    """Filter (..., frames) input with a delayed integrator; preserve frame order.

    The response solves u[k]=u[k-1]-gain*b*u[k-delay]+gain*x[k-delay].
    Equilibrium initialization supplies constant first-input prehistory.
    Zero initialization is exposed only for independent settling checks.
    """
    dtype = _input_dtype(history, dtype)
    history = np.asarray(history, dtype=dtype)
    gain, strength = dtype.type(gain), dtype.type(strength)
    numerator = np.zeros(delay + 1, dtype=dtype)
    numerator[-1] = gain
    denominator = np.zeros(delay + 1, dtype=dtype)
    denominator[0], denominator[1] = 1.0, -1.0
    denominator[-1] += gain * strength
    if not equilibrium:
        return lfilter(numerator, denominator, history, axis=-1)
    zi = history[..., :1] * lfilter_zi(numerator, denominator)
    return lfilter(numerator, denominator, history, axis=-1, zi=zi)[0]


def filter_input(
    x: np.ndarray, B: np.ndarray, gain: float, delay: int, *, dtype=np.float64
) -> np.ndarray:
    """Zero-history delayed integrator response to shape (2,realizations,frames).

    A complex Schur basis is a numerically stable exact 2D decomposition.
    Coupled off-diagonal response is retained; no symmetric-B approximation or
    potentially ill-conditioned eigenvector inversion enters this simulation.
    """
    from scipy.linalg import schur

    dtype = _input_dtype(x, dtype)
    complex_dtype = np.complex64 if dtype.itemsize == 4 else np.complex128
    x = np.asarray(x, dtype=dtype)
    B = np.asarray(B, dtype=dtype)
    gain = dtype.type(gain)
    integrator_transition(B, gain, delay, dtype=dtype)
    triangular, basis = schur(B, output="complex")
    triangular, basis = triangular.astype(complex_dtype), basis.astype(complex_dtype)
    y = np.einsum("ab,brt->art", basis.conj().T, x)
    numerator = np.r_[np.zeros(delay, dtype=dtype), gain]
    denominator2 = np.r_[1.0, -1.0, np.zeros(max(delay - 1, 0), dtype=dtype)].astype(
        complex_dtype
    )
    denominator2[delay] += gain * triangular[1, 1]
    second = lfilter(numerator, denominator2, y[1], axis=-1)
    first_input = y[0] - triangular[0, 1] * second
    denominator1 = np.r_[1.0, -1.0, np.zeros(max(delay - 1, 0), dtype=dtype)].astype(
        complex_dtype
    )
    denominator1[delay] += gain * triangular[0, 0]
    first = lfilter(numerator, denominator1, first_input, axis=-1)
    result = np.einsum("ab,brt->art", basis, np.stack((first, second)))
    bound = 32 * np.finfo(dtype).eps * max(1.0, float(np.max(np.abs(result.real))))
    if np.max(np.abs(result.imag)) > bound:
        raise ValueError("Noise simulation has a nonreal numerical remainder.")
    return result.real


def polc_transfer(
    frequency_hz: np.ndarray, gain: float, frame_interval_s: float, delay: int = 2
) -> np.ndarray:
    """Evaluate the scalar leaky pseudo-open-loop response on signed frequencies."""
    z = np.exp(-2j * np.pi * frequency_hz * frame_interval_s)
    return gain * z**delay / (1 - (1 - gain) * z)
