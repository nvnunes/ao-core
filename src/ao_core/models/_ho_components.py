"""High-order reconstruction, control and scientific component assembly."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import numpy as np
from astropy import units as u
from scipy import linalg

from ._ho_reconstruction import (
    DirectionalCommandFit,
    lenslet_covariances,
    phase_slope_covariance,
    solve_commands,
)
from ._ho_responses import (
    CommandLayout,
    GaussianDm,
    PreparedDm,
    PreparedSensing,
    ShackHartmannSensing,
    tangent_directions,
)
from .config import FittingConfig, LoopConfig
from .control import polc_transfer


def _fitting_weights(weights, count):
    weights = np.ones(count) if weights is None else np.asarray(weights, dtype=float)
    if (
        weights.shape != (count,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError(
            "Mirror fitting weights must match the directions and be nonnegative with positive sum"
        )
    return weights


class SlopePrior:
    """Phase–slope moments and the regularized slope solve for a prediction.

    AO Core prepares this prior and passes it to the reconstructor. The
    factorization is shared by phase-fit reconstructors for both stages.
    """

    def __init__(self, grid, layers, sensing, inversion_floor, regularization, dtype):
        self._grid = grid
        self._layers = layers
        self._lenslets = sensing._lenslets
        self._inversion_floor = inversion_floor
        self._regularization = regularization
        self._dtype = dtype
        self._temporal = None

    @cached_property
    def _factor(self):
        regular, self._temporal = lenslet_covariances(
            self._grid,
            self._layers,
            self._lenslets,
            self._inversion_floor,
            self._regularization,
        )
        return linalg.cho_factor(
            regular, lower=True, overwrite_a=True, check_finite=False
        )

    @property
    def _temporal_covariance(self):
        if self._temporal is None:
            _, self._temporal = lenslet_covariances(
                self._grid, self._layers, self._lenslets, None, 0.0
            )
        return self._temporal

    def phase_slope_covariance(
        self,
        points: u.Quantity,
        directions: u.Quantity,
        *,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return covariance of weighted target phase at each point with each raw slope."""
        return self._phase_slope_covariance(
            points.to_value(u.m), directions, weights, np.float64
        )

    def _phase_slope_covariance(self, points, directions, weights, dtype):
        rays = tangent_directions(directions.to_value(u.rad))
        if rays.ndim != 2 or rays.shape[1] != 2 or not len(rays):
            raise ValueError("directions must contain one or more angular x/y pairs")
        weights = _fitting_weights(weights, len(rays))
        return phase_slope_covariance(
            self._grid,
            self._layers,
            self._lenslets,
            points,
            rays,
            weights / weights.sum(),
            dtype,
        )

    def solve(self, target_slope_covariance: np.ndarray) -> np.ndarray:
        """Return target-by-slope reconstructor rows from target–slope covariances."""
        return solve_commands(self._factor, target_slope_covariance)


def _command_matrix(matrix, shape):
    if isinstance(matrix, u.Quantity):
        raise TypeError("matrix must be a numerical array with values in metres")
    matrix = np.asarray(matrix)
    if matrix.dtype.kind not in "fiu" or matrix.shape != shape:
        raise ValueError(f"matrix must be real with actuator-by-slope shape {shape}")
    matrix = matrix.astype(np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("matrix must contain finite values")
    return matrix


class PreparedReconstructor(ABC):
    """Produces actuator-by-slope operators in the prepared geometry."""

    @abstractmethod
    def matrix(
        self, directions: u.Quantity, *, weights: np.ndarray | None = None
    ) -> np.ndarray:
        """Return one real operator for angular x/y directions and optional weights.

        The shape is (active actuators, raw WFS slopes) in the resolved layout.
        Numerical values in metres map phase gradients in rad/m to actuator
        phase in rad at the same wavelength. The predictor copies this result
        before another call, so the implementation may reuse its storage.
        """

    def _prediction_matrix(self, directions, shape, *, weights=None):
        return _command_matrix(self.matrix(directions, weights=weights), shape).copy()

    def _common_matrix(self, target, shape):
        directions, weights = target
        return self._prediction_matrix(directions, shape, weights=weights)

    def _directional_matrices(self, directions, shape, *, backend):
        return [
            self._prediction_matrix(direction[None], shape) for direction in directions
        ]


class Reconstructor(Protocol):
    """Reconstruction component assembled with resolved sensing and a DM."""

    def prepare(
        self,
        *,
        sensing: PreparedSensing,
        dm: PreparedDm,
        prior: SlopePrior,
    ) -> PreparedReconstructor:
        """Bind the reconstructor to the geometry and the available design prior."""
        ...


class _PreparedPhaseFit(PreparedReconstructor):
    def __init__(self, phase_fit, prior, factor):
        self._phase_fit = phase_fit
        self._prior = prior
        self._factor = factor

    def matrix(self, directions, *, weights=None):
        return self._matrix(directions, weights, np.float64)

    def _prediction_matrix(self, directions, shape, *, weights=None):
        return self.matrix(directions, weights=weights)

    def _matrix(self, directions, weights, dtype):
        cross = self._prior._phase_slope_covariance(
            self._phase_fit.points, directions, weights, dtype
        )
        return solve_commands(self._factor, self._phase_fit.projection @ cross)

    @cached_property
    def _directional_fit(self):
        prior = self._prior
        return DirectionalCommandFit(
            prior._grid,
            prior._layers,
            prior._lenslets,
            self._phase_fit,
            self._factor,
            prior._dtype,
        )

    def _directional_matrices(self, directions, shape, *, backend):
        if backend == "mps":
            return self._directional_fit.fit_batch(
                tangent_directions(directions.to_value(u.rad))
            )
        return [
            self._matrix(direction[None], None, self._prior._dtype)
            for direction in directions
        ]


class PhaseFitReconstructor:
    """Estimate target phase from raw slopes and fit it to the actuator basis.

    The covariance regression follows Eq. (5) of
    [Correia et al. (2018)](https://arxiv.org/pdf/1806.07938).
    The target is the weighted field phase projected onto the calibrated DM.
    """

    def prepare(
        self,
        *,
        sensing: PreparedSensing,
        dm: PreparedDm,
        prior: SlopePrior,
    ) -> PreparedReconstructor:
        """Prepare phase fitting and the shared regularized slope factor."""
        return _PreparedPhaseFit(dm._phase_fit(sensing), prior, prior._factor)


class MatrixReconstructor(PreparedReconstructor):
    """A fixed actuator-by-slope operator in a declared physical layout.

    Slopes are phase gradients in radians per metre and commands are phase
    coefficients in radians at the same wavelength. The supplied operator
    carries any command-space tilt projection. Owns a read-only float64 copy.
    """

    def __init__(self, *, matrix: np.ndarray, layout: CommandLayout):
        if not isinstance(layout, CommandLayout):
            raise TypeError("layout must be a CommandLayout")
        matrix = _command_matrix(matrix, layout.shape)
        self._matrix = np.frombuffer(matrix.tobytes(), dtype=np.float64).reshape(
            layout.shape
        )
        self._layout = layout

    def prepare(
        self,
        *,
        sensing: PreparedSensing,
        dm: PreparedDm,
        prior: SlopePrior,
    ) -> PreparedReconstructor:
        """Check channel and actuator order against the requested geometry."""
        if not self._layout._matches(dm.actuator_positions, sensing.layout):
            raise ValueError(
                "matrix layout does not match the resolved actuator and slope order"
            )
        return self

    def _common_matrix(self, target, shape):
        return self._matrix

    def _directional_matrices(self, directions, shape, *, backend):
        return [self._matrix] * len(directions)

    def matrix(
        self, directions: u.Quantity, *, weights: np.ndarray | None = None
    ) -> np.ndarray:
        """Return the same read-only operator for every direction and weight."""
        return self._matrix


def common_fit_target(reconstructor, fitting: FittingConfig | None):
    """Validate a fitted common stage's target before building the phase prior."""
    if isinstance(reconstructor, MatrixReconstructor):
        return None
    if fitting is None or fitting.directions is None or fitting.weights is None:
        raise ValueError("HO correction requires mirror fitting directions and weights")
    return fitting.directions, _fitting_weights(
        fitting.weights.value, len(fitting.directions)
    )


class ProjectedPolc:
    """Closed-loop pseudo-open-loop control with sensor-mean removal."""

    def transfer(self, frequencies: u.Quantity, *, loop: LoopConfig) -> np.ndarray:
        """Return the scalar complex response on signed temporal frequencies in Hz."""
        if (
            loop is None
            or loop.gain is None
            or loop.frame_interval is None
            or loop.delay_frames is None
        ):
            raise ValueError(
                "The common high-order loop requires gain, frame rate and delay"
            )
        if not 0 <= loop.gain.value <= 1 or loop.delay_frames % 1:
            raise ValueError(
                "Projected POLC requires gain in [0, 1] and whole-frame delay"
            )
        hz = frequencies.to_value(u.Hz)
        if loop.gain.value == 0:
            return np.zeros_like(hz)
        return polc_transfer(
            hz,
            loop.gain.value,
            loop.frame_interval.to_value(u.s),
            int(loop.delay_frames),
        )

    def sensor_mean_response(
        self,
        transfer: np.ndarray,
        *,
        reduced_feedback: np.ndarray,
        atmospheric_means: np.ndarray,
        feedback_means: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the common-loop mean correction and its driving term.

        The feedback matrix acts in the mean basis, one mean per guide and
        slope axis. The spectral inputs and both outputs lead with that axis.
        The delayed science loop uses the same driving term.
        """
        drive = -atmospheric_means + transfer * feedback_means
        count = len(reduced_feedback)
        frequencies = transfer.ravel()
        rhs = drive.reshape(count, -1)
        correction = np.empty_like(rhs)
        for first in range(0, len(frequencies), 4096):
            block = slice(first, first + 4096)
            gain = frequencies[block, None, None]
            system = np.eye(count) - gain * reduced_feedback
            correction[:, block] = (
                np.linalg.solve(system, rhs[:, block].T[..., None]) * gain
            )[..., 0].T
        return correction.reshape(drive.shape), drive


@dataclass(frozen=True)
class DelayedResidualResponse:
    """Science DM gain and time delay applied to the residual slopes."""

    gain: float
    delay: u.Quantity


class DelayedResidualControl:
    """Apply gain and delay to the slopes left by the common correction."""

    def response(
        self,
        *,
        loop: LoopConfig,
        common_frame_interval: u.Quantity,
    ) -> DelayedResidualResponse:
        """Resolve the residual gain and delay on the common loop clock."""
        if loop.gain is None or loop.gain.value < 0:
            raise ValueError("The residual high-order loop requires a nonnegative gain")
        gain = loop.gain.value
        if not gain:
            return DelayedResidualResponse(gain, 0 * u.s)
        if (
            loop.delay_frames is None
            or loop.delay_frames % 1
            or (
                loop.frame_interval is not None
                and loop.frame_interval != common_frame_interval
            )
        ):
            raise ValueError(
                "The residual high-order loop requires a whole-frame delay at the common high-order loop frame rate"
            )
        return DelayedResidualResponse(
            gain, (loop.delay_frames * common_frame_interval).to(u.s)
        )


@dataclass(frozen=True, kw_only=True)
class HoPsfStage:
    """DM response, reconstruction and control for one HO correction stage."""

    dm: GaussianDm
    reconstructor: Reconstructor
    control: ProjectedPolc | DelayedResidualControl


@dataclass(frozen=True, kw_only=True)
class HoPsfComponents:
    """Scientific components for the common DM and optional science DM."""

    sensing: ShackHartmannSensing
    common_stage: HoPsfStage
    science_stage: HoPsfStage | None = None

    def __post_init__(self):
        if not isinstance(self.common_stage.control, ProjectedPolc):
            raise TypeError("The common stage requires a ProjectedPolc controller")
        if self.science_stage is not None and not isinstance(
            self.science_stage.control, DelayedResidualControl
        ):
            raise TypeError(
                "The science stage requires a DelayedResidualControl controller"
            )


def default_components(two_stage):
    return HoPsfComponents(
        sensing=ShackHartmannSensing(),
        common_stage=HoPsfStage(
            dm=GaussianDm(),
            reconstructor=PhaseFitReconstructor(),
            control=ProjectedPolc(),
        ),
        science_stage=HoPsfStage(
            dm=GaussianDm(),
            reconstructor=PhaseFitReconstructor(),
            control=DelayedResidualControl(),
        )
        if two_stage
        else None,
    )
