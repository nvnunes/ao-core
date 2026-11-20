"""Moment propagation for the projected common high-order controller.

The small sensor-mean solve follows the project-authored P3 finiteMeanSlopes
calculation. Products are contracted in the applied actuator basis.
"""

import numpy as np
from scipy import ndimage

from ._ho_reconstruction import (
    CovarianceQueries,
    covariance_kernel,
    fourier_sum,
    gaussian_slopes,
)


def _slope_response(rows, grid, layer, lenslets):
    """Evaluate weighted lenslet slopes on all represented atmospheric modes."""
    samples = len(lenslets.cells)
    count = len(grid.fx)
    pitch = lenslets.pitch
    step = layer.cone_scale * grid.spacing
    first = np.full(2, -(count // 2) * step)
    centre = lenslets.count // 2 - (lenslets.count - 1) / 2
    phase = np.exp(2j * np.pi * centre * pitch * layer.cone_scale * (grid.fx + grid.fy))
    result = np.zeros((len(rows), count, count), dtype=complex)
    x, y = lenslets.indices.T
    for guide, footprint in enumerate(layer.footprints):
        offset = phase * np.exp(
            2j * np.pi * (grid.fx * footprint[0] + grid.fy * footprint[1])
        )
        for direction, gradient in enumerate(layer.gradient):
            first_column = (2 * guide + direction) * samples
            weights = np.zeros((len(rows), lenslets.count, lenslets.count))
            weights[:, x, y] = rows[:, first_column : first_column + samples]
            result += (
                fourier_sum(weights, pitch, first, step, count) * offset * gradient
            )
    return result


def projected_modes(temporal, feedback, sensor_means, feedback_means):
    """Small exact POLC solve for the removed measurement means."""
    return _solve_projected_modes(
        temporal, feedback, -sensor_means + temporal * feedback_means
    )


def _solve_projected_modes(temporal, feedback, rhs):
    count = len(feedback)
    flat = temporal.ravel()
    shape = rhs.shape
    rhs = rhs.reshape(count, -1)
    output = np.empty_like(rhs)
    for first in range(0, len(flat), 4096):
        block = slice(first, first + 4096)
        control = flat[block, None, None]
        matrix = np.eye(count) - control * feedback
        output[:, block] = (
            np.linalg.solve(matrix, rhs[:, block].T[..., None]) * control
        )[..., 0].T
    return output.reshape(shape)


def add_slope_mean_cross(result, grid, layer, lenslets, weighted):
    """Accumulate slope/mean-transfer covariance in the valid-cell order."""
    samples = len(lenslets.cells)
    x, y = lenslets.indices.T
    step = layer.cone_scale * lenslets.pitch
    for guide, footprint in enumerate(layer.footprints):
        start = footprint - step * (lenslets.count - 1) / 2
        for axis, gradient in enumerate(layer.gradient):
            values = fourier_sum(
                weighted * gradient, grid.spacing, start, step, lenslets.count
            ).real
            first = (2 * guide + axis) * samples
            result[first : first + samples] += values[:, x, y].T


def command_moments(grid, layers, lenslets, mirror, weights, measurement_covariance):
    """Return applied covariance, mean-transfer modes and their command map."""
    samples = len(lenslets.cells)
    channels = 2 * len(lenslets.offsets)
    means = np.kron(np.eye(channels), np.ones((samples, 1)) / np.sqrt(samples))
    sensor_mirror = gaussian_slopes(
        lenslets.cells, mirror.centres, mirror.width, lenslets.pitch, lenslets.offsets
    )
    # Form only the rows of H = sensor_mirror @ weights needed by the mean solve.
    mean_feedback = (means.T @ sensor_mirror) @ weights
    command_means = weights @ means
    feedback = mean_feedback @ means
    rows = np.concatenate((means.T, mean_feedback))
    command_covariance = (weights @ measurement_covariance) @ weights.T
    covariance = np.zeros((channels, channels))
    sensor_cross = np.zeros((samples * channels, channels))
    modes = []
    for layer in layers:
        unprojected = _slope_response(rows, grid, layer, lenslets)
        mean_modes = projected_modes(
            layer.temporal, feedback, unprojected[:channels], unprojected[channels:]
        )
        modes.append(mean_modes)
        flat = mean_modes.reshape(channels, -1)
        covariance += ((flat * layer.power.ravel()) @ flat.conj().T).real
        add_slope_mean_cross(
            sensor_cross,
            grid,
            layer,
            lenslets,
            layer.power * layer.temporal * mean_modes.conj(),
        )
    cross = (weights @ sensor_cross) @ command_means.T
    command_covariance += (
        cross + cross.T + (command_means @ covariance) @ command_means.T
    )
    return command_covariance, modes, command_means


class ScienceCommandQueries:
    """Evaluate phase/slopes and phase/sensor-mean cross covariances."""

    def __init__(self, grid, lenslets, pupil, tilt_modes, *, dtype=np.float64):
        self.grid = grid
        self.lenslets = lenslets
        self.pupil = pupil
        self.tilt_modes = tilt_modes
        self.dtype = np.dtype(dtype)
        self.queries = CovarianceQueries(
            pupil.points, lenslets.cells, grid.kernel_side, grid.period
        )
        self.values = np.empty((2, len(pupil.points), len(lenslets.cells)), dtype=dtype)
        self.slope_ramps = pupil.ramps.astype(dtype, copy=False)

    def slope_kernels(self, layer, transfer):
        return slope_kernels(
            layer, transfer, self.tilt_modes, self.grid.kernel_side, self.dtype
        )

    def add_slope_cross(self, result, layer, kernels, footprint, rows, shift=0):
        phase_kernels, fit_kernels = kernels
        rows = rows.astype(self.dtype, copy=False)
        lenslets = self.lenslets
        samples = len(lenslets.cells)
        for guide, guide_footprint in enumerate(layer.footprints):
            origin = footprint - guide_footprint - shift
            self.queries.evaluate(phase_kernels, origin, layer.cone_scale, self.values)
            if fit_kernels:
                lags = origin - layer.cone_scale * lenslets.cells
                fitted = slope_fit_values(
                    fit_kernels, lags, self.grid.kernel_side / self.grid.period
                )
                for value, projected in zip(self.values, fitted, strict=True):
                    value -= self.slope_ramps.T @ projected
            for axis, value in enumerate(self.values):
                first = (2 * guide + axis) * samples
                result += value @ rows[:, first : first + samples].T

    def mean_cross(self, layer, modes, footprint):
        grid, pupil = self.grid, self.pupil
        real_dtype = self.values.dtype
        complex_dtype = np.result_type(real_dtype, np.complex64)
        phase = np.exp(2j * np.pi * (grid.fx * footprint[0] + grid.fy * footprint[1]))
        weighted_phase = layer.power.astype(real_dtype, copy=False) * phase.astype(
            complex_dtype, copy=False
        )
        cross = np.empty((len(pupil.points), len(modes)))
        for index, mode in enumerate(modes):
            density = weighted_phase * mode.astype(complex_dtype, copy=False).conj()
            lag = covariance_kernel(density, pupil.side)
            fitted = np.einsum("ixy,xy->i", self.tilt_modes, density).real
            cross[:, index] = lag[pupil.x_index, pupil.y_index] - pupil.ramps.T @ fitted
        return cross


def slope_kernels(layer, transfer, tilt_modes, side, dtype):
    """Prepare periodic spline coefficients for the phase and pupil fit."""

    def prepare(spectrum):
        return ndimage.spline_filter(
            covariance_kernel(spectrum, side), mode="grid-wrap"
        ).astype(dtype, copy=False)

    spectra = layer.power * layer.gradient.conj() * np.conj(transfer)
    phase = [prepare(spectrum) for spectrum in spectra]
    fit = [[prepare(spectrum * mode) for spectrum in spectra] for mode in tilt_modes]
    return phase, fit


def slope_fit_values(kernels, lags, scale):
    """Sample the tilt-fit kernels at lenslet lags with an optional field axis."""
    coordinates = np.moveaxis(lags, -1, 0) * scale
    return np.array(
        [
            [
                ndimage.map_coordinates(
                    fitted[axis],
                    coordinates,
                    order=3,
                    mode="grid-wrap",
                    prefilter=False,
                )
                for fitted in kernels
            ]
            for axis in range(2)
        ]
    )


def science_command(
    grid,
    layers,
    lenslets,
    pupil,
    weights,
    tilt_modes,
    mean_modes,
    command_means,
    directions,
    *,
    dtype=np.float64,
    queries=None,
):
    """Contract phase/command cross moments for the requested lines of sight."""
    if queries is not None:
        return queries.common_cross(directions, mean_modes)
    queries = ScienceCommandQueries(grid, lenslets, pupil, tilt_modes, dtype=dtype)
    cross = np.zeros((len(directions), len(pupil.points), len(weights)))
    for layer, modes in zip(layers, mean_modes, strict=True):
        kernels = queries.slope_kernels(layer, layer.temporal)
        for result, direction in zip(cross, directions, strict=True):
            footprint = layer.height * direction
            queries.add_slope_cross(result, layer, kernels, footprint, weights)
            result += queries.mean_cross(layer, modes, footprint) @ command_means.T
    return cross
