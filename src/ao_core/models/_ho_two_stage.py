"""Joint command covariance for the common DM and the science DM."""

from dataclasses import dataclass

import numpy as np

from ._ho_covariance import (
    ScienceCommandQueries,
    _slope_response,
    _solve_projected_modes,
    add_slope_mean_cross,
)
from ._ho_reconstruction import fourier_sum, gaussian_slopes, slope_matrix


def _delay_transfer(grid, layer, gain):
    x, y = layer.residual_shift
    return gain * np.exp(2j * np.pi * (grid.fx * x + grid.fy * y))


def _raw_temporal_moments(grid, layers, lenslets, delay_transfers, include_primary):
    """Integrate the raw slope moments required by both corrected streams."""
    channels = 2 * len(lenslets.offsets)
    span = 2 * lenslets.count - 1
    lags = np.zeros((4 + include_primary, channels, channels, span, span))
    for layer, delay in zip(layers, delay_transfers, strict=True):
        primary = layer.temporal
        factors = [
            np.ones_like(primary),
            primary.conj(),
            primary * delay.conj(),
            abs(primary) ** 2 * delay.conj(),
        ]
        if include_primary:
            factors.insert(0, abs(primary) ** 2)
        factors = np.stack(factors)
        pitch = layer.cone_scale * lenslets.pitch
        for guide, footprint in enumerate(layer.footprints):
            for other, other_footprint in enumerate(layer.footprints):
                start = footprint - other_footprint - pitch * (lenslets.count - 1)
                for x, y in ((0, 0), (0, 1), (1, 1)):
                    spectrum = (
                        layer.power * layer.gradient[x] * layer.gradient[y].conj()
                    )
                    values = fourier_sum(
                        factors * spectrum, grid.spacing, start, pitch, span
                    ).real
                    lags[:, 2 * guide + x, 2 * other + y] += values
                    if x != y:
                        lags[:, 2 * guide + y, 2 * other + x] += values
    return tuple(slope_matrix(values, lenslets) for values in lags)


def _stream_covariances(raw, mean_cross, mean_covariance, maps, feedback, gain):
    """Expand the common and science DM moments, including the WFS mean terms."""
    primary, free, free_primary, primary_delay, primary_product = raw
    primary_mean, delayed_mean, product_mean = mean_cross
    common_map, science_mean_map = maps
    # The unprojected delayed residual is h2 * (I - H h1) m.
    # Keep the products and subtraction in float64 to retain small
    # residual moments when the streams are strongly correlated.
    free_feedback = free_primary @ feedback.T
    science_slope_covariance = gain**2 * (
        free + (feedback @ primary) @ feedback.T - free_feedback - free_feedback.T
    )
    common_science_cross = primary_delay - primary_product @ feedback.T
    science_mean_cross = delayed_mean - feedback @ product_mean

    common_update = primary_mean @ common_map.T
    science_update = science_mean_cross @ science_mean_map.T
    return (
        primary
        + common_update
        + common_update.T
        + common_map @ mean_covariance @ common_map.T,
        science_slope_covariance
        + science_update
        + science_update.T
        + science_mean_map @ mean_covariance @ science_mean_map.T,
        common_science_cross
        + primary_mean @ science_mean_map.T
        + common_map @ science_mean_cross.T
        + common_map @ mean_covariance @ science_mean_map.T,
    )


@dataclass(frozen=True)
class TwoStageCovariance:
    """Shared slope moments and field-dependent commands for the two DMs."""

    feedback: np.ndarray
    common_covariance: np.ndarray
    science_slope_covariance: np.ndarray
    common_science_cross: np.ndarray
    common_means: np.ndarray
    science_mean_map: np.ndarray
    mean_modes: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]

    @classmethod
    def prepare(cls, grid, layers, lenslets, mirror, rows, primary, gain):
        count = 2 * len(lenslets.offsets)
        samples = len(lenslets.cells)
        means = np.kron(np.eye(count), np.ones((samples, 1)) / np.sqrt(samples))
        slopes = gaussian_slopes(
            lenslets.cells,
            mirror.centres,
            mirror.width,
            lenslets.pitch,
            lenslets.offsets,
        )
        feedback = slopes @ rows
        mean_feedback = means.T @ feedback
        reduced_feedback = mean_feedback @ means
        common_map = np.concatenate((means, np.zeros((len(means), 2 * count))), axis=1)
        science_mean_map = np.concatenate(
            (np.zeros_like(means), means, means @ reduced_feedback - feedback @ means),
            axis=1,
        )

        transfers = [_delay_transfer(grid, layer, gain) for layer in layers]
        raw = _raw_temporal_moments(grid, layers, lenslets, transfers, primary is None)
        if primary is not None:
            raw = (primary, *raw)
        covariance = np.zeros((3 * count, 3 * count))
        cross = np.zeros((3, len(means), 3 * count))
        layer_modes = []
        sensor_rows = np.concatenate((means.T, mean_feedback))
        for layer, delay in zip(layers, transfers, strict=True):
            primary_transfer = layer.temporal
            measured = _slope_response(sensor_rows, grid, layer, lenslets)
            # The three mean blocks couple the primary and delayed residual streams.
            rhs = -measured[:count] + primary_transfer * measured[count:]
            projected = _solve_projected_modes(primary_transfer, reduced_feedback, rhs)
            delayed_rhs = delay * rhs
            modes = np.concatenate((projected, delayed_rhs, delay * projected))
            # Combining the last block on demand avoids retaining the same modes twice.
            layer_modes.append((projected, delayed_rhs, delay))
            flat = modes.reshape(3 * count, -1)
            covariance += ((flat * layer.power.ravel()) @ flat.conj().T).real
            base = layer.power * modes.conj()
            for result, transfer in zip(
                cross, (primary_transfer, delay, primary_transfer * delay), strict=True
            ):
                add_slope_mean_cross(result, grid, layer, lenslets, base * transfer)

        common, science_slope_covariance, paired = _stream_covariances(
            raw,
            cross,
            covariance,
            (common_map, science_mean_map),
            feedback,
            gain,
        )
        # Keep joint command moments in float64 for the optical retry.
        return cls(
            feedback,
            (rows @ common) @ rows.T,
            science_slope_covariance,
            rows @ paired,
            rows @ common_map,
            science_mean_map,
            tuple(layer_modes),
        )

    def command_covariance(self, rows):
        cross = self.common_science_cross @ rows.T
        return np.block(
            [
                [self.common_covariance, cross],
                [cross.T, (rows @ self.science_slope_covariance) @ rows.T],
            ]
        )

    def field_cross(
        self,
        grid,
        layers,
        lenslets,
        pupil,
        tilt_modes,
        common_rows,
        science_command_weights,
        directions,
        gain,
        *,
        dtype=np.float64,
    ):
        queries = ScienceCommandQueries(grid, lenslets, pupil, tilt_modes, dtype=dtype)
        common = np.zeros((len(directions), len(pupil.points), len(common_rows)))
        science_cross = np.zeros(
            (len(directions), len(pupil.points), len(science_command_weights[0]))
        )
        feedback_rows = [-rows @ self.feedback for rows in science_command_weights]
        mean_rows = [rows @ self.science_mean_map for rows in science_command_weights]
        for layer, (projected, delayed_rhs, delay) in zip(
            layers, self.mean_modes, strict=True
        ):
            primary = queries.slope_kernels(layer, layer.temporal)
            free = queries.slope_kernels(layer, 1)
            modes = np.concatenate((projected, delayed_rhs, delay * projected))
            for index, direction in enumerate(directions):
                footprint = layer.height * direction
                queries.add_slope_cross(
                    common[index], layer, primary, footprint, common_rows
                )
                queries.add_slope_cross(
                    science_cross[index],
                    layer,
                    free,
                    footprint,
                    gain * science_command_weights[index],
                    layer.residual_shift,
                )
                queries.add_slope_cross(
                    science_cross[index],
                    layer,
                    primary,
                    footprint,
                    gain * feedback_rows[index],
                    layer.residual_shift,
                )
                phase_mean = queries.mean_cross(layer, modes, footprint)
                common[index] += phase_mean @ self.common_means.T
                science_cross[index] += phase_mean @ mean_rows[index].T
        return np.concatenate((common, science_cross), axis=-1)
