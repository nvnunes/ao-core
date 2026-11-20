"""Forward input-band diagnostics for validating the two-stage HO residual."""

from dataclasses import replace

import numpy as np
from astropy import units as u
from scipy import fft

from ._ho_atmosphere import chromatic_power, science_azimuths
from ._ho_common import chromatic_geometry, prepare_correction
from ._ho_optics import NATIVE_OPTICAL_SIDE, OpticalPupil
from ._ho_responses import tangent_directions
from ._ho_tail import atmospheric_tail
from ._ho_two_stage import TwoStageCovariance


def input_band_budget(config):
    """Return common and joint dynamic terms in nm² with fixed mirror estimators.

    Split the forward atmosphere at the common DM cutoff. The low-band
    residual is spatio-temporal error. The high-band phase and its correction
    give the fitting and aliasing terms and their signed covariance.
    """
    wavelength = config.science_sources.wavelengths[0].to_value(u.m)
    angular_step = (config.science_detector.pixel_scale * u.pix).to_value(u.rad)
    period = wavelength / angular_step
    side = NATIVE_OPTICAL_SIDE
    correction = prepare_correction(
        config, period, wavelength, side, include_inactive_science=True
    )
    grid, layers = correction.grid, correction.layers
    lenslets = correction.sensing._lenslets
    common_rows = correction.common_rows
    temporal = correction.prior._temporal_covariance
    gain = correction.gain
    interaction = correction.common_dm.slope_response(correction.sensing)
    amplitude = config.pupil.amplitude
    pupil = OpticalPupil(
        amplitude,
        config.pupil.pitch.to_value(u.m) * len(amplitude),
        period,
        side,
        config.tilt_removal,
    )
    common_basis = correction.common_dm._projected_phase(pupil)
    joint_basis = np.column_stack(
        (common_basis, correction.science_dm._projected_phase(pupil))
    )
    weights = pupil.amplitude**2 / pupil.pupil_power
    weight_grid = np.zeros((side, side))
    weight_grid[pupil.x_index, pupil.y_index] = weights
    weight_pairs = fft.ifft2(abs(fft.fft2(weight_grid)) ** 2).real

    def trace(left, right):
        """Piston-removed intensity-weighted trace of left @ right.T."""
        return np.einsum("i,ij,ij->", weights, left, right) - (weights @ left) @ (
            weights @ right
        )

    def phase_variance(moments):
        lags, phase_tilt, tilt = moments
        ramps = pupil.ramps.T
        return (
            lags[0, 0]
            - np.sum(lags * weight_pairs)
            - 2 * trace(ramps, phase_tilt.T)
            + trace(ramps @ tilt, ramps)
        )

    cutoff = 0.5 / config.mirrors[0].pitch.to_value(u.m)
    inside = np.hypot(grid.fx, grid.fy) < cutoff
    low_layers = [replace(layer, power=layer.power * inside) for layer in layers]
    tilt = pupil.fit_modes(len(grid.fx))
    source = pupil.stationary_moments(sum(layer.power for layer in layers), tilt)
    total_phase = phase_variance(source) + phase_variance(atmospheric_tail(grid, pupil))
    low_phase = phase_variance(
        pupil.stationary_moments(sum(layer.power for layer in low_layers), tilt)
    )
    zenith, difference, fraction = chromatic_geometry(config, wavelength)
    sky_directions = config.science_sources.directions.to(u.rad)
    directions = tangent_directions(sky_directions.value)
    field_rows = correction.science_reconstructor._directional_matrices(
        sky_directions,
        (len(correction.science_dm.actuator_positions), lenslets.slope_count),
        backend="cpu",
    )

    def stage_moments(forward_layers, temporal_covariance):
        joint = TwoStageCovariance.prepare(
            grid,
            forward_layers,
            lenslets,
            interaction,
            common_rows,
            temporal_covariance,
            gain,
            correction.common_control,
        )
        result = []
        # Evaluate one field at a time to bound the phase/command work arrays.
        for direction, rows in zip(directions, field_rows, strict=True):
            cross = joint.field_cross(
                grid,
                forward_layers,
                lenslets,
                pupil,
                tilt,
                common_rows,
                [rows],
                direction[None],
                gain,
            )[0]
            cov = joint.command_covariance(rows)
            result.append(
                (
                    (
                        trace(common_basis @ joint.common_covariance, common_basis),
                        trace(cross[:, : len(common_rows)], common_basis),
                    ),
                    (trace(joint_basis @ cov, joint_basis), trace(cross, joint_basis)),
                )
            )
        return result

    full = stage_moments(layers, temporal)
    low = stage_moments(low_layers, None)
    auxiliary_modes = pupil.fit_modes(side)
    scale = (wavelength * 1e9 / (2 * np.pi)) ** 2
    records = []
    for azimuth, full_field, low_field in zip(
        science_azimuths(config.science_sources), full, low, strict=True
    ):
        refraction, chromatism = [
            phase_variance(pupil.stationary_moments(power, auxiliary_modes))
            for power in chromatic_power(
                grid,
                layers,
                side,
                cutoff,
                difference,
                fraction,
                zenith,
                azimuth,
            )
        ]
        record = {}
        for name, (total_command, total_cross), (low_command, low_cross) in zip(
            ("common", "two_stage"), full_field, low_field, strict=True
        ):
            fitting = total_phase - low_phase
            spatio_temporal = low_phase + low_command - 2 * low_cross
            aliasing = total_command - low_command
            fitting_aliasing_covariance = -2 * (total_cross - low_cross)
            dynamic = (
                total_phase + total_command - 2 * total_cross + refraction + chromatism
            )
            record[name] = {
                "fitting": fitting * scale,
                "spatio_temporal": spatio_temporal * scale,
                "aliasing": aliasing * scale,
                "fitting_aliasing_covariance": fitting_aliasing_covariance * scale,
                "differential_refraction": refraction * scale,
                "chromatism": chromatism * scale,
                "dynamic": dynamic * scale,
            }
        records.append(record)
    return records
