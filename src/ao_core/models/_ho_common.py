"""Prepare the high-order residual and render the science field."""

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from scipy import ndimage

from ._ho_atmosphere import (
    chromatic_power,
    phase_layers,
    refractivity,
    science_azimuths,
    tangent_directions,
)
from ._ho_covariance import command_moments, science_command
from ._ho_optics import (
    NATIVE_OPTICAL_SIDE,
    OpticalPupil,
    ResidualCovariance,
    coherence_image,
)
from ._ho_reconstruction import (
    MirrorSurface,
    covariance_kernel,
    estimate_commands,
    fit_commands,
)
from ._ho_static import OriginalMapOptics, original_map_side, resampled_field
from ._ho_tail import atmospheric_tail, atmospheric_tail_lags
from ._ho_two_stage import TwoStageCovariance


@dataclass(frozen=True)
class Lenslets:
    pitch: float
    count: int
    cells: np.ndarray
    indices: np.ndarray
    offsets: np.ndarray


def _lenslets(config, diameter):
    lenslets = config.high_order_lenslets
    guides = config.high_order_sources
    sensors = len(guides.directions)
    if lenslets is None or lenslets.pitches is None:
        raise ValueError("HO correction requires lenslet pitches")
    pitches = lenslets.pitches.to_value(u.m)
    pitch = pitches[0]
    if len(pitches) not in (1, sensors) or not np.allclose(pitches, pitch):
        raise ValueError("HO correction requires equal lenslet pitches for all sensors")
    count = round(diameter / pitch)
    if not np.isclose(count * pitch, diameter, rtol=1e-10, atol=0) or (
        lenslets.counts is not None
        and (
            len(lenslets.counts) not in (1, sensors)
            or any(n != count for n in lenslets.counts)
        )
    ):
        raise ValueError("The nominal lenslet grid must span the telescope diameter")
    offsets = lenslets.grid_offsets
    offsets = np.zeros((sensors, 2)) if offsets is None else offsets.to_value(u.m)
    if len(offsets) != sensors:
        raise ValueError("HO lenslet offsets must match the guide stars")
    axis = (np.arange(count) - (count - 1) / 2) * pitch
    x, y = np.meshgrid(axis, axis, indexing="ij")
    pupil = config.pupil.amplitude
    pitch_of_pupil = config.pupil.pitch.to_value(u.m)
    indices_yx = np.array([y, x]) / pitch_of_pupil + len(pupil) / 2
    valid = ndimage.map_coordinates(pupil, indices_yx, order=0) > 0.5
    if not valid.any():
        raise ValueError("The illuminated pupil contains no valid lenslets")
    return Lenslets(
        pitch, count, np.column_stack((x[valid], y[valid])), np.argwhere(valid), offsets
    )


def prepare_common(config, period, wavelength, fft_side):
    """Resolve per-layer spectra, the calibrated common DM and its estimator."""
    if len(config.mirrors) not in (1, 2):
        raise NotImplementedError(
            "HO turbulence currently requires one common DM and accepts at most one science DM"
        )
    guides = config.high_order_sources
    if guides is None or any(
        value is None
        for value in (guides.height, guides.directions, guides.wavelengths)
    ):
        raise ValueError(
            "HO correction requires guide-star height, directions and wavelength"
        )
    if len(guides.wavelengths) != 1:
        raise ValueError("HO correction requires one guide-star wavelength")
    common_loop = config.high_order_loop
    if (
        common_loop is None
        or common_loop.gain is None
        or common_loop.frame_rate is None
        or common_loop.delay_frames is None
    ):
        raise ValueError(
            "The common high-order loop requires gain, frame rate and delay"
        )
    if not 0 <= common_loop.gain.value <= 1 or common_loop.delay_frames % 1:
        raise ValueError("Projected POLC requires gain in [0, 1] and whole-frame delay")
    fitting = config.fitting
    if fitting is None or fitting.directions is None or fitting.weights is None:
        raise ValueError("HO correction requires mirror fitting directions and weights")
    weights = fitting.weights.value
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Mirror fitting weights must be nonnegative with positive sum")
    residual_loop = config.residual_high_order_loop
    if len(config.mirrors) == 2 and residual_loop is None:
        raise ValueError(
            "A science DM requires a residual high-order loop configuration"
        )
    if residual_loop is not None:
        if residual_loop.gain is None or residual_loop.gain.value < 0:
            raise ValueError("The residual high-order loop requires a nonnegative gain")
        if residual_loop.gain.value > 0:
            if len(config.mirrors) != 2:
                raise ValueError(
                    "An active residual high-order loop requires a science DM"
                )
            if (
                residual_loop.delay_frames is None
                or residual_loop.delay_frames % 1
                or (
                    residual_loop.frame_rate is not None
                    and residual_loop.frame_rate != common_loop.frame_rate
                )
            ):
                raise ValueError(
                    "The residual high-order loop requires a whole-frame delay at the common high-order loop frame rate"
                )
    diameter = config.telescope.diameter.to_value(u.m)
    lenslets = _lenslets(config, diameter)
    grid, layers = phase_layers(
        config,
        period,
        diameter,
        lenslets.pitch,
        lenslets.count,
        lenslets.offsets,
        wavelength,
        fft_side,
    )
    mirror = _mirror_surface(config, lenslets, 0)
    commands, measurement_covariance, factor = estimate_commands(
        grid,
        layers,
        lenslets,
        mirror,
        tangent_directions(fitting.directions.to_value(u.rad)),
        weights / weights.sum(),
        config.inversion_floor,
        config.estimator_regularization,
    )
    return grid, layers, lenslets, mirror, commands, measurement_covariance, factor


def _mirror_surface(config, lenslets, index):
    nominal = config.mirrors[index]
    if nominal.influence_model != "gaussian" or (
        nominal.conjugate_height is not None and nominal.conjugate_height.value != 0
    ):
        raise NotImplementedError(
            "HO correction currently requires a ground-conjugate Gaussian mirror"
        )
    if (
        nominal.pitch is None
        or nominal.actuator_count is None
        or nominal.influence_coupling is None
    ):
        raise ValueError(
            "HO correction requires nominal mirror pitch, count and influence coupling"
        )
    if nominal.actuator_count < 2 or not 0 < nominal.influence_coupling.value < 1:
        raise ValueError(
            "The Gaussian mirror requires at least two actuators per axis and coupling in (0, 1)"
        )
    mirror_pitch = nominal.pitch.to_value(u.m)
    geometry = config.mirror_geometries[index] if config.mirror_geometries else None
    return MirrorSurface.prepare(
        lenslets.cells,
        lenslets.pitch,
        (nominal.actuator_count, mirror_pitch, nominal.influence_coupling.value),
        config.pupil.pitch.to_value(u.m),
        config.pupil.amplitude,
        geometry,
    )


def chromatic_geometry(config, wavelength):
    """Resolve physical inputs common to the chromatic terms at all field points."""
    zenith = config.telescope.zenith_angle
    zenith = 0.0 if zenith is None else zenith.to_value(u.rad)
    guide = config.high_order_sources.wavelengths[0].to_value(u.m)
    refractivity_difference = fractional_difference = 0.0
    if guide != wavelength:
        air_conditions = {
            "temperature_k": config.air_temperature.to_value(u.K),
            "pressure_pa": config.air_pressure.to_value(u.Pa),
            "relative_humidity": config.relative_humidity.to_value(u.one),
            "ciddor_compressibility": config.ciddor_compressibility,
        }
        guide_index = refractivity(guide, **air_conditions)
        refractivity_difference = (
            refractivity(wavelength, **air_conditions) - guide_index
        )
        fractional_difference = refractivity_difference / guide_index
    return zenith, refractivity_difference, fractional_difference


def predict_common(config, period, wavelength, image_side, phase=None):
    """Evaluate the corrected residual across the science field."""
    # Keep the pupil quadrature fixed for detector crops of the native image.
    original_map = phase is not None and config.static_opd_sampling == "original_map"
    fft_side = (
        NATIVE_OPTICAL_SIDE if original_map else max(NATIVE_OPTICAL_SIDE, image_side)
    )
    grid, layers, lenslets, mirror, weights, temporal_covariance, factor = (
        prepare_common(
            config,
            period,
            wavelength,
            fft_side,
        )
    )
    pupil = OpticalPupil(
        config.pupil.amplitude,
        config.pupil.pitch.to_value(u.m) * len(config.pupil.amplitude),
        period,
        fft_side,
        config.tilt_removal,
    )
    basis = pupil.fitted_mirror(mirror)
    tilt_modes = pupil.fit_modes(len(grid.fx))
    auxiliary_modes = pupil.fit_modes(fft_side)
    atmospheric = pupil.stationary_moments(
        sum(layer.power for layer in layers), tilt_modes
    )
    completion = atmospheric_tail(grid, pupil)
    source = tuple(a + b for a, b in zip(atmospheric, completion, strict=True))
    cutoff = 1 / (2 * config.mirrors[0].pitch.to_value(u.m))
    optics = field = high_lags = None
    if original_map:
        pitch = config.pupil.pitch.to_value(u.m)
        fine_side = original_map_side(
            period, pitch, len(config.pupil.amplitude), image_side
        )
        optics = OriginalMapOptics(
            pupil, config.pupil.amplitude, phase, pitch, fine_side
        )
        high = sum(layer.power for layer in layers) * (
            np.hypot(grid.fx, grid.fy) > cutoff
        )
        high_lags = (
            covariance_kernel(high, fft_side) + completion[0],
            covariance_kernel(high, fine_side) + atmospheric_tail_lags(grid, fine_side),
        )
    elif phase is not None:
        field = resampled_field(pupil, config.pupil.amplitude, phase)
    residual_loop = config.residual_high_order_loop
    gain = 0.0 if residual_loop is None else residual_loop.gain.value
    if gain:
        science_dm = _mirror_surface(config, lenslets, 1)
        joint = TwoStageCovariance.prepare(
            grid, layers, lenslets, mirror, weights, temporal_covariance, gain
        )
        basis = np.column_stack((basis, pupil.fitted_mirror(science_dm)))
    else:
        covariance, modes, command_means = command_moments(
            grid, layers, lenslets, mirror, weights, temporal_covariance
        )

    directions = config.science_sources.directions.to_value(u.rad)
    azimuths = science_azimuths(config.science_sources)
    directions = tangent_directions(directions)
    zenith, refractivity_difference, fractional_difference = chromatic_geometry(
        config, wavelength
    )
    images = np.empty((len(directions), image_side, image_side))
    # Bound the intermediate phase/command arrays for larger fields.
    batch_size = max(1, (256 * 1024**2) // (8 * len(basis) * len(weights)))
    for first in range(0, len(directions), batch_size):
        batch = slice(first, first + batch_size)
        if gain:
            science_command_weights = [
                fit_commands(
                    grid,
                    layers,
                    lenslets,
                    science_dm,
                    direction[None],
                    (1.0,),
                    factor,
                )
                for direction in directions[batch]
            ]
            cross = joint.field_cross(
                grid,
                layers,
                lenslets,
                pupil,
                tilt_modes,
                weights,
                science_command_weights,
                directions[batch],
                gain,
            )
            covariances = [
                joint.command_covariance(command_weights)
                for command_weights in science_command_weights
            ]
        else:
            cross = science_command(
                grid,
                layers,
                lenslets,
                pupil,
                weights,
                tilt_modes,
                modes,
                command_means,
                directions[batch],
            )
            covariances = [covariance] * len(cross)
        for image, azimuth, phase_command, command_covariance in zip(
            images[batch], azimuths[batch], cross, covariances, strict=True
        ):
            chromatic = chromatic_power(
                grid,
                layers,
                fft_side,
                cutoff,
                refractivity_difference,
                fractional_difference,
                zenith,
                azimuth,
            )
            extra = pupil.stationary_moments(chromatic.sum(axis=0), auxiliary_modes)
            field_source = tuple(a + b for a, b in zip(source, extra, strict=True))
            residual = ResidualCovariance(
                pupil, field_source, basis, command_covariance, phase_command
            )
            if optics is None:
                image[:] = coherence_image(
                    pupil, residual, image_side, config.detector_origin, field
                )
            else:
                image[:] = optics.image(
                    residual, image_side, config.detector_origin, high_lags
                )
    return images, pupil.full_flux if optics is None else optics.full_flux
