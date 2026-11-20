"""Prepare the high-order residual and render the science field."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import numpy as np
from astropy import units as u

from ._ho_atmosphere import (
    PhaseGrid,
    PhaseLayer,
    chromatic_power,
    phase_layers,
    refractivity,
    science_azimuths,
)
from ._ho_components import (
    DelayedResidualControl,
    PreparedReconstructor,
    ProjectedPolc,
    SlopePrior,
    common_fit_target,
    default_components,
)
from ._ho_covariance import command_moments, science_command
from ._ho_metal import MpsPupilOptics
from ._ho_metal_queries import MpsScienceCommandQueries
from ._ho_optics import (
    NATIVE_OPTICAL_SIDE,
    InvalidCovarianceOptics,
    OpticalPupil,
    ResidualCovariance,
    coherence_image,
    image_from_signed_lags,
    phase_pairs_need_float64,
    precision_probe_pairs,
)
from ._ho_reconstruction import covariance_kernel
from ._ho_responses import PreparedDm, PreparedSensing, tangent_directions
from ._ho_static import OriginalMapOptics, original_map_side, resampled_field
from ._ho_tail import atmospheric_tail, atmospheric_tail_lags
from ._ho_two_stage import TwoStageCovariance


@dataclass(frozen=True)
class _PreparedCorrection:
    grid: PhaseGrid
    layers: list[PhaseLayer]
    sensing: PreparedSensing
    common_dm: PreparedDm
    common_rows: np.ndarray
    prior: SlopePrior
    common_control: ProjectedPolc
    science_dm: PreparedDm | None
    science_reconstructor: PreparedReconstructor | None
    gain: float


def prepare_sensing(config, components):
    return components.sensing.prepare(
        pupil=config.pupil,
        telescope=config.telescope,
        guides=config.high_order_sources,
        lenslets=config.high_order_lenslets,
    )


def prepare_dm(config, stage, index):
    geometry = config.mirror_geometries[index] if config.mirror_geometries else None
    return stage.dm.prepare(
        config=config.mirrors[index], pupil=config.pupil, geometry=geometry
    )


def correction_components(config, components):
    mirrors = len(config.mirrors)
    if mirrors not in (1, 2):
        raise NotImplementedError(
            "HO turbulence currently requires one common DM and accepts at most one science DM"
        )
    if components is None:
        return default_components(mirrors == 2)
    if (components.science_stage is not None) != (mirrors == 2):
        raise ValueError("The science stage must match the configured science DM")
    return components


def prepare_correction(
    config,
    period,
    wavelength,
    fft_side,
    components=None,
    *,
    dtype=np.float64,
    include_inactive_science=False,
):
    """Bind the components to a common atmospheric and sensing geometry."""
    components = correction_components(config, components)
    common_stage, science_stage = components.common_stage, components.science_stage
    common_interval = (
        None
        if config.high_order_loop is None
        else config.high_order_loop.frame_interval
    )
    if common_interval is None:
        raise ValueError(
            "The common high-order loop requires gain, frame rate and delay"
        )
    residual_loop = config.residual_high_order_loop
    if science_stage is not None and residual_loop is None:
        raise ValueError(
            "A science DM requires a residual high-order loop configuration"
        )
    residual = None
    if residual_loop is not None:
        control = (
            DelayedResidualControl() if science_stage is None else science_stage.control
        )
        residual = control.response(
            loop=residual_loop, common_frame_interval=common_interval
        )
    gain = 0.0 if residual is None else residual.gain
    if gain and science_stage is None:
        raise ValueError("An active residual high-order loop requires a science DM")

    target = common_fit_target(common_stage.reconstructor, config.fitting)
    sensing = prepare_sensing(config, components)

    grid, layers = phase_layers(
        config,
        period,
        config.telescope.diameter.to_value(u.m),
        sensing,
        common_stage.control,
        residual,
        wavelength,
        fft_side,
    )
    prior = SlopePrior(
        grid,
        layers,
        sensing,
        config.inversion_floor,
        config.estimator_regularization,
        dtype,
    )
    common_dm = prepare_dm(config, common_stage, 0)
    common = common_stage.reconstructor.prepare(
        sensing=sensing, dm=common_dm, prior=prior
    )
    common_shape = (len(common_dm.actuator_positions), sensing._lenslets.slope_count)
    common_rows = common._common_matrix(target, common_shape)
    science_dm = science_reconstructor = None
    if science_stage is not None and (gain or include_inactive_science):
        science_dm = prepare_dm(config, science_stage, 1)
        science_reconstructor = science_stage.reconstructor.prepare(
            sensing=sensing, dm=science_dm, prior=prior
        )
    return _PreparedCorrection(
        grid,
        layers,
        sensing,
        common_dm,
        common_rows,
        prior,
        common_stage.control,
        science_dm,
        science_reconstructor,
        gain,
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


@contextmanager
def _field_batches(directions, batch_size, prepare_science):
    """Yield ordered field slices and let the consumer decide when to take the fits."""
    batches = (
        slice(first, first + batch_size)
        for first in range(0, len(directions), batch_size)
    )
    if prepare_science is None:
        yield ((batch, None) for batch in batches)
        return
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ao-science-dm")

    def fit_ahead():
        batch = next(batches)
        pending = executor.submit(prepare_science, directions[batch])
        for next_batch in batches:
            upcoming = executor.submit(prepare_science, directions[next_batch])
            yield batch, pending.result
            batch, pending = next_batch, upcoming
        yield batch, pending.result

    try:
        yield fit_ahead()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def predict_common(
    config, period, wavelength, image_side, phase, *, components, dtype, backend
):
    """Evaluate the corrected residual across the science field."""
    # Keep the pupil quadrature fixed for detector crops of the native image.
    original_map = phase is not None and config.static_opd_sampling == "original_map"
    fft_side = (
        NATIVE_OPTICAL_SIDE if original_map else max(NATIVE_OPTICAL_SIDE, image_side)
    )
    correction = prepare_correction(
        config, period, wavelength, fft_side, components, dtype=dtype
    )
    grid, layers = correction.grid, correction.layers
    lenslets = correction.sensing._lenslets
    weights = correction.common_rows
    temporal_covariance = correction.prior._temporal_covariance
    interaction = correction.common_dm.slope_response(correction.sensing)
    pupil = OpticalPupil(
        config.pupil.amplitude,
        config.pupil.pitch.to_value(u.m) * len(config.pupil.amplitude),
        period,
        fft_side,
        config.tilt_removal,
    )
    basis = correction.common_dm._projected_phase(pupil)
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
    gain = correction.gain
    joint = None
    if gain:
        joint = TwoStageCovariance.prepare(
            grid,
            layers,
            lenslets,
            interaction,
            weights,
            temporal_covariance,
            gain,
            correction.common_control,
        )
        basis = np.column_stack((basis, correction.science_dm._projected_phase(pupil)))
        command_means = joint.common_means
    else:
        covariance, modes, command_means = command_moments(
            grid,
            layers,
            lenslets,
            interaction,
            weights,
            temporal_covariance,
            correction.common_control,
        )
    working_basis = basis.astype(dtype, copy=False)
    probe_pairs = precision_probe_pairs(pupil) if dtype == np.float32 else None
    metal_queries = None
    if backend == "mps":
        metal_queries = MpsScienceCommandQueries(
            grid,
            layers,
            lenslets,
            pupil,
            tilt_modes,
            weights,
            command_means,
            joint,
            gain,
        )

    def phase_command_cross(field_directions, science_weights, precision):
        if gain:
            return joint.field_cross(
                grid,
                layers,
                lenslets,
                pupil,
                tilt_modes,
                weights,
                science_weights,
                field_directions,
                gain,
                dtype=precision,
            )
        query_options = {}
        if metal_queries is not None and precision == np.float32:
            query_options["queries"] = metal_queries
        return science_command(
            grid,
            layers,
            lenslets,
            pupil,
            weights,
            tilt_modes,
            modes,
            command_means,
            field_directions,
            dtype=precision,
            **query_options,
        )

    if optics is None:
        render = partial(
            coherence_image,
            pupil,
            image_side=image_side,
            origin=config.detector_origin,
            field=field,
        )
        render_lags = partial(
            image_from_signed_lags,
            side=pupil.side,
            image_side=image_side,
            origin=config.detector_origin,
        )
    else:
        render = partial(
            optics.image,
            image_side=image_side,
            origin=config.detector_origin,
            high_lags=high_lags,
        )
        render_lags = partial(
            optics.image_from_lags,
            image_side=image_side,
            origin=config.detector_origin,
            high_lags=high_lags,
        )
    metal = None
    if backend == "mps":
        metal = MpsPupilOptics(
            pupil, working_basis, field if optics is None else optics.field
        )
        probe_rows, probe_cross_pairs = np.unique(probe_pairs, return_inverse=True)
        probe_cross_pairs = probe_cross_pairs.reshape(probe_pairs.shape)
        probe_rows = metal.x.new_tensor(probe_rows)

    sky_directions = config.science_sources.directions.to(u.rad)
    azimuths = science_azimuths(config.science_sources)
    directions = tangent_directions(sky_directions.value)
    zenith, refractivity_difference, fractional_difference = chromatic_geometry(
        config, wavelength
    )
    images = np.empty((len(directions), image_side, image_side))
    # Bound the intermediate phase/command arrays for larger fields.
    batch_size = max(1, (256 * 1024**2) // (8 * len(basis) * len(weights)))
    if metal_queries is not None:
        batch_size = min(batch_size, metal_queries.direction_batch_size)
    science_matrices = None
    if gain:
        science_shape = (
            len(correction.science_dm.actuator_positions),
            lenslets.slope_count,
        )
        science_matrices = partial(
            correction.science_reconstructor._directional_matrices,
            shape=science_shape,
            backend=backend,
        )
    fit_ahead = science_matrices if metal is not None else None
    with _field_batches(sky_directions, batch_size, fit_ahead) as batches:
        for batch, take_science_weights in batches:
            batch_directions = directions[batch]
            if take_science_weights is not None:
                streams = metal_queries.two_stage_streams(batch_directions, joint)
                science_weights = take_science_weights()
                cross = metal_queries.contract_two_stage_cross(streams, science_weights)
                del streams
            else:
                science_weights = None
                if gain:
                    science_weights = science_matrices(sky_directions[batch])
                cross = phase_command_cross(batch_directions, science_weights, dtype)
            if gain:
                covariances = [
                    joint.command_covariance(command_weights)
                    for command_weights in science_weights
                ]
            else:
                covariances = [covariance] * len(batch_directions)
            for index, (image, azimuth, phase_command, command_covariance) in enumerate(
                zip(images[batch], azimuths[batch], cross, covariances, strict=True)
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
                use_float64 = False
                if probe_pairs is not None:
                    probe_cross = phase_command
                    probe_options = {}
                    if metal is not None:
                        probe_cross = phase_command[probe_rows].cpu().numpy()
                        probe_options["cross_pairs"] = probe_cross_pairs
                    use_float64 = phase_pairs_need_float64(
                        pupil,
                        probe_pairs,
                        field_source,
                        basis,
                        command_covariance,
                        probe_cross,
                        **probe_options,
                    )
                if not use_float64:
                    try:
                        if metal is None:
                            image[:] = render(
                                ResidualCovariance(
                                    pupil,
                                    field_source,
                                    working_basis,
                                    command_covariance,
                                    phase_command,
                                    dtype=dtype,
                                )
                            )
                        else:
                            image[:] = render_lags(
                                metal.coherence_lags(
                                    field_source, command_covariance, phase_command
                                )
                            )
                    except InvalidCovarianceOptics:
                        if probe_pairs is None:
                            raise
                    else:
                        continue

                # Rebuild the cross moments for this point before rendering in
                # float64. The fitted commands and joint covariance can be reused.
                point = slice(index, index + 1)
                point_weights = (
                    None if science_weights is None else science_weights[point]
                )
                precise_cross = phase_command_cross(
                    batch_directions[point], point_weights, np.float64
                )[0]
                image[:] = render(
                    ResidualCovariance(
                        pupil, field_source, basis, command_covariance, precise_cross
                    )
                )
    return images, pupil.full_flux if optics is None else optics.full_flux
