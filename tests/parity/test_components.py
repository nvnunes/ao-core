"""Regression checks against the pinned Hybrid and sampling calculations."""

from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from ao_core.models.config import (
    AtmosphereConfig,
    Configuration,
    FittingConfig,
    LoopConfig,
    SourceConfig,
)
from ao_core.models.control import (
    FixedGainIntegrator,
    ScalarFeedback,
    TipTiltControl,
    integrator_noise_lags,
    reconstructed_noise_covariance,
    tip_tilt_feedback,
)
from ao_core.models.pupil import prepare_modal_pupil
from ao_core.models.reconstruction import solve_lo_reconstructor
from ao_core.models.residual import (
    FieldCovariance,
    TiltCommandCovariances,
    assemble_tilt_residual_covariances,
    response_covariance,
    source_minus_correction_covariance,
)
from ao_core.models.sensing import DirectCentroid, PixelSpot, clipped_normal_laplace


@pytest.fixture(scope="module")
def detector_reference():
    # Captured from GIRMOS dc013340 using the A.3 source inventory.
    # Both independent implementations agreed on the common outputs.
    path = Path(__file__).with_name("detector_reference.npz")
    with np.load(path, allow_pickle=False) as data:
        yield data


def assert_numerical_parity(actual, expected, *, scale=None, tau=1e-12):
    """Use the A.3 float64 component budget on one block in matching units."""
    error = np.abs(actual - expected)
    reference_scale = np.max(np.abs(expected))
    scale = reference_scale if scale is None else scale
    np.testing.assert_array_less(
        error, np.nextafter(tau * (scale + np.abs(expected)), np.inf)
    )
    norm = np.linalg.norm(expected.ravel())
    if norm and scale == reference_scale:
        assert np.linalg.norm(error.ravel()) <= tau * norm


def test_clipped_normal_laplace_matches_pinned_sources(detector_reference):
    ref = detector_reference
    result = np.array(clipped_normal_laplace(ref["counts"], ref["times"]))
    assert_numerical_parity(result, ref["laplace"])
    np.testing.assert_array_equal(result[0, :, 0], np.ones(len(ref["counts"])))


@pytest.mark.parametrize("index", range(4))
def test_detector_flux_and_conditional_moments_match_pinned_sources(
    detector_reference, index
):
    ref = detector_reference
    incident = ref["photons"][index] * u.photon
    spot = PixelSpot(
        ref["image"],
        ref["pixel_mas"] * u.mas / u.pix,
        incident,
        ref["pitch_mas"] * u.mas / u.pix,
    )
    flux = spot.flux(ref["offsets"] * u.mas)
    assert_numerical_parity(flux.to_value(u.photon), ref["flux"][index])
    if incident.value == 0:
        np.testing.assert_array_equal(flux.value, np.zeros_like(flux.value))

    detector = DirectCentroid(incident, ref["conversion"] * u.mas / u.pix)
    actual = detector.calculate(flux, jacobian=True)
    for name, unit in (
        ("mean", u.mas),
        ("covariance", u.mas**2),
        ("jacobian", u.mas / u.photon),
    ):
        assert_numerical_parity(actual[name].to_value(unit), ref[name][index])
    # The diagnostic subtracts two probabilities, each at scale one.
    assert_numerical_parity(
        actual["mass_error"].value, ref["mass_error"][index], scale=1
    )


def test_bright_zero_threshold_centroid_converges():
    flux = np.vstack((np.full(25, 10.0), np.linspace(0.0, 9400.0, 25))) * u.photon
    conversion = np.eye(2) * u.mas / u.pix
    detector = DirectCentroid(flux.max(), conversion, threshold=0 * u.electron)
    actual = detector.calculate(flux, jacobian=True)
    refined = DirectCentroid(
        flux.max(), conversion, threshold=0 * u.electron, nodes=256
    ).calculate(flux, jacobian=True)

    np.testing.assert_allclose(actual["mass_error"].value, 0, atol=1e-10)
    np.testing.assert_allclose(actual["mean"].value[0], 0, atol=1e-13)
    assert np.all(np.linalg.eigvalsh(actual["covariance"].value) > 0)
    # The covariance subtracts two raw moments much larger than its bright value.
    for name in ("mean", "covariance", "jacobian"):
        np.testing.assert_allclose(
            actual[name].value, refined[name].value, rtol=1e-8, atol=1e-12
        )


def test_signed_optical_derivative_matches_sampling_route(detector_reference):
    ref = detector_reference
    incident = ref["photons"][-1] * u.photon
    spot = PixelSpot(
        ref["image"],
        ref["pixel_mas"] * u.mas / u.pix,
        incident,
        ref["pitch_mas"] * u.mas / u.pix,
    )
    result = spot.derivative_flux(ref["derivative"])
    assert_numerical_parity(result.to_value(u.photon), ref["signed_flux"])


def test_detector_calibration_owns_values_and_accepts_equivalent_units():
    calibration = np.array([[20.0, 1.0], [2.0, 15.0]]) * u.mas / u.pix
    flux = np.arange(25)[None] * u.photon
    detector = DirectCentroid(300 * u.photon, calibration)
    original = detector.calculate(flux)
    converted = DirectCentroid(300 * u.photon, calibration.to(u.arcsec / u.pix))
    calibration[:] = 0 * u.mas / u.pix
    repeated = detector.calculate(flux)
    alternate_units = converted.calculate(flux)
    for name, expected in original.items():
        np.testing.assert_array_equal(repeated[name], expected)
        assert_numerical_parity(alternate_units[name].value, expected.value)


@pytest.fixture(scope="module")
def control_reference():
    path = Path(__file__).with_name("control_reference.npz")
    with np.load(path, allow_pickle=False) as data:
        yield data


@pytest.mark.parametrize("dtype, tau", [(np.float64, 1e-12), (np.float32, 2e-6)])
def test_shared_controller_matches_pinned_loop_and_impulse(
    control_reference, dtype, tau
):
    ref = control_reference
    gain, delay = float(ref["gain"]), int(ref["delay"])
    controller = FixedGainIntegrator(
        gain=gain, delay=delay, frame_rate=500 * u.Hz, dtype=dtype
    )
    response, noise = ref["response"], ref["noise"]
    transition = controller.transition(response)
    covariance = controller.noise_covariance(response, noise * u.mas**2)
    lags = integrator_noise_lags(
        response, noise, gain, delay, len(ref["lags"]), dtype=dtype
    )
    filtered = controller.filter(ref["samples"] * u.mas, response)
    for actual, expected in (
        (transition, ref["transition"]),
        (covariance.to_value(u.mas**2), ref["covariance"]),
        (lags, ref["lags"]),
        (filtered.to_value(u.mas), ref["filtered"]),
    ):
        assert actual.dtype == dtype
        assert_numerical_parity(actual, expected, tau=tau)
    assert not np.allclose(lags[1], lags[1].T)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tip_tilt_control_uses_actual_feedback_for_noise(dtype):
    integrator = FixedGainIntegrator(
        gain=0.3, delay=2, frame_rate=500 * u.Hz, dtype=dtype
    )
    weights = np.array([[0.7, 0.2, 0.1, 0.0], [-0.2, 0.7, 0.0, 0.1]], dtype=dtype)
    variances = np.array([2.0, 4.0], dtype=dtype)
    control = TipTiltControl(
        reconstructor=weights * u.one,
        sensing_gains=[100, 50] * u.percent,
        controller=integrator,
    )
    expected_feedback = weights[:, :2] + dtype(0.5) * weights[:, 2:]
    innovation = (weights * np.repeat(variances, 2)) @ weights.T
    expected = integrator.noise_covariance(expected_feedback, innovation * u.mas**2)

    assert control.scalar_feedback.symmetric_fallback
    np.testing.assert_array_equal(control.feedback, expected_feedback)
    np.testing.assert_array_equal(
        control.scalar_feedback.effective_feedback,
        (expected_feedback + expected_feedback.T) / 2,
    )
    covariance = control.command_noise_covariance(variances * u.mas**2)
    assert covariance.dtype == dtype
    assert covariance.unit == u.mas**2
    np.testing.assert_allclose(covariance.value, expected.value, rtol=2e-6)
    for values in (
        control.feedback,
        control.reconstructor_blocks,
        control.scalar_feedback.projectors,
    ):
        assert not values.flags.writeable
    with pytest.raises(ValueError, match="one nonnegative value per NGS"):
        control.command_noise_covariance([1, -1] * u.mas**2)
    assert control.feedback is not control.feedback
    assert control.scalar_feedback.projectors is not control.scalar_feedback.projectors


def test_scalar_feedback_requires_two_real_rank_one_responses():
    identity = np.eye(2)
    arguments = {
        "effective_feedback": identity,
        "strengths": [1.0, -5.0],
        "projectors": [identity, np.zeros((2, 2))],
        "symmetric_fallback": False,
        "approximation_relative_error": 0.0,
        "eigenbasis_condition": 1.0,
    }
    with pytest.raises(ValueError, match="rank-one projectors"):
        ScalarFeedback(**arguments).validate(identity)
    with pytest.raises(ValueError, match="strengths must have shape"):
        ScalarFeedback(**(arguments | {"strengths": [1.0]}))
    with pytest.raises(ValueError, match="projectors must be finite real values"):
        projectors = np.asarray(arguments["projectors"], complex)
        ScalarFeedback(**(arguments | {"projectors": projectors}))
    projector = np.array([[1, 16], [16, 0]], dtype=np.int8)
    complementary = np.eye(2, dtype=np.int8) - projector
    with pytest.raises(ValueError, match="idempotent projectors"):
        ScalarFeedback(
            **(
                arguments
                | {"strengths": [1, 1], "projectors": [projector, complementary]}
            )
        ).validate(identity)


def test_scalar_feedback_validates_wider_dtype():
    control = TipTiltControl(
        reconstructor=np.array([[0.7, 0.2], [0.15, 0.9]], dtype=np.float32) * u.one,
        sensing_gains=[1] * u.one,
        controller=FixedGainIntegrator(
            gain=0.3, delay=2, frame_rate=500 * u.Hz, dtype=np.float32
        ),
    )
    control.scalar_feedback.validate(control.feedback.astype(np.float64))


@pytest.mark.parametrize("dtype", [np.dtype("f4"), np.dtype("f8")])
def test_tip_tilt_array_kernels_accept_non_native_byte_order(dtype):
    non_native = dtype.newbyteorder("S")
    weights = np.eye(2, dtype=non_native)
    scalar = np.array([2], dtype=non_native)
    for kernel in (tip_tilt_feedback, reconstructed_noise_covariance):
        actual = kernel(weights, scalar)
        assert actual.dtype == dtype
        np.testing.assert_array_equal(actual, 2 * np.eye(2))


def test_tip_tilt_array_kernels_avoid_integer_overflow_and_unit_loss():
    weights = 12 * np.eye(2, dtype=np.int8)
    gains = np.array([12], dtype=np.int8)
    variances = np.array([1], dtype=np.int8)
    np.testing.assert_array_equal(tip_tilt_feedback(weights, gains), 144 * np.eye(2))
    np.testing.assert_array_equal(
        reconstructed_noise_covariance(weights, variances), 144 * np.eye(2)
    )
    with pytest.raises(TypeError, match="units already converted"):
        tip_tilt_feedback(weights, [50] * u.percent)
    with pytest.raises(TypeError, match="units already converted"):
        reconstructed_noise_covariance(weights * u.percent, variances)
    with pytest.raises(TypeError, match="units already converted"):
        reconstructed_noise_covariance(weights, variances * u.mas**2)


@pytest.mark.parametrize("dtype, tau", [(np.float32, 2e-6), (np.float64, 1e-12)])
def test_tilt_residual_matches_covariance_of_joint_residual(dtype, tau):
    science = np.array(
        [[[3, 1, 0], [0.5, 2, 1]], [[4, -1, 1], [0.75, 3, 2]]], dtype=dtype
    )
    atmosphere = np.array([[1, 0, 0.5], [-0.2, 0.5, 0]], dtype=dtype)
    high_order = np.array([[0.7, 0.5, 0], [0, 0.5, -0.25]], dtype=dtype)
    noise = np.array([[0.5, 0.125], [0.125, 0.25]], dtype=dtype)

    terms = TiltCommandCovariances(
        science_tilt=(science @ science.swapaxes(-1, -2)) * u.mas**2,
        atmospheric_command=(atmosphere @ atmosphere.T) * u.mas**2,
        ho_command=(high_order @ high_order.T) * u.mas**2,
        atmospheric_ho_command=(atmosphere @ high_order.T) * u.mas**2,
        science_atmospheric_command=(science @ atmosphere.T) * u.mas**2,
        science_ho_command=(science @ high_order.T) * u.mas**2,
    )
    residual = assemble_tilt_residual_covariances(
        terms, command_noise=(noise * u.mas**2).to(u.arcsec**2)
    )

    atmospheric_response = science - atmosphere
    joint_response = atmospheric_response - high_order
    atmospheric = atmospheric_response @ atmospheric_response.swapaxes(-1, -2)
    noiseless = joint_response @ joint_response.swapaxes(-1, -2)
    for actual, expected in (
        (residual.atmospheric_residual, atmospheric),
        (residual.ho_contribution, noiseless - atmospheric),
        (residual.command_noise, noise),
        (residual.noiseless, noiseless),
        (residual.total, noiseless + noise),
    ):
        assert actual.dtype == dtype
        assert actual.unit == u.mas**2
        np.testing.assert_allclose(actual.value, expected, rtol=tau, atol=tau)
    assert residual.ho_contribution.value[0, 0, 0] < 0


@pytest.mark.parametrize("dtype, tau", [(np.float64, 1e-12), (np.float32, 2e-6)])
def test_shared_reconstructor_matches_pinned_solve(control_reference, dtype, tau):
    ref = control_reference
    weights = solve_lo_reconstructor(
        ref["css"] * u.mas**2,
        ref["cas"] * u.mas**2,
        ref["prior"] * u.mas**2,
        dtype=dtype,
    )
    assert weights.dtype == dtype
    assert weights.unit == u.one
    assert_numerical_parity(weights.value, ref["weights"], tau=tau)


def test_shared_lo_single_star_and_singular_policies():
    one = np.zeros((2, 2)) * u.mas**2
    weights = solve_lo_reconstructor(one, one, one)
    np.testing.assert_array_equal(weights.value, np.eye(2))
    assert weights.dtype == np.float64
    assert not weights.flags.writeable

    css = np.diag([4.0, 0.0, 2.0, 0.0]) * u.mas**2
    cas = np.arange(8).reshape(2, 4) * u.mas**2
    noise = np.zeros((4, 4)) * u.mas**2
    with pytest.raises(np.linalg.LinAlgError):
        solve_lo_reconstructor(css, cas, noise)
    weights = solve_lo_reconstructor(css, cas, noise, method="pinv")
    np.testing.assert_array_equal(weights.value, cas.value * [0.25, 0, 0.5, 0])


@pytest.fixture(scope="module")
def modal_reference():
    path = Path(__file__).with_name("modal_reference.npz")
    with np.load(path, allow_pickle=False) as data:
        yield data


def _polar(radii, azimuths):
    return radii[:, None] * np.column_stack((np.cos(azimuths), np.sin(azimuths)))


@pytest.fixture(scope="module")
def modal_configuration(modal_reference):
    ref = modal_reference
    return Configuration(
        atmosphere=AtmosphereConfig(
            reference_wavelength=ref["reference_wave"] * u.m,
            fried_parameter=ref["r0"] * u.m,
            outer_scale=ref["outer_scale"] * u.m,
            layer_heights=ref["layer_heights"] * u.m,
            layer_weights=ref["layer_weights"] * u.one,
            wind_speeds=ref["wind"] * u.m / u.s,
            wind_directions=ref["directions"] * u.rad,
        ),
        fitting=FittingConfig(
            directions=_polar(ref["fit_radii"], ref["fit_angles"]) * u.rad,
            weights=ref["fit_weights"] * u.one,
        ),
        high_order_sources=SourceConfig(
            directions=_polar(
                np.full(4, ref["lgs_radius"]),
                float(ref["lgs_angle"]) + 2 * np.pi * np.arange(4) / 4,
            )
            * u.rad,
            height=ref["projected_beacon"] * u.m,
        ),
        low_order_sources=SourceConfig(
            directions=_polar(ref["ngs_radii"], np.deg2rad(ref["ngs_angles"]))
            * u.arcsec
        ),
        high_order_loop=LoopConfig(
            gain=ref["ho_gain"] * u.one, frame_rate=500 * u.Hz, delay_frames=2
        ),
        low_order_loop=LoopConfig(
            gain=ref["lo_gain"] * u.one, frame_rate=500 * u.Hz, delay_frames=2
        ),
    )


@pytest.fixture(scope="module")
def modal_pupil(modal_reference):
    ref = modal_reference
    return prepare_modal_pupil(
        ref["image"],
        ref["opd"] * u.nm,
        8 * u.m,
        7.9 * u.m,
        ref["reference_wave"] * u.m,
        int(ref["side"]),
        int(ref["order"]),
    )


def _modal_field(configuration, pupil, ref, *, derive_bins=False):
    return FieldCovariance(
        configuration,
        pupil,
        ref["wave"] * u.m,
        int(ref["radial"]),
        int(ref["angular"]),
        float(ref["period"]),
        None if derive_bins else float(ref["dt"]),
        tilt_noise_prior=ref["prior"],
        sensing_gains=ref["gains"],
        sensing_ho=ref["sensing"],
    )


def test_interpolated_modal_spectrum_and_ngs_covariance_match_pinned_sampling(
    modal_configuration, modal_pupil, modal_reference
):
    ref = modal_reference
    pupil = modal_pupil
    for name in ("amplitude", "coordinates", "basis", "weights"):
        assert_numerical_parity(getattr(pupil, name), ref[name])
    field = _modal_field(modal_configuration, pupil, ref)
    assert_numerical_parity(field.tilt_reconstructor, ref["reconstructor"])
    assert_numerical_parity(
        field.residual_response(ref["theta"]), ref["residual_response"]
    )
    assert_numerical_parity(field.spectrum(ref["theta"]), ref["spectrum"])
    derived = _modal_field(modal_configuration, pupil, ref, derive_bins=True)
    spectrum = derived.spectrum(ref["theta"])
    assert len(spectrum) < len(ref["spectrum"])
    padded = np.zeros_like(ref["spectrum"])
    padded[:len(spectrum)] = spectrum
    assert_numerical_parity(padded, ref["spectrum"])
    ngs = (np.array([10.0, -7.0]) * u.arcsec).to_value(u.rad)
    assert_numerical_parity(field.ngs_ho_covariance(ngs), ref["guide_covariance"])


def test_modal_covariance_blocks_match_direct_fourier_waves(
    modal_configuration, modal_pupil, modal_reference
):
    pupil = modal_pupil

    class DirectWaves:
        def __getattr__(self, name):
            return getattr(pupil, name)

        def response(self, fx, fy):
            wave = np.exp(2j * np.pi * pupil.coordinates @ np.vstack((fx, fy)))
            return pupil.weights @ wave

    ref = modal_reference
    field = _modal_field(modal_configuration, DirectWaves(), ref)
    _, source, higher_order = field.direction_terms(ref["theta"])
    correction = np.vstack([field.lo, higher_order])
    weights = field.weights
    scale = field.scale**2
    blocks = (
        response_covariance(source, source, weights) * scale,
        response_covariance(correction, correction, weights) * scale,
        response_covariance(source, correction, weights) * scale,
        response_covariance(correction, source, weights) * scale,
    )
    for actual, expected in zip(blocks, ref["blocks"], strict=True):
        assert_numerical_parity(actual, expected)
    residual = source_minus_correction_covariance(*blocks)
    assert_numerical_parity(residual, ref["direct_covariance"])


def test_restored_coherence_and_lags_match_pinned_sampling(
    modal_configuration, modal_pupil, modal_reference
):
    from ao_core.models._restored_pupil import FullScienceMean
    from ao_core.models._temporal_pupil import full_pupil_lags
    from ao_core.models.optics import PupilPairGeometry
    from ao_core.models.pupil import ARCSEC, optical_pupil

    ref = modal_reference
    path = Path(__file__).with_name("restored_reference.npz")
    with np.load(path, allow_pickle=False) as saved:
        wavelength = float(ref["wave"])
        optical = optical_pupil(
            ref["image"], ref["opd"], 8.0, 7.9, wavelength, 10, int(ref["order"])
        )
        assert_numerical_parity(optical.basis, saved["sampled_basis"])
        field = FieldCovariance(
            modal_configuration,
            modal_pupil,
            wavelength * u.m,
            40,
            48,
            float(ref["period"]),
            float(ref["dt"]),
            tilt_noise_prior=ref["prior"],
            sensing_gains=ref["gains"],
            sensing_ho=ref["sensing"],
        )
        geometry = PupilPairGeometry(optical, 7.8)
        atmosphere = modal_configuration.atmosphere
        noise_lags = saved["noise_lags"]
        offsets, indices = saved["offsets"], saved["offset_indices"]
        science = FullScienceMean(
            field,
            geometry,
            atmosphere,
            noise_lags[0],
            radial_order=40,
            angular_order=48,
            pupil_offsets=(offsets, indices),
        )
        terms = field.direction_terms(ref["theta"])
        coherence = science.coherence(ref["theta"] / ARCSEC, terms=terms)
        assert_numerical_parity(coherence, saved["coherence"])
        lags = np.array(
            list(
                full_pupil_lags(
                    science,
                    terms,
                    saved["indices"],
                    offsets,
                    indices,
                    atmosphere=atmosphere,
                    frame_interval_s=float(ref["dt"]),
                    noise_lags=noise_lags,
                )
            )
        )
        assert_numerical_parity(lags, saved["pupil_lags"])
        assert not np.allclose(lags[1], lags[1].T)


@pytest.mark.parametrize("index, photons", [(0, 200.0), (1, 2_000.0)])
def test_sensing_matches_pinned_optics_and_detector(modal_reference, index, photons):
    from ao_core.models.pupil import optical_pupil
    from ao_core.models.sensing import (
        diffraction_calibration,
        optical_mean,
        sensor_response,
    )

    ref = modal_reference
    path = Path(__file__).with_name("sensor_reference.npz")
    with np.load(path, allow_pickle=False) as saved:
        pupil = optical_pupil(
            ref["image"],
            np.zeros_like(ref["opd"]),
            8.0,
            7.9,
            710e-9,
            20,
            int(ref["order"]),
        )
        pixel, pitch = float(saved["pixel"]), float(saved["pitch"])
        calibration, _ = diffraction_calibration(pupil, pixel, pitch)
        assert_numerical_parity(calibration, saved["conversion"])
        image, derivatives = optical_mean(pupil, saved["covariances"][index], (2, 3))
        spot = PixelSpot(
            image,
            pixel * u.mas / u.pix,
            photons * u.photon,
            pitch * u.mas / u.pix,
        )
        signed_flux = [spot.derivative_flux(mode).value for mode in derivatives]
        gain, noise, response, details = sensor_response(
            image,
            derivatives,
            pixel,
            pitch,
            photons,
            calibration,
            2,
        )
        for name, value in (
            ("image", image),
            ("derivatives", derivatives),
            ("signed_flux", signed_flux),
            ("gain", gain),
            ("noise", noise),
            ("response", response),
        ):
            assert_numerical_parity(np.asarray(value), saved[f"{name}_{index}"])
        np.testing.assert_array_equal(
            noise, details["conditional_covariance_mas2"].mean(axis=0)
        )


def test_shared_gaussian_priors_keep_each_pinned_prescription():
    from ao_core.models.sensing import centered_measurement_noise, centroid_noise_prior

    path = Path(__file__).with_name("sensor_reference.npz")
    with np.load(path, allow_pickle=False) as saved:
        prior = centroid_noise_prior(150.0, 0.63, 210.0, 100.0, 5, 0.5, 0.5)
        assert_numerical_parity(prior["covariance_pixel2"], saved["instrument_prior"])
        assert prior["centroid_mass_e"] == pytest.approx(
            float(saved["instrument_prior_mass"]), rel=1e-12
        )
        noise, details = centered_measurement_noise(
            [150.0, 1200.0],
            [0.63, 0.4],
            [210.0, 190.0],
            pixel_mas=100.0,
            angular_scale_mas=111.0,
            read_noise=0.5,
            threshold=0.5,
        )
        assert_numerical_parity(noise, saved["centered_prior"])
        assert_numerical_parity(
            details["spotSecondMomentPixel2"], saved["centered_prior_moments"]
        )


def test_common_ho_only_field_retains_uncorrected_tilt(
    modal_configuration, modal_pupil, modal_reference
):
    ref = modal_reference
    field = _modal_field(modal_configuration, modal_pupil, ref)
    common = FieldCovariance(
        modal_configuration,
        modal_pupil,
        ref["wave"] * u.m,
        int(ref["radial"]),
        int(ref["angular"]),
        float(ref["period"]),
        float(ref["dt"]),
        common_only=True,
    )
    _, source, _ = field.direction_terms(ref["theta"])
    source[2:] -= field.common
    assert_numerical_parity(
        common.residual_response(ref["theta"]), source * field.scale
    )
    assert np.isfinite(common.spectrum(ref["theta"])).all()
    assert_numerical_parity(
        common.ngs_ho_covariance(ref["theta"]), field.ngs_ho_covariance(ref["theta"])
    )

    zero_lag = FieldCovariance(
        modal_configuration,
        modal_pupil,
        ref["wave"] * u.m,
        int(ref["radial"]),
        int(ref["angular"]),
        period=None,
        common_only=True,
    )
    assert_numerical_parity(
        zero_lag.ngs_ho_covariance(ref["theta"]),
        field.ngs_ho_covariance(ref["theta"]),
    )


def test_bright_clipped_detector_uses_exact_moments_and_empty_response_shape():
    from ao_core.models.sensing import sensor_response

    image = np.full((51, 51), 1 / 51**2)
    conversion = np.eye(2) * 50
    spot = PixelSpot(image, 10 * u.mas / u.pix, 7000 * u.photon, 50 * u.mas / u.pix)
    flux = spot.flux(np.zeros(2) * u.mas)
    result = DirectCentroid(
        7000 * u.photon, conversion * u.mas / u.pix, threshold=80 * u.electron
    ).calculate(flux)
    # Paired clipped-normal integration on the half-line for 67.28 photons/pixel.
    np.testing.assert_allclose(result["covariance"].value[0, 0, 0], 2947.28, rtol=3e-6)
    assert np.max(np.abs(result["mass_error"].value)) < 1e-10
    _, noise, response, _ = sensor_response(
        image, (), 10, 50, 7000, conversion, 1, threshold_e=80
    )
    np.testing.assert_allclose(
        noise, result["covariance"].value[0], atol=1e-9, rtol=1e-12
    )
    assert response.shape == (2, 0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"difference_step": 0}, "difference_step"),
        ({"difference_step": np.nan}, "difference_step"),
        ({"offsets_mas": [[0, 0], [10, 0]]}, "span both axes"),
    ],
)
def test_sensor_rejects_unidentifiable_gain_or_phase_response(kwargs, message):
    from ao_core.models.sensing import sensor_response

    image = np.full((51, 51), 1 / 51**2)
    with pytest.raises(ValueError, match=message):
        sensor_response(image, (), 10, 50, 200, np.eye(2) * 50, 1, **kwargs)
