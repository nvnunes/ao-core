"""Input contracts for reusable optical, detector and covariance calculations."""

import json

import numpy as np
import pytest
from astropy import units as u

from ao_core.models.config import AtmosphereConfig
from ao_core.models.control import FixedGainIntegrator
from ao_core.models.optics import (
    SuppliedImageModalResponse,
    covariance_matched_displacement_otfs,
)
from ao_core.models.pupil import (
    LocalCubicPupilResponse,
    pupil_tilt_weights,
    sampled_pupil_fourier,
)
from ao_core.models.reconstruction import factor_modal_covariance
from ao_core.models.residual import wind_aligned_tilt_covariance
from ao_core.models.sensing import sample_centroid_errors


def test_pupil_response_supports_scalar_queries_and_guards_nyquist_stencil():
    weights = np.array([[[1.0, 0.0], [0.0, 0.0]]])
    local = LocalCubicPupilResponse(weights, 1.0, fft_size=16)
    fx, fy = 0.25, 0.125
    expected = [np.exp(-1j * np.pi * (fx + fy))]
    np.testing.assert_allclose(local(fx, fy), expected, rtol=1e-12)
    np.testing.assert_allclose(
        sampled_pupil_fourier(weights, 1.0, fx, fy, fft_size=16),
        expected,
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="stencil"):
        local(0.48, 0.0)
    with pytest.raises(ValueError, match="real"):
        sampled_pupil_fourier(weights * np.nan, 1.0, fx, fy, fft_size=16)


@pytest.mark.parametrize("side", [4, 7])
def test_angular_tilt_fit_and_fourier_response_share_the_native_pitch(side):
    pitch = 0.024 * u.m
    wavelength = 500 * u.nm
    fft_size = 32
    axis = (np.arange(side) - (side - 1) / 2) * pitch.value
    x, y = np.meshgrid(axis, axis)
    amplitude = np.triu(np.ones((side, side)))
    amplitude[:, -1] *= 0.6
    weights = pupil_tilt_weights(
        amplitude, sample_pitch=pitch, reference_wavelength=wavelength
    ).to_value(u.mas / u.rad)
    gradient = np.array([3.0, -2.0])
    phase = 1.23 + gradient[0] * x + gradient[1] * y
    radians_per_mas = u.mas.to(u.rad)
    expected = gradient * (wavelength.to_value(u.m) / (2 * np.pi * radians_per_mas))
    np.testing.assert_allclose(
        np.einsum("ayx,yx->a", weights, phase), expected, rtol=1e-12
    )
    np.testing.assert_allclose(weights.sum(axis=(1, 2)), 0, atol=1e-12)

    response = LocalCubicPupilResponse(weights, pitch_m=pitch.value, fft_size=fft_size)
    span_response = LocalCubicPupilResponse(
        weights, (side - 1) * pitch.value, fft_size=fft_size
    )
    fx, fy = np.array([2.0, -1.0]) / (fft_size * pitch.value)
    direct = np.einsum("ayx,yx->a", weights, np.exp(2j * np.pi * (fx * x + fy * y)))
    np.testing.assert_allclose(response(fx, fy), direct, rtol=1e-12)
    np.testing.assert_array_equal(response(fx, fy), span_response(fx, fy))
    with pytest.raises(
        ValueError, match="either the sample-centre span or the sample pitch"
    ):
        LocalCubicPupilResponse(
            weights, (side - 1) * pitch.value, pitch_m=pitch.value, fft_size=fft_size
        )


def test_wind_aligned_covariance_rejects_unstable_strength_and_bad_channel_shape():
    atmosphere = AtmosphereConfig(
        fried_parameter=0.2 * u.m,
        outer_scale=25 * u.m,
        layer_heights=[100.0] * u.m,
        layer_weights=[1.0] * u.one,
        wind_speeds=[10.0] * u.m / u.s,
        wind_directions=[0.0] * u.rad,
    )
    controller = FixedGainIntegrator(gain=0.3, delay=2, frame_rate=500 * u.Hz)
    args = (atmosphere, [[0.0, 0.0]] * u.arcsec)

    def response(fx, fy):
        return np.ones((2, 1))

    with pytest.raises(ValueError, match="Unstable"):
        wind_aligned_tilt_covariance(*args, response, controller, [0.0])
    with pytest.raises(ValueError, match="two finite tilt amplitudes"):
        wind_aligned_tilt_covariance(*args, response, controller, [1.0])


def test_modal_eigenvalue_floor_requires_a_positive_scale():
    with pytest.raises(ValueError, match="eigenvalue floor"):
        factor_modal_covariance(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="eigenvalue floor"):
        factor_modal_covariance(np.eye(2), relative_floor=0)
    with pytest.raises(ValueError, match="symmetric"):
        factor_modal_covariance(1e-12 * np.array([[2.0, 1.9], [-1.9, 2.0]]))
    inverse, factor = factor_modal_covariance(np.diag([1.0, -1e-14]))
    np.testing.assert_allclose(inverse, np.diag([1.0, 1e8]), rtol=1e-12)
    np.testing.assert_allclose(factor @ factor.T, inverse, rtol=1e-12)


def test_supplied_optics_rejects_broadcast_kernels_and_nonfinite_images():
    optical = SuppliedImageModalResponse(
        np.ones((2, 2)),
        np.zeros((2, 2, 1)),
        fft_size=4,
        field_crop=4,
        detector_side=2,
        phase_per_unit_rad=1.0,
    )
    image = np.full((4, 4), 1 / 16)
    kernels = optical.kernel_otfs()
    flux, derivative = optical.psf_derivatives(image, kernels)
    np.testing.assert_array_equal(flux, np.full(4, 0.25))
    np.testing.assert_array_equal(derivative, np.zeros((4, 1)))
    with pytest.raises(ValueError, match="one finite optical Fourier kernel"):
        optical.psf_derivatives(image, kernels[:, :, :1])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        optical.psf_derivatives(image * np.nan, kernels)


def test_centroid_sampler_rejects_broadcast_means_before_using_rng():
    rng = np.random.default_rng(10)
    before = json.dumps(rng.bit_generator.state)
    with pytest.raises(ValueError, match="matching state and pixel axes"):
        sample_centroid_errors(
            rng, [[1.0]], [1.0], [[1.0, 2.0]], np.eye(2), 0.5, 0.5, 1
        )
    assert json.dumps(rng.bit_generator.state) == before


def test_displacement_otfs_reject_material_asymmetry():
    samples = np.random.default_rng(1).normal(size=(20, 2))
    covariance = [[1.0, 0.2], [0.2000001, 1.0]]
    with pytest.raises(ValueError, match="covariance"):
        covariance_matched_displacement_otfs(samples, covariance, 1.0, 8)


def test_displacement_otfs_promote_float32_inputs():
    samples = np.random.default_rng(1).normal(size=(20, 2)).astype(np.float32)
    covariance = np.array([[1.0, 0.2], [0.2, 2.0]], dtype=np.float32)
    otf, halves = covariance_matched_displacement_otfs(samples, covariance, 1.0, 8)
    for actual, expected in zip(
        (otf, halves),
        covariance_matched_displacement_otfs(
            samples.astype(float), covariance.astype(float), 1.0, 8
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
