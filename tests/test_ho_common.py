from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from scipy.integrate import quad
from scipy.linalg import cho_factor

from ao_core.models._ho_atmosphere import (
    PhaseGrid,
    chromatic_power,
    refraction_power,
    refractivity,
    science_azimuths,
)
from ao_core.models._ho_covariance import projected_modes
from ao_core.models._ho_optics import (
    InvalidCovarianceOptics,
    OpticalPupil,
    ResidualCovariance,
    coherence_image,
    phase_pairs_need_float64,
    precision_probe_pairs,
)
from ao_core.models._ho_reconstruction import (
    MirrorSurface,
    footprint_shift,
    gaussian_slopes,
    lenslet_gradient,
    solve_commands,
)
from ao_core.models._ho_static import OriginalMapOptics
from ao_core.models.config import SourceConfig


def test_finite_shifted_lenslet_matches_direct_integration():
    frequency = np.array([0.71, -0.47])
    cone = 0.84
    pitch = 0.35
    width = pitch / 2
    height = 1200
    direction = np.array([8e-5, -3e-5])
    offset = np.array([0.12, -0.09])
    shift = footprint_shift(height, cone, direction, offset)

    mean = np.prod(
        [
            quad(
                lambda x, f=f, s=s: np.exp(2j * np.pi * f * (cone * x + s)),
                -width,
                width,
                complex_func=True,
            )[0]
            / pitch
            for f, s in zip(frequency, shift, strict=True)
        ]
    )
    actual = lenslet_gradient(*frequency, pitch, cone) * np.exp(
        2j * np.pi * (frequency @ shift)
    )
    np.testing.assert_allclose(actual, 2j * np.pi * cone * frequency * mean)
    np.testing.assert_array_equal(
        footprint_shift(height, cone, np.zeros(2), np.zeros(2)), [0, 0]
    )
    np.testing.assert_allclose(
        lenslet_gradient(*frequency, pitch, 1.0 - height / np.inf),
        lenslet_gradient(*frequency, pitch, 1.0),
    )


def test_gaussian_actuator_matches_direct_square_cell_integration():
    pitch, width = 0.38, 2.5
    delta = np.array([0.21, -0.33])
    slopes = gaussian_slopes(
        np.zeros((1, 2)), np.zeros((1, 2)), width, pitch, delta[None]
    )
    means = [
        quad(lambda p: np.exp(-(width**2) * p**2), t - pitch / 2, t + pitch / 2)[0]
        / pitch
        for t in delta
    ]
    gradient = [
        quad(
            lambda p: -2 * p * width**2 * np.exp(-(width**2) * p**2),
            t - pitch / 2,
            t + pitch / 2,
        )[0]
        / pitch
        for t in delta
    ]
    np.testing.assert_allclose(
        slopes[:, 0], [gradient[0] * means[1], means[0] * gradient[1]]
    )


def test_one_measurement_regularized_command():
    phase_covariance, regularization, target_cross = 4.0, 0.1, 1.5
    result = solve_commands(
        cho_factor(np.array([[phase_covariance * (1 + regularization)]])),
        np.array([[target_cross]]),
    )
    assert result.item() == pytest.approx(
        target_cross / (phase_covariance * (1 + regularization))
    )


def test_projected_polc_matches_full_sensor_solve():
    slopes = np.array([0.5 + 0.3j, -0.2 + 0.4j, 0.6 + 0.1j])
    mean = np.full((3, 1), 3**-0.5)
    feedback = np.array([[0.2, 0.1, 0.05], [0.05, 0.3, 0.08], [-0.2, 0.15, 0.2]])
    temporal = np.array(0.7 - 0.2j)
    measured = np.eye(3) - mean @ mean.T
    expected = np.linalg.solve(
        np.eye(3) - temporal * mean @ mean.T @ feedback, temporal * measured @ slopes
    )
    update = projected_modes(
        temporal, mean.T @ feedback @ mean, mean.T @ slopes, mean.T @ feedback @ slopes
    )
    np.testing.assert_allclose(temporal * slopes + mean @ update, expected)


@pytest.mark.parametrize("apply_commands", [False, True])
def test_blocked_covariance_matches_direct_residual_phase(apply_commands):
    rng = np.random.default_rng(42)
    pupil = OpticalPupil(np.ones((3, 3)), 3.0, 4.0, 4, "none")
    count = len(pupil.x)
    mode = 2 * np.pi * pupil.points[:, 0] / 4
    phase = np.column_stack((np.cos(mode), np.sin(mode)))
    commands = rng.normal(size=(3, 2)) * int(apply_commands)
    basis = rng.normal(size=(count, 3))
    lags = np.broadcast_to(np.cos(2 * np.pi * np.arange(4)[:, None] / 4), (4, 4))
    stationary = (lags, np.empty((0, count)), np.empty((0, 0)))
    residual = ResidualCovariance(
        pupil, stationary, basis, commands @ commands.T, phase @ commands.T
    )
    direct = phase - basis @ commands
    expected = direct @ direct.T
    np.testing.assert_allclose(residual.diagonal, np.diag(expected))
    for first, block in residual.blocks(rows=3):
        np.testing.assert_allclose(
            block, expected[first : first + len(block), first:], atol=1e-14
        )


@pytest.mark.parametrize("origin,offset", [("fft", 0.0), ("geometric", 0.5)])
@pytest.mark.parametrize("dtype, tolerance", [(np.float64, 1e-13), (np.float32, 2e-7)])
def test_pair_coherence_matches_direct_optical_sum(origin, offset, dtype, tolerance):
    pupil = OpticalPupil(
        np.array([[0.0, 1, 0], [1, 1, 0.6], [0, 1, 0]]), 3.0, 4.0, 4, "pupil_fit"
    )
    power = np.diag([0.2, 0.3, 0.4, 0.3])
    moments = pupil.stationary_moments(power, pupil.fit_modes(4))
    count = len(pupil.points)
    inputs = (
        pupil,
        moments,
        np.zeros((count, 1)),
        np.zeros((1, 1)),
        np.zeros((count, 1)),
    )
    residual = ResidualCovariance(*inputs, dtype=dtype)
    _, covariance = next(ResidualCovariance(*inputs).blocks())
    structure = np.diag(covariance)[:, None] + np.diag(covariance) - 2 * covariance
    coherence = np.exp(-0.5 * structure) * pupil.amplitude[:, None] * pupil.amplitude
    azimuth = np.arange(4) - 2 + offset
    field = np.array(np.meshgrid(azimuth, azimuth)).reshape(2, -1).T
    wave = np.exp(-2j * np.pi * field @ pupil.points.T / 4)
    direct = np.einsum("ij,jk,ik->i", wave, coherence, wave.conj()).real.reshape(4, 4)
    image = coherence_image(pupil, residual, 4, origin)
    assert next(residual.blocks())[1].dtype == dtype
    assert image.dtype == np.float64
    np.testing.assert_allclose(
        image / pupil.full_flux, direct / pupil.full_flux, rtol=0, atol=tolerance
    )
    assert image.sum() == pytest.approx(pupil.full_flux)


def test_float32_pair_integration_preserves_the_diffraction_limit():
    pupil = OpticalPupil(np.full((8, 8), 0.8793), 8.0, 16.0, 16, "none")
    count = len(pupil.points)
    moments = (np.zeros((16, 16)), np.zeros((0, count)), np.zeros((0, 0)))
    basis = np.zeros((count, 1))
    inputs = pupil, moments, basis, np.zeros((1, 1)), basis
    assert not phase_pairs_need_float64(
        pupil, precision_probe_pairs(pupil), *inputs[1:]
    )
    reference = coherence_image(pupil, ResidualCovariance(*inputs), 16, "fft")
    actual = coherence_image(
        pupil, ResidualCovariance(*inputs, dtype=np.float32), 16, "fft"
    )
    np.testing.assert_allclose(actual, reference, atol=1e-11, rtol=0)
    assert actual.sum() == pytest.approx(pupil.full_flux)


@pytest.mark.parametrize(
    "scale,message",
    [(1e-5, "negative difference variance"), (1e-3, "negative PSF intensity")],
)
def test_near_diffraction_probe_flags_covariance_that_requires_float64(scale, message):
    pupil = OpticalPupil(np.ones((8, 8)), 8.0, 16.0, 16, "none")
    count = len(pupil.points)
    source = (np.zeros((16, 16)), np.zeros((0, count)), np.zeros((0, 0)))
    basis = np.exp(-scale * np.sum(pupil.points**2, axis=1))[:, None]
    inputs = pupil, source, basis, np.ones((1, 1)), np.zeros_like(basis)
    assert phase_pairs_need_float64(pupil, precision_probe_pairs(pupil), *inputs[1:])
    image = coherence_image(pupil, ResidualCovariance(*inputs), 16, "fft")
    assert image.sum() == pytest.approx(pupil.full_flux)
    residual = ResidualCovariance(*inputs, dtype=np.float32)
    with pytest.raises(InvalidCovarianceOptics, match=message):
        coherence_image(pupil, residual, 16, "fft")
    invalid = ResidualCovariance(
        pupil, source, basis, -np.ones((1, 1)), np.zeros_like(basis)
    )
    with pytest.raises(InvalidCovarianceOptics, match="negative difference variance"):
        coherence_image(pupil, invalid, 16, "fft")


@pytest.mark.parametrize("image_side", [8, 16])
def test_original_map_near_rank_one_command_covariance_requires_float64(
    image_side,
):
    amplitude = np.ones((8, 8))
    pupil = OpticalPupil(amplitude, 8.0, 16.0, 16, "none")
    optics = OriginalMapOptics(pupil, amplitude, np.zeros_like(amplitude), 1.0, 16)
    count = len(pupil.points)
    source = (np.zeros((16, 16)), np.zeros((0, count)), np.zeros((0, 0)))
    weight = np.linspace(0, 1, count)
    basis = np.column_stack((weight, 1 - weight))
    commands = np.array([1.0, 1.00000006])
    inputs = pupil, source, basis, np.outer(commands, commands), np.zeros_like(basis)
    assert phase_pairs_need_float64(pupil, precision_probe_pairs(pupil), *inputs[1:])
    image = optics.image(ResidualCovariance(*inputs), image_side, "fft")
    centre = image_side // 2
    assert image[centre, centre] == pytest.approx(amplitude.sum() ** 2)
    with pytest.raises(InvalidCovarianceOptics, match="negative difference variance"):
        optics.image(ResidualCovariance(*inputs, dtype=np.float32), image_side, "fft")


def test_chromatic_limits_and_directly_translated_mode():
    grid = PhaseGrid(4.0, 3.0, 3, 0.5, 30.0, 4, np.empty((0, 0)), np.empty((0, 0)), 4)
    layer = SimpleNamespace(weight=1.0, height=4000.0, cone_scale=0.95)
    args = grid, [layer], 8, 0.6
    assert not chromatic_power(*args, 0.0, 0.0, 0.4, 0.5).any()
    relative_difference = -1.4e-6 / 2.8e-4
    args += (-1.4e-6, relative_difference)
    refraction, chromatism = chromatic_power(*args, 0, 0.5)
    assert not refraction.any()
    assert chromatism.sum() > 0
    fx, fy = 0.49, -0.21
    shift = np.array([0.34, 0.18])
    coordinate = np.array([0.23, -0.28])
    phase = lambda xy: np.exp(2j * np.pi * np.array([fx, fy]) @ xy)
    expected = abs(phase(coordinate) - phase(coordinate - shift)) ** 2
    assert refraction_power(fx, fy, shift) == pytest.approx(expected)

    angle = 73 * u.deg
    radius = 30 * u.arcsec
    polar = SourceConfig(zenith=radius.reshape(1), azimuth=angle.reshape(1))
    cartesian = SourceConfig(
        directions=radius * np.array([[np.cos(angle), np.sin(angle)]])
    )
    x, y = cartesian.directions[0].value
    by_azimuth = chromatic_power(*args, 0.4, polar.azimuth.to_value(u.rad)[0])
    by_position = chromatic_power(*args, 0.4, np.arctan2(y, x))
    np.testing.assert_allclose(by_azimuth, by_position)


def test_refractivity_at_mathar_reference_conditions():
    air = {"temperature_k": 290.65, "pressure_pa": 75000, "relative_humidity": 0.10}
    # At the Table I reference wavelength and atmosphere, only c_0 contributes.
    assert refractivity(2.25e-6, **air, ciddor_compressibility=True) == pytest.approx(
        2.00192e-4, rel=1e-14, abs=0
    )
    # Ciddor Eqs. (4), (5), and (12) at 589 nm under the same conditions.
    assert refractivity(589e-9, **air, ciddor_compressibility=True) == pytest.approx(
        2.0328212590050155e-4, rel=1e-13, abs=0
    )


@pytest.mark.parametrize("dtype, tolerance", [(np.float64, 3e-14), (np.float32, 2e-7)])
def test_original_map_matches_nonstationary_pairs_when_the_grid_resolves_it(
    dtype, tolerance
):
    y, x = np.indices((16, 16))
    radius = (x - 7.5) ** 2 + (y - 7.5) ** 2
    amplitude = ((radius < 49) & (radius > 4)).astype(float)
    phase = 0.5 * np.sin(x / 2) + 0.4 * np.cos(y / 3) + 0.2 * (x - 7.5) / 8
    pupil = OpticalPupil(amplitude, 4.0, 16.0, 64, "pupil_fit")
    optics = OriginalMapOptics(pupil, amplitude, phase, 0.25, 64)
    basis = np.column_stack(
        (0.2 * np.sin(pupil.points[:, 0]), 0.15 * np.cos(pupil.points[:, 1]))
    )
    stationary = (np.zeros((64, 64)), np.zeros_like(pupil.ramps), np.zeros((2, 2)))
    residual = ResidualCovariance(
        pupil, stationary, basis, np.eye(2), np.zeros_like(basis), dtype=dtype
    )
    image = optics.image(residual, 64, "fft")
    ramps = np.stack((x - 7.5, y - 7.5), axis=-1) / 8
    valid = amplitude > 0
    tilt = np.linalg.lstsq(ramps[valid], phase[valid], rcond=None)[0]
    phasor = np.exp(1j * (phase[valid] - ramps[valid] @ tilt))
    differences = basis[:, None, :] - basis[None, :, :]
    coherence = np.exp(-0.5 * np.sum(differences**2, axis=-1)) * (
        phasor[:, None] * phasor.conj()
    )
    coordinates = np.arange(64) - 32
    sample = np.array(np.meshgrid(coordinates, coordinates)).reshape(2, -1).T
    wave = np.exp(-2j * np.pi * sample @ pupil.points.T / 16)
    direct = np.einsum(
        "ij,jk,ik->i", wave, coherence, wave.conj(), optimize=True
    ).real.reshape(64, 64)
    np.testing.assert_allclose(
        image / optics.full_flux, direct / optics.full_flux, atol=tolerance
    )


def test_science_azimuths_preserve_polar_direction_at_field_centre():
    sources = SourceConfig(zenith=[0, 1, 0] * u.arcsec, azimuth=[90, 180, 15] * u.deg)
    np.testing.assert_allclose(science_azimuths(sources), np.deg2rad([90, 180, 15]))
    cartesian = SourceConfig(directions=[[-0.0, 0.0], [0.0, -1.0]] * u.arcsec)
    np.testing.assert_allclose(science_azimuths(cartesian), [0, -np.pi / 2])


def test_nominal_dm_calibrates_actuators_on_the_illuminated_pupil():
    y, x = np.indices((32, 32)) - 16
    pupil = ((x * x + y * y) <= 12**2).astype(float)
    cells = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    mirror = MirrorSurface.prepare(cells, 1.0, (25, 0.5, 0.25), 0.125, pupil, None)
    assert len(mirror.centres) < 25**2
    assert np.max(np.linalg.norm(mirror.centres, axis=1)) <= 1.5
    assert np.isfinite(mirror.fit).all()
    assert np.max(np.abs(mirror.fit)) < 100
