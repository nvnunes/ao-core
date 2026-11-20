from types import SimpleNamespace

import numpy as np

from ao_core.models._ho_atmosphere import PhaseLayer
from ao_core.models._ho_common import Lenslets
from ao_core.models._ho_optics import OpticalPupil, ResidualCovariance
from ao_core.models._ho_reconstruction import gaussian_slopes, lenslet_gradient
from ao_core.models._ho_two_stage import TwoStageCovariance, _stream_covariances
from ao_core.models.control import polc_transfer


def test_stream_covariance_retains_small_positive_variance_after_cancellation():
    variance = np.array([[1.00000007]])
    feedback = np.array([[0.9999]])
    zero = np.zeros((1, 1))
    _, science, _ = _stream_covariances(
        (variance,) * 5, (zero,) * 3, zero, (zero, zero), feedback, 1.0
    )
    expected = (1.0 - feedback) ** 2 * variance
    np.testing.assert_allclose(science, expected, rtol=1e-7, atol=0)


def test_two_stage_residual_covariance_matches_direct_phase_subtraction():
    """Solve the full WFS streaming equations on each mode and subtract both DMs."""
    period = 8.0
    frequencies = (np.arange(8) - 4) / period
    fx, fy = np.meshgrid(frequencies, frequencies, indexing="ij")
    grid = SimpleNamespace(
        fx=fx, fy=fy, period=period, spacing=1 / period, kernel_side=64
    )
    indices = np.array([[0, 0], [0, 1], [1, 1]])
    cells = indices - 0.5
    offsets = np.array([[0.0, 0.0], [0.125, -0.25]])
    lenslets = Lenslets(1.0, 2, cells, indices, offsets)
    mirror = SimpleNamespace(centres=np.array([[-0.5, -0.5], [0.5, 0.5]]), width=1.25)

    power = np.exp(-8 * (fx**2 + fy**2))
    power[0] = power[:, 0] = 0
    power[4, 4] = 0
    interval, delay = 0.05, 2
    wind = np.array([2.5, -1.25])
    speed = -(wind[0] * fx + wind[1] * fy)
    primary = polc_transfer(speed, 0.35, interval, 2)
    gain = 0.8
    science_transfer = gain * np.exp(-2j * np.pi * speed * delay * interval)
    gradient = lenslet_gradient(fx, fy, lenslets.pitch, 1.0)
    layer = PhaseLayer(
        0.0, 1.0, 1.0, power, gradient, primary, offsets, wind * delay * interval
    )
    measured = np.concatenate(
        [
            (
                gradient[:, None]
                * np.exp(
                    2j
                    * np.pi
                    * (point[:, 0, None, None] * fx + point[:, 1, None, None] * fy)
                )
            ).reshape(2 * len(cells), -1)
            for point in (cells + offset for offset in offsets)
        ]
    )
    rng = np.random.default_rng(905)
    rows = rng.normal(scale=0.08, size=(2, len(measured)))
    science_command_weights = rng.normal(scale=0.08, size=(3, len(measured)))

    def covariance(left, right):
        return ((left * power.ravel()) @ right.conj().T).real

    unprojected = primary.ravel() * measured
    prepare_inputs = (
        grid,
        [layer],
        lenslets,
        mirror,
        rows,
        covariance(unprojected, unprojected),
        gain,
    )
    result = TwoStageCovariance.prepare(*prepare_inputs)
    feedback = gaussian_slopes(cells, mirror.centres, mirror.width, 1.0, offsets) @ rows
    average = np.kron(np.eye(4), np.full((3, 3), 1 / 3))
    identity = np.eye(len(measured))
    projector = identity - average
    common_slopes = np.column_stack(
        [
            np.linalg.solve(identity - h1 * average @ feedback, h1 * projector @ mode)
            for h1, mode in zip(primary.ravel(), measured.T, strict=True)
        ]
    )
    science_slopes = science_transfer.ravel() * (
        projector @ (measured - feedback @ common_slopes)
    )
    common = rows @ common_slopes
    science_command = science_command_weights @ science_slopes
    command = np.concatenate((common, science_command))
    joint = result.command_covariance(science_command_weights)
    assert joint.dtype == np.float64
    np.testing.assert_allclose(joint, covariance(command, command), atol=2e-13)
    assert np.linalg.norm(joint[:2, 2:]) > 0.1

    pupil = OpticalPupil(np.ones((3, 3)), 3.0, period, 8, "none")
    tilt_modes = pupil.fit_modes(len(fx))
    phase = np.exp(
        2j
        * np.pi
        * (
            pupil.points[:, 0, None] * fx.ravel()
            + pupil.points[:, 1, None] * fy.ravel()
        )
    )
    field_inputs = (
        grid,
        [layer],
        lenslets,
        pupil,
        tilt_modes,
        rows,
        [science_command_weights],
        np.zeros((1, 2)),
        gain,
    )
    phase_command = result.field_cross(*field_inputs)[0]
    np.testing.assert_allclose(
        phase_command[:, :2], covariance(phase, common), atol=2e-13
    )
    np.testing.assert_allclose(
        phase_command[:, 2:], covariance(phase, science_command), atol=2e-13
    )
    single = result.field_cross(*field_inputs, dtype=np.float32)[0]
    assert single.dtype == np.float64
    np.testing.assert_allclose(single, phase_command, rtol=1e-6, atol=2e-7)

    basis = rng.normal(size=(len(pupil.points), len(command)))
    residual = ResidualCovariance(
        pupil, pupil.stationary_moments(power, tilt_modes), basis, joint, phase_command
    )
    direct = phase - basis[:, :2] @ common - basis[:, 2:] @ science_command
    expected = covariance(direct, direct)
    np.testing.assert_allclose(residual.diagonal, np.diag(expected))
    for first, block in residual.blocks(rows=3):
        np.testing.assert_allclose(
            block, expected[first : first + len(block), first:], atol=2e-13
        )
