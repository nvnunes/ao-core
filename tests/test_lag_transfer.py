"""Check the fixed-flux cubic transfer and its metric response."""

import numpy as np

from ao_core.models._compact_optics import StaticTransfer
from ao_core.models._lag_transfer import CubicLagTransfer


def test_cubic_lag_transfer_preserves_static_diffraction():
    static = np.zeros((8, 8), complex)
    static[2:7, 2:7] = (1 + 1j) / 2
    static[4, 4] = 1
    native = np.arange(49).reshape(7, 7).astype(complex)
    transfer = CubicLagTransfer(static, np.arange(-3, 4) * 0.6)

    np.testing.assert_allclose(
        transfer.forward(static, native), native, rtol=1e-13, atol=1e-13
    )


def test_cubic_lag_transfer_adjoint_reverses_stationary_factors():
    rng = np.random.default_rng(27)
    static = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    static[0] = 0
    coarse_factor = np.exp(rng.normal(size=(8, 8)) / 4)
    native_factor = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
    kernel = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
    delta = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    transfer = CubicLagTransfer(static, np.arange(-3, 4) * 0.6)

    change = transfer.forward(delta, native_factor, coarse_factor) - transfer.forward(
        np.zeros_like(delta), native_factor, coarse_factor
    )
    response = transfer.adjoint(kernel, native_factor, coarse_factor)

    assert change[3, 3] == 0
    np.testing.assert_allclose(
        (kernel * change).sum().real,
        (response * delta).sum().real,
        rtol=1e-12,
        atol=1e-12,
    )


def test_sampling_lag_transfer_adjoint_on_signed_hermitian_lags():
    rng = np.random.default_rng(71)
    pupil = np.exp(1j * rng.normal(size=(9, 9)) / 4)
    coarse_pupil = pupil[::2, ::2]
    axis = np.arange(len(pupil)) / len(pupil)
    high = axis[:, None] ** 2 + axis**2
    transfer = StaticTransfer(pupil, coarse_pupil, high)
    delta = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    delta = (delta + delta[::-1, ::-1].conj()) / 2
    delta[4, 4] = 0
    kernels = rng.normal(size=(2, 17, 17)) + 1j * rng.normal(size=(2, 17, 17))

    change = transfer.mean(delta) - transfer.mean(np.zeros_like(delta))
    response = transfer.responses(kernels)

    assert change[8, 8] == 0
    np.testing.assert_allclose(
        (kernels * change).sum(axis=(1, 2)).real,
        (response * delta).sum(axis=(1, 2)).real,
        rtol=1e-12,
        atol=1e-12,
    )
