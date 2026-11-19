"""Public NGS photometry contracts and a fixed numerical baseline."""

from dataclasses import replace

import numpy as np
import pytest
from astropy import units as u

from ao_core.photometry import (
    WFSPhotometryConfig,
    magnitudes_to_effective_photons_per_second,
    magnitudes_to_photon_flux,
    magnitudes_to_photons_per_frame,
    photon_flux_to_magnitudes,
    photon_flux_to_photons_per_frame,
    photons_per_frame_to_magnitudes,
    photons_per_frame_to_photon_flux,
)

PHOTON_FLUX_UNIT = u.photon / (u.m**2 * u.s)


@pytest.fixture
def calibration() -> WFSPhotometryConfig:
    return WFSPhotometryConfig(
        telescope_diameter=8 * u.m,
        n_channels=16,
        frame_rate=500 * u.Hz,
        zeropoint=3e10 * PHOTON_FLUX_UNIT,
    )


def test_fixed_wfs_photon_baseline(calibration: WFSPhotometryConfig) -> None:
    magnitudes = np.array([0, 14]) * u.mag

    per_frame = magnitudes_to_photons_per_frame(magnitudes, calibration)
    effective_rate = magnitudes_to_effective_photons_per_second(magnitudes, calibration)

    np.testing.assert_array_equal(per_frame.to_value(u.photon), [15000000, 37.67829647264365])
    np.testing.assert_array_equal(
        effective_rate.to_value(u.photon / u.s), [7500000000, 18839.148236321827]
    )
    np.testing.assert_array_equal(
        photons_per_frame_to_magnitudes(np.rint(per_frame.value) * u.photon, calibration).value,
        [0, 13.990769156097178],
    )


def test_photon_flux_and_effective_rate_are_distinct(calibration: WFSPhotometryConfig) -> None:
    flux = magnitudes_to_photon_flux(0 * u.mag, calibration.zeropoint)
    effective = magnitudes_to_effective_photons_per_second(0 * u.mag, calibration)

    assert flux.shape == ()
    assert flux.unit == PHOTON_FLUX_UNIT
    assert flux.to_value(PHOTON_FLUX_UNIT) == 3e10
    assert effective.shape == (1,)
    assert effective.to_value(u.photon / u.s)[0] == 7.5e9
    assert magnitudes_to_effective_photons_per_second(
        0 * u.mag, replace(calibration, frame_rate=1000 * u.Hz)
    ).value[0] == effective.value[0]


def test_forward_inverse_round_trip_and_shapes(calibration: WFSPhotometryConfig) -> None:
    magnitudes = np.array([[8.0, 12.0]]) * u.mag
    flux = magnitudes_to_photon_flux(magnitudes, calibration.zeropoint)
    per_frame = photon_flux_to_photons_per_frame(flux, calibration)

    assert flux.shape == (1, 2)
    assert per_frame.shape == (1, 2)
    np.testing.assert_allclose(
        photon_flux_to_magnitudes(flux, calibration.zeropoint).value,
        magnitudes.value,
    )
    np.testing.assert_allclose(
        photons_per_frame_to_photon_flux(per_frame, calibration).value,
        flux.value,
    )
    assert photons_per_frame_to_photon_flux(per_frame, calibration).unit == PHOTON_FLUX_UNIT
    assert magnitudes_to_photons_per_frame(magnitudes, calibration).shape == (2,)
    assert photons_per_frame_to_magnitudes(per_frame, calibration).shape == (2,)


def test_equivalent_astropy_units(calibration: WFSPhotometryConfig) -> None:
    equivalent = WFSPhotometryConfig(
        telescope_diameter=0.008 * u.km,
        n_channels=16.0,
        frame_rate=0.5 * u.kHz,
        zeropoint=1.8e18 * (u.photon / (u.km**2 * u.min)),
    )
    magnitudes = np.array([9.0, 14.0]) * u.mag

    np.testing.assert_allclose(
        magnitudes_to_photons_per_frame(magnitudes, equivalent).value,
        magnitudes_to_photons_per_frame(magnitudes, calibration).value,
    )


def test_inverse_zero_floor(calibration: WFSPhotometryConfig) -> None:
    expected = -2.5 * np.log10(1e-30 / 3e10)
    assert photon_flux_to_magnitudes(0 * PHOTON_FLUX_UNIT, calibration.zeropoint).value == expected
    assert photons_per_frame_to_magnitudes(0 * u.photon, calibration).value[0] == expected


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"telescope_diameter": np.nan * u.m}, "telescope_diameter must be finite"),
        ({"telescope_diameter": 0 * u.m}, "telescope_diameter must be > 0"),
        ({"frame_rate": np.inf * u.Hz}, "frame_rate must be finite"),
        ({"frame_rate": -1 * u.Hz}, "frame_rate must be > 0"),
        ({"zeropoint": np.nan * PHOTON_FLUX_UNIT}, "zeropoint must be finite"),
        ({"zeropoint": 0 * PHOTON_FLUX_UNIT}, "zeropoint must be > 0"),
        ({"n_channels": np.inf}, "n_channels must be finite"),
        ({"n_channels": 0}, "n_channels must be finite and > 0"),
    ],
)
def test_invalid_calibration(
    calibration: WFSPhotometryConfig, replacement: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(calibration, **replacement)


@pytest.mark.parametrize("magnitude", [np.nan, np.inf, -np.inf])
def test_nonfinite_magnitude_rejected(calibration: WFSPhotometryConfig, magnitude: float) -> None:
    with pytest.raises(ValueError, match="magnitudes must be finite"):
        magnitudes_to_photons_per_frame(magnitude * u.mag, calibration)


@pytest.mark.parametrize("flux", [-1.0, np.nan, np.inf])
def test_invalid_photon_flux_rejected(calibration: WFSPhotometryConfig, flux: float) -> None:
    with pytest.raises(ValueError):
        photon_flux_to_magnitudes(flux * PHOTON_FLUX_UNIT, calibration.zeropoint)


def test_photon_rate_cannot_be_used_as_pre_aperture_flux(calibration: WFSPhotometryConfig) -> None:
    rate = 3e10 * (u.photon / u.s)
    with pytest.raises(ValueError, match="units equivalent"):
        replace(calibration, zeropoint=rate)
    with pytest.raises(ValueError, match="units equivalent"):
        magnitudes_to_photon_flux(0 * u.mag, rate)
    with pytest.raises(ValueError, match="units equivalent"):
        photon_flux_to_magnitudes(rate, calibration.zeropoint)
    with pytest.raises(ValueError, match="units equivalent"):
        photon_flux_to_photons_per_frame(rate, calibration)


def test_quantity_and_unit_contract(calibration: WFSPhotometryConfig) -> None:
    with pytest.raises(TypeError, match="dimensionless real scalar"):
        replace(calibration, n_channels=16 * u.m)
    with pytest.raises(TypeError, match="Astropy Quantity"):
        magnitudes_to_photons_per_frame(10.0, calibration)
    with pytest.raises(ValueError, match="units equivalent"):
        magnitudes_to_photons_per_frame(10 * u.m, calibration)
    with pytest.raises(ValueError, match="units equivalent"):
        photons_per_frame_to_magnitudes(10 * u.s, calibration)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"telescope_diameter": (8 + 1j) * u.m}, "telescope_diameter must be real"),
        ({"frame_rate": (500 + 1j) * u.Hz}, "frame_rate must be real"),
        ({"zeropoint": (3e10 + 1j) * PHOTON_FLUX_UNIT}, "zeropoint must be real"),
    ],
)
def test_complex_calibration_rejected(
    calibration: WFSPhotometryConfig, replacement: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(calibration, **replacement)


def test_complex_photometry_inputs_rejected(calibration: WFSPhotometryConfig) -> None:
    with pytest.raises(ValueError, match="magnitudes must be real"):
        magnitudes_to_photons_per_frame((14 + 1j) * u.mag, calibration)
    with pytest.raises(ValueError, match="photon_flux must be real"):
        photon_flux_to_magnitudes((10 + 1j) * PHOTON_FLUX_UNIT, calibration.zeropoint)
    with pytest.raises(ValueError, match="photons_per_frame must be real"):
        photons_per_frame_to_magnitudes((10 + 1j) * u.photon, calibration)


def test_boolean_quantity_rejected(calibration: WFSPhotometryConfig) -> None:
    with pytest.raises(ValueError, match="magnitudes must be real"):
        magnitudes_to_photons_per_frame(u.Quantity(True, u.mag), calibration)


@pytest.mark.parametrize("n_channels", ["16", True, 16 + 1j, np.array("16")])
def test_non_numeric_channel_count_rejected(
    calibration: WFSPhotometryConfig, n_channels: object
) -> None:
    with pytest.raises(TypeError, match="n_channels must be a real scalar"):
        replace(calibration, n_channels=n_channels)
