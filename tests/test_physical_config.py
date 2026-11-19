"""Check equivalent physical input records from files and quantities."""

from dataclasses import fields

import numpy as np
import pytest
from astropy import units as u

from ao_core.models.config import (
    AOConfiguration,
    AtmosphereConfig,
    DetectorConfig,
    FittingConfig,
    LensletConfig,
    LoopConfig,
    SourceConfig,
    TelescopeConfig,
    load_ao_configuration,
    parse_ini,
    read_ini,
)


def test_file_and_direct_inputs_agree(tmp_path):
    path = tmp_path / "system.ini"
    path.write_text(
        """[telescope]
TelescopeDiameter = 8
ObscurationRatio = 0.15
ZenithAngle = 30
Resolution = 256
PathPupil = pupil.fits

[atmosphere]
Wavelength = 5e-7
r0_Value = 0.18
Seeing = 0.65
L0 = 30
Cn2Weights = [0.4, 0.6]
Cn2Heights = [0, 12500]
WindSpeed = [5, 15]
WindDirection = [0, 90]

[sources_HO]
Zenith = [30, 30]
Azimuth = [0, 90]
Height = 90000
Wavelength = 5.89e-7

[sources_LO]
Zenith = [10, 15]
Azimuth = [180, 270]
Wavelength = [7.1e-7]

[sources_science]
Zenith = 0
Azimuth = 0
Wavelength = [1.65e-6, 2.2e-6]

[DM]
OptimizationZenith = [0, 20]
OptimizationAzimuth = [0, 180]
OptimizationWeight = [2, 1]

[sensor_HO]
NumberLenslets = [20, 16]
SizeLenslets = [0.42, 0.52]
GridOffsets = [[0.25, -0.5], [-0.25, 0.5]]

[sensor_LO]
PixelScale = 250
FieldOfView = 5
SigmaRON = 0.5
ThresholdWCoG = 0.5
NumberPhotons = [80, 100]

[sensor_science]
PixelScale = 5

[RTC]
LoopGain_HO = 0.3
SensorFrameRate_HO = 500
LoopDelaySteps_HO = 1.7
LoopGain_LO = 0.4
SensorFrameRate_LO = 400
LoopDelaySteps_LO = 1
""",
        encoding="utf-8",
    )
    _, parsed = read_ini(path)
    expected = AOConfiguration(
        telescope=TelescopeConfig(
            diameter=800 * u.cm,
            obscuration_ratio=15 * u.percent,
            zenith_angle=np.pi / 6 * u.rad,
            pupil_resolution=256,
        ),
        atmosphere=AtmosphereConfig(
            reference_wavelength=500 * u.nm,
            fried_parameter=18 * u.cm,
            seeing=650 * u.mas,
            outer_scale=30 * u.m,
            layer_weights=[40, 60] * u.percent,
            layer_heights=[0, 12.5] * u.km,
            wind_speeds=[5, 15] * u.m / u.s,
            wind_directions=[0, np.pi / 2] * u.rad,
        ),
        high_order_sources=SourceConfig(
            directions=[[30, 0], [0, 30]] * u.arcsec,
            wavelengths=[589] * u.nm,
            height=90 * u.km,
        ),
        low_order_sources=SourceConfig(
            directions=[[-10, 0], [0, -15]] * u.arcsec, wavelengths=[710] * u.nm
        ),
        science_sources=SourceConfig(
            directions=[[0, 0]] * u.arcsec, wavelengths=[1.65, 2.2] * u.um
        ),
        fitting=FittingConfig(
            directions=[[0, 0], [-20, 0]] * u.arcsec, weights=[2, 1] * u.one
        ),
        high_order_lenslets=LensletConfig(
            counts=(20, 16),
            pitches=[42, 52] * u.cm,
            grid_offsets=[[10.5, -21], [-13, 26]] * u.cm,
        ),
        low_order_detector=DetectorConfig(
            pixel_scale=0.25 * u.arcsec / u.pix,
            field_of_view=5 * u.pix,
            read_noise=0.5 * u.electron,
            threshold=0.5 * u.electron,
            photons_per_frame=[80, 100] * u.photon,
        ),
        science_detector=DetectorConfig(pixel_scale=5 * u.mas / u.pix),
        high_order_loop=LoopConfig(
            gain=0.3 * u.one, frame_rate=0.5 * u.kHz, delay_frames=1.7
        ),
        low_order_loop=LoopConfig(
            gain=0.4 * u.one, frame_rate=400 * u.Hz, delay_frames=1
        ),
    )
    for actual in (load_ao_configuration(path), AOConfiguration.from_ini(parsed)):
        for group in fields(expected):
            left = getattr(actual, group.name)
            right = getattr(expected, group.name)
            assert type(left) is type(right)
            for field in fields(right):
                read = getattr(left, field.name)
                direct = getattr(right, field.name)
                if isinstance(direct, u.Quantity):
                    assert read.unit == direct.unit
                    assert read.dtype == np.float64
                    np.testing.assert_allclose(read.value, direct.value, atol=1e-14)
                else:
                    assert read == direct
        assert actual.high_order_loop.frame_interval == 2 * u.ms
        assert actual.high_order_lenslets.grid_offsets.unit == u.m
        assert actual.atmosphere.reference_wavelength.unit == u.m
        assert actual.low_order_sources.directions.unit == u.arcsec
    assert parsed["telescope"]["PathPupil"] == "pupil.fits"


def test_partial_inputs_and_derived_pitch():
    config = AOConfiguration.from_ini(
        parse_ini(
            "[telescope]\nTelescopeDiameter = 8\n"
            "[sensor_HO]\nNumberLenslets = 20\nGridOffsets = [[0.25, 0.5]]\n"
            "[sources_LO]\nWavelength = 710e-9\n"
            "[atmosphere]\nr0_value = 0.2\nr0_Value = 0.18\n"
            "[hybrid]\nfoo = 'application setting'"
        )
    )
    assert config.high_order_lenslets.pitches[0] == 0.4 * u.m
    np.testing.assert_array_equal(
        config.high_order_lenslets.grid_offsets.to_value(u.m), [[0.1, 0.2]]
    )
    assert config.atmosphere.fried_parameter == 0.2 * u.m
    assert config.atmosphere.reference_wavelength is None
    assert config.low_order_sources.directions is None
    assert config.high_order_sources is None
    assert config.science_detector is None


def test_direct_inputs_own_the_layer_values():
    heights = [0, 1800] * u.m
    atmosphere = AtmosphereConfig(layer_heights=heights)
    heights[1] = 10 * u.m
    assert atmosphere.layer_heights[1] == 1800 * u.m
    with pytest.raises(ValueError, match="read-only"):
        atmosphere.layer_heights[1] = 10 * u.m


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[sources_HO]\nZenith = 30", "requires both Zenith and Azimuth"),
        ("[atmosphere]\nCn2Heights = [0, 500]\nWindSpeed = 4", "matching lengths"),
        ("[RTC]\nLoopDelaySteps_LO = -1.5", "must be finite and nonnegative"),
        ("[sensor_HO]\nGridOffsets = [[0.1, 0.1]]", "requires a shared pitch"),
    ],
)
def test_ini_rejects_inconsistent_physical_fields(text, message):
    with pytest.raises(ValueError, match=message):
        AOConfiguration.from_ini(parse_ini(text))


def test_direct_inputs_reject_incompatible_units_and_shapes():
    with pytest.raises(u.UnitConversionError):
        AtmosphereConfig(reference_wavelength=5 * u.s)
    with pytest.raises(ValueError, match="matching lengths"):
        AtmosphereConfig(layer_heights=[0, 500] * u.m, wind_speeds=[4] * u.m / u.s)
    with pytest.raises(ValueError, match="match the sensors"):
        LensletConfig(counts=(20, 16), grid_offsets=[[0.1, 0.1]] * u.m)


def test_existing_ini_placeholders_and_native_policy_preserve_physical_inputs():
    ini = parse_ini(
        "[atmosphere]\nSeeing = 0.8\nr0_Value = 0\n"
        "[RTC]\nLoopGain_LO = 'optimize'\n"
        "SensorFrameRate_LO = 500\nLoopDelaySteps_LO = 1.5"
    )
    config = AOConfiguration.from_ini(ini)
    assert config.atmosphere.seeing == 0.8 * u.arcsec
    assert config.atmosphere.fried_parameter is None
    assert config.low_order_loop.gain is None
    assert config.low_order_loop.frame_rate == 500 * u.Hz
    assert config.low_order_loop.delay_frames == 1.5
    assert ini["atmosphere"]["r0_Value"] == "0"
    assert ini["RTC"]["LoopGain_LO"] == "'optimize'"
