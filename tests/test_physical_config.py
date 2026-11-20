"""Check equivalent physical input records from files and quantities."""

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits

from ao_core.models.config import (
    AtmosphereConfig,
    Configuration,
    DeformableMirrorConfig,
    DetectorConfig,
    FittingConfig,
    LensletConfig,
    LoopConfig,
    Pupil,
    SourceConfig,
    TelescopeConfig,
    load_ao_configuration,
    parse_ini,
    read_ini,
)
from ao_core.models.uncertainty import SamplingSettings


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
FieldOfView = 512

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
    amplitude = np.ones((256, 256))
    fits.writeto(tmp_path / "pupil.fits", amplitude)
    ini_path, parsed = read_ini(path)
    expected = Configuration(
        pupil=Pupil(amplitude=amplitude, pitch=(8 / 256) * u.m),
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
            zenith=[30, 30] * u.arcsec,
            azimuth=[0, 90] * u.deg,
            wavelengths=[589] * u.nm,
            height=90 * u.km,
        ),
        low_order_sources=SourceConfig(
            zenith=[10, 15] * u.arcsec,
            azimuth=[180, 270] * u.deg,
            wavelengths=[710] * u.nm,
        ),
        science_sources=SourceConfig(
            zenith=[0] * u.arcsec,
            azimuth=[0] * u.deg,
            wavelengths=[1.65, 2.2] * u.um,
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
        science_detector=DetectorConfig(
            pixel_scale=5 * u.mas / u.pix, field_of_view=512 * u.pix
        ),
        high_order_loop=LoopConfig(
            gain=0.3 * u.one, frame_rate=0.5 * u.kHz, delay_frames=1.7
        ),
        low_order_loop=LoopConfig(
            gain=0.4 * u.one, frame_rate=400 * u.Hz, delay_frames=1
        ),
    )
    for actual in (
        load_ao_configuration(path),
        Configuration.from_ini(parsed, ini_path=ini_path),
    ):
        for group in fields(expected):
            left = getattr(actual, group.name)
            right = getattr(expected, group.name)
            assert type(left) is type(right)
            if right is None or isinstance(right, (tuple, SamplingSettings)):
                assert left == right
                continue
            for field in fields(right):
                read = getattr(left, field.name)
                direct = getattr(right, field.name)
                if isinstance(direct, u.Quantity):
                    assert read.unit == direct.unit
                    assert read.dtype == np.float64
                    np.testing.assert_allclose(read.value, direct.value, atol=1e-14)
                elif isinstance(direct, np.ndarray):
                    np.testing.assert_array_equal(read, direct)
                else:
                    assert read == direct
        assert actual.high_order_loop.frame_interval == 2 * u.ms
        assert actual.high_order_lenslets.grid_offsets.unit == u.m
        assert actual.atmosphere.reference_wavelength.unit == u.m
        assert actual.low_order_sources.directions.unit == u.arcsec
    assert parsed["telescope"]["PathPupil"] == "pupil.fits"


@dataclass(frozen=True, kw_only=True, eq=False)
class ModelConfig(Configuration):
    regularization: float


@dataclass(frozen=True, kw_only=True, eq=False)
class RunConfig(ModelConfig):
    precision: str


@pytest.mark.parametrize("extended", [False, True])
def test_loader_decodes_shared_inputs_once_and_extends_in_order(
    tmp_path, monkeypatch, extended
):
    path = tmp_path / "system.ini"
    text = (
        "[telescope]\nTelescopeDiameter = 8\n"
        "[sensor_HO]\nReconstructionRegularization = 0.1\n"
        "[hybrid]\nprecision = float64\n"
    )
    path.write_text(text, encoding="utf-8")
    events = []
    seen_parsers = []
    read_text = Path.read_text
    from_ini = Configuration.from_ini

    def track_read(source_path, *args, **kwargs):
        assert source_path == path.resolve()
        events.append("read")
        return read_text(source_path, *args, **kwargs)

    def track_shared_decode(parser, *, ini_path, load_pupil):
        events.append("shared")
        seen_parsers.append(parser)
        return from_ini(parser, ini_path=ini_path, load_pupil=load_pupil)

    def decode_model(source, base):
        events.append("model")
        assert source.path == path.resolve()
        assert source.text == text
        assert source.parser is seen_parsers[0]
        assert base.telescope.diameter == 8 * u.m
        regularization = source.parser.getfloat("sensor_HO", "ReconstructionRegularization")
        return ModelConfig.from_config(base, regularization=regularization)

    def decode_run(source, base):
        model_config = decode_model(source, base)
        events.append("run")
        return RunConfig.from_config(
            model_config, precision=source.parser["hybrid"]["precision"]
        )

    monkeypatch.setattr(Path, "read_text", track_read)
    monkeypatch.setattr(Configuration, "from_ini", staticmethod(track_shared_decode))
    if extended:
        result = load_ao_configuration(path, decode=decode_run)
        assert events == ["read", "shared", "model", "run"]
        assert result.regularization == 0.1
        assert result.precision == "float64"
        updated = result.with_updates(zenith_angle=20 * u.deg)
        assert type(updated) is RunConfig
        assert updated.telescope.zenith_angle == 20 * u.deg
        assert updated.regularization == result.regularization
        assert updated.precision == result.precision
    else:
        result = load_ao_configuration(path)
        assert events == ["read", "shared"]
        assert type(result) is Configuration
    assert result.telescope.diameter == 8 * u.m


def test_loader_rebinds_relative_pupil_before_decoding(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "system.ini"
    text = "[telescope]\nTelescopeDiameter = 8\nResolution = 4\nPathPupil = archived.fits\n"
    path.write_text(text, encoding="utf-8")
    amplitude = np.arange(16, dtype=float).reshape(4, 4)
    fits.writeto(config_dir / "retained.fits", amplitude)
    monkeypatch.chdir(tmp_path)

    def prepare(source):
        source.parser["telescope"]["PathPupil"] = "retained.fits"

    def decode(source, config):
        assert source.text == text
        assert source.parser["telescope"]["PathPupil"] == "retained.fits"
        np.testing.assert_array_equal(config.pupil.amplitude, amplitude)
        return ModelConfig.from_config(config, regularization=0.1)

    with pytest.raises(FileNotFoundError, match=r"telescope\.PathPupil"):
        load_ao_configuration(
            "config/system.ini", decode=lambda *_: pytest.fail("decoded early")
        )
    result = load_ao_configuration(
        "config/system.ini", prepare=prepare, decode=decode
    )
    assert isinstance(result, ModelConfig)
    assert result.pupil.pitch == 2 * u.m
    assert path.read_text(encoding="utf-8") == text


def test_configuration_extension_rejects_field_replacement_and_erasure():
    config = Configuration(telescope=TelescopeConfig(diameter=8 * u.m))
    with pytest.raises(TypeError, match="telescope"):
        ModelConfig.from_config(config, telescope=None, regularization=0.1)
    model_config = ModelConfig.from_config(config, regularization=0.1)
    assert model_config.telescope is config.telescope
    with pytest.raises(TypeError, match="must extend"):
        Configuration.from_config(model_config)


def test_loader_requires_decoder_to_return_a_configuration(tmp_path):
    path = tmp_path / "system.ini"
    path.write_text("[telescope]\nTelescopeDiameter = 8\n", encoding="utf-8")
    with pytest.raises(TypeError, match="decode must return a Configuration"):
        load_ao_configuration(path, decode=lambda source, config: object())


def test_ini_decodes_nominal_mirrors_in_order():
    config = Configuration.from_ini(
        parse_ini(
            "[DM]\nNumberActuators = [32, 17]\nDmPitchs = [0.25, 0.5]\n"
            "DmHeights = [0, 11800]\nInfModel = 'gaussian'\nInfCoupling = [0.25, 0.3]\n"
            "[DM_science]\nNumberActuators = [21]"
        )
    )
    direct = (
        DeformableMirrorConfig(
            actuator_count=32,
            pitch=25 * u.cm,
            conjugate_height=0 * u.km,
            influence_model="gaussian",
            influence_coupling=25 * u.percent,
        ),
        DeformableMirrorConfig(
            actuator_count=17,
            pitch=50 * u.cm,
            conjugate_height=11.8 * u.km,
            influence_model="gaussian",
            influence_coupling=30 * u.percent,
        ),
    )
    for loaded, expected in zip(config.mirrors, direct, strict=True):
        for item in fields(expected):
            actual_value = getattr(loaded, item.name)
            expected_value = getattr(expected, item.name)
            assert actual_value == expected_value
            if isinstance(actual_value, u.Quantity):
                assert actual_value.unit == expected_value.unit

    partial = Configuration.from_ini(parse_ini("[DM]\nDmHeights = 0"))
    assert len(partial.mirrors) == 1
    assert partial.mirrors[0].conjugate_height == 0 * u.m
    assert partial.mirrors[0].actuator_count is None
    assert Configuration.from_ini(
        parse_ini("[DM]\nNumberReconstructedLayers = 7")
    ).mirrors == ()


def test_partial_inputs_and_derived_pitch():
    config = Configuration.from_ini(
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


def test_metadata_load_keeps_polar_coordinates_and_leaves_pupil_to_caller(tmp_path):
    path = tmp_path / "system.ini"
    path.write_text(
        "[telescope]\nTelescopeDiameter = 8\nPathPupil = unavailable.fits\n"
        "[sources_HO]\nZenith = []\nAzimuth = []\n"
        "[sources_LO]\nZenith = [0, 10]\nAzimuth = [-20, 360]\n"
        "[sensor_LO]\nNumberLenslets = [1, 2]\n"
    )
    system_config = load_ao_configuration(path, load_pupil=False)
    assert system_config.pupil is None
    assert system_config.high_order_sources is None
    np.testing.assert_array_equal(system_config.low_order_sources.zenith.value, [0, 10])
    np.testing.assert_array_equal(system_config.low_order_sources.azimuth.value, [-20, 360])
    np.testing.assert_allclose(
        system_config.low_order_sources.directions.value, [[0, 0], [10, 0]], atol=1e-14
    )
    np.testing.assert_array_equal(
        system_config.low_order_detector.lenslet_counts.value, [1, 2]
    )
    with pytest.raises(FileNotFoundError, match="PathPupil"):
        load_ao_configuration(path)


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
        ("[sensor_science]\nFieldOfView = 10.5", "whole number of pixels"),
        ("[DM]\nNumberActuators = 10.5", r"DM\.NumberActuators must be an integer"),
        (
            "[DM]\nNumberActuators = [32, 17]\nDmHeights = [0]",
            r"\[DM\].*must match",
        ),
    ],
)
def test_ini_rejects_inconsistent_physical_fields(text, message):
    with pytest.raises(ValueError, match=message):
        Configuration.from_ini(parse_ini(text))


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
    config = Configuration.from_ini(ini)
    assert config.atmosphere.seeing == 0.8 * u.arcsec
    assert config.atmosphere.fried_parameter is None
    assert config.low_order_loop.gain is None
    assert config.low_order_loop.frame_rate == 500 * u.Hz
    assert config.low_order_loop.delay_frames == 1.5
    assert ini["atmosphere"]["r0_Value"] == "0"
    assert ini["RTC"]["LoopGain_LO"] == "'optimize'"
