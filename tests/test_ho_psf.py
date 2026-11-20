from dataclasses import replace

import numpy as np
import pytest
from astropy import units as u

from ao_core.models.config import (
    AtmosphereConfig,
    DeformableMirrorConfig,
    DetectorConfig,
    FittingConfig,
    LoopConfig,
    Pupil,
    SourceConfig,
    TelescopeConfig,
    load_ao_configuration,
)
from ao_core.models.ho_psf import (
    HoPsfConfiguration,
    HoPsfModel,
    HoPsfRequest,
    decode_ho_psf,
)


@pytest.fixture
def config():
    amplitude = np.array([[0.0, 0.2, 0.0], [0.9, 1.0, 0.3], [1.0, 0.5, 0.0]])
    return HoPsfConfiguration(
        telescope=TelescopeConfig(diameter=3 * u.m),
        pupil=Pupil(amplitude=amplitude, pitch=1 * u.m),
        science_sources=SourceConfig(
            directions=[[10, -5], [0, 0], [-3, 7]] * u.arcsec,
            wavelengths=[1.6] * u.um,
        ),
        science_detector=DetectorConfig(
            pixel_scale=1e-7 * u.rad / u.pix,
            field_of_view=16 * u.pix,
        ),
        inversion_floor=1e-6,
        estimator_regularization=0,
        tilt_removal="pupil_fit",
    )


def test_refractivity_configuration(config):
    assert config.air_temperature == 290.65 * u.K
    assert config.air_pressure == 75000 * u.Pa
    assert config.relative_humidity == 10 * u.percent
    assert config.ciddor_compressibility is True
    compatible = replace(
        config,
        air_temperature=15 * u.deg_C,
        air_pressure=1013.25 * u.hPa,
        relative_humidity=0.2 * u.one,
        ciddor_compressibility=False,
    )
    assert compatible.air_temperature == 288.15 * u.K
    assert compatible.air_pressure == 101325 * u.Pa
    assert compatible.relative_humidity == 20 * u.percent
    assert compatible.ciddor_compressibility is False
    with pytest.raises(ValueError, match="relative_humidity"):
        replace(config, relative_humidity=101 * u.percent)


@pytest.mark.parametrize("origin, zero", [("fft", 8), ("geometric", 7.5)])
def test_diffraction_matches_direct_fourier_sum(config, origin, zero):
    config = replace(config, detector_origin=origin)
    model = HoPsfModel()
    prediction = model.predict(HoPsfRequest(config=config))
    amplitude = config.pupil.amplitude
    coordinates = np.arange(len(amplitude)) - (len(amplitude) - 1) / 2
    frequencies = (np.arange(16) - zero) / 16
    transform = np.exp(-2j * np.pi * np.outer(frequencies, coordinates))
    expected = np.abs(transform @ amplitude @ transform.T) ** 2

    assert prediction.psfs.shape == (3, 16, 16)
    np.testing.assert_allclose(prediction.psfs, np.repeat(expected[None], 3, axis=0))
    assert prediction.psfs.dtype == np.float64
    assert not prediction.psfs.flags.writeable
    assert not prediction.positions.flags.writeable
    np.testing.assert_array_equal(
        prediction.positions, config.science_sources.directions
    )
    assert prediction.metadata.wavelength.to_value(u.um) == pytest.approx(1.6)
    assert prediction.metadata.pixel_scale.to_value(u.rad) == pytest.approx(1e-7)
    assert prediction.metadata.tel_diameter == 3 * u.m
    np.testing.assert_array_equal(prediction.metadata.tel_pupil.value, amplitude)
    assert prediction.full_flux == pytest.approx(16**2 * np.sum(amplitude**2))
    np.testing.assert_allclose(
        prediction.psfs.sum(axis=(1, 2)) / prediction.full_flux, 1
    )

    crop_config = replace(
        config,
        science_sources=replace(config.science_sources, directions=[[0, 0]] * u.arcsec),
        science_detector=replace(config.science_detector, field_of_view=8 * u.pix),
    )
    crop = model.predict(HoPsfRequest(config=crop_config))
    assert crop.psfs.shape == (1, 8, 8)
    assert crop.full_flux == prediction.full_flux
    np.testing.assert_array_equal(crop.psfs[0], prediction.psfs[1, 4:12, 4:12])
    fraction = crop.psfs.sum() / crop.full_flux
    assert 0 < fraction < 1
    assert fraction == pytest.approx(expected[4:12, 4:12].sum() / expected.sum())


def test_resampled_grid_preserves_full_flux_across_origins(config):
    config = replace(config, pupil=replace(config.pupil, pitch=1.05 * u.m))
    model = HoPsfModel()
    fft = model.predict(HoPsfRequest(config=config))
    geometric = model.predict(
        HoPsfRequest(config=replace(config, detector_origin="geometric"))
    )
    assert fft.full_flux == geometric.full_flux
    for prediction in (fft, geometric):
        assert prediction.psfs[0].sum() == pytest.approx(prediction.full_flux)


@pytest.mark.parametrize("origin", ["fft", "geometric"])
@pytest.mark.parametrize("sampling", ["resampled", "original_map"])
def test_static_opd_matches_direct_complex_pupil_sum(config, origin, sampling):
    y, x = np.indices((16, 16))
    radius = (x - 7.5) ** 2 + (y - 7.5) ** 2
    amplitude = ((radius < 49) & (radius > 4)) * (0.7 + x / 30)
    phase = 0.5 * np.sin(x / 2) + 0.4 * np.cos(y / 3) + 0.2 * (x - 7.5) / 8
    wavelength = config.science_sources.wavelengths[0]
    period = (1024 if sampling == "resampled" else 64) * u.m
    config = replace(
        config,
        pupil=Pupil(
            amplitude=amplitude,
            pitch=1 * u.m,
            static_opd=wavelength * phase / (2 * np.pi),
        ),
        science_detector=replace(
            config.science_detector,
            pixel_scale=(wavelength / period).to_value(u.one) * u.rad / u.pix,
        ),
        detector_origin=origin,
        static_opd_sampling=sampling,
    )
    ramps = np.stack((x - 7.5, y - 7.5), axis=-1) / 8
    piston_and_ramps = np.concatenate((np.ones((*phase.shape, 1)), ramps), axis=-1)
    tilt = np.linalg.lstsq(
        (piston_and_ramps * amplitude[..., None]).reshape(-1, 3),
        (phase * amplitude).ravel(),
        rcond=None,
    )[0][1:]
    fitted = phase - ramps @ tilt
    if sampling == "resampled":
        native = amplitude[::2, ::2]
        coarse_phase = phase[::2, ::2]
        original_variance = np.average(
            (phase - np.average(phase, weights=amplitude**2)) ** 2, weights=amplitude**2
        )
        coarse_variance = np.average(
            (coarse_phase - np.average(coarse_phase, weights=native**2)) ** 2,
            weights=native**2,
        )
        field = native * np.exp(
            1j * fitted[::2, ::2] * np.sqrt(original_variance / coarse_variance)
        )
        side = 512
    else:
        field = amplitude * np.exp(1j * fitted)
        side = 64

    model = HoPsfModel()
    crop = model.predict(HoPsfRequest(config=config))
    offset = 0.5 if origin == "geometric" else 0.0
    kernel = np.exp(
        -2j * np.pi * np.outer(np.arange(16) - 8 + offset, np.arange(len(field))) / side
    )
    expected = abs(kernel @ field @ kernel.T) ** 2
    np.testing.assert_allclose(crop.psfs[0], expected, rtol=2e-12, atol=2e-12)
    assert crop.full_flux == pytest.approx(side**2 * np.sum(abs(field) ** 2))
    shifted = replace(
        config,
        pupil=replace(config.pupil, static_opd=config.pupil.static_opd + 100 * u.nm),
    )
    shifted_prediction = model.predict(HoPsfRequest(config=shifted))
    assert shifted_prediction.full_flux == crop.full_flux
    np.testing.assert_allclose(
        shifted_prediction.psfs, crop.psfs, rtol=2e-12, atol=2e-12
    )
    if sampling == "original_map":
        full_detector = replace(config.science_detector, field_of_view=side * u.pix)
        full = model.predict(
            HoPsfRequest(config=replace(config, science_detector=full_detector))
        )
        window = slice(side // 2 - 8, side // 2 + 8)
        np.testing.assert_allclose(
            crop.psfs[0], full.psfs[0, window, window], rtol=2e-12, atol=2e-12
        )
        assert full.full_flux == crop.full_flux
        assert full.psfs[0].sum() / full.full_flux == pytest.approx(1)


@pytest.mark.parametrize(
    "regularization_key",
    ["ReconstructionRegularization", "ReconstructionNoiseFraction"],
)
def test_decode_ho_psf(tmp_path, regularization_key):
    ini = tmp_path / "system.ini"
    ini.write_text(
        "[telescope]\nTelescopeDiameter = 8.0\n"
        "[sensor_HO]\nTipTiltRemoval = 'pupil_fit'\n"
        "TomographicRegularizationFloor = 1e-6\n"
        f"{regularization_key} = 0.1\n"
    )
    config = load_ao_configuration(ini, decode=decode_ho_psf)
    assert config.telescope.diameter == 8 * u.m
    assert config.tilt_removal == "pupil_fit"
    assert config.inversion_floor == 1e-6
    assert config.estimator_regularization == 0.1
    assert config.detector_origin == "fft"


def test_prediction_reports_missing_or_unsupported_inputs(config):
    model = HoPsfModel()
    with pytest.raises(ValueError, match="detector_origin"):
        replace(config, detector_origin="invalid")
    with pytest.raises(ValueError, match="resolved pupil"):
        model.predict(HoPsfRequest(config=replace(config, pupil=None)))
    odd_detector = replace(config.science_detector, field_of_view=7 * u.pix)
    with pytest.raises(ValueError, match="even pixel count"):
        model.predict(
            HoPsfRequest(config=replace(config, science_detector=odd_detector))
        )
    with pytest.raises(NotImplementedError, match="one common DM"):
        model.predict(
            HoPsfRequest(config=replace(config, atmosphere=AtmosphereConfig()))
        )
    pupil_with_opd = replace(config.pupil, static_opd=np.arange(9).reshape(3, 3) * u.nm)
    with pytest.raises(ValueError, match="requires static_opd_sampling"):
        model.predict(HoPsfRequest(config=replace(config, pupil=pupil_with_opd)))


def test_decode_science_dm_and_residual_loop(tmp_path):
    ini = tmp_path / "system.ini"
    ini.write_text(
        "[DM]\nNumberActuators = [31]\nDmPitchs = [0.25]\n"
        "[DM_science]\nNumberActuators = [17]\nDmPitchs = [0.5]\n"
        "[RTC]\nSensorFrameRate_HO = 500\n"
        "[RTC_science]\nLoopGain_HO = 1.0\nLoopDelaySteps_HO = 1\n"
    )
    shared = load_ao_configuration(ini)
    configured = load_ao_configuration(ini, decode=decode_ho_psf)
    assert len(shared.mirrors) == 1
    assert [mirror.actuator_count for mirror in configured.mirrors] == [31, 17]
    assert configured.high_order_loop.frame_rate == 500 * u.Hz
    assert configured.residual_high_order_loop.frame_rate is None
    assert configured.residual_high_order_loop.gain == 1 * u.one
    assert configured.residual_high_order_loop.delay_frames == 1


@pytest.mark.parametrize("sampling", ["resampled", "original_map"])
def test_uniform_static_opd_leaves_diffraction_unchanged(config, sampling):
    model = HoPsfModel()
    piston = replace(
        config,
        static_opd_sampling=sampling,
        pupil=replace(config.pupil, static_opd=np.full((3, 3), 100) * u.nm),
    )
    reference = model.predict(HoPsfRequest(config=config))
    prediction = model.predict(HoPsfRequest(config=piston))
    assert prediction.full_flux == reference.full_flux
    np.testing.assert_array_equal(prediction.psfs, reference.psfs)


@pytest.mark.parametrize("sampling", ["resampled", "original_map"])
def test_static_opd_ignores_dark_samples_and_illuminated_piston(config, sampling):
    # The resampled path interpolates this 3 × 3 pupil to 96 × 96 samples.
    opd = np.array([[0, 10, 0], [10, 20, -15], [5, 15, 0]]) * u.nm
    config = replace(
        config,
        static_opd_sampling=sampling,
        pupil=replace(config.pupil, static_opd=opd),
    )
    model = HoPsfModel()
    reference = model.predict(HoPsfRequest(config=config))
    lit = config.pupil.amplitude > 0
    for mask, offset in ((lit, 100), (~lit, 100), (~lit, 1e19)):
        pupil = replace(config.pupil, static_opd=opd + (mask * offset) * u.nm)
        changed = model.predict(HoPsfRequest(config=replace(config, pupil=pupil)))
        assert changed.full_flux == reference.full_flux
        np.testing.assert_allclose(changed.psfs, reference.psfs, rtol=2e-12, atol=2e-12)


def test_reject_negative_residual_gain_before_calculating_phase(config):
    config = replace(
        config,
        mirrors=(DeformableMirrorConfig(),),
        atmosphere=AtmosphereConfig(),
        high_order_sources=SourceConfig(
            directions=[[0, 0]] * u.arcsec,
            wavelengths=[589] * u.nm,
            height=0 * u.m,
        ),
        fitting=FittingConfig(directions=[[0, 0]] * u.arcsec, weights=[1] * u.one),
        high_order_loop=LoopConfig(
            gain=0.3 * u.one, frame_rate=500 * u.Hz, delay_frames=2
        ),
        residual_high_order_loop=LoopConfig(gain=-0.3 * u.one),
    )
    with pytest.raises(
        ValueError, match="residual high-order loop requires a nonnegative gain"
    ):
        HoPsfModel().predict(HoPsfRequest(config=config))


@pytest.mark.parametrize(
    "section, key, value",
    [
        ("sensor_HO", "WfsType", "'Pyramid'"),
        ("sensor_HO", "ReconstructionModel", "'standard'"),
        ("sensor_HO", "RemoveMeanSlopes", "False"),
        ("sensor_HO", "NoiseVariance", "[0.1]"),
        ("telescope", "extraErrorNm", "100"),
    ],
)
def test_decode_rejects_explicit_unsupported_physics(tmp_path, section, key, value):
    ini = tmp_path / "system.ini"
    ini.write_text(f"[{section}]\n{key} = {value}\n")
    with pytest.raises(
        NotImplementedError,
        match=key if key != "NoiseVariance" else "injected HO noise",
    ):
        load_ao_configuration(ini, decode=decode_ho_psf)
