from dataclasses import replace
from weakref import ref

import numpy as np
import pytest
from astropy import units as u

from ao_core.models import (
    _ho_atmosphere,
    _ho_common,
    _ho_metal,
    _ho_metal_queries,
    _ho_optics,
    ho_psf,
)
from ao_core.models.config import (
    AtmosphereConfig,
    DeformableMirrorConfig,
    DetectorConfig,
    FittingConfig,
    LensletConfig,
    LoopConfig,
    Pupil,
    SourceConfig,
    TelescopeConfig,
    load_ao_configuration,
)
from ao_core.models.ho_psf import (
    DelayedResidualControl,
    GaussianDm,
    HoPsfComponents,
    HoPsfConfiguration,
    HoPsfModel,
    HoPsfRequest,
    HoPsfStage,
    MatrixReconstructor,
    PhaseFitReconstructor,
    PreparedReconstructor,
    ProjectedPolc,
    ShackHartmannSensing,
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


def test_model_precision_selection(monkeypatch):
    default = HoPsfModel()
    assert default.backend == "cpu"
    assert default.precision == "float64"
    assert HoPsfModel(backend="cpu", precision="float32").precision == "float32"
    with pytest.raises(AttributeError):
        default.backend = "mps"
    with pytest.raises(AttributeError):
        default.precision = "float16"
    with pytest.raises(ValueError, match="backend must be"):
        HoPsfModel(backend="cuda")
    with pytest.raises(ValueError, match="precision must be"):
        HoPsfModel(precision="float16")
    with pytest.raises(ValueError, match="requires precision='float32'"):
        HoPsfModel(backend="mps", precision="float64")

    def unavailable():
        raise RuntimeError("Metal execution unavailable")

    monkeypatch.setattr(ho_psf, "require_mps", unavailable)
    with pytest.raises(RuntimeError, match="Metal execution unavailable"):
        HoPsfModel(backend="mps")


@pytest.fixture
def corrected_config(config, monkeypatch):
    monkeypatch.setattr(_ho_common, "NATIVE_OPTICAL_SIDE", 32)
    common_dm = DeformableMirrorConfig(
        actuator_count=5,
        pitch=1 * u.m,
        influence_model="gaussian",
        influence_coupling=0.25 * u.one,
    )
    return replace(
        config,
        telescope=TelescopeConfig(diameter=4 * u.m, zenith_angle=20 * u.deg),
        pupil=Pupil(amplitude=np.ones((8, 8)), pitch=0.5 * u.m),
        atmosphere=AtmosphereConfig(
            reference_wavelength=500 * u.nm,
            fried_parameter=0.15 * u.m,
            outer_scale=25 * u.m,
            layer_weights=[1] * u.one,
            layer_heights=[4000] * u.m,
            wind_speeds=[8] * u.m / u.s,
            wind_directions=[35] * u.deg,
        ),
        high_order_sources=SourceConfig(
            directions=[[0, 0]] * u.arcsec,
            height=90 * u.km,
            wavelengths=[589] * u.nm,
        ),
        high_order_lenslets=LensletConfig(pitches=[1] * u.m, counts=(4,)),
        fitting=FittingConfig(directions=[[0, 0]] * u.arcsec, weights=[1] * u.one),
        mirrors=(common_dm, common_dm),
        high_order_loop=LoopConfig(
            gain=0.3 * u.one, frame_rate=500 * u.Hz, delay_frames=2
        ),
        residual_high_order_loop=LoopConfig(gain=1 * u.one, delay_frames=1),
        estimator_regularization=0.1,
    )


@pytest.mark.parametrize(
    "route", ["probe", "float64_preflight", "optical_retry", "float64_failure"]
)
@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_mixed_precision_corrected_prediction(
    corrected_config, monkeypatch, request, route, backend
):
    if backend == "mps":
        request.getfixturevalue("mps_device")
    prediction_request = HoPsfRequest(config=corrected_config)
    phase_pairs_need_float64 = _ho_common.phase_pairs_need_float64
    probe_calls = []

    def track_probe(*args, **options):
        probe_calls.append(None)
        if route == "float64_preflight":
            return len(probe_calls) == 1
        if route in ("optical_retry", "float64_failure"):
            return False
        return phase_pairs_need_float64(*args, **options)

    monkeypatch.setattr(_ho_common, "phase_pairs_need_float64", track_probe)
    default = HoPsfModel().predict(prediction_request)
    assert not probe_calls

    cross_calls = []
    field_cross = _ho_common.TwoStageCovariance.field_cross

    def record_cross(cross, dtype):
        cross_calls.append((np.dtype(dtype).name, len(cross)))
        if route == "float64_failure" or (
            route == "optical_retry" and len(cross_calls) == 1
        ):
            cross[0] = np.nan
        return cross

    def track_cross(self, *args, dtype):
        if backend == "mps":
            assert dtype == np.float64
        return record_cross(field_cross(self, *args, dtype=dtype), dtype)

    optical_dtypes = []
    coherence_lags = _ho_optics.coherence_lags

    def track_optics(pupil, residual, field):
        optical_dtypes.append(residual.diagonal.dtype.name)
        return coherence_lags(pupil, residual, field)

    monkeypatch.setattr(_ho_common.TwoStageCovariance, "field_cross", track_cross)
    monkeypatch.setattr(_ho_optics, "coherence_lags", track_optics)
    if backend == "mps":
        integrate = _ho_metal.MpsPupilOptics.coherence_lags
        query_type = _ho_metal_queries.MpsScienceCommandQueries
        contract = query_type.contract_two_stage_cross

        def track_metal_optics(self, *args):
            optical_dtypes.append("float32")
            return integrate(self, *args)

        def track_metal_cross(self, *args):
            cross = contract(self, *args)
            assert cross.device.type == "mps"
            return record_cross(cross, np.float32)

        monkeypatch.setattr(query_type, "contract_two_stage_cross", track_metal_cross)
        monkeypatch.setattr(
            _ho_metal.MpsPupilOptics, "coherence_lags", track_metal_optics
        )
    model = HoPsfModel(backend=backend, precision="float32")
    first_batch = 2 if backend == "mps" else 3
    if route == "float64_failure":
        with pytest.raises(
            _ho_optics.InvalidCovarianceOptics, match="nonfinite PSF intensity"
        ):
            model.predict(prediction_request)
        assert optical_dtypes == ["float32", "float64"]
        assert cross_calls == [("float32", first_batch), ("float64", 1)]
        return

    mixed = model.predict(prediction_request)
    assert mixed.psfs.dtype == np.float64
    assert mixed.full_flux == default.full_flux
    np.testing.assert_array_equal(mixed.positions, default.positions)
    assert np.isfinite(mixed.psfs).all()
    relative_image_error = (
        np.abs(mixed.psfs - default.psfs).sum(axis=(1, 2)) / default.full_flux
    )
    assert np.all(relative_image_error < 1e-4)
    assert np.all(relative_image_error[1:] > 0)
    assert len(probe_calls) == 3
    expected_dtypes = {
        "probe": ["float32", "float32", "float32"],
        "float64_preflight": ["float64", "float32", "float32"],
        "optical_retry": ["float32", "float64", "float32", "float32"],
    }
    assert optical_dtypes == expected_dtypes[route]
    expected_cross_calls = [("float32", first_batch)]
    if route != "probe":
        expected_cross_calls.append(("float64", 1))
    if backend == "mps":
        expected_cross_calls.append(("float32", 1))
    assert cross_calls == expected_cross_calls


@pytest.fixture
def mps_device():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available() or not hasattr(
        torch.mps, "compile_shader"
    ):
        pytest.skip("Metal shader execution is unavailable")


@pytest.fixture
def ho_components():
    return HoPsfComponents(
        sensing=ShackHartmannSensing(),
        common_stage=HoPsfStage(
            dm=GaussianDm(),
            reconstructor=PhaseFitReconstructor(),
            control=ProjectedPolc(),
        ),
        science_stage=HoPsfStage(
            dm=GaussianDm(),
            reconstructor=PhaseFitReconstructor(),
            control=DelayedResidualControl(),
        ),
    )


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("two_stage", [False, True])
def test_explicit_ho_components_match_defaults(
    corrected_config, ho_components, request, backend, two_stage
):
    if backend == "mps":
        request.getfixturevalue("mps_device")
    config = corrected_config
    if not two_stage:
        config = replace(
            config, mirrors=config.mirrors[:1], residual_high_order_loop=None
        )
        ho_components = replace(ho_components, science_stage=None)
    prediction_request = HoPsfRequest(config=config)
    default = HoPsfModel(backend=backend).predict(prediction_request)
    explicit = HoPsfModel(components=ho_components, backend=backend).predict(
        prediction_request
    )
    np.testing.assert_array_equal(explicit.positions, default.positions)
    assert explicit.full_flux == default.full_flux
    scaled_error = (
        abs(explicit.psfs - default.psfs).sum(axis=(1, 2)) / default.full_flux
    )
    # Metal combines pupil pairs with parallel atomic additions.
    assert np.max(scaled_error) <= (2e-7 if backend == "mps" else 0)


@pytest.mark.parametrize("weights", [None, [-1.0]])
def test_invalid_common_fit_target_is_rejected_before_phase(
    corrected_config, monkeypatch, weights
):
    def phase_layers(*args):
        pytest.fail("Built atmospheric spectra before validating the fit target")

    monkeypatch.setattr(_ho_common, "phase_layers", phase_layers)
    fitting = (
        None
        if weights is None
        else replace(corrected_config.fitting, weights=weights * u.one)
    )
    prediction_request = HoPsfRequest(config=replace(corrected_config, fitting=fitting))
    with pytest.raises(ValueError, match="[Mm]irror fitting"):
        HoPsfModel().predict(prediction_request)


def test_supplied_slope_matrix_layout_and_ownership(corrected_config, ho_components):
    model = HoPsfModel(components=ho_components)
    request = HoPsfRequest(config=corrected_config)
    layout = model.command_layout(request, stage="common")
    assert layout.slopes.size == 2 * len(layout.slopes.lenslet_positions)
    with pytest.raises(ValueError, match="actuator-by-slope shape"):
        MatrixReconstructor(matrix=np.zeros(layout.shape[::-1]), layout=layout)
    with pytest.raises(TypeError, match="numerical array with values in metres"):
        MatrixReconstructor(matrix=np.full(layout.shape, 250.0) * u.mm, layout=layout)
    values = np.ones(layout.shape)
    reconstructor = MatrixReconstructor(matrix=values, layout=layout)
    values[0] = 0
    supplied = reconstructor.matrix(corrected_config.fitting.directions)
    assert np.all(supplied == 1)
    assert not supplied.flags.writeable

    reversed_layout = replace(
        layout, actuator_positions=layout.actuator_positions[::-1]
    )
    calibrated = replace(
        ho_components,
        common_stage=replace(
            ho_components.common_stage,
            reconstructor=MatrixReconstructor(matrix=values, layout=reversed_layout),
        ),
    )
    with pytest.raises(ValueError, match="actuator and slope order"):
        HoPsfModel(components=calibrated).predict(request)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_reconstructor_can_reuse_matrix_storage(
    corrected_config, ho_components, request, backend
):
    if backend == "mps":
        request.getfixturevalue("mps_device")
    prediction_request = HoPsfRequest(config=corrected_config)
    layout = HoPsfModel().command_layout(prediction_request, stage="common")
    rng = np.random.default_rng(94)
    common_rows = rng.normal(scale=0.04, size=layout.shape)
    science_rows = rng.normal(scale=0.04, size=layout.shape)

    class ReusedMatrix(PreparedReconstructor):
        def __init__(self, rows, storage, reuse):
            self.rows = rows
            self.storage = storage
            self.reuse = reuse

        def prepare(self, *, sensing, dm, prior):
            return self

        def matrix(self, directions, *, weights=None):
            scale = 1 + 0.04 * directions.to_value(u.arcsec)[0, 0]
            np.multiply(self.rows, scale, out=self.storage)
            return self.storage if self.reuse else self.storage.copy()

    def predict(reuse):
        storage = np.empty(layout.shape)
        components = replace(
            ho_components,
            common_stage=replace(
                ho_components.common_stage,
                reconstructor=ReusedMatrix(common_rows, storage, reuse),
            ),
            science_stage=replace(
                ho_components.science_stage,
                reconstructor=ReusedMatrix(science_rows, storage, reuse),
            ),
        )
        return HoPsfModel(components=components, backend=backend).predict(
            prediction_request
        )

    reference = predict(False)
    reused = predict(True)
    error = abs(reused.psfs - reference.psfs).sum(axis=(1, 2)) / reference.full_flux
    assert np.max(error) <= (2e-7 if backend == "mps" else 0)


def test_supplied_common_matrix_matches_direct_covariance_and_psf(
    corrected_config, ho_components, monkeypatch, record_property
):
    config = replace(
        corrected_config,
        fitting=None,
        tilt_removal="none",
        detector_origin="geometric",
        telescope=replace(corrected_config.telescope, zenith_angle=0 * u.deg),
        science_sources=replace(
            corrected_config.science_sources, directions=[[0, 0]] * u.arcsec
        ),
        high_order_sources=replace(
            corrected_config.high_order_sources,
            height=0 * u.m,
            wavelengths=corrected_config.science_sources.wavelengths,
        ),
        atmosphere=replace(
            corrected_config.atmosphere,
            layer_heights=[0] * u.m,
            wind_speeds=[12.5] * u.m / u.s,
            wind_directions=[0] * u.deg,
        ),
        residual_high_order_loop=LoopConfig(gain=0.8 * u.one, delay_frames=5),
    )
    period = 16.0
    bins = np.array([[1, 2], [2, -1], [4, 0], [-1, -2], [-2, 1], [-4, 0]])
    power = np.array([0.08, 0.12, 0.06] * 2)

    def spectrum(fx, fy, fried, outer_scale):
        result = np.zeros_like(fx)
        origin = len(fx) // 2
        result[tuple((bins + origin).T)] = power * period**2
        return result

    def no_tail(grid, pupil):
        return (
            np.zeros((pupil.side, pupil.side)),
            np.zeros_like(pupil.ramps),
            np.empty((0, 0)),
        )

    monkeypatch.setattr(_ho_atmosphere, "von_karman_spectrum", spectrum)
    monkeypatch.setattr(_ho_common, "atmospheric_tail", no_tail)

    captured = {}
    render = _ho_common.coherence_image

    def capture(pupil, residual, *args, **kwargs):
        captured["residual"] = residual
        captured["pupil"] = pupil
        return render(pupil, residual, *args, **kwargs)

    class CaptureScienceFit(PhaseFitReconstructor):
        def prepare(self, *, sensing, dm, prior):
            fit = super().prepare(sensing=sensing, dm=dm, prior=prior)
            captured["science_rows"] = fit.matrix(config.science_sources.directions)
            captured["science_dm"] = dm
            return fit

    monkeypatch.setattr(_ho_common, "coherence_image", capture)
    prediction_request = HoPsfRequest(config=config)
    layout = HoPsfModel().command_layout(prediction_request, stage="common")
    positions = layout.actuator_positions.to_value(u.m)
    rounded_positions = np.nextafter(positions, 0.0) * u.m
    calibration_layout = replace(layout, actuator_positions=rounded_positions)
    rows = np.random.default_rng(151).normal(scale=0.05, size=layout.shape)
    components = replace(
        ho_components,
        common_stage=replace(
            ho_components.common_stage,
            reconstructor=MatrixReconstructor(matrix=rows, layout=calibration_layout),
        ),
        science_stage=replace(
            ho_components.science_stage, reconstructor=CaptureScienceFit()
        ),
    )
    prediction = HoPsfModel(components=components).predict(prediction_request)
    science_rows = captured["science_rows"]
    pupil = captured["pupil"]

    # Directly solve the two slope streams on each retained Fourier mode.
    frequencies = bins / period
    cells = layout.slopes.lenslet_positions.to_value(u.m)
    wave = np.exp(2j * np.pi * (cells @ frequencies.T))
    gradients = 2j * np.pi * frequencies.T * np.prod(np.sinc(frequencies), axis=1)
    raw = np.concatenate([wave * gradient for gradient in gradients])
    mean = np.kron(np.eye(2), np.full((len(cells), len(cells)), 1 / len(cells)))
    identity = np.eye(len(raw))
    common_dm = components.common_stage.dm.prepare(
        config=config.mirrors[0], pupil=config.pupil
    )
    sensing = components.sensing.prepare(
        pupil=config.pupil,
        telescope=config.telescope,
        guides=config.high_order_sources,
        lenslets=config.high_order_lenslets,
    )
    feedback = common_dm.slope_response(sensing) @ rows
    seconds_per_frame = 1 / 500
    frame_phase = np.exp(2j * np.pi * frequencies[:, 0] * 12.5 * seconds_per_frame)
    common_transfer = 0.3 * frame_phase**2 / (1 - 0.7 * frame_phase)
    common_slopes = np.column_stack(
        [
            np.linalg.solve(
                identity - transfer * mean @ feedback,
                transfer * (identity - mean) @ mode,
            )
            for transfer, mode in zip(common_transfer, raw.T, strict=True)
        ]
    )
    science_transfer = 0.8 * frame_phase**5
    science_slopes = science_transfer * (
        (identity - mean) @ (raw - feedback @ common_slopes)
    )
    assert (
        np.linalg.norm(science_slopes - science_transfer * ((identity - mean) @ raw))
        > 0.01
    )

    def dm_basis(dm):
        centres = dm.actuator_positions.to_value(u.m)
        separation = pupil.points[:, None, :] - centres
        return 0.25 ** np.sum(separation**2, axis=2)

    residual_modes = (
        np.exp(2j * np.pi * (pupil.points @ frequencies.T))
        - dm_basis(common_dm) @ rows @ common_slopes
        - dm_basis(captured["science_dm"]) @ science_rows @ science_slopes
    )
    covariance = ((residual_modes * power) @ residual_modes.conj().T).real
    residual = captured["residual"]
    covariance_error = 0.0
    for first, block in residual.blocks(rows=8):
        expected = covariance[first : first + len(block), first:]
        covariance_error = max(covariance_error, np.max(abs(block - expected)))
    assert covariance_error < 2e-12

    # Transform the pair-weighted coherence directly at the geometric pixel centres.
    variance = np.diag(covariance)
    coherence = np.exp(covariance - 0.5 * (variance[:, None] + variance))
    coherence *= np.outer(pupil.amplitude, pupil.amplitude)
    pixels = np.arange(16) - 7.5
    x, y = np.meshgrid(pixels, pixels, indexing="xy")
    detector_angles = np.column_stack((x.ravel(), y.ravel())) / period
    detector_wave = np.exp(-2j * np.pi * (detector_angles @ pupil.points.T))
    expected_image = np.sum(
        (detector_wave @ coherence) * detector_wave.conj(), axis=1
    ).real
    expected_image = expected_image.reshape(16, 16)
    image_error = (
        np.abs(prediction.psfs[0] - expected_image).sum() / prediction.full_flux
    )
    record_property("max_abs_covariance_error_rad2", float(covariance_error))
    record_property("psf_l1_error_over_full_flux", float(image_error))
    assert image_error < 2e-12


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_corrected_prediction_without_tilt_removal(corrected_config, request, backend):
    if backend == "mps":
        request.getfixturevalue("mps_device")
    config = replace(corrected_config, tilt_removal="none")
    prediction_request = HoPsfRequest(config=config)
    reference = HoPsfModel().predict(prediction_request)
    prediction = HoPsfModel(backend=backend, precision="float32").predict(
        prediction_request
    )
    image_error = abs(prediction.psfs - reference.psfs).sum(axis=(1, 2))
    assert prediction.full_flux == reference.full_flux
    assert np.all(image_error / reference.full_flux < 1e-4)


@pytest.mark.parametrize("correction", ["common-only", "science-dm", "zero-gain"])
def test_metal_residual_lags(
    corrected_config, mps_device, monkeypatch, record_property, correction
):
    monkeypatch.setattr(_ho_common, "NATIVE_OPTICAL_SIDE", 64)
    config = replace(
        corrected_config,
        pupil=Pupil(amplitude=np.ones((32, 32)), pitch=0.125 * u.m),
        atmosphere=replace(
            corrected_config.atmosphere,
            layer_weights=[0.6, 0.4] * u.one,
            layer_heights=[4000, 9500] * u.m,
            wind_speeds=[8, 14] * u.m / u.s,
            wind_directions=[35, 75] * u.deg,
        ),
        high_order_sources=replace(
            corrected_config.high_order_sources, directions=[[-5, 4], [7, 9]] * u.arcsec
        ),
        high_order_lenslets=replace(
            corrected_config.high_order_lenslets,
            grid_offsets=[[0.01, -0.02], [0.03, 0.04]] * u.m,
        ),
    )
    mirrors = config.mirrors
    common_config = replace(config, mirrors=mirrors[:1], residual_high_order_loop=None)
    if correction == "common-only":
        config = common_config
    elif correction == "zero-gain":
        config = replace(config, residual_high_order_loop=LoopConfig(gain=0 * u.one))
        # Allow one layer on Metal to exercise eviction and reuse of host planes.
        monkeypatch.setattr(
            _ho_metal_queries.MpsScienceCommandQueries,
            "kernel_cache_bytes",
            2 * 256**2 * 4,
        )
    else:
        # Exercise complex pupil weights and the original-map detector path.
        y, x = np.indices(config.pupil.amplitude.shape) - 15.5
        config = replace(
            config,
            pupil=replace(config.pupil, static_opd=x * y * 0.3 * u.nm),
            static_opd_sampling="original_map",
            detector_origin="geometric",
        )

    covariance_errors = []
    lag_errors = []
    lag_rms_errors = []
    cpu_cross = []
    mps_cross = []
    probe_samples = []
    kernel_preparations = []
    preflight = _ho_common.phase_pairs_need_float64
    integrate = _ho_metal.MpsPupilOptics.coherence_lags
    slope_kernels = _ho_metal_queries.slope_kernels
    kernel_pair = _ho_metal_queries.MpsScienceCommandQueries._kernel_pair
    upload = _ho_metal_queries.MpsScienceCommandQueries._upload

    def track_cross_moments(
        pupil, pairs, source, basis, covariance, cross, *, cross_pairs=None
    ):
        if cross_pairs is None:
            cpu_cross.append(cross.copy())
        else:
            assert len(cross) < len(pupil.points)
            probe_samples.append((pairs, cross_pairs, cross))
        return preflight(
            pupil, pairs, source, basis, covariance, cross, cross_pairs=cross_pairs
        )

    def track_slope_kernels(*args):
        kernel_preparations.append(None)
        return slope_kernels(*args)

    def check_kernel_budget(queries, index):
        cache = queries._device_kernels
        if correction == "zero-gain" and cache and index not in cache:
            evicted = ref(next(iter(cache.values()))[0][0])

            def check_upload(values, dtype):
                assert evicted() is None
                return upload(queries, values, dtype)

            with monkeypatch.context() as scoped:
                scoped.setattr(queries, "_upload", check_upload)
                prepared = kernel_pair(queries, index)
        else:
            prepared = kernel_pair(queries, index)
        resident_bytes = sum(
            phase.numel() * phase.element_size()
            for layer in cache.values()
            for phase, _ in layer
        )
        assert resident_bytes <= queries.kernel_cache_bytes
        return prepared

    def compare_lags(optics, source, command_covariance, source_command):
        assert source_command.device.type == "mps"
        cross = source_command.cpu().numpy()
        pairs, cross_pairs, sampled = probe_samples[-1]
        np.testing.assert_array_equal(sampled[cross_pairs], cross[pairs])
        mps_cross.append(cross)
        residual = _ho_optics.ResidualCovariance(
            optics.pupil,
            source,
            optics.basis.cpu().numpy(),
            command_covariance,
            cross,
            dtype=np.float32,
        )
        metal = _ho_metal._MpsResidualCovariance(
            optics, source, command_covariance, source_command
        )
        scale = np.max(abs(residual.diagonal))
        for (_, block), (_, device_block) in zip(
            residual.blocks(), metal.blocks(), strict=True
        ):
            covariance_errors.append(
                np.max(abs(block - device_block.cpu().numpy())) / scale
            )
        cpu = _ho_optics.coherence_lags(optics.pupil, residual, optics.field)
        mps = integrate(optics, source, command_covariance, source_command)
        center = tuple(size // 2 for size in cpu.shape)
        assert mps[center] == cpu[center]
        np.testing.assert_allclose(
            mps, mps[::-1, ::-1].conj(), rtol=0, atol=1e-14 * cpu[center].real
        )
        lag_errors.append(np.max(abs(mps - cpu)) / cpu[center].real)
        lag_rms_errors.append(np.linalg.norm(mps - cpu) / np.linalg.norm(cpu))
        return mps

    monkeypatch.setattr(_ho_metal.MpsPupilOptics, "coherence_lags", compare_lags)
    monkeypatch.setattr(_ho_common, "phase_pairs_need_float64", track_cross_moments)
    monkeypatch.setattr(_ho_metal_queries, "slope_kernels", track_slope_kernels)
    monkeypatch.setattr(
        _ho_metal_queries.MpsScienceCommandQueries, "_kernel_pair", check_kernel_budget
    )
    model = HoPsfModel(backend="mps")
    assert model.precision == "float32"
    request = HoPsfRequest(config=config)
    cpu = HoPsfModel(precision="float32").predict(request)
    device = model.predict(request)
    assert device.full_flux == cpu.full_flux
    assert device.psfs.dtype == np.float64
    assert len(lag_errors) == len(cpu.positions)
    assert len(covariance_errors) > len(lag_errors)
    assert len(kernel_preparations) == (4 if correction == "science-dm" else 2)
    cpu_moments = np.stack(cpu_cross)
    mps_moments = np.stack(mps_cross)
    cross_error = np.max(
        np.linalg.norm(mps_moments - cpu_moments, axis=(1, 2))
        / np.linalg.norm(cpu_moments, axis=(1, 2))
    )
    image_error = np.max(abs(device.psfs - cpu.psfs).sum(axis=(1, 2)) / cpu.full_flux)
    record_property("relative_phase_command_cross_error", float(cross_error))
    record_property("relative_max_covariance_error", float(max(covariance_errors)))
    record_property("max_lag_error_over_zero_lag", float(max(lag_errors)))
    record_property("relative_rms_lag_error", float(max(lag_rms_errors)))
    record_property("image_error_over_full_flux", float(image_error))
    assert max(covariance_errors) < 1e-5
    assert max(lag_errors) < 1e-5
    assert max(lag_rms_errors) < 1e-5
    assert cross_error < 1e-5
    assert image_error < 1e-5
    if correction == "zero-gain":
        common = model.predict(HoPsfRequest(config=common_config))
        np.testing.assert_allclose(device.psfs, common.psfs, rtol=1e-5, atol=1e-5)


def test_metal_weak_coherence_and_invalid_variance(mps_device):
    pupil = _ho_optics.OpticalPupil(np.ones((4, 4)), 4, 4, 4, "none")
    count = len(pupil.x)
    zero_lags = np.zeros((4, 4))
    stationary = (zero_lags.copy(), np.empty((0, count)), np.empty((0, 0)))
    stationary[0][0, 0] = 1e-9
    basis = np.zeros((count, 1))
    covariance = np.zeros((1, 1))
    optics = _ho_metal.MpsPupilOptics(pupil, basis, None)
    cross = optics.to_device(basis)
    static = optics.coherence_lags((zero_lags, *stationary[1:]), covariance, cross)
    residual = _ho_optics.ResidualCovariance(
        pupil, stationary, basis, covariance, basis, dtype=np.float32
    )
    cpu = _ho_optics.coherence_lags(pupil, residual)
    mps = optics.coherence_lags(stationary, covariance, cross)
    assert np.max(abs(cpu - static)) > 1e-9
    np.testing.assert_allclose(mps, cpu, rtol=0, atol=1e-13)

    stationary[0][0, 0] = -1e-3
    with pytest.raises(_ho_optics.InvalidCovarianceOptics, match="negative difference"):
        optics.coherence_lags(stationary, covariance, cross)
    stationary[0][0, 0] = np.nan
    with pytest.raises(
        _ho_optics.InvalidCovarianceOptics, match="nonfinite difference"
    ):
        optics.coherence_lags(stationary, covariance, cross)


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
