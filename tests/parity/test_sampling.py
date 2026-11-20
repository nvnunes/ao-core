"""Independent checks of exposure averaging for the sampling model."""

import numpy as np
import pytest
from astropy import units as u

from ao_core.models.exposure import exposure_variance


@pytest.mark.parametrize(
    "lags", [[4.0, 0, 0, 0], [4.0, 4.0, 4.0, 4.0], [4.0, -4.0, 4.0, -4.0]]
)
def test_finite_exposure_variance_matches_frame_pairs(lags):
    count = len(lags)
    frames = np.arange(count)
    covariance = np.asarray(lags)[np.abs(frames[:, None] - frames)]
    expected = covariance.mean() / 3
    actual = exposure_variance(lags * u.mas**2, 2 * u.ms, count, realizations=3)
    assert actual.to_value(u.mas**2) == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_single_frame_uses_only_zero_lag():
    actual = exposure_variance(np.array([[4, 1], [9, -2]]) * u.one, 2 * u.ms, 1)
    np.testing.assert_array_equal(actual.value, [4, 9])


def gaussian_pupil_fixture():
    # A damped rotation gives a stationary process with signed, asymmetric lags.
    rng = np.random.default_rng(741)
    basis = np.linalg.qr(np.column_stack((np.ones(5), rng.normal(size=(5, 4)))))[0][
        :, 1:
    ]
    angle = 0.27
    rotation = np.eye(4)
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    rotation *= 0.71
    full = (
        np.array(
            [basis @ np.linalg.matrix_power(rotation, k) @ basis.T for k in range(2)]
        )
        * 0.13
    )
    modal = (
        np.array(
            [
                basis[:, :3]
                @ (np.linalg.matrix_power(rotation, k)[:3, :3])
                @ basis[:, :3].T
                for k in range(2)
            ]
        )
        * 0.13
    )
    weights = rng.normal(size=(3, 5, 5)) + 1j * rng.normal(size=(3, 5, 5))
    weights = (weights + weights.swapaxes(1, 2).conj()) / 2
    return np.array([modal, full]), weights


def pair_covariances(covariances, weights):
    nodes = covariances.shape[-1]
    a, b = np.indices((nodes, nodes)).reshape(2, -1)
    signed = weights[:, a, b]
    cross = (
        covariances[:, a[:, None], a]
        - covariances[:, a[:, None], b]
        - covariances[:, b[:, None], a]
        + covariances[:, b[:, None], b]
    )
    pair = np.einsum("mp,mq->mpq", signed, signed)
    closed = np.einsum("mpq,vpq->vm", pair, np.expm1(-cross)).real
    remainder = np.einsum(
        "mpq,vpq->vm", pair, np.expm1(-cross) + cross - 0.5 * cross**2
    ).real
    return closed, remainder


def test_exhaustive_gaussian_pupil_covariance_distinguishes_balanced_and_full():
    from ao_core.models._all_order import (
        quadratic_covariance,
        quadratic_response,
        restored_covariance,
    )

    covariances, weights = gaussian_pupil_fixture()
    for lag in range(2):
        expected, remainder = pair_covariances(covariances[:, lag], weights)
        analytic = quadratic_covariance(
            covariances[:, lag], *quadratic_response(weights)
        )
        np.testing.assert_allclose(
            analytic + remainder, expected, rtol=1e-12, atol=1e-14
        )
        assert np.linalg.norm(expected[0] - expected[1]) > 0.01
    for covariance in covariances[:, 0]:
        diagonal = np.diag(covariance)
        coherence = np.exp(-0.5 * (diagonal[:, None] + diagonal - 2 * covariance))
        restored = restored_covariance(coherence, np.ones_like(coherence))
        np.testing.assert_allclose(restored, covariance, rtol=1e-12, atol=1e-14)


def test_cached_sobol_draws_match_direct_evaluation():
    from ao_core.models._all_order import RemainderQuadrature

    covariances, weights = gaussian_pupil_fixture()
    shared = np.broadcast_to(weights, (len(covariances), *weights.shape))
    quadrature = RemainderQuadrature(covariances[:, 0], shared)
    options = {"power": 14, "replicates": 4, "seed": 20261105}
    for lag in range(2):
        actual = quadrature.evaluate(
            covariances[:, lag], shared, **options, reuse_draws=True
        )
        repeat = quadrature.evaluate(covariances[:, lag], shared, **options)
        np.testing.assert_array_equal(actual, repeat)
        _, expected = pair_covariances(covariances[:, lag], weights)
        assert np.linalg.norm(
            actual[0].mean(axis=0) - expected
        ) <= 0.03 * np.linalg.norm(expected)


def test_metal_gaussian_contraction_matches_cpu_with_identical_quartets():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal is unavailable")
    from ao_core.models._all_order import (
        RemainderQuadrature,
        quadratic_covariance,
        quadratic_response,
    )
    from ao_core.models._metal_covariance import (
        MpsQuadraticCovariance,
        MpsRemainderQuadrature,
    )

    covariances, weights = gaussian_pupil_fixture()
    shared = np.broadcast_to(weights, (len(covariances), *weights.shape))
    quadrature = RemainderQuadrature(covariances[:, 0], shared)
    options = {"power": 12, "replicates": 4, "seed": 20261105, "reuse_draws": True}
    quadrature.evaluate(covariances[:, 0], shared, **options)
    metal = MpsRemainderQuadrature(
        quadrature.metal_indices, weights, quadrature.proposal, len(covariances)
    )
    kernels = quadratic_response(weights)
    quadratic = MpsQuadraticCovariance(*kernels)
    for lag in range(2):
        expected = quadrature.evaluate(covariances[:, lag], shared, **options)
        actual = metal(covariances[:, lag])
        for value, reference in zip(actual, expected, strict=True):
            assert np.linalg.norm(value - reference) <= 2e-6 * np.linalg.norm(reference)
        analytic = np.array([quadratic(item) for item in covariances[:, lag]])
        np.testing.assert_allclose(
            analytic,
            quadratic_covariance(covariances[:, lag], *kernels),
            rtol=2e-6,
            atol=1e-9,
        )


def test_fast_covariance_matches_direct_pupil_pairs_through_order_six():
    from math import factorial

    from ao_core.models.metrics import metric_covariance, phase_responses

    _, kernels = gaussian_pupil_fixture()
    rng = np.random.default_rng(872)
    modes, _ = np.linalg.qr(np.column_stack((np.ones(5), rng.normal(size=(5, 3)))))
    basis = modes[:, 1:].T
    zero = np.diag([0.4, 0.18, 0.08])
    lag = np.array(
        [
            zero,
            [[0.27, 0.03, 0.02], [-0.09, 0.12, -0.01], [0.01, 0.02, 0.06]],
        ]
    )
    differences = (basis[:, :, None] - basis[:, None, :]).reshape(3, -1)
    structure = np.einsum("ip,ij,jp->p", differences, zero, differences)
    coherence = np.exp(-0.5 * structure).reshape(5, 5)
    weights = (kernels * coherence).reshape(3, -1)
    pair_products = weights[:, :, None] * weights[:, None, :]

    full = np.einsum("ip,tij,jq->tpq", differences, lag, differences)
    top_two = np.einsum(
        "ip,tij,jq->tpq", differences[:2], lag[:, :2, :2], differences[:2]
    )
    expected = sum(
        np.einsum("mpq,tpq->mt", pair_products, (-cross) ** order).real
        / factorial(order)
        for order, cross in enumerate(
            (full, full, top_two, top_two, top_two, top_two), 1
        )
    )
    dense = phase_responses(basis, zero, kernels, coherence, tail_order=6)
    tiled = phase_responses(
        basis,
        zero,
        kernels.reshape(3, -1),
        coherence,
        tail_order=6,
        lag_indices=np.arange(25).reshape(5, 5),
    )
    for response in (dense, tiled):
        actual = metric_covariance(lag, *response)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_metal_quadratic_covariance_with_large_pupil():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Metal is unavailable")
    from ao_core.models._metal_covariance import MpsQuadraticCovariance

    # The original flattened reduction intermittently spoiled the small metric
    # contribution at this pupil size when another metric had a much larger one.
    nodes = 2152
    diagonal = np.linspace(0.2, 0.8, nodes, dtype=np.float32)
    amplitudes = np.linspace(-1, 1, nodes, dtype=np.float32)
    scales = np.array([0.001, 0.0004, 0.1], dtype=np.float32)
    covariance = np.diag(diagonal)
    first = scales[:, None] * amplitudes
    second = scales[:, None, None] * np.outer(amplitudes, amplitudes)

    # For diagonal C, each product (H C)_ij (C H)_ij is H_ij² C_ii C_jj.
    linear = np.sum(first.astype(float) ** 2 * diagonal, axis=1)
    quadratic = np.einsum(
        "mij,mij,i,j->m", second.astype(float), second, diagonal, diagonal
    )
    expected = linear + 0.5 * quadratic
    model = MpsQuadraticCovariance(first, second)
    samples = np.array([model(covariance) for _ in range(240)])
    np.testing.assert_allclose(
        samples, np.broadcast_to(expected, samples.shape), rtol=1e-5, atol=0
    )


@pytest.fixture(scope="module")
def sampling_reference():
    # Pinned e011 CPU Fast with linear optical transfer at K and Gb,
    # indices 55/95, 2/4 ms.
    # Capture recipe: GIRMOS verification capture_public_sampling_reference.py.
    from pathlib import Path

    with np.load(Path(__file__).with_name("sampling_reference.npz")) as data:
        yield data


@pytest.fixture(scope="module")
def public_sampling_request(sampling_reference):
    from ao_core.models.config import (
        AtmosphereConfig,
        Configuration,
        DetectorConfig,
        FittingConfig,
        LoopConfig,
        NaturalGuideStar,
        Pupil,
        SourceConfig,
        TelescopeConfig,
    )
    from ao_core.models.uncertainty import (
        UncertaintyRequest,
        gaussian_ngs_noise_covariance,
    )

    d = sampling_reference
    loops = [
        LoopConfig(
            gain=gain * u.one,
            frame_rate=(1 / float(d["frame_s"])) * u.Hz,
            delay_frames=int(delay),
        )
        for gain, delay in zip(d["gains"], d["delay_frames"], strict=True)
    ]
    configuration = Configuration(
        pupil=Pupil(
            amplitude=d["pupil_amplitude"],
            pitch=d["support_m"] / (d["pupil_amplitude"].shape[0] - 1) * u.m,
            static_opd=d["opd_nm"] * u.nm,
        ),
        ngs=tuple(
            NaturalGuideStar(
                position=xy * u.arcsec, flux=count / d["frame_s"] * u.photon / u.s
            )
            for xy, count in zip(d["guide_xy_arcsec"], d["photons"], strict=True)
        ),
        telescope=TelescopeConfig(diameter=d["diameter_m"] * u.m),
        atmosphere=AtmosphereConfig(
            reference_wavelength=d["reference_m"] * u.m,
            fried_parameter=d["fried_m"] * u.m,
            outer_scale=d["outer_m"] * u.m,
            layer_heights=d["heights_m"] * u.m,
            layer_weights=d["weights"] * u.one,
            wind_speeds=d["wind_mps"] * u.m / u.s,
            wind_directions=d["wind_deg"] * u.deg,
        ),
        fitting=FittingConfig(
            directions=d["fit_xy_arcsec"] * u.arcsec,
            weights=d["fit_weights"] * u.one,
        ),
        high_order_sources=SourceConfig(
            directions=d["beacon_xy_arcsec"] * u.arcsec,
            height=d["beacon_height_m"] * u.m,
        ),
        low_order_sources=SourceConfig(
            wavelengths=d["guide_wave_m"] * u.m,
        ),
        low_order_detector=DetectorConfig(
            pixel_scale=d["pitch_mas"] * u.mas / u.pix,
            read_noise=d["read_noise"] * u.electron,
            threshold=d["threshold"] * u.electron,
        ),
        high_order_loop=loops[0],
        low_order_loop=loops[1],
    )
    noise = gaussian_ngs_noise_covariance(
        configuration,
        ee=d["guide_ee"] * u.one,
        fwhm=d["guide_fwhm_mas"] * u.mas,
    )
    return UncertaintyRequest(
        config=configuration,
        ngs_noise_covariance=noise,
        positions=d["points"] * u.arcsec,
        exposures=d["exposures_s"] * u.s,
        wavelength=d["wavelength_m"][0] * u.m,
    )


def test_public_sampling_predictor_preserves_order_and_agrees_with_linear_reference(
    public_sampling_request,
    sampling_reference,
    monkeypatch,
):
    from dataclasses import replace

    from ao_core.models import uncertainty
    from ao_core.models.uncertainty import (
        SamplingSettings,
        SamplingUncertaintyModel,
        UncertaintyDiagnostics,
        UncertaintyMetricPrediction,
        UncertaintyPrediction,
        UncertaintyRequest,
    )

    def historical_grid(settings, exposure, **_):
        return (*settings._science_nodes(exposure), 20.0)

    monkeypatch.setattr(SamplingSettings, "_science_grid", historical_grid)

    k = public_sampling_request
    assert isinstance(k, UncertaintyRequest)
    assert uncertainty.UncertaintyRequest is UncertaintyRequest
    assert uncertainty.UncertaintyMetricPrediction is UncertaintyMetricPrediction
    assert uncertainty.UncertaintyPrediction is UncertaintyPrediction
    d = sampling_reference
    order = [1, 0, 1]
    gb = replace(
        k,
        wavelength=d["wavelength_m"][1] * u.m,
        positions=k.positions[order],
        exposures=k.exposures[order],
    )
    predictor = SamplingUncertaintyModel()
    results = tuple(predictor.predict_many((k, gb), diagnostics=True))
    metrics = ("sr", "ee100", "fwhm_mas")
    for band, (request, result, index) in enumerate(
        zip((k, gb), results, ([0, 1], order), strict=True)
    ):
        assert isinstance(result, UncertaintyPrediction)
        assert result.request is request
        assert isinstance(result.diagnostics, UncertaintyDiagnostics)
        window_and_point_order = np.ix_(index, index)
        for metric_index, name in enumerate(metrics):
            metric = result.metrics[name]
            assert isinstance(metric, UncertaintyMetricPrediction)
            expected = d[f"predicted_mean_{band}"][..., metric_index][
                window_and_point_order
            ]
            coefficient = (
                d[f"coefficient_{band}"][..., metric_index][window_and_point_order]
                / 100
            )
            unit = u.mas if name == "fwhm_mas" else u.one
            means = result.diagnostics.mean_by_exposure[name].to_value(unit)
            coefficients = metric.coefficient.to_value(u.s**0.5)
            # The cubic optics shift the reference means by up to 0.54%
            # and the fractional SD coefficients by up to 1.04%.
            np.testing.assert_allclose(means, expected, rtol=0.006)
            np.testing.assert_allclose(coefficients, coefficient, rtol=0.012)
            np.testing.assert_allclose(metric.mean.to_value(unit), means[0], rtol=1e-10)
            variance = (coefficients * means) ** 2 / request.exposures.value[:, None]
            np.testing.assert_allclose(
                metric.variance.to_value(unit**2), variance, rtol=2e-10
            )

    single = replace(k, exposures=k.exposures[:1])
    prediction = predictor.predict(single)
    assert prediction.diagnostics is None
    for name in metrics:
        expected = results[0].metrics[name].mean.value
        np.testing.assert_allclose(
            prediction.metrics[name].mean.value, expected, rtol=1e-10
        )


def test_reference_mean_is_independent_of_temporal_grids(
    public_sampling_request, monkeypatch
):
    from dataclasses import replace

    from ao_core.models.uncertainty import SamplingSettings, SamplingUncertaintyModel

    def temporal_grid(settings, exposure, **_):
        return (80 if exposure == 0.002 else 96, 192, 20.0)

    monkeypatch.setattr(SamplingSettings, "_science_grid", temporal_grid)
    model = SamplingUncertaintyModel()
    request = replace(
        public_sampling_request, positions=public_sampling_request.positions[1:]
    )
    reversed_request = replace(request, exposures=request.exposures[::-1])
    first, second = model.predict_many((request, reversed_request), diagnostics=True)
    single = model.predict(replace(request, exposures=request.exposures[1:]))

    assert first.diagnostics.grids == ((80, 192, 20.0), (96, 192, 20.0))
    assert second.diagnostics.grids == first.diagnostics.grids[::-1]
    assert single.diagnostics is None
    for name, metric in first.metrics.items():
        np.testing.assert_array_equal(metric.mean, second.metrics[name].mean)
        np.testing.assert_array_equal(metric.mean, single.metrics[name].mean)
        grid_means = first.diagnostics.mean_by_exposure[name]
        assert np.any(grid_means[0] != grid_means[1])
        np.testing.assert_array_equal(
            grid_means, second.diagnostics.mean_by_exposure[name][::-1]
        )
        coefficient = (
            np.sqrt(metric.variance * request.exposures[:, None]) / metric.mean
        )
        np.testing.assert_allclose(metric.coefficient, coefficient, rtol=1e-14)
        np.testing.assert_allclose(
            metric.coefficient[1:], single.metrics[name].coefficient, rtol=1e-12
        )


def test_stationary_wind_grid_uses_two_exposures_and_only_the_zero_bin(
    public_sampling_request,
):
    from dataclasses import replace

    from ao_core.models.uncertainty import SamplingUncertaintyModel

    source = public_sampling_request
    atmosphere = source.config.atmosphere
    config = replace(
        source.config,
        atmosphere=replace(
            atmosphere, wind_speeds=np.zeros_like(atmosphere.wind_speeds)
        ),
    )
    request = replace(source, config=config, positions=source.positions[:1])
    result = SamplingUncertaintyModel().predict(request, diagnostics=True)

    assert result.diagnostics.grids == ((80, 192, 0.004), (80, 192, 0.008))
    for metric in result.metrics.values():
        assert np.all(np.isfinite(metric.variance.value))
        assert np.all(metric.variance.value > 0)


def test_long_fast_grid_is_compact_with_unrelated_short_balanced_request(
    public_sampling_request,
    monkeypatch,
):
    from dataclasses import replace

    from ao_core.models.uncertainty import SamplingUncertaintyModel

    class ReachedCompactGrid(Exception):
        pass

    def check_route(grid, windows, bands, representatives, device, *, compact, spectral_interval):
        assert spectral_interval is None
        assert grid == (320, 768, 24)
        assert device == "cpu"
        assert compact
        raise ReachedCompactGrid

    def dense_route(*args, **kwargs):
        pytest.fail("Long Fast request entered the dense grid")

    request = public_sampling_request
    predictor = SamplingUncertaintyModel()
    monkeypatch.setattr(predictor, "_fast_points", check_route)
    monkeypatch.setattr(predictor, "_prepare_grid", dense_route)
    with pytest.raises(ReachedCompactGrid):
        tuple(
            predictor.predict_many(
                (
                    replace(request, exposures=[12] * u.s),
                    replace(request, model_type="balanced", exposures=[0.002] * u.s),
                )
            )
        )


def test_accelerated_fast_bounds_simultaneous_band_optics(
    public_sampling_request, monkeypatch
):
    from dataclasses import replace
    from types import SimpleNamespace

    from ao_core.models.uncertainty import SamplingUncertaintyModel

    batch_shapes = []

    def evaluate(grid, windows, bands, representatives, device, *, compact, spectral_interval):
        assert spectral_interval is None
        assert device == "mps"
        batch_shapes.append((len(bands), windows))
        prepared = SimpleNamespace(
            dt=0.002, windows={time: np.array([0.5]) for time in windows}
        )
        return prepared, (
            (band, point, np.ones(3), {"fast": (np.array([0]), np.ones((3, 1)))})
            for band, points in bands.items()
            for point in points
        )

    predictor = SamplingUncertaintyModel(device="mps")
    monkeypatch.setattr(predictor, "_fast_points", evaluate)
    monkeypatch.setattr(
        predictor,
        "_reference_means",
        lambda requests: {
            (float(request.wavelength.value), False): dict.fromkeys(
                map(tuple, request.positions.value), np.ones(3)
            )
            for request in requests
        },
    )
    requests = tuple(
        replace(
            public_sampling_request,
            wavelength=wave * u.um,
            exposures=[0.002 if band < 8 else 1.0] * u.s,
        )
        for band, wave in enumerate(np.linspace(1, 2, 10))
    )
    assert tuple(item.request for item in predictor.predict_many(requests)) == requests
    assert batch_shapes == [(8, (0.002,)), (2, (1.0,))]


@pytest.fixture
def sampling_grid_settings():
    from ao_core.models._explicit_optics import ExplicitGridSettings

    return ExplicitGridSettings(
        physical_side=64,
        optical_side=48,
        sensor_side=96,
        modal_order=5,
        radial_order=40,
        angular_order=96,
        spectral_period_s=20,
        spectral_interval_s=0.01,
    )


def test_sampling_settings_require_integer_orders_and_worker_counts(
    sampling_grid_settings,
):
    from dataclasses import replace

    from ao_core.models._explicit_optics import AllOrderSettings, LimitedOrderSettings

    grid = sampling_grid_settings
    with pytest.raises(ValueError, match="modal_order"):
        replace(grid, modal_order=5.0)
    with pytest.raises(ValueError, match="tail_order"):
        LimitedOrderSettings(grid_settings=grid, tail_order=6.0)
    with pytest.raises(ValueError, match="sensor_workers"):
        LimitedOrderSettings(grid_settings=grid, sensor_workers=1.5)
    with pytest.raises(ValueError, match="sobol_workers"):
        AllOrderSettings(grid_settings=grid, sobol_workers=True)


def test_sampling_request_accepts_only_settings_for_selected_model(
    public_sampling_request,
):
    from dataclasses import replace

    from ao_core.models.uncertainty import FastSettings, UncertaintyModelSettings

    settings = FastSettings(remove_science_tilt=True)
    assert isinstance(settings, UncertaintyModelSettings)
    request = public_sampling_request
    assert replace(request, model_settings=settings).model_settings is settings
    for model in ("balanced", "full"):
        assert replace(request, model_type=model).model_settings is None
        with pytest.raises(TypeError, match="model_settings=None"):
            replace(request, model_type=model, model_settings=settings)

    with pytest.raises(TypeError, match="FastSettings"):
        replace(request, model_settings=UncertaintyModelSettings())


@pytest.mark.parametrize("model", ["limited-order", "all-order", "full-pupil"])
def test_sampling_request_rejects_former_explicit_model_names(
    public_sampling_request, model
):
    from dataclasses import replace

    with pytest.raises(ValueError, match="Unknown sampling model_type"):
        replace(public_sampling_request, model_type=model)


def test_sampling_configuration_ini_and_memory_predict_identically(
    public_sampling_request, sampling_reference, tmp_path
):
    from dataclasses import replace

    from astropy.io import fits

    from ao_core.models.config import Configuration, Pupil, read_ini
    from ao_core.models.uncertainty import (
        SamplingSettings,
        SamplingUncertaintyModel,
        _project_sampling_configuration,
        gaussian_ngs_noise_covariance,
    )

    request = public_sampling_request
    prototype = request.config
    telescope = prototype.telescope
    atmosphere = prototype.atmosphere
    side = len(prototype.pupil.amplitude)
    pupil = Pupil(
        amplitude=prototype.pupil.amplitude,
        pitch=telescope.diameter / side,
        static_opd=prototype.pupil.static_opd,
    )
    # Align detector edges with samples of the sensing PSF in this fixture.
    wavelength = prototype.low_order_sources.wavelengths[0]
    psf_pixel = (wavelength / (4 * pupil.span)).to_value(u.one) * u.rad
    detector = replace(
        prototype.low_order_detector,
        pixel_scale=50 * psf_pixel.to(u.mas) / u.pix,
    )
    fits.writeto(tmp_path / "pupil.fits", pupil.amplitude)
    fits.writeto(tmp_path / "static.fits", pupil.static_opd.to_value(u.nm))

    def polar(positions):
        x, y = positions.value.T
        return np.hypot(x, y).tolist(), np.rad2deg(np.arctan2(y, x)).tolist()

    lgs = prototype.high_order_sources
    laser_r, laser_theta = polar(lgs.directions)
    fit_r, fit_theta = polar(prototype.fitting.directions)
    ho, lo = prototype.high_order_loop, prototype.low_order_loop
    path = tmp_path / "system.ini"
    path.write_text(
        f"""[telescope]
TelescopeDiameter = {telescope.diameter.value}
Resolution = {side}
ZenithAngle = 10
PathPupil = 'pupil.fits'
PathStaticOn = static.fits

[atmosphere]
Wavelength = {atmosphere.reference_wavelength.value}
Seeing = 0.7
r0_Value = {atmosphere.fried_parameter.value}
L0 = {atmosphere.outer_scale.value}
Cn2Heights = {atmosphere.layer_heights.value.tolist()}
Cn2Weights = {atmosphere.layer_weights.value.tolist()}
WindSpeed = {atmosphere.wind_speeds.value.tolist()}
WindDirection = {atmosphere.wind_directions.value.tolist()}

[sources_HO]
Zenith = {laser_r}
Azimuth = {laser_theta}
Height = {lgs.height.value}

[sources_LO]
Wavelength = {prototype.low_order_sources.wavelengths.value.tolist()}

[sensor_LO]
PixelScale = {detector.pixel_scale.value}
SigmaRON = {detector.read_noise.value}
ThresholdWCoG = {detector.threshold.value}

[DM]
OptimizationZenith = {fit_r}
OptimizationAzimuth = {fit_theta}
OptimizationWeight = {prototype.fitting.weights.value.tolist()}

[RTC]
LoopGain_HO = {ho.gain.value}
LoopGain_LO = {lo.gain.value}
SensorFrameRate_HO = {ho.frame_rate.value}
SensorFrameRate_LO = {lo.frame_rate.value}
LoopDelaySteps_HO = {ho.delay_frames}
LoopDelaySteps_LO = {lo.delay_frames}

[sampling_uncertainty_model]
modal_pupil_samples = 48
science_pupil_samples = 32
sensing_pupil_samples = 92
zernike_radial_order = 5
sensing_radial_nodes = 24
sensing_angular_nodes = 36
science_min_radial_nodes = 96
""",
        encoding="utf-8",
    )
    ini_path, parser = read_ini(path)
    base = Configuration.from_ini(parser, ini_path=ini_path)
    loaded = base.with_updates(ngs=prototype.ngs, zenith_angle=20 * u.deg)
    direct = replace(
        prototype,
        atmosphere=replace(atmosphere, seeing=0.7 * u.arcsec),
        telescope=replace(telescope, pupil_resolution=side, zenith_angle=20 * u.deg),
        pupil=pupil,
        low_order_detector=detector,
        sampling_uncertainty_model=SamplingSettings(
            modal_pupil_samples=48,
            science_pupil_samples=32,
            sensing_pupil_samples=92,
            zernike_radial_order=5,
            sensing_radial_nodes=24,
            sensing_angular_nodes=36,
            science_min_radial_nodes=96,
        ),
    )
    assert base.telescope.zenith_angle == 10 * u.deg
    assert base.ngs == ()
    assert loaded.ngs == direct.ngs
    assert loaded.sampling_uncertainty_model == direct.sampling_uncertainty_model
    assert loaded.pupil.pitch == direct.pupil.pitch
    np.testing.assert_array_equal(loaded.pupil.amplitude, direct.pupil.amplitude)
    np.testing.assert_array_equal(
        loaded.pupil.static_opd.value, direct.pupil.static_opd.value
    )

    projected = _project_sampling_configuration(loaded)
    airmass = 1 / np.cos(np.deg2rad(20))
    assert projected.high_order_sources.height.value == pytest.approx(
        lgs.height.value * airmass
    )
    np.testing.assert_allclose(
        projected.atmosphere.layer_heights.value,
        atmosphere.layer_heights.value * airmass,
    )
    seeing_r0 = (
        0.976 * atmosphere.reference_wavelength / (0.7 * u.arcsec).to_value(u.rad)
    )
    assert projected.atmosphere.fried_parameter.value == pytest.approx(
        seeing_r0.value * airmass ** (-3 / 5)
    )

    requests = []
    for config in (loaded, direct):
        covariance = gaussian_ngs_noise_covariance(
            config,
            ee=sampling_reference["guide_ee"] * u.one,
            fwhm=sampling_reference["guide_fwhm_mas"] * u.mas,
        )
        requests.append(
            replace(
                request,
                config=config,
                ngs_noise_covariance=covariance,
                positions=request.positions[:1],
            )
        )
    np.testing.assert_array_equal(
        requests[0].ngs_noise_covariance.value,
        requests[1].ngs_noise_covariance.value,
    )
    first, second = SamplingUncertaintyModel().predict_many(requests, diagnostics=True)
    assert (
        first.diagnostics.grids == second.diagnostics.grids == ((96, 192, 5.644),) * 2
    )
    for metric in first.metrics:
        a, b = first.metrics[metric], second.metrics[metric]
        np.testing.assert_allclose(a.mean.value, b.mean.value, rtol=1e-10)
        np.testing.assert_allclose(a.coefficient.value, b.coefficient.value, rtol=1e-10)


def test_sampling_settings_missing_section_and_partial_override():
    from dataclasses import replace

    from ao_core.models.config import Configuration, parse_ini
    from ao_core.models.uncertainty import SamplingSettings

    defaults = SamplingSettings()
    assert (
        Configuration.from_ini(parse_ini("[telescope]")).sampling_uncertainty_model
        == defaults
    )
    parsed = parse_ini("[sampling_uncertainty_model]\nzernike_radial_order = 9")
    assert Configuration.from_ini(parsed).sampling_uncertainty_model == replace(
        defaults, zernike_radial_order=9
    )


@pytest.mark.parametrize(
    "key", ["sensing_period_s", "spectral_interval_s", "science_angular_offset"]
)
def test_sampling_ini_rejects_internal_grid_settings(key):
    from ao_core.models.config import Configuration, parse_ini

    with pytest.raises(ValueError, match=key):
        Configuration.from_ini(parse_ini(f"[sampling_uncertainty_model]\n{key} = 1"))


def test_sampling_request_requires_valid_ngs_covariance(public_sampling_request):
    from dataclasses import replace

    request = public_sampling_request
    with pytest.raises(ValueError, match="Multiple guide stars"):
        replace(request, ngs_noise_covariance=None)
    with pytest.raises(ValueError, match="two axes per NGS"):
        replace(request, ngs_noise_covariance=np.eye(2) * u.mas**2)
    one_star = request.config.with_updates(ngs=request.config.ngs[:1])
    assert (
        replace(
            request, config=one_star, ngs_noise_covariance=None
        ).ngs_noise_covariance
        is None
    )
