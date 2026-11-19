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
    # Pinned e011 CPU Fast at K and Gb, indices 55/95, 2/4 ms.
    # Capture recipe: GIRMOS verification capture_public_sampling_reference.py.
    from pathlib import Path

    with np.load(Path(__file__).with_name("sampling_reference.npz")) as data:
        yield data


@pytest.fixture(scope="module")
def public_sampling_request(sampling_reference):
    from ao_core.models.config import (
        AOConfiguration,
        AtmosphereConfig,
        DetectorConfig,
        FittingConfig,
        LoopConfig,
        SourceConfig,
        TelescopeConfig,
    )
    from ao_core.models.uncertainty import SamplingConfiguration, SamplingRequest

    d = sampling_reference
    loops = [
        LoopConfig(
            gain=gain * u.one,
            frame_rate=(1 / float(d["frame_s"])) * u.Hz,
            delay_frames=int(delay),
        )
        for gain, delay in zip(d["gains"], d["delay_frames"], strict=True)
    ]
    physical = AOConfiguration(
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
            directions=d["guide_xy_arcsec"] * u.arcsec,
            wavelengths=d["guide_wave_m"] * u.m,
        ),
        low_order_detector=DetectorConfig(
            pixel_scale=d["pitch_mas"] * u.mas / u.pix,
            photons_per_frame=d["photons"] * u.photon,
            read_noise=d["read_noise"] * u.electron,
            threshold=d["threshold"] * u.electron,
        ),
        high_order_loop=loops[0],
        low_order_loop=loops[1],
    )
    configuration = SamplingConfiguration(
        physical=physical,
        pupil_amplitude=d["pupil_amplitude"],
        pupil_support=d["support_m"] * u.m,
        static_opd=d["opd_nm"] * u.nm,
        guide_star_ee=d["guide_ee"] * u.one,
        guide_star_fwhm=d["guide_fwhm_mas"] * u.mas,
        frame_interval=d["frame_s"] * u.s,
    )
    return SamplingRequest(
        configuration=configuration,
        positions=d["points"] * u.arcsec,
        exposures=d["exposures_s"] * u.s,
        wavelength=d["wavelength_m"][0] * u.m,
        pixel_scale=d["pixel_mas"][0] * u.mas / u.pix,
    )


def test_public_sampling_predictor_matches_pinned_fields_and_preserves_order(
    public_sampling_request,
    sampling_reference,
):
    from dataclasses import replace

    from ao_core.models.uncertainty import SamplingPredictor

    k = public_sampling_request
    d = sampling_reference
    order = [1, 0, 1]
    gb = replace(
        k,
        wavelength=d["wavelength_m"][1] * u.m,
        pixel_scale=d["pixel_mas"][1] * u.mas / u.pix,
        positions=k.positions[order],
        exposures=k.exposures[order],
    )
    predictor = SamplingPredictor()
    results = tuple(predictor.predict_many((k, gb)))
    metrics = ("sr", "ee100", "fwhm_mas")
    for band, (request, result, index) in enumerate(
        zip((k, gb), results, ([0, 1], order), strict=True)
    ):
        assert result.request is request
        window_and_point_order = np.ix_(index, index)
        for metric_index, name in enumerate(metrics):
            metric = result.metrics[name]
            expected = d[f"predicted_mean_{band}"][..., metric_index][
                window_and_point_order
            ]
            coefficient = (
                d[f"coefficient_{band}"][..., metric_index][window_and_point_order]
                / 100
            )
            unit = u.mas if name == "fwhm_mas" else u.one
            np.testing.assert_allclose(
                metric.mean.to_value(unit), expected[0], rtol=1e-10
            )
            np.testing.assert_allclose(
                metric.mean_by_exposure.to_value(unit), expected, rtol=1e-10
            )
            np.testing.assert_allclose(
                metric.coefficient.to_value(u.s**0.5), coefficient, rtol=1e-10
            )
            variance = (coefficient * expected) ** 2 / request.exposures.value[:, None]
            np.testing.assert_allclose(
                metric.variance.to_value(unit**2), variance, rtol=2e-10
            )

    single = replace(k, exposures=k.exposures[:1])
    prediction = predictor.predict(single)
    mean_only = predictor.predict_fast_means(single)
    for index, name in enumerate(metrics):
        expected = d["predicted_mean_0"][0, :, index]
        np.testing.assert_allclose(
            prediction.metrics[name].mean.value, expected, rtol=1e-10
        )
        np.testing.assert_allclose(mean_only[name].value, expected, rtol=1e-10)


def test_long_fast_grid_is_compact_with_unrelated_short_balanced_request(
    public_sampling_request,
    monkeypatch,
):
    from dataclasses import replace

    from ao_core.models.uncertainty import SamplingPredictor

    class ReachedCompactGrid(Exception):
        pass

    def check_route(grid, windows, bands, representatives, device, *, compact):
        assert grid == (320, 768, 40)
        assert device == "cpu"
        assert compact
        raise ReachedCompactGrid

    def dense_route(*args, **kwargs):
        pytest.fail("Long Fast request entered the dense grid")

    request = public_sampling_request
    predictor = SamplingPredictor()
    monkeypatch.setattr(predictor, "_fast_points", check_route)
    monkeypatch.setattr(predictor, "_prepare_grid", dense_route)
    with pytest.raises(ReachedCompactGrid):
        tuple(
            predictor.predict_many(
                (
                    replace(request, exposures=[20] * u.s),
                    replace(request, model="balanced", exposures=[0.002] * u.s),
                )
            )
        )


def test_accelerated_fast_bounds_simultaneous_band_optics(
    public_sampling_request, monkeypatch
):
    from dataclasses import replace
    from types import SimpleNamespace

    from ao_core.models.uncertainty import SamplingPredictor

    batch_shapes = []

    def evaluate(grid, windows, bands, representatives, device, *, compact):
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

    predictor = SamplingPredictor(device="mps")
    monkeypatch.setattr(predictor, "_fast_points", evaluate)
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


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("modal_order", 5.0),
        ("tail_order", 6.0),
        ("sensor_workers", 1.5),
        ("sobol_workers", True),
    ],
)
def test_grid_settings_require_integer_orders_and_worker_counts(name, value):
    from ao_core.models.uncertainty import SamplingNumerics

    settings = {
        "physical_side": 64,
        "optical_side": 48,
        "sensor_side": 96,
        "modal_order": 5,
        "radial_order": 40,
        "angular_order": 96,
        "spectral_period_s": 20,
        "spectral_interval_s": 0.01,
    }
    with pytest.raises(ValueError, match=name):
        SamplingNumerics(**(settings | {name: value}))
