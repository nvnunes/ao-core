# AO Scientific Models

[Model configuration](model-configuration.md) supplies unit-aware physical
inputs to these calculations. Import each component from its module.
See the [Python reference](reference/api.md) for their public APIs.

## Scientific Components

`PixelSpot` integrates an incident-normalized optical image over a 5×5
detector at supplied angular offsets. The integral retains light lost
outside the image or detector window. `DirectCentroid` calculates
thresholded centroid means and covariances using the conditional Poisson
and read-noise moments. Its optional photon Jacobian uses the same
denominator integral.

```python
import numpy as np
from astropy import units as u

from ao_core.models.sensing import DirectCentroid, PixelSpot

y, x = np.indices((31, 31)) - 15
image = np.exp(-(x**2 + y**2) / 24)
image *= 0.82 / image.sum()
spot = PixelSpot(image, 10 * u.mas / u.pix, 200 * u.photon, 50 * u.mas / u.pix)
flux = spot.flux(np.array([[0, 0], [20, -10]]) * u.mas)
calibration = np.eye(2) * 50 * u.mas / u.pix
moments = DirectCentroid(200 * u.photon, calibration).calculate(flux, jacobian=True)
```

`FixedGainIntegrator` evaluates delayed command propagation using the
actual two-axis feedback matrix. `solve_lo_reconstructor` accepts
atmospheric and design-noise covariances as angular-squared quantities. Single-star
reconstruction returns the identity. Multistar reconstruction supports a
direct solve or an explicit SVD cutoff. These operations default to
float64 and support explicit float32.

`TipTiltControl` combines the solved reconstructor and NGS sensing gains
with a `FixedGainIntegrator`. Its scalar feedback describes signal filtering,
and `command_noise_covariance` propagates noise through the actual feedback.
`assemble_tilt_residual_covariances` combines the supplied angular command
and cross-covariance terms with that command noise. Applications evaluate
those inputs and use the resulting residual covariances.

`prepare_modal_pupil` builds piston-free modes on the sampled aperture.
`FieldCovariance` uses the supplied atmospheric layers, projected beacon
height, fitting weights and controller settings. It retains the shared
pupil sampling and phase reference when evaluating a different science band.
Supply exactly four laser guide stars and a finite, positive beacon height
that already includes the projection for the observing geometry.

For custom metric covariance, use `phase_responses` to combine the modal
basis, zero-lag covariance, metric influence kernels and pupil coherence.
Pass those responses and the modal lag covariance to `metric_covariance`.

The lower-level controller kernels operate on numerical arrays.
`integrator_transition` and `integrator_noise_covariance` assemble the
command-history transition and stationary noise covariance. `state_matrix`
and `settling_cut` cover scalar feedback and its settling window.
`scalar_filter` and `filter_input` apply scalar and coupled delayed feedback.
`lo_reconstructor` accepts numerical covariance arrays in mas². Use
`solve_lo_reconstructor` for the equivalent quantity-based API.

## Sampling Uncertainty

`SamplingUncertaintyModel` predicts finite-exposure sampling uncertainty. Supply a
shared `Configuration` and specify the science wavelength, Cartesian field
positions, and exposure durations on `UncertaintyRequest`. The model derives
the PSF pixel scale from the wavelength and pupil Fourier grid.

```python
from astropy import units as u

from ao_core.models.config import Configuration, NaturalGuideStar, read_ini
from ao_core.models.uncertainty import SamplingUncertaintyModel, UncertaintyRequest

ini_path, ini_parser = read_ini("system.ini")
base_config = Configuration.from_ini(ini_parser, ini_path=ini_path)
ngs = [NaturalGuideStar(position=[15, -5] * u.arcsec, flux=150_000 * u.photon / u.s)]
config = base_config.with_updates(ngs=ngs, zenith_angle=20 * u.deg)
request = UncertaintyRequest(
    config=config,
    wavelength=1.65 * u.um,
    positions=[[0, 0], [25, 0]] * u.arcsec,
    exposures=[0.5, 1.0] * u.s,
    model_type="fast",
)
result = SamplingUncertaintyModel(device="cpu").predict(request)
```

`with_updates()` returns a new configuration. Supply the NGS collection
in Cartesian angular order with incident photon rates at the NGS sensor.
The model derives the photon counts and native frame interval from the
common HO and LO loop rate. The five-pixel detector uses `[sensor_LO]`
`PixelScale`, `SigmaRON` and `ThresholdWCoG`.

`Configuration.from_ini()` resolves relative pupil, apodizer and static
OPD paths against `ini_path`. It uses `[telescope] TelescopeDiameter`
and `Resolution` for sample pitch. For in-memory use, pass a `Pupil`
with amplitude, pitch and optional static OPD to `Configuration()` or
`with_updates(pupil=pupil)`. An explicit zenith angle takes precedence
over `[telescope] ZenithAngle`. During preparation the sampling model
resolves zenith `Seeing` or the Fried parameter using the P3 convention,
then projects the atmosphere and finite beacon height once.

Multiple NGSs require `ngs_noise_covariance` on the request. This is
the angular measurement-noise covariance used to determine the
reconstructor weights. Its axes are interleaved x/y in configured
NGS order. A calibrated covariance can be passed directly. To calculate
one from a supplied EE/FWHM convention, use
`gaussian_ngs_noise_covariance(config, ee=ee, fwhm=fwhm)`.

`predict_many()` retains input request order and reuses compatible
preparation across points, wavelengths and exposures. Fast contracts the
modal covariance through finite Gaussian order. Balanced keeps all orders
of the modal covariance. Full uses the restored pupil covariance for all
orders. All three use the restored mean PSF and measure EE100 in a 100 mas square
centered at the nominal source. The sampling calculation uses four laser
beacons. Fast optionally accepts `FastSettings(remove_science_tilt=True)`
as `model_settings`.

CPU uses float64. The `metal` installation extra supplies PyTorch for
`device="mps"`, which uses float32 for Metal contractions. `dtype` can
explicitly confirm the selected device precision.
The spatial grid schedule covers whole native-frame exposures up to 20 s.
Metal is used through 10 s, with CPU quadrature for nonfinite Metal remainders.

### Sampling Settings

The optional `[sampling_uncertainty_model]` section maps directly to
`config.sampling_uncertainty_model`, a `SamplingSettings` record. Both
construction paths use the defaults below for omitted values.

```ini
[sampling_uncertainty_model]
modal_pupil_samples = 64
science_pupil_samples = 64
sensing_pupil_samples = 96
zernike_radial_order = 11
sensing_radial_nodes = 40
sensing_angular_nodes = 48
science_min_radial_nodes = 80
science_min_angular_nodes = 192
```

Pupil sample counts are per side. The science node counts are lower
bounds applied to the exposure-dependent spatial grid. The science Fourier
period is the larger of twice the exposure and four pupil crossing times
at the slowest nonzero wind speed. It is rounded up to whole native frames.
For an entirely stationary atmosphere, the period is twice the exposure.
The model derives the occupied frequency bins and evaluates the lags at
the native frame cadence.

### Predictions

An `UncertaintyPrediction` retains point and exposure order. Each
`UncertaintyMetricPrediction.mean` gives one reference mean per field point.
The model evaluates it at zero lag on the configured baseline spatial Fourier
grid using `science_min_radial_nodes` and `science_min_angular_nodes`. Variance
and sampling uncertainty coefficient have `(exposure, point)` axes. For each
duration `T`, the coefficient is `sqrt(T * variance) / mean` in square-root
seconds. Divide by `sqrt(T)` for the relative standard deviation of one
realization.

Pass `diagnostics=True` to `predict()` or `predict_many()` to inspect the
temporal quadrature grids and their grid-specific means. The diagnostic grids
give radial and angular node counts and periods in seconds, in exposure order.
`diagnostics.mean_by_exposure` maps each metric to its means on those grids
with `(exposure, point)` axes. By default `diagnostics` is `None`.

To average a supplied lag covariance over independent realizations, use
`exposure_variance`.
