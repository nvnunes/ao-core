# AO Scientific Models

[Model configuration](model-configuration.md) supplies unit-aware physical
inputs to these calculations. Import each component from its module.
See the [Python reference](reference/api.md) for their public APIs.

## Scientific Components

`PixelSpot` integrates an incident-normalized optical image over a 5×5
detector at supplied angular offsets. The integral leaves flux in
incident-photon units, preserving cropping losses. `DirectCentroid` calculates
thresholded centroid means and covariances using the conditional Poisson
and read-noise moments. Its optional photon Jacobian uses the same
denominator integral. For an incident-normalized supplied spot,
`SuppliedImageModalResponse` transfers selected pupil-mode derivatives to
the image and integrates them over aligned detector pixels. Multiply the
integrated flux and its derivatives by the incident photons per frame.
Evaluate `DirectCentroid.calculate(..., jacobian=True)` at those expected
counts and contract its Jacobian with the photon-count derivatives. The sampling uncertainty model
uses `sensor_response` with image-grid derivatives. It integrates those
derivatives over the detector before applying the centroid Jacobian.

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
`wind_aligned_tilt_covariance` integrates unfiltered and filtered channels
over the continuous frozen-flow spectrum. `sample_centroid_errors` draws
conditional detector errors for loop replay. `covariance_matched_displacement_otfs`
converts a command-displacement record to an OTF at the supplied covariance.

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

## High-Order PSFs

`HoPsfModel.predict()` takes a `HoPsfRequest` with resolved science
directions, one wavelength, and detector sampling. For diffraction, supply
the pupil with `atmosphere=None` and no mirrors. For corrected PSFs, supply
the atmosphere, guide stars, square lenslets, a ground-conjugate Gaussian
common DM, and the common high-order loop (`high_order_loop`). The default
phase-fit reconstructor also uses a fitting field. The calculation uses
instantaneous noise-free slope measurements and projected
pseudo-open-loop control. It includes differential refraction and correction
chromatism.

Use square lenslets of equal pitch whose nominal grid spans the telescope
diameter. The Fourier period, given by wavelength divided by angular pixel
scale in radians, must span a positive even number of lenslet pitches.

The common high-order loop requires a gain in [0, 1], a positive frame rate,
and a whole-frame delay. If the guide and science wavelengths differ, both
must be in the 0.3–2.5 µm refractivity range. The predictor derives the
refraction direction from the Cartesian science offset. For an on-axis
point, it uses the supplied polar azimuth, or zero if the input was purely
Cartesian.

For two-stage correction, supply a ground-conjugate Gaussian science channel
DM and set the residual high-order loop (`residual_high_order_loop`). The
science DM applies an open-loop command computed for each science direction.
Its gain must be nonnegative. With positive gain, give the delay in whole
common-loop frames. Any residual high-order loop frame rate supplied for
this case must match the common high-order loop. The model retains the
cross-covariance between the DM commands. Zero gain disables the science
DM correction.

For an OPD map that varies across the illuminated pupil, use
`static_opd_sampling="resampled"` to sample the map on the native optical
pupil with RMS compensation. Set
`"original_map"` to retain the original static diffraction and transfer the
coarse atmospheric coherence onto that grid. The latter requires the Fourier
period to span an even integer number of original pupil samples and to
cover both the original pupil and the output detector grid. The model uses
a chirp-Z transform for crops. `full_flux` refers to the complete original-map
plane.

Use `detector_origin="fft"` to sample at integer offsets from the native
FFT origin. Use `"geometric"` to sample at half-integer offsets around the
centre of an even detector window. The model samples directly on either
grid and returns `float64` images. Divide `prediction.psfs` by the shared
`prediction.full_flux` to express intensities as fractions of the total
uncropped flux.

Use `HoPsfModel(backend="cpu", precision="float32")` for mixed-precision
execution. It uses float32 for science DM spline sampling, phase–command
slope products, residual covariance blocks and atmospheric pupil-pair
integration. The phase–sensor-mean spectral products and inverse FFTs use
complex64.
CPU defaults to `precision="float64"`. Physical coordinates, spline
preparation, reconstruction solves, and two-stage command covariance
assembly use float64. CPU execution also sums cross moments in float64.
Static diffraction, original-map transfer and detector transforms also use
float64. Before image formation in mixed precision, the model evaluates up
to 64 pupil-pair phase variances in float64. If a variance falls within
eight times the estimated float32 roundoff, or if a float32 render fails
a phase-variance or intensity check, the model recomputes that point's
cross moments and renders it once in float64. A failed float64 check
raises an error.

Select `backend="mps"` to run phase–command slope queries and spectral
products, residual covariance assembly, and pupil-pair integration on
Metal. MPS defaults to and requires `precision="float32"`. It requires the
`metal` extra, an available MPS device, and a PyTorch build with Metal
shader compilation. Metal queries two field points at a time and retains
phase spline planes within a 512 MiB device cache. Cross moments stay on
Metal through pupil integration, with only the sampled rows copied for
the CPU precision check. The CPU also handles spline preparation,
reconstruction, float64 recomputation, and the final detector transform.

### Assemble the High-Order Components

Pass `components` to `HoPsfModel` to select the sensing, mirror,
reconstruction, and control calculations. The request still carries the
resolved physical inputs, including loop timing and mirror calibration.
The following assembly selects the defaults for a common DM and a science DM.
For common-only correction, omit `science_stage` and configure one DM.

```python
from dataclasses import replace

from ao_core.models.ho_psf import (
    DelayedResidualControl,
    GaussianDm,
    HoPsfComponents,
    HoPsfModel,
    HoPsfRequest,
    HoPsfStage,
    MatrixReconstructor,
    PhaseFitReconstructor,
    ProjectedPolc,
    ShackHartmannSensing,
)

components = HoPsfComponents(
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
request = HoPsfRequest(config=config)
model = HoPsfModel(components=components)
prediction = model.predict(request)

layout = model.command_layout(request, stage="common")
calibrated = replace(
    components,
    common_stage=replace(
        components.common_stage,
        reconstructor=MatrixReconstructor(matrix=matrix, layout=layout),
    ),
)
prediction = HoPsfModel(components=calibrated).predict(request)
```

Arrange the supplied `matrix` with one row per active actuator and columns
ordered by guide, then x or y slope block, then valid lenslet. Use
`command_layout()` to get those orders from the resolved geometry. The
operator maps phase gradients in radians per metre to actuator phase in
radians at the same wavelength. Pass numerical matrix values in metres.
A fixed common reconstructor can omit the fitting field. Include any
command-space tilt projection in the supplied matrix. Sensor-mean removal,
loop gain and delay, and the
configured optical tilt policy apply during prediction. When the common
reconstructor changes, the model also updates the residual slopes driving
the science DM and propagates both mirrors' correlations with the atmosphere.

This example uses a resolved INI with nominal DM geometry and no varying
static OPD. For a configuration with calibrated DM geometry, supply
`mirror_geometries` before prediction. [Model configuration](model-configuration.md)

```python
from ao_core.models.config import load_ao_configuration
from ao_core.models.ho_psf import HoPsfModel, HoPsfRequest, decode_ho_psf

config = load_ao_configuration("system.ini", decode=decode_ho_psf)
prediction = HoPsfModel().predict(HoPsfRequest(config=config))
psfs = prediction.psfs / prediction.full_flux
```

## Sampling Uncertainty

`SamplingUncertaintyModel` predicts finite-exposure sampling uncertainty. Supply a
shared `Configuration` and specify the science wavelength, Cartesian field
positions, and exposure durations on `UncertaintyRequest`. The model derives
the PSF pixel scale from the wavelength and pupil Fourier grid.

```python
from astropy import units as u

from ao_core.io.ini import read_ini
from ao_core.models.config import Configuration, NaturalGuideStar
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
