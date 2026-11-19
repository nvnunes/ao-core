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

## Sampling Predictor

`SamplingConfiguration` combines the shared physical configuration with
sampled pupil amplitude, static OPD, a native frame interval and guide-star
descriptors for the design prior. `SamplingRequest` specifies Cartesian
positions, wavelength, science pixel scale and exposure windows.
`SamplingPredictor` reuses detector and Fourier-grid preparation across
compatible requests. Fast contracts the modal covariance through finite
Gaussian order. Balanced keeps all orders of the modal covariance. Full
uses the restored pupil covariance for all orders. These automatic models
share the restored mean PSF and measure EE100 in a 100 mas square centered
at the nominal source. They use the four-beacon geometry of `FieldCovariance`.
Balanced and Full submitted together share a Sobol proposal where physical
configuration, science wavelength, pixel scale, field point and grid match.

For peak-centered EE100 and an explicit Fourier grid, pass
`SamplingNumerics` and select `limited-order`, `all-order` or `full-pupil`.
The explicit all-order and full-pupil options evaluate one native frame.
Those options and limited order with `restore_mean=True` require CPU.

CPU uses float64. The `metal` installation extra supplies PyTorch for
`device="mps"`, which uses float32 for Metal contractions. `dtype` can
explicitly confirm the selected device precision.
Automatic grids accept whole native-frame exposures up to 20 s, with
Metal available through 10 s. Nonfinite Metal remainders use CPU quadrature.

Use `predict_fast_means` with one Fast exposure to obtain only the mean
metric fields.

Results retain point and exposure order. A metric's `mean` holds the
first requested exposure's estimate at each point. `mean_by_exposure`,
one-realization variance and fractional sampling coefficient use
`(exposure, point)` axes. The coefficient is in square-root seconds. For a single
realization of duration (T), divide the coefficient by
`sqrt(T)` for relative standard deviation. To average a supplied
lag covariance over independent realizations, use `exposure_variance`.
