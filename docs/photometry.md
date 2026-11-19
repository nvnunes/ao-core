# NGS Photometry

`ao_core.photometry` converts natural guide-star (NGS) magnitudes to
pre-aperture photon flux and wavefront-sensor (WFS) photon counts and rates.
The caller supplies the photometric calibration. See the
[Python reference](reference/api.md) for signatures and docstrings.

## Convert Magnitudes to WFS Photon Counts

```python
import numpy as np
from astropy import units as u

from ao_core.photometry import (
    WFSPhotometryConfig,
    magnitudes_to_effective_photons_per_second,
    magnitudes_to_photons_per_frame,
)

calibration = WFSPhotometryConfig(
    telescope_diameter=8 * u.m,
    n_channels=16,
    frame_rate=500 * u.Hz,
    zeropoint=3e10 * u.photon / (u.m**2 * u.s),
)
magnitudes = np.array([0, 14]) * u.mag
per_frame = magnitudes_to_photons_per_frame(magnitudes, calibration)
effective_rate = magnitudes_to_effective_photons_per_second(magnitudes, calibration)
```

The magnitude-zero source gives 15,000,000 photons per frame and
7,500,000,000 effective photons per second.

## Calibration

Magnitude-to-flux conversions use only the supplied zeropoint. The WFS
conversions use `WFSPhotometryConfig`.

| Field | Required value |
| --- | --- |
| `telescope_diameter` | Positive scalar Astropy quantity convertible to metres |
| `n_channels` | Positive finite real numeric scalar giving the number of WFS channels across the diameter |
| `frame_rate` | Positive scalar Astropy quantity convertible to hertz |
| `zeropoint` | Positive scalar zero-magnitude photon flux convertible to `photon / (m2 s)`. It may include instrument throughput. |

Pre-aperture flux is the zeropoint multiplied by `10**(-0.4 * magnitude)`.
Multiply that flux by the nominal subaperture area
`(telescope_diameter / n_channels)**2` and divide by the frame rate to get
unrounded photons per frame. Multiply the per-frame counts by the frame
rate to get the WFS-effective photon rate. The inverse functions reverse
the corresponding conversion.

## Units and Shapes

| Function | Input → output |
| --- | --- |
| `magnitudes_to_photon_flux` | Magnitude → pre-aperture flux |
| `photon_flux_to_magnitudes` | Pre-aperture flux → magnitude |
| `photon_flux_to_photons_per_frame` | Pre-aperture flux → photons per frame |
| `photons_per_frame_to_photon_flux` | Photons per frame → pre-aperture flux |
| `magnitudes_to_photons_per_frame` | Magnitude → photons per frame |
| `photons_per_frame_to_magnitudes` | Photons per frame → magnitude |
| `magnitudes_to_effective_photons_per_second` | Magnitude → WFS-effective photon rate |

Magnitudes use `mag`, pre-aperture flux uses `photon / (m2 s)`, and the
WFS-effective rate uses `photon / s`. Per-frame counts use `photon` because
frames are dimensionless sample counts. Inputs accept equivalent Astropy
units. Magnitude/flux and flux/frame conversions retain the input shape.
The composed magnitude/frame and effective-rate conversions return
one-dimensional vectors, including a length-one vector for scalar input.

Inputs must be real and finite. Photon inputs must be nonnegative. Inverse
magnitude conversions apply a floor of `1e-30 photon / (m2 s)` to
pre-aperture flux before taking its logarithm.
