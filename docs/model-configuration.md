# AO Model Configuration

## Physical Inputs

`ao_core.models.config` provides unit-aware physical records for adaptive
optics calculations. Load the shared fields from an existing AO INI or
construct the records from quantities in memory.

```python
from astropy import units as u

from ao_core.models.config import load_ao_configuration

system = load_ao_configuration("system.ini")
phase_wavelength = system.atmosphere.reference_wavelength.to(u.nm)
layer_heights = system.atmosphere.layer_heights.to(u.km)
```

Construct the same records directly when the physical inputs are already in
memory. For example, an adapter can pass its supplied layer heights and Fried
parameter with their original units.

```python
from astropy import units as u

from ao_core.models.config import Configuration, AtmosphereConfig, SourceConfig

system = Configuration(
    atmosphere=AtmosphereConfig(
        reference_wavelength=500 * u.nm,
        fried_parameter=18 * u.cm,
        outer_scale=30 * u.m,
        layer_weights=[0.4, 0.6] * u.one,
        layer_heights=[0, 12.5] * u.km,
        wind_speeds=[5, 15] * u.m / u.s,
        wind_directions=[0, 90] * u.deg,
    ),
    high_order_sources=SourceConfig(
        directions=[[30, 0], [0, 30]] * u.arcsec,
        wavelengths=[589] * u.nm,
        height=90 * u.km,
    ),
)
```

Both paths store supplied quantities in the calculation units listed in the
[Python reference](reference/api.md). Omitted groups and fields are `None`.
Callers determine the inputs and defaults required by their model.

- `[atmosphere] Wavelength` defines the phase reference. Source wavelengths
  remain separate. Layer heights, wind directions and turbulence weights
  retain their supplied geometry and order. Seeing and the explicit Fried
  parameter are exposed separately so models can apply their own conversion
  and precedence. An INI with positive seeing and a zero Fried parameter
  leaves the Fried parameter unspecified. If both existing Fried parameter
  spellings occur, the shared reader uses `r0_value` before `r0_Value`. Both
  keys stay available to native readers.
- Source positions and DM fitting directions use sky-plane `[x, y]` pairs
  in arcseconds. INI azimuth increases from positive x toward positive y.
  Callers apply any projection their calculations require.
- `[sensor_HO] GridOffsets` uses lenslet-pitch units in the INI. The reader
  returns physical `[x, y]` offsets in metres using `SizeLenslets` or the
  telescope diameter divided by `NumberLenslets`. Add the offsets to the
  nominal lenslet positions.
- Detector inputs include sampling and noise. Loop inputs include scalar
  gain and frame rate, delay in frames, and the corresponding frame interval.
  A native low-order gain policy such as `optimize` leaves the numeric gain
  unspecified. The original policy stays available in the INI parser.

## Reusing a Parsed INI

Call `read_ini` once when a consumer needs physical and application fields
from the same file. It returns the resolved source path and the
case-sensitive parser.

```python
from ao_core.models.config import Configuration, read_ini

ini_path, ini_parser = read_ini("system.ini")
system = Configuration.from_ini(ini_parser, ini_path=ini_path)
```

`parse_ini` reads INI text. `read_ini_scalar` and `read_ini_array` read finite
numeric values. `resolve_ini_path` resolves a file path relative to the
source INI. `ini_to_mapping` builds an owned mapping for a downstream
consumer using a decoder supplied by the caller.
