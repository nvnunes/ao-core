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
[Python reference](reference/api.md). Omitted optional records and fields are
`None`. The mirror tuple is empty when there are no mirror inputs. Callers
determine the inputs and defaults required by their model.

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
  Source records also retain the supplied zenith offsets and azimuths.
  Callers apply any projection their calculations require.
- `[sensor_HO] GridOffsets` uses lenslet-pitch units in the INI. The reader
  returns physical `[x, y]` offsets in metres using `SizeLenslets` or the
  telescope diameter divided by `NumberLenslets`. Add the offsets to the
  nominal lenslet positions.
- Detector inputs include sampling and noise. The low-order detector also
  exposes `NumberLenslets` per sensor. Loop inputs include scalar
  gain and frame rate, delay in frames, and the corresponding frame interval.
  A native low-order gain policy such as `optimize` leaves the numeric gain
  unspecified. The original policy stays available in the INI parser.
- `[sensor_science] PixelScale` and `FieldOfView` give the science sampling
  in mas per pixel and the image side length in pixels.
- `[DM]` supplies `NumberActuators`, `DmPitchs`, `DmHeights`, `InfModel`,
  and `InfCoupling` to `Configuration.mirrors` in INI order. The reader
  applies one influence model name to the listed mirrors. Supplied numeric
  vectors have one value per mirror. The existing `[DM] Optimization*`
  fields supply the separate fitting directions and weights.

## Extending Configuration

Pass a decoder to `load_ao_configuration()` when a model or application
needs additional fields from the same INI. The loader decodes the shared
`Configuration` once, then passes it and the parsed source to the decoder.

```python
from dataclasses import dataclass

from ao_core.models.config import (
    Configuration,
    IniSource,
    load_ao_configuration,
    read_ini_scalar,
)


@dataclass(frozen=True, kw_only=True, eq=False)
class ReconstructionConfig(Configuration):
    regularization: float


def decode_reconstruction(source: IniSource, config: Configuration) -> ReconstructionConfig:
    regularization = read_ini_scalar(
        source.parser["sensor_HO"],
        "ReconstructionRegularization",
        label="sensor_HO.ReconstructionRegularization",
    )
    return ReconstructionConfig.from_config(config, regularization=regularization)


config = load_ao_configuration("system.ini", decode=decode_reconstruction)
```

`config.telescope` and `config.regularization` are both available on the
returned object. A further decoder can extend `ReconstructionConfig` using
its own subtype's `from_config()` method. Each extension supplies its added
fields and carries forward the resolved values from the preceding config.

Use `prepare_ini(source)` to update the parser before shared decoding, for
example to bind an archived pupil path to its retained file. File references
are resolved relative to `source.path`. The original INI text remains in
`source.text` for provenance.

Set `load_pupil=False` when another provider supplies the pupil. The loader
still decodes the shared INI metadata and passes it to the extension.

## Reusing a Parsed INI

Use a record decoder such as `TelescopeConfig.from_ini(parser)` or
`SourceConfig.from_ini(parser, "sources_science")` when a consumer needs
one group of shared fields from an already parsed INI.

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

Use `read_ini_source()` when a caller also needs the original text. For an
already parsed effective INI, pass its parser and source path to
`Configuration.from_ini()`. An `IniSource` built from those values can be
passed directly to an extension decoder with the resulting configuration.
