# HDF5 Result Format

This reference describes the storage contract for
[`ao_core.io`](../io.md). It specifies the standard dataset paths, units
and axes, and how producers declare additional fields.

## Observation Axis

`/options/wavelength` is required and holds real numbers in micrometres.
Its shape determines the layout for the file. A batch has at least one
observation.

| Layout | Shape of `/options/wavelength` | Where the observation axis appears |
| --- | --- | --- |
| Standalone | Scalar `()` | Per-observation datasets contain the data for one observation directly. |
| Batch | `(observation,)`, including `(1,)` | Each per-observation dataset has the same leading axis. |

The wavelength is the only field required for every AO Core file. A stored
output can require related fields, such as apertures for EE.

Shared datasets such as the base science grid and telescope pupil have the
same shape in either layout. The wavelength rule also works when the file
contains no PSF pixels.

The tables below give per-observation shapes. A batch prepends the observation
axis to each field in `/options`, per-observation metadata and output datasets.
`science` is the number of points in the base science grid, `aperture` is the
number of EE apertures, and `ngs` is the number of NGS slots.

## Units and Precision

A quantity dataset carries a `units` attribute with its canonical Astropy
unit string. A scientific dimensionless value uses `1`. HDF5 values without
units, including PSF pixels, identifiers, flags and strings, are read as plain
values or NumPy arrays. Text is stored as UTF-8.

Standard physical writes accept Astropy quantities equivalent to the units
below. For new files, `/simulation/precision` selects `float32` or `float64`
for PSFs, statistics and numeric metadata and defaults to `float32`. Input
settings and options retain their field-specific types. Diagnostics follow
their own dtype declarations. Reads use the data types stored in the file.

## Run Settings

The `/simulation` group contains settings shared by a file. A scalar field
within this group has the same shape for a standalone file and a batch.

| Field | Meaning |
| --- | --- |
| `name`, `version` | Producer name and version. |
| `base_config` | Base configuration text supplied by the producer. |
| `precision` | Floating-point result precision. |
| `diagnostics_level` | Producer-selected diagnostic level, using `none`, `validation` or `debug`. |
| `ngs_mag_standard` | Photometric standard of `/options/ngs_magnitude`. |
| `extra_stat_fields`, `meta_fields`, `diagnostic_fields` | Extension declarations, each stored as a group of named fields. |

Producer-specific settings, such as input file paths or model parameters,
appear alongside these fields under producer-defined names.

## Shared Setup

| Field under `/setup` | Shape | Unit | Meaning |
| --- | --- | --- | --- |
| `sci_r` | `(science,)` | arcsec | Base science-grid radius, in output order. |
| `sci_theta` | `(science,)` | deg | Base science-grid angle, in output order. |
| `ee_apertures` | `(aperture,)` | mas | Aperture sizes in the order used by `/stats/ee`. |
| `ee_geometry`, `ee_center` | Scalar text | | Aperture geometry and centring used for EE. |
| `peak_method` | Scalar text | | Peak estimator used for the PSF measurements. |
| `fwhm_summary` | Scalar text | | Rule used to reduce FWHM to one value per science point. |
| `ngs_magnitude_zeropoint` | Scalar | photon / (s m²) | Optional incident photon flux per area at magnitude zero. |

Provide `sci_r` and `sci_theta` together with matching lengths for
science-indexed outputs. Provide `ee_apertures` when storing EE. The other
measurement settings record the methods used by the producer. AO Stats
defines the measurement methods and their setting values.

Producers can add shared model or instrument inputs to this group.

## Observation Options

| Field under `/options` | Shape per observation | Unit | Meaning |
| --- | --- | --- | --- |
| `wavelength` | Scalar | `um` | Required science wavelength. |
| `zenith_angle` | Scalar | deg | Zenith angle for the observation. |
| `ngs_r`, `ngs_theta` | `(ngs,)` | arcsec, deg | Natural guide-star positions. |
| `ngs_magnitude` | `(ngs,)` | mag | Magnitudes in the same slot order as the NGS positions. |

The NGS fields form an optional triplet. When present, they share a shape.
A batch can pad shorter asterisms with slots that have `NaN` in all three
arrays. Producers can add further per-observation options as datasets directly
under `/options`. In a batch, the first axis of each option matches the length
of `/options/wavelength`.

## Physical Metadata

| Field under `/meta` | Shape | Unit | Scope | Initial value |
| --- | --- | --- | --- | --- |
| `pixel_scale` | Scalar | mas | Per observation | `NaN` for each observation |
| `tel_diameter` | Scalar | m | Shared across the file | `NaN` |
| `tel_pupil` | `(pupil_y, pupil_x)` | `1` | Shared across the file | Empty `(0, 0)` array |

Initialization creates these three datasets whether or not PSF storage is
enabled. When PSF storage is enabled, every write supplies all three values.
Otherwise a write may supply any of the physical fields. An omitted pixel
scale uses the fill value for that observation. Shared fields take their
values from the first write that supplies them.

Additional scalar metadata is declared with `/simulation/meta_fields`, which
maps names to unit strings. An empty unit string in the declaration represents
a plain numeric scalar whose dataset has no `units` attribute. This differs
from a dimensionless quantity declared with `1`.

## PSFs and Statistics

| Path | Shape per observation | Unit | Contents |
| --- | --- | --- | --- |
| `/psfs/data` | `(science, image_y, image_x)` | | PSFs in science-grid order. |
| `/stats/ee` | `(science, aperture)` | `1` | Fraction of energy in each aperture. |
| `/stats/fwhm` | `(science,)` | mas | Full width at half maximum. |
| `/stats/sr` | `(science,)` | `1` | Strehl ratio. |

The producer selects which outputs to store. The EE aperture axis is
present even for a single aperture. The PSF and pupil grids have independent
sampling.

Additional statistics use `/simulation/extra_stat_fields`, which maps field
names to unit strings. Each extra statistic is stored under `/stats/<name>`
with shape `(science,)` per observation. A producer can, for example,
declare `jitter` in milliarcseconds.

## Diagnostics

`/simulation/diagnostic_fields` declares producer-defined datasets below
`/diagnostics`. Each declaration has a path relative to `/diagnostics` and
leaves named `dtype`, `shape` and, for floating-point fields, `unit`.
The declared shape excludes the observation axis. An empty shape describes a
scalar. Shape elements can be positive integers or the tokens `num_sci`
and `num_ngs`. The tokens resolve from the science grid in `/setup` and the
NGS slot count in `/options`, including padding. An unresolved token or a
resolved size of zero is a schema error.

On disk, `shape` is a one-dimensional dataset. Shapes with only integer
dimensions are stored as integer vectors. If a shape includes a symbolic
token, all entries are written as UTF-8 strings, so `("num_sci", 2, 2)`
is stored as `["num_sci", "2", "2"]`. The reader interprets decimal strings
as integers before resolving symbolic sizes. A scalar uses an empty vector.

| Declared dtype | Unit declaration | Initial or cleared value |
| --- | --- | --- |
| `float32`, `float64` | Required, including `1` for dimensionless values | `NaN` |
| `int32`, `int64` | Absent | `0` |
| `bool` | Absent | `False` |
| `str` | Absent | Empty UTF-8 string |

For example, the declaration for `quality/valid_mask` can specify
`dtype="bool"` and `shape=("num_sci",)`. The dataset then appears at
`/diagnostics/quality/valid_mask` with shape `(observation, science)` in a
batch. The values supplied on a write must match the declared paths exactly.
The producer chooses the fields for each level. When no fields are declared,
the group is omitted.

## Extension Names

Additional statistics and metadata use direct dataset names beginning with
an ASCII letter or underscore and containing only ASCII letters, digits and
underscores. Standard names in `/meta` and `/stats` are reserved. Relative
diagnostic paths may use subgroups. Each path must be unique, and a dataset
path cannot also serve as a subgroup. Conflicting declarations raise
`ValueError`.

Readers use the declarations to recover extension values and units. The
producer defines their scientific meanings.

## Output Allocation

When PSF storage is enabled, initialization creates an empty `/psfs` group.
The first PSF write supplies the dimensions of `/psfs/data`. The shared
`/meta/tel_pupil` dataset takes its shape from the first pupil write. Reads
return the empty group or array as stored. Later writes in the same batch
must agree with the stored PSF shape and any shared telescope metadata they
supply.

Unwritten or cleared batch rows contain fill values in their allocated
per-observation outputs. Floating PSFs, statistics and per-observation
metadata use `NaN`. Diagnostic fill values follow the dtype table above.
