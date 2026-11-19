# Result I/O

This guide describes the `ao_core.io` interface for standalone
observations and batches. See the [HDF5 result format](reference/hdf5.md) for
dataset paths, units and axis order.

## Open a File

Pass an open `h5py.File` to the I/O functions. The caller selects the file
mode and controls when it closes. Use mode `"x"` to make a new file or
`"r+"` to write into an existing batch. The functions accept mappings and
arrays with Astropy quantities for physical values. The same functions work
with both layouts.

| Function | Use |
| --- | --- |
| `initialize_result(file, *, simulation, setup, options, save_psfs=False, standard_stats=())` | Write the shared settings and observation options into an empty file. Allocate output datasets whose dimensions are known. |
| `write_observation(file, *, meta, psfs=None, stats=None, diagnostics=None, index=None)` | Store the outputs for a standalone observation or a selected batch row. |
| `read_result(file, *, index=None, sections=None)` | Read the stored values as mappings keyed by HDF5 group and dataset name. |
| `clear_observation(file, *, index)` | Reset one batch row's outputs to their fill values. |
| `collect_schema_issues(file)` | Return a list of problems with the stored result schema. An empty list means the shared checks passed. |

## Write a Standalone Observation

This example stores two PSFs with a producer-declared jitter statistic and
diagnostic.
A scalar wavelength selects the standalone layout. The science coordinates,
statistics and PSF cube use the same point order.

```python
import h5py
import numpy as np
from astropy import units as u

from ao_core.io import initialize_result, read_result, write_observation

simulation = {
    "name": "example",
    "version": "0.1.0",
    "precision": "float32",
    "diagnostics_level": "validation",
    "extra_stat_fields": {"jitter": "mas"},
    "diagnostic_fields": {
        "quality/valid_mask": {"dtype": "bool", "shape": ("num_sci",)},
    },
}
setup = {
    "sci_r": np.array([0.0, 15.0]) * u.arcsec,
    "sci_theta": np.array([0.0, 45.0]) * u.deg,
}
meta = {
    "pixel_scale": 4.0 * u.mas,
    "tel_diameter": 8.0 * u.m,
    "tel_pupil": np.ones((16, 16), dtype=np.float32) * u.one,
}
psfs = np.full((2, 8, 8), 1.0 / 64, dtype=np.float32)
stats = {"jitter": np.array([1.2, 1.4]) * u.mas}
diagnostics = {"quality": {"valid_mask": np.array([True, True])}}

with h5py.File("standalone.h5", "x") as file:
    initialize_result(
        file,
        simulation=simulation,
        setup=setup,
        options={"wavelength": 1.65 * u.um},
        save_psfs=True,
    )
    write_observation(
        file, meta=meta, psfs=psfs, stats=stats, diagnostics=diagnostics
    )

with h5py.File("standalone.h5", "r") as file:
    result = read_result(file)
    assert result["psfs"]["data"].shape == (2, 8, 8)
    assert result["stats"]["jitter"].unit == u.mas
    assert result["diagnostics"]["quality"]["valid_mask"].shape == (2,)
```

`standard_stats` selects which of `ee`, `fwhm` and `sr` to store. Supply the
selected statistics and declared extra statistics in `stats`. EE uses shape
`(science, aperture)` even for one aperture. Use `stats=None` when there are
no selected or declared statistics, and `diagnostics=None` when there are no
declared diagnostics. Supply each declared extension value in its respective
output mapping. Missing selected or declared outputs, unselected statistics
and undeclared extension values raise `ValueError`.

With `save_psfs=True`, each write requires a PSF cube and all three
[physical metadata fields](reference/hdf5.md#physical-metadata). With
`save_psfs=False`, supply any physical metadata you want to record. A supplied
cube is discarded. Use `meta={}` for a PSF-free write with neither physical
nor declared extra metadata to supply.

The first write that supplies PSFs or a pupil establishes their dimensions.
Later PSF cubes use the same shape, and any supplied telescope diameter and
pupil must match the stored values. See
[output allocation](reference/hdf5.md#output-allocation) for initial values.

## Write and Read a Batch Row

To store the same example as a one-row batch, pass a wavelength array with
shape `(1,)` and add `index=0` to the write. Pass the same per-observation
arrays to `write_observation()`. It handles the leading axis in the file.

```python
with h5py.File("batch.h5", "x") as file:
    initialize_result(
        file,
        simulation=simulation,
        setup=setup,
        options={"wavelength": np.array([1.65]) * u.um},
        save_psfs=True,
    )
    write_observation(
        file, index=0, meta=meta, psfs=psfs, stats=stats, diagnostics=diagnostics
    )

with h5py.File("batch.h5", "r") as file:
    batch = read_result(file)
    row = read_result(file, index=0, sections=("options", "psfs", "stats"))
    assert batch["psfs"]["data"].shape == (1, 2, 8, 8)
    assert row["psfs"]["data"].shape == (2, 8, 8)
    assert row["options"]["wavelength"].isscalar
    np.testing.assert_array_equal(row["stats"]["jitter"], result["stats"]["jitter"])
```

A read with `index=None` returns the stored arrays, including the observation
axis in a batch. A batch write requires an integer `index`. For a single-row
read or write, the index ranges from zero to the observation count minus one.
Reading a row takes one slice from each per-observation dataset and returns
the full values of the shared datasets. For a standalone file, use
`index=None` for both reading and writing. Supplying an integer index with
the standalone layout raises `ValueError`.

`sections` selects result groups by name without a leading slash. The names are
`simulation`, `setup`, `options`, `meta`, `psfs`, `stats` and `diagnostics`.
By default, a read returns all available result groups. An unknown section
name raises `ValueError`. An absent optional group is omitted, and an allocated
group with no datasets is returned as an empty mapping. For example,
`sections=("meta", "stats")` reads those groups without loading PSF pixels.
The file contains the declarations needed to read extensions.

## Use Producer Fields

Put run-wide inputs into `simulation` or `setup`. Put observation-specific
inputs into `options`, with one dataset per option. Use the three declarations
under `simulation` for extra statistics, metadata and diagnostics.

| Declaration | Producer supplies |
| --- | --- |
| `extra_stat_fields` | A unit for each extra per-science-point statistic. |
| `meta_fields` | A unit or `None` for each extra scalar per observation. |
| `diagnostic_fields` | A dtype, per-observation shape and applicable unit for each diagnostic path. |

Pass the corresponding values in `stats`, `meta` or `diagnostics`. Diagnostic
declarations and values accept nested mappings or slash-delimited keys. Reads
return nested mappings for nested HDF5 groups.

## Validate a File

A read checks the requested result groups and the information needed to
interpret them, such as wavelength for layout and the relevant extension
declarations. To check every result group, call `collect_schema_issues()`.
It reports problems with paths, shapes, dtypes and units without loading PSF
pixels or opening referenced input files. An empty list means those checks
passed. Reads return stored values, including fill values for unwritten rows.

Invalid declaration values, shapes, units and dataset layouts raise
`ValueError`. A wrong argument type, such as a declaration that requires a
mapping or a batch index supplied as text, raises `TypeError`. An out-of-range
batch index raises `IndexError`. File-open errors come from h5py.

## Clear the Outputs for a Batch Row

`clear_observation(file, index=...)` requires an integer batch index from
zero to the observation count minus one. It writes the
[fill values](reference/hdf5.md#output-allocation) to that row's allocated
outputs. The caller decides when to clear a row.
