# Architecture

This document defines AO Core's ownership and public API boundaries.

AO Core is a shared library for adaptive optics software. The `ao_core.io`
module defines the result-file format and reads and writes its datasets.
Producers calculate their outputs, define additional fields and manage batch
execution and resumption.

`ao_core.photometry` defines unit-aware NGS magnitude and WFS photon
conversions. The caller supplies the calibration and applies any
instrument-specific rounding. See the [photometry guide](photometry.md).

`ao_core.models.config` defines shared physical input records and decodes
their AO INI fields. Consumers apply the defaults and projections required
by their calculations and interpret their own INI fields. See the
[model configuration guide](model-configuration.md).

`ao_core.models` provides reusable calculations for atmosphere, pupil
projection, sensing, reconstruction, control, residual covariance, optics,
and metric response. The sampling predictor composes these components to
estimate the uncertainty of PSF metrics averaged over finite exposures.
Applications own instrument prescriptions, native data interpretation and
simulation orchestration. AO Stats owns ordinary PSF metric measurements.
AO Core owns the physical response and covariance propagation used in the
predictor.

The package root exposes `__version__`. The
[API guide](api.md) and [Python reference](reference/api.md) describe
supported interfaces.

The storage adapter for an existing producer can use the internal
`results.initialize_layout()` and `results.diagnostic_shape()` helpers
after applying its own input checks. The `schema` and `hdf5` modules provide
the storage definitions and dataset access those adapters share. Changes
to these helpers are coordinated with their consuming packages. The
[result I/O guide](io.md) defines the public API for producers and readers.
