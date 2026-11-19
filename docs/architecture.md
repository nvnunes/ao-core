# Architecture

This document defines AO Core's ownership and public API boundaries.

AO Core is a shared library for adaptive optics software. The `ao_core.io`
module defines the result-file format and reads and writes its datasets.
Producers calculate their outputs, define additional fields and manage batch
execution and resumption.

The package root exposes `__version__`. The
[API guide](api.md) and [Python reference](reference/api.md) describe
supported interfaces.

The storage adapter for an existing producer can use the internal
`results.initialize_layout()` and `results.diagnostic_shape()` helpers
after applying its own input checks. The `schema` and `hdf5` modules provide
the storage definitions and dataset access those adapters share. Changes
to these helpers are coordinated with their consuming packages. The
[result I/O guide](io.md) defines the public API for producers and readers.
