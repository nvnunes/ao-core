# AO Core

AO Core is a Python library for shared adaptive optics infrastructure. Its
[result-file interface](io.md) reads and writes HDF5 observations. Its
[photometry interface](photometry.md) converts NGS magnitudes, flux and WFS
photon counts. Its [PSF measurement interface](stats.md) measures Strehl
ratio, enclosed energy, and FWHM. The
[model configuration interface](model-configuration.md) provides unit-aware
inputs for AO calculations.

Start with the [API guide](api.md) for the current interface or
[Development](development.md) to set up a checkout. See
[Architecture](architecture.md) for the boundary between AO Core and its
consumers.
