# API Guide

The package root exposes the installed AO Core version.

```python
from ao_core import __version__

print(__version__)
```

The [Python reference](reference/api.md) lists the installed exports. The
[result I/O guide](io.md) explains how to use `ao_core.io`. The
[photometry guide](photometry.md) explains how to use `ao_core.photometry`.
The [PSF measurement guide](stats.md) explains how to use `ao_core.stats`.
The [model configuration guide](model-configuration.md) explains how to use
`ao_core.models.config`.
The [scientific models guide](scientific-models.md) describes shared model
components and the [high-order PSF](scientific-models.md#high-order-psfs)
and [sampling uncertainty](scientific-models.md#sampling-uncertainty) predictors.
