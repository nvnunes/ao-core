# AO Core

AO Core provides infrastructure shared by adaptive optics software.
`ao_core.io` reads and writes HDF5 results for standalone observations and
batches. `ao_core.photometry` converts natural guide-star magnitudes, photon
flux and wavefront-sensor photon counts. `ao_core.stats` measures Strehl ratio,
enclosed energy, and FWHM from PSF images.

## Installation

Install AO Core from GitHub.

```bash
python -m pip install 'git+https://github.com/nvnunes/ao-core.git'
python -c 'import ao_core.io, ao_core.photometry, ao_core.stats'
```

To replace an earlier AO Core `0.1.0` build in an existing environment,
rerun the install command with `--force-reinstall` to install the current
build and its dependencies.

## Start Here

- [Development](docs/development.md) covers installation and local setup.
- [API guide](docs/api.md) describes the available Python interface.
- [Photometry](docs/photometry.md) explains the conversions and calibration inputs.
- [PSF measurements](docs/stats.md) explains how to measure single PSFs and cubes.
- [Architecture](docs/architecture.md) defines package ownership.
- [Testing](docs/testing.md) lists the verification commands.
