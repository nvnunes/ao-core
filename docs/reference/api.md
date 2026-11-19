# Python Reference

::: ao_core
    options:
      members:
        - __version__

::: ao_core.io
    options:
      members:
        - initialize_result
        - write_observation
        - read_result
        - clear_observation
        - collect_schema_issues

::: ao_core.photometry
    options:
      members:
        - WFSPhotometryConfig
        - magnitudes_to_photon_flux
        - photon_flux_to_magnitudes
        - photon_flux_to_photons_per_frame
        - photons_per_frame_to_photon_flux
        - magnitudes_to_photons_per_frame
        - photons_per_frame_to_magnitudes
        - magnitudes_to_effective_photons_per_second

::: ao_core.models.config
    options:
      members:
        - AOConfiguration
        - AtmosphereConfig
        - DetectorConfig
        - FittingConfig
        - LensletConfig
        - LoopConfig
        - SourceConfig
        - TelescopeConfig
        - ini_to_mapping
        - load_ao_configuration
        - parse_ini
        - polar_to_cartesian
        - read_ini
        - read_ini_array
        - read_ini_scalar
        - resolve_ini_path
