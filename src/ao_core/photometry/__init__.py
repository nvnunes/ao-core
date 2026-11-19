"""Natural guide-star magnitude and wavefront-sensor photon conversions."""

from ._conversions import (
    WFSPhotometryConfig,
    magnitudes_to_effective_photons_per_second,
    magnitudes_to_photon_flux,
    magnitudes_to_photons_per_frame,
    photon_flux_to_magnitudes,
    photon_flux_to_photons_per_frame,
    photons_per_frame_to_magnitudes,
    photons_per_frame_to_photon_flux,
)

__all__ = [
    "WFSPhotometryConfig",
    "magnitudes_to_effective_photons_per_second",
    "magnitudes_to_photon_flux",
    "magnitudes_to_photons_per_frame",
    "photon_flux_to_magnitudes",
    "photon_flux_to_photons_per_frame",
    "photons_per_frame_to_magnitudes",
    "photons_per_frame_to_photon_flux",
]
