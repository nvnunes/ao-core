"""Natural guide star image formation and centroid statistics."""

import numpy as np

from ._centroid import (
    DirectCentroid,
    PixelSpot,
    clipped_normal_laplace,
    sample_centroid_errors,
)
from ._gaussian_prior import centered_measurement_noise, centroid_noise_prior
from ._sensor import diffraction_calibration, optical_mean, sensor_response

__all__ = [
    "DirectCentroid",
    "PixelSpot",
    "centered_measurement_noise",
    "centroid_noise_prior",
    "clipped_normal_laplace",
    "diffraction_calibration",
    "lgs_footprint_phase",
    "modal_lgs_response",
    "optical_mean",
    "sample_centroid_errors",
    "sensor_response",
]


def lgs_footprint_phase(fx, fy, layer_height, directions, *, cone=1.0, offsets=None):
    """Return beacon phase factors with centre h*direction + cone*offset.

    Provide spatial frequencies in cycles/m, layer height and offsets in m,
    and directions in radians. The result is beacon by spatial frequency.
    """
    if offsets is None:
        return np.exp(
            2j
            * np.pi
            * layer_height
            * (directions[:, :1] * fx + directions[:, 1:] * fy)
        )
    centres = layer_height * directions + cone * offsets
    return np.exp(2j * np.pi * (centres[:, :1] * fx + centres[:, 1:] * fy))


def modal_lgs_response(modes, phase, *, out=None):
    """Expand shared pupil modes and footprint phases into beacon-major WFS rows."""
    shape = (len(phase), *modes.shape)
    if out is not None:
        out = out.reshape(shape)
    response = np.multiply(phase[:, None], modes[None], out=out)
    return response.reshape(shape[0] * shape[1], shape[2])
