"""Selected time lags of the restored pupil phase."""

from functools import lru_cache
from itertools import pairwise

import numpy as np
from astropy import units as u

from .atmosphere import von_karman_correlation, von_karman_variance


def _correction_lags(science, terms, indices, native_count):
    """Transform the correction and its two cross terms with bounded pupil tiles."""
    context = science.context
    phase, _, high = terms
    bins = context.bin_values <= native_count // 2
    groups = [np.flatnonzero(context.zero)]
    groups.extend(
        context.positive[start:stop]
        for start, stop in zip(context.starts[bins], context.ends[bins], strict=True)
    )
    samples = np.concatenate(groups)
    boundaries = np.r_[0, np.cumsum([len(group) for group in groups])]
    blocks = [slice(start, stop) for start, stop in pairwise(boundaries)]
    frequencies = np.r_[0, context.bin_values[bins]]
    factors = 2 * np.exp(2j * np.pi / native_count * np.outer(indices, frequencies))
    factors[:, 0] = 1
    if native_count % 2 == 0 and frequencies[-1] == native_count // 2:
        factors[:, -1] *= 0.5
    reverse = factors.conj()

    correction = np.vstack((context.lo[:, samples], high[:, samples]))
    conjugate = correction.conj()
    weighted = conjugate * (context.weights[samples] * phase[samples])
    modal_spectrum = np.array(
        [
            (correction[:, block] * context.weights[samples[block]])
            @ conjugate[:, block].T
            for block in blocks
        ]
    )
    modes = context.nmodes
    modal = (
        (factors @ modal_spectrum.reshape(len(blocks), -1))
        .real.copy()
        .reshape(-1, modes, modes)
    )
    del correction, conjugate, modal_spectrum

    angular = science.angular
    wave_count = angular // 2 if science.pair_waves else angular
    ring, azimuth = np.divmod(samples % (science.radial * angular), angular)
    columns = ring * wave_count + azimuth % wave_count
    opposite = azimuth >= wave_count
    points = len(science.waves)
    cross = np.empty((len(indices), points, modes))
    opposite_cross = np.empty_like(cross)
    for start in range(0, points, 64):
        tile = slice(start, min(start + 64, points))
        waves = science.waves[tile, columns]
        np.conjugate(waves, out=waves, where=opposite)
        spectrum = np.array(
            [waves[:, block] @ weighted[:, block].T for block in blocks]
        )
        shape = (len(indices), len(waves), modes)
        spectrum = spectrum.reshape(len(blocks), -1)
        cross[:, tile] = (factors @ spectrum).real.reshape(shape)
        opposite_cross[:, tile] = (reverse @ spectrum).real.reshape(shape)
    return modal, cross, opposite_cross


def atmospheric_offset_covariance(atmosphere, dt, offsets):
    """Cache reference-wavelength offset vectors for one atmosphere and pupil grid."""
    speed = atmosphere.wind_speeds.to_value(u.m / u.s)
    direction = atmosphere.wind_directions.to_value(u.rad)
    velocity = speed[:, None] * np.column_stack((np.cos(direction), np.sin(direction)))
    fractions = atmosphere.layer_weights.value
    outer = 1 / float(atmosphere.outer_scale.to_value(u.m))
    amplitude = von_karman_variance(
        float(atmosphere.fried_parameter.to_value(u.m)), outer
    )
    offsets = np.asarray(offsets, dtype=np.float64).copy()

    @lru_cache(maxsize=256)
    def at_frame(frame):
        shifted = offsets[None] - (frame * dt) * velocity[:, None]
        distance = np.hypot(shifted[..., 0], shifted[..., 1])
        correlation = von_karman_correlation(distance, outer)
        return amplitude * (fractions @ correlation)

    return at_frame


def full_pupil_lags(
    science,
    terms,
    indices,
    offsets,
    offset_indices,
    *,
    atmosphere,
    frame_interval_s,
    noise_lags,
    atmosphere_at_frame=None,
):
    """Yield piston-free pupil covariances, holding one dense time lag at a time.

    The atmospheric term translates the von Kármán covariance at each
    layer's velocity. The correction and both atmospheric cross terms use
    the same temporal bins as the modal predictor.
    """
    native_count = round(science.context.period / frame_interval_s)
    modal, cross, reverse = _correction_lags(science, terms, indices, native_count)
    basis = science.geometry.pupil.basis
    scale2 = science.context.scale**2
    if atmosphere_at_frame is None:
        atmosphere_at_frame = atmospheric_offset_covariance(
            atmosphere, frame_interval_s, offsets
        )
    for frame, correction, left, right in zip(
        indices, modal, cross, reverse, strict=True
    ):
        covariance = (
            science.atmospheric.copy()
            if frame == 0
            else (scale2 * atmosphere_at_frame(frame))[offset_indices]
        )
        correction *= scale2
        correction[:2, :2] += noise_lags[frame]
        covariance += (basis.T @ correction - scale2 * left) @ basis
        covariance -= ((scale2 * right) @ basis).T
        covariance -= covariance.mean(axis=1, keepdims=True)
        covariance -= covariance.mean(axis=0, keepdims=True)
        yield covariance
