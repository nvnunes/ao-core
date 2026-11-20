"""High-frequency continuation of the uncorrected science phase spectrum.

Positive spectral shells and projected white closure follow the
project-authored atmosphericTail implementation in P3.
"""

import numpy as np
from scipy import fft

from .atmosphere import VK_COEFFICIENT


def _shell_strips(count, side, represented, spacing, fried, outer_scale):
    modes = np.arange(count) - count // 2
    frequency = modes * spacing
    outside = abs(modes) >= represented // 2
    coefficient = VK_COEFFICIENT * fried ** (-5 / 3) * spacing**2
    for first in range(0, count, 128):
        rows = slice(first, min(first + 128, count))
        squared_radius = frequency[rows, None] ** 2 + frequency**2 + outer_scale**-2
        power = coefficient * squared_radius ** (-11 / 6)
        power[~(outside[rows, None] | outside)] = 0
        power[:, 0] = 0
        if first == 0:
            power[0] = 0
        indices = (modes[rows, None] % side) * side + (modes % side)
        yield rows, power, indices.ravel()


def _white_power(cutoff, fried, outer_scale):
    nodes, weights = np.polynomial.legendre.leggauss(32)
    radius = cutoff / np.cos((nodes + 1) * np.pi / 8)
    normalization = np.pi * VK_COEFFICIENT * fried ** (-5 / 3) * (3 / 5)
    return normalization * (weights @ (radius**2 + outer_scale**-2) ** (-5 / 6))


def _tail_lags(grid, side):
    represented = len(grid.fx)
    count = side * max(4, 1 << int(np.ceil(np.log2(represented / side + 1))))
    previous = None
    for _ in range(7):
        args = (count, side, represented, grid.spacing, grid.fried, grid.outer_scale)
        folded = np.zeros(side**2)
        for _, power, indices in _shell_strips(*args):
            folded += np.bincount(indices, power.ravel(), minlength=side**2)
        white = _white_power(
            (count / 2 - 0.5) * grid.spacing, grid.fried, grid.outer_scale
        )
        folded += white / side**2
        if (
            previous is not None
            and np.sum(abs(folded - previous)) < 1e-5
            and white < 1e-3
        ):
            break
        previous = folded
        count *= 2
    else:
        raise ValueError("The atmospheric spectral tail did not converge")

    return fft.ifft2(folded.reshape(side, side)).real * side**2, args, white


def atmospheric_tail_lags(grid, side):
    """Return the unprojected unresolved lag covariance at an optical sampling."""
    lags, _, _ = _tail_lags(grid, side)
    return lags


def atmospheric_tail(grid, pupil):
    """Return lag, sample/tilt and tilt/tilt moments of unresolved phase."""
    side = pupil.side
    lags, args, white = _tail_lags(grid, side)
    if len(pupil.tilt_fit) == 0:
        return lags, np.zeros_like(pupil.ramps.T), np.empty((0, 0))
    count = args[0]
    frequencies = (np.arange(count) - count // 2) * grid.spacing
    waves = np.exp(2j * np.pi * frequencies[:, None] * pupil.fit_axis)
    cross = np.zeros((2, side**2), dtype=complex)
    tilt = np.zeros((2, 2))
    for rows, power, indices in _shell_strips(*args):
        kernels = np.asarray([waves[rows] @ fit.T @ waves.T for fit in pupil.tilt_fit])
        for axis, kernel in enumerate(kernels):
            values = (power * kernel.conj()).ravel()
            cross[axis].real += np.bincount(indices, values.real, minlength=side**2)
            cross[axis].imag += np.bincount(indices, values.imag, minlength=side**2)
        tilt += np.einsum(
            "xy,ixy,jxy->ij", power, kernels, kernels.conj(), optimize=True
        ).real
    point_grid = fft.ifft2(cross.reshape(2, side, side)).real * side**2
    point = point_grid[:, pupil.x_index, pupil.y_index]
    coordinates = (pupil.points - pupil.fit_axis[0]) / np.diff(pupil.fit_axis)[0]
    nearest = np.rint(coordinates).astype(int)
    inside = np.all((nearest >= 0) & (nearest < len(pupil.fit_axis)), axis=1)
    coincident = inside & np.all(abs(coordinates - nearest) < 1e-10, axis=1)
    point[:, coincident] += (
        white * pupil.tilt_fit[:, nearest[coincident, 1], nearest[coincident, 0]]
    )
    tilt += white * np.einsum("iyx,jyx->ij", pupil.tilt_fit, pupil.tilt_fit)
    return lags, point, tilt
