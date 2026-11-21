"""Square lenslet measurements and least-squares Gaussian mirror commands.

The finite-cell estimator and mirror fit follow the project-authored P3
finiteAperture, finiteActuator and mirrorSurface work (ab39298, 8909b96,
ddfa46b). The calculation takes resolved AO Core geometry and spectra.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import fft, linalg, ndimage, special
from scipy.signal import CZT

from ._lag_transfer import cubic_stencil
from .atmosphere import von_karman_spectrum


def fourier_sum(spectrum, spacing, start, step, size):
    """Evaluate a centred spectrum on a translated Cartesian lattice.

    Use P3's rectangular products for small output grids. The products and
    the chirp Z-transform evaluate the same finite Fourier sum.
    See Eqs. (7a)–(7c) for the chirp Z-transform sampling grid.
    [Rabiner et al. (1969)](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/015_czt.pdf)
    """
    count = spectrum.shape[-1]
    modes = np.arange(count) - count // 2
    if count >= 4 * size:
        frequency = modes * spacing
        positions = np.asarray(start)[:, None] + np.arange(size) * step
        transform = np.exp(2j * np.pi * positions[..., None] * frequency)
        return (transform[0] @ spectrum) @ transform[1].T

    output = spectrum * np.exp(2j * np.pi * spacing * start[0] * modes)[:, None]
    output *= np.exp(2j * np.pi * spacing * start[1] * modes)
    transform = CZT(count, m=size, w=np.exp(2j * np.pi * spacing * step))
    output = transform(transform(output, axis=-2), axis=-1)
    phase = np.exp(-2j * np.pi * spacing * step * (count // 2) * np.arange(size))
    return output * phase[:, None] * phase


def fold_spectrum(spectrum, side):
    indices = (np.arange(spectrum.shape[-1]) - spectrum.shape[-1] // 2) % side
    folded = np.zeros((side, side), dtype=spectrum.dtype)
    np.add.at(folded, (indices[:, None], indices[None, :]), spectrum)
    return folded


def covariance_kernel(spectrum, side):
    return (fft.ifft2(fold_spectrum(spectrum, side)) * side**2).real


def piston_filter(diameter, radial_frequency):
    """Return the piston-rejection power factor for a circular aperture.

    See the single-aperture limit following Eq. (24). [Plantet et al. (2022)](https://arxiv.org/pdf/2209.00931)
    """
    argument = np.pi * diameter * radial_frequency
    amplitude = np.ones_like(argument)
    np.divide(2 * special.j1(argument), argument, out=amplitude, where=argument != 0)
    return 1 - amplitude**2


def footprint_shift(height, cone_scale, direction, offset):
    return height * direction + cone_scale * offset


def lenslet_gradient(fx, fy, pitch, cone_scale):
    """Complex x/y slope of a plane wave averaged over the square lenslet.

    See the spatial gradient and square-cell averaging factors in
    Eq. (10). [Correia & Teixeira (2014)](https://arxiv.org/pdf/1410.6055)
    """
    x, y = cone_scale * fx, cone_scale * fy
    transfer = 2j * np.pi * np.sinc(pitch * x) * np.sinc(pitch * y)
    return np.stack((x * transfer, y * transfer))


def gaussian_basis(points, centres, width):
    separation = (points[:, None, :] - centres[None, :, :]) * width
    return np.exp(-np.sum(separation**2, axis=-1))


def gaussian_slopes(cells, centres, width, pitch, offsets):
    """Square-cell averages of the two derivatives of each Gaussian."""
    blocks = []
    for offset in offsets:
        delta = (cells[:, None] + offset - centres) * width
        left, right = delta - pitch * width / 2, delta + pitch * width / 2
        mean = (
            (special.erf(right) - special.erf(left))
            * np.sqrt(np.pi)
            / (2 * width * pitch)
        )
        edge = (np.exp(-(right**2)) - np.exp(-(left**2))) / pitch
        blocks.extend((edge[..., 0] * mean[..., 1], mean[..., 0] * edge[..., 1]))
    return np.concatenate(blocks)


def pinverse(response, cutoff):
    left, singular, right = linalg.svd(response, full_matrices=False)
    keep = singular > cutoff
    if not keep.any():
        raise ValueError("The mirror phase-fit cutoff removes all actuator modes")
    return (right[keep].T / singular[keep]) @ left[:, keep].T


def _command_tilt(axis, valid, calibration_x, calibration_y, weights, width):
    y, x = np.nonzero(valid)
    response_x = np.exp(-((width * (calibration_x[:, None] - axis)) ** 2))
    response_y = np.exp(-((width * (calibration_y[:, None] - axis)) ** 2))
    pairs_x = (response_x[:, :, None] * response_x[:, None, :]).reshape(
        len(response_x), -1
    )
    pairs_y = (response_y[:, :, None] * response_y[:, None, :]).reshape(
        len(response_y), -1
    )
    separable_gram = (pairs_x.T @ weights.T @ pairs_y).reshape((len(axis),) * 4)
    gram = separable_gram[x[:, None], x, y[:, None], y]
    ramps = np.array(np.meshgrid(calibration_x, calibration_y))
    ramps -= np.einsum("iyx,yx->i", ramps, weights)[:, None, None] / weights.sum()
    cross = np.array(
        [(response_x.T @ (weights * ramp).T @ response_y)[x, y] for ramp in ramps]
    ).T
    tilt_gram = np.einsum("iyx,jyx,yx->ij", ramps, ramps, weights)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", linalg.LinAlgWarning)
            shape = linalg.solve(gram, cross, assume_a="pos")
            tilt_projection = linalg.solve(tilt_gram, cross.T, assume_a="pos")
    except (linalg.LinAlgError, linalg.LinAlgWarning) as exc:
        raise ValueError(
            "The pupil cannot calibrate a stable DM command tilt fit"
        ) from exc
    if np.linalg.norm(gram @ shape - cross) > 1e-9 * np.linalg.norm(cross):
        raise ValueError("Mirror command tilt fit failed its normal equation")
    return np.eye(len(x)) - shape @ tilt_projection


@dataclass(frozen=True)
class MirrorPhaseFit:
    """Sample positions and their phase-to-command projection."""

    points: np.ndarray
    projection: np.ndarray


def mirror_phase_fit(cells, lenslet_pitch, centres, width, tilt_projection, cutoff):
    # The fit uses the union of closed 3 × 3 nodes of every valid lenslet.
    first = cells.min(axis=0) - lenslet_pitch / 2
    cell_indices = np.rint((cells - first) / lenslet_pitch - 0.5).astype(int)
    offsets = np.stack(np.meshgrid(np.arange(3), np.arange(3)), axis=-1).reshape(9, 2)
    nodes = np.unique((2 * cell_indices[:, None] + offsets).reshape(-1, 2), axis=0)
    points = first + nodes * (lenslet_pitch / 2)
    fit = pinverse(gaussian_basis(points, centres, width), cutoff)
    return MirrorPhaseFit(points, tilt_projection @ fit)


class CovarianceQueries:
    """Sample separable lag kernels from pupil-minus-lenslet coordinates."""

    def __init__(self, points, cells, side, period):
        self.x, x_index = np.unique(points[:, 0], return_inverse=True)
        self.y, y_index = np.unique(points[:, 1], return_inverse=True)
        self.cell_x, cell_x_index = np.unique(cells[:, 0], return_inverse=True)
        self.cell_y, cell_y_index = np.unique(cells[:, 1], return_inverse=True)
        self.side = side
        self.scale = side / period
        width = len(self.x) * len(self.cell_x)
        rows = max(1, (32 * 1024**2) // (8 * len(self.cell_y) * (2 * side + width)))
        self.blocks = []
        for first in range(0, len(self.y), rows):
            last = min(first + rows, len(self.y))
            selected = np.flatnonzero((y_index >= first) & (y_index < last))
            height = (last - first) * len(self.cell_y)
            indices = (
                x_index[selected, None] * len(self.cell_x) + cell_x_index
            ) * height
            indices += (y_index[selected, None] - first) * len(
                self.cell_y
            ) + cell_y_index
            self.blocks.append((self.y[first:last], selected, indices))

    def evaluate(self, kernels, origin, cone_scale, output):
        """Apply stencils in the dtype shared by the kernels and output."""
        x = self.x[:, None] + origin[0] - cone_scale * self.cell_x
        points_x = cubic_stencil(x * self.scale, self.side).astype(
            output.dtype, copy=False
        )
        for y, selected, indices in self.blocks:
            points_y = self._y_stencil(y, origin[1], cone_scale, output.dtype)
            for direction, kernel in enumerate(kernels):
                values = points_x @ (points_y @ kernel.T).T
                output[direction, selected] = values.ravel()[indices]

    def _y_stencil(self, y, offset, cone_scale, dtype):
        coordinates = y[:, None] + offset - cone_scale * self.cell_y
        return cubic_stencil(coordinates * self.scale, self.side).astype(
            dtype, copy=False
        )

    def evaluate_group(self, kernels, origins, cone_scale, output):
        """Query contiguous [y, x] planes and reuse identical y passes."""
        x = self.x[None, :, None] + origins[:, 0, None, None] - cone_scale * self.cell_x
        matrices = [
            cubic_stencil(coordinates * self.scale, self.side).astype(
                output.dtype, copy=False
            )
            for coordinates in x
        ]
        guides_by_y = {}
        for guide, offset in enumerate(origins[:, 1]):
            guides_by_y.setdefault(offset, []).append(guide)
        for y, selected, indices in self.blocks:
            for offset, guides in guides_by_y.items():
                points_y = self._y_stencil(y, offset, cone_scale, output.dtype)
                for axis, kernel in enumerate(kernels):
                    columns = np.ascontiguousarray((points_y @ kernel).T)
                    for guide in guides:
                        values = matrices[guide] @ columns
                        output[axis, guide, selected] = values.ravel()[indices]


def slope_matrix(values, lenslets):
    """Expand lag covariances in guide, slope axis, and valid-cell order."""
    offsets = lenslets.indices[:, None] - lenslets.indices + lenslets.count - 1
    result = values[:, :, offsets[..., 0], offsets[..., 1]].transpose(0, 2, 1, 3)
    return result.reshape(len(values) * len(lenslets.indices), -1)


def lenslet_covariances(grid, layers, lenslets, inversion_floor, regularization):
    """Return the common-loop slope covariance and the optional design prior.

    See [Gendron et al. (2014), §§2–3](https://hdl.handle.net/10754/346823)
    for more detail on assembling covariance across pairs of projected lenslets.
    """
    count = lenslets.count
    span = 2 * count - 1
    channels = 2 * len(lenslets.offsets)
    temporal = np.zeros((channels, channels, span, span))
    prior = None if inversion_floor is None else np.zeros_like(temporal)
    for layer in layers:
        pitch = lenslets.pitch * layer.cone_scale
        for guide, initial in enumerate(layer.footprints):
            for other in range(guide, len(layer.footprints)):
                start = initial - layer.footprints[other] - pitch * (count - 1)
                for i, j in ((0, 0), (0, 1), (1, 1)):
                    cross = layer.power * layer.gradient[i] * layer.gradient[j].conj()
                    if prior is not None:
                        free = fourier_sum(cross, grid.spacing, start, pitch, span).real
                    tracked = fourier_sum(
                        cross * abs(layer.temporal) ** 2,
                        grid.spacing,
                        start,
                        pitch,
                        span,
                    ).real
                    pairs = [(2 * guide + i, 2 * other + j)]
                    if i != j and guide != other:
                        pairs.append((2 * guide + j, 2 * other + i))
                    for row, col in pairs:
                        temporal[row, col] += tracked
                        if prior is not None:
                            prior[row, col] += free
                        if row != col:
                            temporal[col, row] += tracked[::-1, ::-1]
                            if prior is not None:
                                prior[col, row] += free[::-1, ::-1]

    if prior is None:
        return None, slope_matrix(temporal, lenslets)
    regular = slope_matrix(prior, lenslets)
    regular.flat[:: len(regular) + 1] += regularization * np.diag(regular).mean()
    if inversion_floor:
        floor = _slope_floor(grid, layers, lenslets.pitch, inversion_floor)
        for channel in range(channels):
            prior[channel, channel] = floor
        for row in range(channels):
            for col in range(channels):
                if row != col:
                    prior[row, col] = 0
        regular += slope_matrix(prior, lenslets)
    return regular, slope_matrix(temporal, lenslets)


def _slope_floor(grid, layers, pitch, inversion_floor):
    """Build a slope covariance floor from the folded lenslet spectrum.

    See Eq. (17) for the folding of spatial frequency replicas. [Correia & Teixeira (2014)](https://arxiv.org/pdf/1410.6055)
    """
    x, y = grid.fx, grid.fy
    density = np.zeros_like(x)
    limit = 3.5 / pitch
    for layer in layers:
        a = layer.cone_scale
        radius = np.hypot(x, y) / a
        raw = von_karman_spectrum(x / a, y / a, grid.fried, grid.outer_scale)
        raw *= (layer.weight / a**2) * piston_filter(grid.diameter, radius)
        raw *= (abs(x / a) < limit) & (abs(y / a) < limit)
        density += raw
    average = np.sinc(pitch * x) * np.sinc(pitch * y)
    for direction in (x, y):
        modes = np.rint(pitch * direction)
        average[(modes != 0) & (abs(pitch * direction - modes) < 1e-12)] = 0
    side = grid.base_side
    energy = [
        np.sum(
            (density * (2 * np.pi * axis * average) ** 2).reshape(7, side, 7, side),
            axis=(0, 2),
        )
        for axis in (x, y)
    ]
    spectrum = np.maximum(*energy) * inversion_floor / pitch**2
    spectrum[side // 2, side // 2] = 0
    spatial = fft.ifft2(fft.ifftshift(spectrum)).real
    indices = np.arange(-(grid.lenslet_count - 1), grid.lenslet_count) % side
    return spatial[np.ix_(indices, indices)]


def solve_commands(factor, target_covariance):
    return linalg.cho_solve(factor, target_covariance.T, check_finite=False).T


def phase_slope_covariance(grid, layers, lenslets, points, directions, weights, dtype):
    """Return unfiltered phase–slope cross moments for a weighted angular target."""
    queries = CovarianceQueries(points, lenslets.cells, grid.kernel_side, grid.period)
    count = len(lenslets.cells)
    cross = np.zeros((len(points), 2 * len(lenslets.offsets) * count))
    sample = np.empty((2, len(points), count), dtype=dtype)
    for layer in layers:
        field_filter = np.zeros_like(grid.fx, dtype=complex)
        for direction, weight in zip(directions, weights, strict=True):
            field_filter += weight * np.exp(
                2j
                * np.pi
                * layer.height
                * (grid.fx * direction[0] + grid.fy * direction[1])
            )
        kernels = _command_kernels(grid, layer, field_filter, dtype)
        for guide, footprint in enumerate(layer.footprints):
            queries.evaluate(kernels, -footprint, layer.cone_scale, sample)
            cross[:, 2 * guide * count : 2 * (guide + 1) * count] += sample.transpose(
                1, 0, 2
            ).reshape(len(points), -1)
    return cross


def _command_kernels(grid, layer, field_filter, dtype):
    spectra = layer.power * layer.gradient.conj() * field_filter
    return [
        ndimage.spline_filter(
            covariance_kernel(spectrum, grid.kernel_side), mode="grid-wrap"
        ).astype(dtype, copy=False)
        for spectrum in spectra
    ]


class DirectionalCommandFit:
    """Prepare the science DM prior once and query translated lenslet lags.

    The cached splines and ground-layer cross moments follow the
    project-authored P3 finiteActuator directional fit. Only one CPU worker
    uses the scratch array and the ground-layer cache during a prediction.
    """

    def __init__(self, grid, layers, lenslets, phase_fit, factor, dtype):
        self.layers = layers
        self.phase_fit = phase_fit
        self.factor = factor
        self.queries = CovarianceQueries(
            phase_fit.points, lenslets.cells, grid.kernel_side, grid.period
        )
        # The first sparse pass consumes rows in y. Store that orientation
        # once so SciPy can flatten each plane without a copy on every pass.
        self.kernels = [
            [
                np.ascontiguousarray(kernel.T)
                for kernel in _command_kernels(grid, layer, 1, dtype)
            ]
            for layer in layers
        ]
        self.cells = len(lenslets.cells)
        self.guides = len(lenslets.offsets)
        self.shape = (len(phase_fit.points), 2 * self.guides * self.cells)
        bytes_per_guide = 2 * self.shape[0] * self.cells * np.dtype(dtype).itemsize
        self.group_size = min(self.guides, max(1, (32 * 1024**2) // bytes_per_guide))
        self.sample = np.empty(
            (2, self.group_size, self.shape[0], self.cells), dtype=dtype
        )
        self.ground_cross = {}

    def fit_batch(self, directions):
        return [self._fit(direction) for direction in directions]

    def _fit(self, direction):
        cross = np.zeros(self.shape)
        for index, (layer, kernels) in enumerate(
            zip(self.layers, self.kernels, strict=True)
        ):
            if index in self.ground_cross:
                cross += self.ground_cross[index]
                continue
            target = cross if layer.height != 0 else np.zeros_like(cross)
            origins = layer.height * direction - layer.footprints
            for first in range(0, self.guides, self.group_size):
                last = min(first + self.group_size, self.guides)
                group = self.sample[:, : last - first]
                self.queries.evaluate_group(
                    kernels, origins[first:last], layer.cone_scale, group
                )
                for guide, values in enumerate(group.swapaxes(0, 1), start=first):
                    start = 2 * guide * self.cells
                    target[:, start : start + 2 * self.cells] += values.transpose(
                        1, 0, 2
                    ).reshape(self.shape[0], 2 * self.cells)
            if layer.height == 0:
                self.ground_cross[index] = target
                cross += target
        return solve_commands(self.factor, self.phase_fit.projection @ cross)
