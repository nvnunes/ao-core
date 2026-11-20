"""Metal phase–command queries for the common and science DMs."""

from collections import OrderedDict
from functools import lru_cache

import numpy as np

from ._ho_covariance import slope_fit_values, slope_kernels
from ._ho_metal import require_mps
from ._lag_transfer import cubic_stencil_weights

# Adapt the project-authored P3 finiteMpsQueries kernel to AO Core's [x, y]
# covariance planes. Each thread queries one valid pupil/lenslet pair.
_CUBIC_PAIRS = r"""
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

inline float along_y(device const float* row, uint4 y, float4 weights) {
    return (row[y.x] * weights.x + row[y.y] * weights.y) +
           (row[y.z] * weights.z + row[y.w] * weights.w);
}

kernel void sample_cubic_pairs(
    device const float* planes [[buffer(0)]],
    device const uint4* x_lags [[buffer(1)]],
    device const float4* x_weights [[buffer(2)]],
    device const uint4* y_lags [[buffer(3)]],
    device const float4* y_weights [[buffer(4)]],
    device const uint* x_groups [[buffer(5)]],
    device const uint* y_groups [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant uint& pupils [[buffer(8)]],
    constant uint& cells [[buffer(9)]],
    constant uint& side [[buffer(10)]],
    constant uint& fields [[buffer(11)]],
    constant uint& x_stride [[buffer(12)]],
    constant uint& y_stride [[buffer(13)]],
    uint index [[thread_position_in_grid]]) {
    uint pair_count = pupils * cells;
    uint group = index / pair_count;
    if (group >= 2 * fields) return;
    uint axis = group / fields;
    uint field = group % fields;
    uint point = (index % pair_count) / cells;
    uint cell = index % cells;
    uint x_group = field * x_stride + x_groups[point] + x_groups[pupils + cell];
    uint y_group = field * y_stride + y_groups[point] + y_groups[pupils + cell];
    uint4 x = x_lags[x_group], y = y_lags[y_group];
    float4 wx = x_weights[x_group], wy = y_weights[y_group];
    device const float* plane = planes + axis * side * side;
    float a = along_y(plane + x.x * side, y, wy) * wx.x;
    float b = along_y(plane + x.y * side, y, wy) * wx.y;
    float c = along_y(plane + x.z * side, y, wy) * wx.z;
    float d = along_y(plane + x.w * side, y, wy) * wx.w;
    output[index] = (a + b) + (c + d);
}
"""


@lru_cache(maxsize=1)
def _cubic_pairs_kernel(torch):
    return torch.mps.compile_shader(_CUBIC_PAIRS).sample_cubic_pairs


class MpsScienceCommandQueries:
    """Keep two fields' slope and mean streams on Metal through contraction."""

    direction_batch_size = 2
    kernel_cache_bytes = 512 * 1024**2

    def __init__(
        self,
        grid,
        layers,
        lenslets,
        pupil,
        tilt_modes,
        rows,
        command_means,
        joint,
        gain,
    ):
        self.grid = grid
        self.layers = layers
        self.lenslets = lenslets
        self.pupil = pupil
        self.tilt_modes = tilt_modes
        self.gain = gain
        torch = self.torch = require_mps()
        self.sample_pairs = _cubic_pairs_kernel(torch)
        self.common_rows = self._upload(rows.T, np.float32)
        self.common_means = self._upload(command_means.T, np.float32)
        if joint is not None:
            self.feedback = self._upload(joint.feedback, np.float32)
            self.science_mean_map = self._upload(joint.science_mean_map, np.float32)
        planes = 2 * (2 if gain else 1)
        self._layer_bytes = planes * grid.kernel_side**2 * np.dtype(np.float32).itemsize
        self._retain_host_phase = (
            len(layers) * self._layer_bytes > self.kernel_cache_bytes
        )
        self._host_kernels = {}
        self._device_kernels = OrderedDict()
        self._stencils = None
        self.axes = []
        for points, cells in zip(pupil.points.T, lenslets.cells.T, strict=True):
            coordinates, point_groups = np.unique(points, return_inverse=True)
            cell_coordinates, cell_groups = np.unique(cells, return_inverse=True)
            groups = np.concatenate((point_groups * len(cell_coordinates), cell_groups))
            self.axes.append(
                (coordinates, cell_coordinates, self._upload(groups, np.int32))
            )
        self.ramps = self._upload(pupil.ramps.T, np.float32)
        self.pupil_x = self._upload(pupil.x_index, np.int64)
        self.pupil_y = self._upload(pupil.y_index, np.int64)
        modes = np.arange(len(grid.fx), dtype=np.int64) - len(grid.fx) // 2
        self.fold_indices = self._upload(modes % pupil.side, np.int64)
        flat_tilt = tilt_modes.reshape(len(tilt_modes), grid.fx.size)
        self.tilt_real = self._upload(flat_tilt.real, np.float32)
        self.tilt_imag = self._upload(flat_tilt.imag, np.float32)

    def _upload(self, values, dtype):
        return self.torch.as_tensor(
            np.require(values, dtype=dtype, requirements=("C", "W")),
            device="mps",
        )

    def _kernel_pair(self, index):
        """Reuse prepared phase planes in the same bounded LRU as P3."""
        cache = self._device_kernels
        if index in cache:
            cache.move_to_end(index)
            return cache[index]
        prepared = self._host_kernels.get(index)
        if prepared is None:
            layer = self.layers[index]
            transfers = (layer.temporal, 1) if self.gain else (layer.temporal,)
            prepared = tuple(
                slope_kernels(
                    layer, transfer, self.tilt_modes, self.grid.kernel_side, np.float32
                )
                for transfer in transfers
            )
            if self._retain_host_phase:
                self._host_kernels[index] = prepared
        # Evict before uploading the next layer to leave room for a full pair.
        limit = self.kernel_cache_bytes // self._layer_bytes
        if limit and len(cache) == limit:
            cache.popitem(last=False)
        pair = tuple((self._upload(phase, np.float32), fit) for phase, fit in prepared)
        if limit:
            cache[index] = pair
        return pair

    def _cubic_stencils(self, origins, cone_scale):
        cached = self._stencils
        if (
            cached is not None
            and cached[1] == cone_scale
            and np.array_equal(cached[0], origins)
        ):
            return cached[2]
        side = self.grid.kernel_side
        scale = side / self.grid.period
        stencils = []
        for offset, (points, cell_axis, _) in zip(origins.T, self.axes, strict=True):
            coordinates = (
                points[:, None] + offset[:, None, None] - cone_scale * cell_axis
            ) * scale
            lags, weights = cubic_stencil_weights(coordinates, side)
            stencils.extend(
                (self._upload(lags, np.int32), self._upload(weights, np.float32))
            )
        self._stencils = (origins, cone_scale, stencils)
        return stencils

    def _sample_slopes(self, kernels, origins, cone_scale, values):
        side = self.grid.kernel_side
        cells = self.lenslets.cells
        phase, fit = kernels
        count = values.numel()
        self.sample_pairs(
            phase,
            *self._cubic_stencils(origins, cone_scale),
            self.axes[0][2],
            self.axes[1][2],
            values,
            len(self.pupil.points),
            len(cells),
            side,
            len(origins),
            len(self.axes[0][0]) * len(self.axes[0][1]),
            len(self.axes[1][0]) * len(self.axes[1][1]),
            threads=count,
            group_size=min(256, count),
        )
        if fit:
            lags = origins[:, None, :] - cone_scale * cells
            projected = slope_fit_values(fit, lags, side / self.grid.period).transpose(
                0, 2, 1, 3
            )
            values -= self.ramps @ self._upload(projected, np.float32)

    def _cross_streams(self, directions, layer_modes):
        """Accumulate slope and sensor-mean streams before applying DM commands."""
        torch = self.torch
        fields, pupils = len(directions), len(self.pupil.points)
        cells = len(self.lenslets.cells)
        channels = 2 * len(self.lenslets.offsets)
        streams = 3 if self.gain else 1
        slopes = torch.zeros(
            (streams, fields, pupils, channels * cells),
            device="mps",
            dtype=torch.float32,
        )
        means = torch.zeros(
            (fields, pupils, streams * channels), device="mps", dtype=torch.float32
        )
        sampled = torch.empty(
            (2, fields, pupils, cells), device="mps", dtype=torch.float32
        )
        for index, (layer, modes) in enumerate(
            zip(self.layers, layer_modes, strict=True)
        ):
            footprints = layer.height * directions
            kernel_pair = self._kernel_pair(index)
            for guide, footprint in enumerate(layer.footprints):
                origins = footprints - footprint
                queries = ((0, origins),)
                if self.gain:
                    delayed = origins - layer.residual_shift
                    queries += ((1, delayed), (0, delayed))
                for stream, (kernel_index, positions) in enumerate(queries):
                    self._sample_slopes(
                        kernel_pair[kernel_index], positions, layer.cone_scale, sampled
                    )
                    for axis in range(2):
                        start = (2 * guide + axis) * cells
                        slopes[stream, ..., start : start + cells] += sampled[axis]
            # Release the local reference so eviction can free the device tensors.
            del kernel_pair
            # A layer's sensor modes are shared by both directions in the batch.
            density = (
                self._upload(layer.power, np.float32)
                * self._upload(modes, np.complex64).conj()
            )
            for result, footprint in zip(means, footprints, strict=True):
                result += self._mean_cross(density, footprint)
        return slopes, means

    def common_cross(self, directions, mean_modes):
        slopes, means = self._cross_streams(directions, mean_modes)
        return slopes[0] @ self.common_rows + means @ self.common_means

    def two_stage_streams(self, directions, joint):
        """Submit the field's Metal queries before science fits are needed."""
        modes = (
            np.concatenate((projected, delayed_rhs, delay * projected))
            for projected, delayed_rhs, delay in joint.mean_modes
        )
        return self._cross_streams(directions, modes)

    def contract_two_stage_cross(self, streams, science_command_weights):
        slopes, means = streams
        common = slopes[0] @ self.common_rows + means @ self.common_means
        rows = self._upload(science_command_weights, np.float32)
        delayed = self.gain * rows
        science = slopes[1] @ delayed.transpose(-1, -2)
        science -= slopes[2] @ (delayed @ self.feedback).transpose(-1, -2)
        science += means @ (rows @ self.science_mean_map).transpose(-1, -2)
        return self.torch.cat((common, science), dim=-1)

    def _mean_cross(self, base_density, footprint):
        torch, grid = self.torch, self.grid
        x = self._upload(
            np.exp(2j * np.pi * grid.fx[:, 0] * footprint[0]), np.complex64
        )
        y = self._upload(
            np.exp(2j * np.pi * grid.fy[0, :] * footprint[1]), np.complex64
        )
        density = base_density * (x[:, None] * y)
        modes = len(density)
        side = self.pupil.side
        folded = torch.zeros((modes, side, side), dtype=torch.complex64, device="mps")
        # Each tile maps to unique bins. Add tiles to retain all alias orders.
        for first in range(0, len(x), side):
            rows = slice(first, first + side)
            for other in range(0, len(y), side):
                columns = slice(other, other + side)
                folded[
                    :, self.fold_indices[rows, None], self.fold_indices[None, columns]
                ] += density[:, rows, columns]
        lags = torch.fft.ifft2(folded).real * side**2
        flat = density.reshape(modes, -1).T
        fitted = self.tilt_real @ flat.real - self.tilt_imag @ flat.imag
        return lags[:, self.pupil_x, self.pupil_y].T - self.ramps @ fitted
