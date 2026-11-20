"""Metal residual covariance and pupil-pair integration."""

from functools import lru_cache

import numpy as np

from ._ho_optics import (
    InvalidCovarianceOptics,
    ResidualCovariance,
    complete_coherence_lags,
)

# Strict upper pairs contribute the change from static diffraction. The
# conjugate half and exact static autocorrelation are restored on the host.
_PUPIL_PAIRS = r"""
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Metal lacks expm1. The sixth-order series keeps the small change to unity
// accurate to float32 roundoff at the switch.
inline float coherence_change(float x) {
    if (x < -0.125f) return exp(x) - 1.f;
    return x * (1.f + x * (0.5f + x * (1.f/6.f + x * (1.f/24.f +
           x * (1.f/120.f + x * (1.f/720.f))))));
}

kernel void accumulate_pupil_pairs(
    device const float* covariance [[buffer(0)]],
    device const float* diagonal [[buffer(1)]],
    device const long* ys [[buffer(2)]],
    device const long* xs [[buffer(3)]],
    device const float* real [[buffer(4)]],
    device const float* imag [[buffer(5)]],
    device atomic_float* lag_real [[buffer(6)]],
    device atomic_float* lag_imag [[buffer(7)]],
    device atomic_uint* max_negative [[buffer(8)]],
    constant uint& first [[buffer(9)]],
    constant uint& columns [[buffer(10)]],
    constant uint& count [[buffer(11)]],
    constant int& lag_y [[buffer(12)]],
    constant int& lag_x [[buffer(13)]],
    constant uint& lag_width [[buffer(14)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    uint i = first + index / columns;
    uint j = first + index % columns;
    if (j <= i) return;

    float variance = (diagonal[i] + diagonal[j]) - 2.f * covariance[index];
    if ((as_type<uint>(variance) & 0x7f800000u) == 0x7f800000u) {
        atomic_fetch_max_explicit(max_negative, 0x7f800000u, memory_order_relaxed);
        return;
    }
    if (variance < 0.f) {
        atomic_fetch_max_explicit(max_negative, as_type<uint>(-variance),
                                  memory_order_relaxed);
        return;
    }
    float coherence = coherence_change(-0.5f * variance);
    uint lag = uint((ys[i] - ys[j] - lag_y) * lag_width + xs[i] - xs[j] - lag_x);
    float weighted_real = real[i] * real[j] + imag[i] * imag[j];
    float weighted_imag = imag[i] * real[j] - real[i] * imag[j];
    atomic_fetch_add_explicit(lag_real + lag, coherence * weighted_real,
                              memory_order_relaxed);
    atomic_fetch_add_explicit(lag_imag + lag, coherence * weighted_imag,
                              memory_order_relaxed);
}
"""


def require_mps():
    """Load the optional Metal dependency and check this backend's requirements."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("HO PSF Metal execution requires ao-core[metal]") from error
    if not torch.backends.mps.is_available():
        raise RuntimeError("HO PSF Metal execution requires an available MPS device")
    if not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("HO PSF Metal execution requires torch.mps.compile_shader")
    return torch


@lru_cache(maxsize=1)
def _pupil_pairs_kernel(torch):
    return torch.mps.compile_shader(_PUPIL_PAIRS).accumulate_pupil_pairs


class MpsPupilOptics:
    """Keep the pupil geometry and mirror basis on Metal across field points."""

    def __init__(self, pupil, basis, field):
        torch = self.torch = require_mps()
        self.pupil = pupil
        self.field = field
        self.x = torch.as_tensor(pupil.x, device="mps", dtype=torch.int64)
        self.y = torch.as_tensor(pupil.y, device="mps", dtype=torch.int64)
        self.indices = torch.arange(len(pupil.x), device="mps")
        self.ramps = self.to_device(pupil.ramps)
        self.basis = self.to_device(basis)
        weights = pupil.amplitude if field is None else field
        self.real = self.to_device(weights.real)
        self.imag = self.to_device(weights.imag)
        self.lag_origin = np.array(
            [pupil.y.min() - pupil.y.max(), pupil.x.min() - pupil.x.max()]
        )
        self.lag_shape = tuple(int(size) for size in 1 - 2 * self.lag_origin)
        self.accumulate_pairs = _pupil_pairs_kernel(torch)

    def to_device(self, values):
        return self.torch.as_tensor(
            np.require(values, dtype=np.float32, requirements=("C", "W")),
            device="mps",
        )

    def coherence_lags(self, source, command_covariance, source_command):
        """Integrate atmospheric coherence into signed lags on Metal."""
        torch = self.torch
        residual = _MpsResidualCovariance(
            self, source, command_covariance, source_command
        )
        lags = torch.zeros(
            (2, self.lag_shape[0] * self.lag_shape[1]),
            device="mps",
            dtype=torch.float32,
        )
        partial_lags = torch.empty_like(lags)
        max_negative = torch.zeros(1, device="mps", dtype=torch.int32)
        for first, pair in residual.blocks():
            # Bound the number of float32 atomic additions into each lag.
            partial_lags.zero_()
            self.accumulate_pairs(
                pair.contiguous(),
                residual.diagonal,
                self.y,
                self.x,
                self.real,
                self.imag,
                partial_lags[0],
                partial_lags[1],
                max_negative,
                first,
                pair.shape[1],
                pair.numel(),
                int(self.lag_origin[0]),
                int(self.lag_origin[1]),
                self.lag_shape[1],
                threads=pair.numel(),
                group_size=min(256, pair.numel()),
            )
            lags += partial_lags
        negative = max_negative.cpu().numpy().view(np.float32).item()
        if not np.isfinite(negative):
            raise InvalidCovarianceOptics(
                "A residual phase pair has nonfinite difference variance"
            )
        if negative > np.float32(1e-8):
            raise InvalidCovarianceOptics(
                "A residual phase pair has negative difference variance"
            )
        real, imag = lags.cpu().numpy().astype(np.float64)
        coherence = real + real[::-1]
        if self.field is not None:
            coherence = coherence + 1j * (imag - imag[::-1])
        return complete_coherence_lags(
            self.pupil, self.field, coherence, self.lag_origin, relative_to_static=True
        )


class _MpsResidualCovariance(ResidualCovariance):
    def __init__(self, optics, source, command_covariance, source_command):
        self.side = optics.pupil.side
        self.x, self.y, self.indices = optics.x, optics.y, optics.indices
        self._prepare(
            tuple(optics.to_device(values) for values in source),
            optics.ramps,
            optics.basis,
            optics.to_device(command_covariance),
            source_command,
            optics.torch.einsum,
        )
