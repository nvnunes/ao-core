"""Metal contractions of the pupil-space Gaussian covariance."""

from functools import lru_cache

import numpy as np
import torch

_SOBOL_SHADER = """
#include <metal_stdlib>
using namespace metal;

kernel void accumulate_quartets(device const ushort4* samples,
                                device const float* covariances,
                                device const float2* weights,
                                device const float* probability,
                                device float* out,
                                constant uint& nodes,
                                constant uint& variants,
                                constant uint& total_draws,
                                uint index [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint width [[threads_per_simdgroup]]) {
    ushort4 q = samples[index];
    uint row_a = q.x * nodes, row_b = q.y * nodes;
    uint ab = row_a + q.y, cd = q.z * nodes + q.w;
    uint stride = nodes * nodes;
    float pair_probability = probability[ab] * probability[cd];
    float multiplier = 4.0f / pair_probability;
    for (uint variant = 0; variant < variants; ++variant) {
        device const float* covariance = covariances + variant * stride;
        float k = covariance[row_a + q.z] - covariance[row_a + q.w]
                - covariance[row_b + q.z] + covariance[row_b + q.w];
        float k2 = k * k, even, odd;
        // Subtraction in the exponential formulas loses precision near zero.
        if (pair_probability < 1e-34f || abs(k) >= 70.0f) {
            // A nonfinite partial sum selects the float64 CPU calculation.
            even = odd = as_type<float>(0x7fc00000u);
        } else if (abs(k) < 0.5f) {
            even = k2*k2 / 24.0f * (1.0f + k2 / 30.0f * (1.0f + k2 / 56.0f * (1.0f + k2 / 90.0f)));
            odd = k*k2 / 6.0f * (1.0f + k2 / 20.0f * (1.0f + k2 / 42.0f * (1.0f + k2 / 72.0f)));
        } else {
            float plus = exp(k) - 1.0f, minus = exp(-k) - 1.0f;
            even = 0.5f * (plus + minus) - 0.5f * k2;
            odd = 0.5f * (plus - minus) - k;
        }
        for (uint metric = 0; metric < 3; ++metric) {
            device const float2* response = weights + metric * stride;
            float2 left = response[ab], right = response[cd];
            float term = left.x * right.x * multiplier * even
                       + left.y * right.y * multiplier * odd;
            float sum = simd_sum(term);
            if (lane == 0) out[((variant * 3 + metric) * total_draws + index) / width] = sum;
        }
    }
}
"""


@lru_cache(maxsize=1)
def _quartet_kernel():
    return torch.mps.compile_shader(_SOBOL_SHADER).accumulate_quartets


class MpsQuadraticCovariance:
    """Keep one science point's responses on Metal and consume one lag at a time."""

    def __init__(self, first, second):
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "Metal is unavailable for the pupil covariance contraction"
            )
        self.first = torch.as_tensor(first, dtype=torch.float32, device="mps")
        self.second = torch.as_tensor(second, dtype=torch.float32, device="mps")

    def __call__(self, covariance):
        covariance = torch.as_tensor(covariance, dtype=torch.float32, device="mps")
        linear = ((self.first @ covariance) * self.first).sum(dim=-1)
        left = self.second @ covariance
        right = covariance @ self.second
        # Large flattened Metal dot products can corrupt the first metric.
        # Reduce the pupil rows separately before summing each metric.
        quadratic = torch.linalg.vecdot(left, right, dim=-1).sum(dim=-1)
        return (linear + 0.5 * quadratic).cpu().numpy().astype(float)


class MpsRemainderQuadrature:
    """Reuse one point's Sobol draws and metric weights across its sampled lags.

    Each Metal SIMD group reduces 32 consecutive terms. The CPU combines
    these partial sums in float64 without mixing scrambles or the coarse net.
    """

    def __init__(self, indices, weights, probability, variants):
        self.kernel = _quartet_kernel()
        self.replicates, self.count = len(indices), len(indices[0])
        self.nodes = len(probability)
        draws = np.stack(indices).view(np.int16)
        self.indices = torch.as_tensor(draws, device="mps")
        self.weights = torch.as_tensor(
            np.ascontiguousarray(weights, dtype=np.complex64).view(np.float32),
            device="mps",
        )
        self.probability = torch.as_tensor(
            probability, dtype=torch.float32, device="mps"
        )
        self.partial = torch.empty(
            (variants, 3, self.replicates, self.count // 32),
            dtype=torch.float32,
            device="mps",
        )

    def __call__(self, covariances):
        variants = len(covariances)
        covariances = torch.as_tensor(covariances, dtype=torch.float32, device="mps")
        draws = self.replicates * self.count
        self.kernel(
            self.indices,
            covariances,
            self.weights,
            self.probability,
            self.partial,
            self.nodes,
            variants,
            draws,
            threads=draws,
            group_size=256,
        )
        partial = self.partial[:variants].cpu().numpy()
        fine = partial.sum(axis=-1, dtype=np.float64) / self.count
        coarse = partial[..., : self.count // 128].sum(axis=-1, dtype=np.float64) / (
            self.count // 4
        )
        return fine.transpose(2, 0, 1), coarse.transpose(2, 0, 1)
