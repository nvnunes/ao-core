"""Bounded tensor batches for the retained modal sampling uncertainty model.

CPU tensors use double precision as a numerical reference. MPS tensors use
single precision; covariance factorization and fixed reconstruction products
are prepared in NumPy double precision before transfer. No device fallback or
coarser spatial/temporal grid is selected here.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.fft import next_fast_len

TENSOR_LAG_WORK_ELEMENTS = 1 << 22
TENSOR_QUADRATIC_WORK_BYTES = 128 * 1024**2
TENSOR_UPLOAD_BLOCK_BYTES = 64 * 1024**2


class TensorBatch:
    """Own one grid's fixed device tensors and evaluate explicit point batches.

    Spectrum shape is (point, frequency, mode, mode), lag shape is
    (point, native lag, mode, mode). Callers bound the number of points. The
    small optical response calculation remains on CPU between lags() and
    integrate(); full lag arrays remain on the selected device.
    """

    def __init__(
        self,
        context,
        native_dt,
        retained,
        noise,
        windows,
        device,
        *,
        consume_reconstruction=False,
    ):
        import torch

        if device not in ("cpu", "mps"):
            raise ValueError("Tensor batches require cpu or mps")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but no Metal device is available")
        self.torch, self.device = torch, torch.device(device)
        self.real_dtype = torch.float64 if device == "cpu" else torch.float32
        self.complex_dtype = torch.complex128 if device == "cpu" else torch.complex64
        self.nmodes, self.bin_count = context.nmodes, context.bin_count
        self.count, self.retained = round(context.period / native_dt), retained
        self.native_dt, self.scale = native_dt, context.scale
        if not 0 < retained <= self.count:
            raise ValueError("Retained lags must lie within the Fourier period")
        self.spatial = self.tensor(context.spatial)
        self.original = self.tensor(context.original)
        self.common, self.lo = self.tensor(context.common), self.tensor(context.lo)
        # Inverse = F F.T. Moving its two fixed factors into the Fourier
        # responses avoids applying an ill-conditioned inverse in float32.
        factor = context.reconstruction_factor
        measured = context.weighted_measurement
        self.measured_factor = self._prepare_factor(measured, factor, transpose=True)
        del measured
        if consume_reconstruction:
            context.weighted_measurement = None
        self.residual_factor = self._prepare_factor(
            context.residual, factor, transpose=False
        )
        if consume_reconstruction:
            context.residual = None
        self.noise = self.tensor(noise)
        self.windows = [
            (exposure, count, self.tensor(weights))
            for exposure, count, weights in windows
        ]
        self._context = context
        self.buckets = self._buckets(context) if retained > 1 else None
        self._spectrum_buffer = None
        self.zero_weights = (
            self.tensor(np.sqrt(context.zero_lag_weights(self.count)))
            if retained == 1
            else None
        )
        self.short_window = 1 < retained < self.count // 4
        if self.short_window:
            bins = min(self.bin_count, self.count // 2 + 1)
            self.transform_bins = bins
            self.transform_size = next_fast_len(bins + retained - 1)
            self.input_chirp = self.tensor(
                np.exp(1j * np.pi * np.arange(bins) ** 2 / self.count)
            )
            self.output_chirp = self.tensor(
                np.exp(1j * np.pi * np.arange(retained) ** 2 / self.count)
            )
            offsets = np.arange(-(bins - 1), retained)
            convolution = self.tensor(np.exp(-1j * np.pi * offsets**2 / self.count))
            self.convolution_fft = torch.fft.fft(convolution, n=self.transform_size)

    def tensor(self, value):
        """Copy a numerical array using this backend's explicit precision."""
        value = np.asarray(value)
        dtype = self.complex_dtype if np.iscomplexobj(value) else self.real_dtype
        return self.torch.as_tensor(value, dtype=dtype, device=self.device)

    def _prepare_factor(self, matrix, factor, *, transpose):
        """Project in float64 on CPU and bound host staging for large Metal buffers."""
        if self.device.type != "mps" or matrix.nbytes <= 8 * TENSOR_UPLOAD_BLOCK_BYTES:
            return self.tensor(matrix.T @ factor if transpose else factor.T @ matrix)
        samples, rank = matrix.shape[1], factor.shape[1]
        shape = (samples, rank) if transpose else (rank, samples)
        projected = self.torch.empty(
            shape, dtype=self.complex_dtype, device=self.device
        )
        block = max(1, TENSOR_UPLOAD_BLOCK_BYTES // (matrix.shape[0] * matrix.itemsize))
        for start in range(0, samples, block):
            rows = slice(start, start + block)
            values = matrix[:, rows]
            if transpose:
                projected[rows].copy_(self.tensor(values.T @ factor))
            else:
                projected[:, rows].copy_(self.tensor(factor.T @ values))
        return projected

    def _buckets(self, context):
        """Pad similar-size temporal bins for a few batched Gram products.

        Power-of-two buckets bound padding below twice the number of samples.
        Zero-frequency samples retain their original real-only contribution.
        """
        torch = self.torch
        groups = {}
        selections = [(0, np.flatnonzero(context.zero))]
        selections += [
            (int(index), context.positive[start:stop])
            for index, start, stop in zip(
                context.bin_values, context.starts, context.ends, strict=True
            )
        ]
        for index, selected in selections:
            if not len(selected):
                continue
            width = 1 << (len(selected) - 1).bit_length()
            groups.setdefault(width, []).append((index, selected))
        buckets = []
        for width, records in sorted(groups.items()):
            indices = np.zeros((len(records), width), dtype=np.int64)
            weights = np.zeros_like(indices, dtype=float)
            for row, (_, selected) in enumerate(records):
                indices[row, : len(selected)] = selected
                weights[row, : len(selected)] = np.sqrt(context.weights[selected])
            bins = torch.tensor([index for index, _ in records], device=self.device)
            buckets.append(
                (
                    bins,
                    torch.as_tensor(indices, device=self.device),
                    self.tensor(weights),
                )
            )
        return buckets

    def direction_terms(self, directions):
        """Return the field phase, uncorrected HO response and HO correction."""
        torch = self.torch
        directions = np.asarray(directions, float)
        if (
            directions.ndim != 2
            or directions.shape[1] != 2
            or not np.isfinite(directions).all()
        ):
            raise ValueError("Require a finite batch-by-two direction array")
        angle = (self.spatial @ self.tensor(directions).T).T
        phase = torch.exp(2j * np.pi * angle)
        high = self.original[None, 2:] * phase[:, None]
        batch, modes, samples = high.shape
        target = (high.reshape(batch * modes, samples) @ self.measured_factor).real
        correction = (target.to(self.complex_dtype) @ self.residual_factor).reshape(
            batch, modes, samples
        )
        correction += self.common[None]
        return phase, high, correction

    def residual_response(self, directions):
        """Return the corrected modal Fourier response of each science direction."""
        phase, high, correction = self.direction_terms(directions)
        high -= correction
        tilt = self.original[None, :2] * phase[:, None] - self.lo[None]
        return self.torch.cat((tilt, high), dim=1) * self.scale

    def point_lags(self, directions):
        """Evaluate only zero lag for single-frame batches."""
        return self.lags_from_response(self.residual_response(directions))

    def lags_from_response(self, response):
        """Evaluate native lags for an already reconstructed modal response."""
        if self.zero_weights is None:
            return self.lags(self.spectrum_from_response(response))
        weighted = response * self.zero_weights
        lag = (weighted @ weighted.mH).real[:, None].contiguous()
        lag[:, :, :2, :2] += self.noise
        return lag

    def spectrum(self, directions):
        """Return a borrowed spectrum buffer, valid until the next spectrum call."""
        return self.spectrum_from_response(self.residual_response(directions))

    def spectrum_from_response(self, response):
        """Bin a reconstructed response into the reusable spectrum buffer."""
        torch = self.torch
        batch = len(response)
        if self.buckets is None:
            self.buckets = self._buckets(self._context)
        if self._spectrum_buffer is None or len(self._spectrum_buffer) < batch:
            self._spectrum_buffer = torch.zeros(
                (batch, self.bin_count, self.nmodes, self.nmodes),
                dtype=self.complex_dtype,
                device=self.device,
            )
        # Active bins are overwritten below; inactive bins stay zero. lags()
        # copies its Fourier input, so the buffer can serve the next batch.
        spectrum = self._spectrum_buffer[:batch]
        for bins, indices, weights in self.buckets:
            values = (
                response[:, :, indices].permute(0, 2, 1, 3) * weights[None, :, None]
            )
            flat = values.reshape(-1, self.nmodes, values.shape[-1])
            gram = torch.bmm(flat, flat.mH).reshape(
                batch, len(bins), self.nmodes, self.nmodes
            )
            spectrum[:, bins] = gram
        # irfft treats the DC bin as real, including its off-diagonal entries.
        spectrum[:, 0] = spectrum[:, 0].real.to(self.complex_dtype)
        return spectrum

    def _lag_block(self, spectrum):
        torch = self.torch
        if self.short_window:
            positive = spectrum[:, : self.transform_bins].movedim(1, -1).contiguous()
            shape = positive.shape[:-1]
            positive = positive.reshape(-1, self.transform_bins)
            positive[..., 0] *= 0.5
            if self.count % 2 == 0 and self.transform_bins == self.count // 2 + 1:
                positive[..., -1] *= 0.5
            transformed = torch.fft.fft(
                positive * self.input_chirp, n=self.transform_size
            )
            transformed *= self.convolution_fft
            convolved = torch.fft.ifft(transformed)
            start = self.transform_bins - 1
            lag = (
                2
                * (
                    convolved[..., start : start + self.retained] * self.output_chirp
                ).real
            )
            lag = lag.reshape(*shape, self.retained)
            lag = lag.movedim(-1, 1).contiguous()
        else:
            positive = spectrum.movedim(1, -1).contiguous()
            shape = positive.shape[:-1]
            flat = positive.reshape(-1, self.bin_count)
            lag = (
                torch.fft.irfft(flat, n=self.count, dim=-1)[:, : self.retained]
                * self.count
            )
            lag = lag.reshape(*shape, self.retained).movedim(-1, 1).contiguous()
        return lag

    def lags(self, spectrum):
        """Evaluate lags in mode blocks to bound device FFT scratch space."""
        batch = len(spectrum)
        block_size = max(
            1,
            TENSOR_LAG_WORK_ELEMENTS
            // (batch * self.nmodes * max(self.count, self.bin_count)),
        )
        if block_size >= self.nmodes:
            lag = self._lag_block(spectrum)
        else:
            lag = self.torch.empty(
                (batch, self.retained, self.nmodes, self.nmodes),
                dtype=self.real_dtype,
                device=self.device,
            )
            for start in range(0, self.nmodes, block_size):
                rows = slice(start, start + block_size)
                lag[:, :, rows] = self._lag_block(spectrum[:, :, rows])
        lag[:, :, :2, :2] += self.noise
        return lag

    def covariances(self, lag):
        """Transfer only zero-lag covariances for the small CPU optical stage."""
        return lag[:, 0].cpu().numpy().astype(float)

    def scalar_covariance(self, lag, responses, *, phase_scale=1.0):
        """Contract metric derivatives, optionally against lags in reference phase units."""
        torch = self.torch
        modes, batch = self.nmodes, len(responses)
        orders = range(3, 3 + len(responses[0][3][1][0]))
        sizes = [
            3 * modes,
            3 * modes * modes,
            modes * 2,
            *(3 * 2**order for order in orders),
        ]
        packed = np.stack(
            [
                np.concatenate(
                    [
                        first.ravel() * phase_scale,
                        second.ravel() * phase_scale**2,
                        projection.ravel(),
                        *(
                            np.stack(terms).ravel() * phase_scale**order
                            for order, terms in zip(
                                orders, zip(*tails, strict=True), strict=True
                            )
                        ),
                    ]
                )
                for _, first, second, (projection, tails) in responses
            ]
        )
        first, second, projection, *higher = self.tensor(packed).split(sizes, dim=1)
        first, second = (
            first.reshape(batch, 3, modes),
            second.reshape(batch, 3, modes, modes),
        )
        projection = projection.reshape(batch, modes, 2)
        total = torch.einsum("bmi,btij,bmj->bmt", first, lag, first)
        bytes_per_lag = batch * 3 * modes * modes * lag.element_size()
        block_size = max(1, TENSOR_QUADRATIC_WORK_BYTES // bytes_per_lag)
        hessian = second[:, :, None]
        for start in range(0, lag.shape[1], block_size):
            window = slice(start, start + block_size)
            block = lag[:, None, window]
            left, right = hessian @ block, hessian @ block.transpose(-1, -2)
            total[:, :, window] += 0.5 * (left * right.transpose(-1, -2)).sum(
                dim=(-1, -2)
            )
        reduced = torch.einsum("bir,btij,bjs->btrs", projection, lag, projection)
        for order, values in zip(orders, higher, strict=True):
            tensor = values.reshape(batch, 3, *((2,) * order))
            transformed = tensor[:, :, None].expand(
                -1, -1, lag.shape[1], *tensor.shape[2:]
            )
            for axis in range(3, order + 3):
                transformed = transformed.movedim(axis, -1)
                transformed = torch.einsum(
                    "btij,bmt...j->bmt...i", reduced, transformed
                )
                transformed = transformed.movedim(-1, axis)
            total += (transformed * tensor[:, :, None]).flatten(3).sum(
                -1
            ) / math.factorial(order)
        return total

    def integrate(self, lag, responses):
        """Return small CPU exposure integrals; keep the modal lags on device."""
        total = self.scalar_covariance(lag, responses)
        return [
            (
                exposure,
                (2 * self.native_dt * (total[:, :, :count] @ weights)).cpu().numpy(),
            )
            for exposure, count, weights in self.windows
        ]
