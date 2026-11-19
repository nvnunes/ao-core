"""Evaluate shared Fast science fields in bounded Metal batches."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from ._tensor_batch import TENSOR_UPLOAD_BLOCK_BYTES, TensorBatch
from .metrics import phase_responses
from .pupil import ARCSEC

if TYPE_CHECKING:
    import torch


@dataclass
class PreparedBatch:
    """Device-resident lag cube and the small inputs to the CPU optical stage."""

    lags: torch.Tensor
    optical_inputs: torch.Tensor
    started: float


class MPSFastBatch:
    """Reuse one device reconstruction for the restored mean and modal lags."""

    def __init__(self, science, predictor):
        self.science = science
        self.wavelength_m = predictor.metadata["wavelength_m"]
        self.backend = TensorBatch(
            self.science.context,
            predictor.dt,
            predictor.max_count,
            predictor._noise_lags,
            (),
            "mps",
            consume_reconstruction=True,
        )
        self.weights = self.backend.tensor(self.science.context.weights)
        nodes = len(science.wave_coordinates)
        features = science.wave_vectors.shape[1]
        self.waves = self.backend.torch.empty(
            (nodes, 2 * features),
            dtype=self.backend.real_dtype,
            device=self.backend.device,
        )
        block = max(1, TENSOR_UPLOAD_BLOCK_BYTES // (16 * features))
        for start in range(0, nodes, block):
            rows = slice(start, start + block)
            self.waves[rows].copy_(
                self.backend.tensor(science.wave_rows(rows).view(float))
            )

    def _restore(self, phase, correction):
        """Calculate the two restoration terms on Metal."""
        backend = self.backend
        torch = backend.torch
        science = self.science
        batch, modes, _ = correction.shape
        tilt = backend.lo[None].expand(batch, -1, -1)
        command = torch.cat((tilt, correction), dim=1)
        weighted = command * self.weights
        covariance = (weighted @ command.mH).real * backend.scale**2
        covariance[:, :2, :2] += backend.noise[0]
        mixed = (
            (weighted.conj() * phase[:, None])
            .reshape(batch, modes + 2, science.layers, science.radial, science.angular)
            .sum(dim=2)
        )
        half = science.angular // 2
        if science.pair_waves:
            mixed = mixed[..., :half] + mixed[..., half:].conj()
        real_mixed = torch.view_as_real(mixed.conj().resolve_conj()).reshape(
            batch * (modes + 2), -1
        )
        projection = self.waves @ real_mixed.T
        return covariance, projection.T.reshape(batch, -1)

    def prepare(self, positions, *, remove_tilt):
        """Enqueue one reconstruction and pack one host handoff."""
        started = perf_counter()
        backend = self.backend
        phase, high, correction = backend.direction_terms(
            np.asarray(positions) * ARCSEC
        )
        covariances, projected = self._restore(phase, correction)
        high -= correction
        del correction
        tilt = backend.original[None, :2] * phase[:, None] - backend.lo[None]
        del phase
        response = backend.torch.cat((tilt, high), dim=1)
        response *= backend.scale
        del tilt, high
        lags = backend.lags_from_response(response)
        del response
        if remove_tilt:
            lags[:, :, :2, :] = 0
            lags[:, :, :, :2] = 0
        packed = backend.torch.cat(
            (covariances.flatten(1), projected, lags[:, 0].flatten(1)), dim=1
        )
        return PreparedBatch(lags, packed, started)

    def means(self, positions, optics, *, phase_scale=1.0):
        """Evaluate means directly from the restored phase, without computing modal lags."""
        phase, high, correction = self.backend.direction_terms(
            np.asarray(positions) * ARCSEC
        )
        del high
        covariances, projection = self._restore(phase, correction)
        del phase, correction
        modes = self.backend.nmodes
        packed = (
            self.backend.torch.cat((covariances.flatten(1), projection), dim=1)
            .cpu()
            .numpy()
            .astype(float)
        )
        covariances, projected = np.split(packed, (modes**2,), axis=1)
        projected = projected.reshape(len(positions), modes, -1).transpose(0, 2, 1)
        means = []
        for cov, proj in zip(
            covariances.reshape(-1, modes, modes), projected, strict=True
        ):
            structure = self.science._phase_structure(cov, proj)
            if phase_scale != 1.0:
                structure *= phase_scale**2
            means.append(optics.metric_response(structure, means_only=True)[0])
        return means

    def optical_inputs(self, batch):
        """Download all data required for CPU optics with one synchronization."""
        modes = self.backend.nmodes
        covariances, projected, modal = np.split(
            batch.optical_inputs.cpu().numpy().astype(float),
            (modes**2, batch.optical_inputs.shape[1] - modes**2),
            axis=1,
        )
        return (
            covariances.reshape(-1, modes, modes),
            projected.reshape(len(projected), modes, -1).transpose(0, 2, 1),
            modal.reshape(-1, modes, modes),
        )

    def evaluate(self, batch, inputs, optics_by_key, users, positions):
        """Score the requested optics using one set of atmospheric lags."""
        backend = self.backend
        responses = {key: [] for key in optics_by_key}
        indices_by_optics = {key: [] for key in optics_by_key}
        for i, (cov, proj, modal) in enumerate(zip(*inputs, strict=True)):
            structure = self.science._phase_structure(cov, proj)
            for key in users[positions[i]]:
                optics = optics_by_key[key]
                phase_scale = self.wavelength_m / key[0]
                scaled = structure if phase_scale == 1.0 else structure * phase_scale**2
                means, kernels, coherence = optics.metric_response(
                    scaled, lag_grid=True
                )
                first, second, tails = phase_responses(
                    optics.pupil.basis,
                    modal,
                    kernels,
                    coherence,
                    tail_order=6,
                    lag_indices=optics.flat_lag.reshape(coherence.shape),
                )
                responses[key].append((means, first, second, tails))
                indices_by_optics[key].append(i)

        contractions = []
        for key, response in responses.items():
            if not response:
                continue
            indices = indices_by_optics[key]
            lags = batch.lags if len(indices) == len(positions) else batch.lags[indices]
            scale = self.wavelength_m / key[0]
            contractions.append(
                (
                    key,
                    indices,
                    response,
                    backend.scalar_covariance(lags, response, phase_scale=scale),
                )
            )
        # All optics use the same native lag grid, so their small results can
        # return in one copy. The modal lag cubes stay on Metal.
        values = backend.torch.cat([item[3] for item in contractions]).cpu().numpy()
        indices = np.arange(backend.retained)
        rows = []
        offset = 0
        for key, selected, response, _ in contractions:
            stop = offset + len(selected)
            rows.extend(
                (key, positions[i], means, (indices, series))
                for i, (means, *_), series in zip(
                    selected, response, values[offset:stop], strict=True
                )
            )
            offset = stop
        return rows

    def evaluate_many(self, batches, optics, users, *, remove_tilt, pipeline):
        """Overlap CPU optics with at most one queued Metal batch."""
        batches = iter(batches)
        positions = next(batches)
        current = self.prepare(positions, remove_tilt=remove_tilt)
        while current is not None:
            inputs = self.optical_inputs(current)
            following = next(batches, None)
            pending = (
                self.prepare(following, remove_tilt=remove_tilt)
                if pipeline and following
                else None
            )
            results = self.evaluate(current, inputs, optics, users, positions)
            yield (
                results,
                (perf_counter() - current.started) / len(positions),
                len(positions),
            )
            current = pending
            if following and current is None:
                current = self.prepare(following, remove_tilt=remove_tilt)
            positions = following
