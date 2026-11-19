"""Restore science pupil coherence from modal control and the layered atmosphere."""

from functools import cached_property

import numpy as np
from astropy import units as u
from scipy.spatial.distance import cdist

from .atmosphere import von_karman_correlation, von_karman_variance
from .pupil import ARCSEC, CORRECTION_CUTOFF, quadrature
from .residual import source_minus_correction_covariance

RESTORATION_TILE_NODES = 8192


def coherence_from_structure(geometry, structure, variance_scale=1.0):
    """Apply science-band variance scaling and the pupil's static phase."""
    return geometry.amplitude_product * np.exp(
        -0.5 * variance_scale * structure + geometry.static_phase
    )


class FullScienceMean:
    """Restore the zero-lag science coherence using the finite AO correction."""

    def __init__(
        self,
        context,
        geometry,
        atmosphere,
        tilt_noise,
        *,
        radial_order,
        angular_order,
        angular_offset=0.0,
        pupil_offsets=None,
    ):
        self.context = context
        self.geometry = geometry
        self.tilt_noise = tilt_noise
        angular = angular_order
        fx, fy, _ = quadrature(radial_order, angular, CORRECTION_CUTOFF, angular_offset)
        pupil = geometry.pupil
        self.angular = getattr(self.context, "angular_samples", angular)
        self.pair_waves = self.angular == angular and angular % 2 == 0
        self.radial = radial_order
        if angular % 2 == 0:
            # Opposite Fourier nodes have conjugate pupil waves. Retain one
            # from each pair and combine their coefficients before projection.
            fx = fx.reshape(self.radial, angular)[:, : angular // 2].ravel()
            fy = fy.reshape(self.radial, angular)[:, : angular // 2].ravel()
        self.wave_vectors = np.array([fx, fy])
        self.wave_coordinates = pupil.coordinates
        self.layers = len(atmosphere.layer_weights)
        if self.radial * self.angular * self.layers != len(self.context.weights):
            raise ValueError(
                "The science mean and time covariance require the same Fourier grid"
            )
        if getattr(self.context, "instantaneous", False):
            weights, lo = self.context.weights, self.context.lo
            self._root_weights = np.sqrt(weights)
            self._lo_correction_covariance = (lo * weights).view(float) @ lo.view(
                float
            ).T
        # The analytic von Kármán transform includes phase above the AO cutoff.
        inverse_outer_scale = 1 / float(atmosphere.outer_scale.to_value(u.m))
        if pupil_offsets is None:
            distances = cdist(pupil.coordinates, pupil.coordinates)
        else:
            offsets, pairs = pupil_offsets
            distances = np.hypot(offsets[:, 0], offsets[:, 1])
        correlation = von_karman_correlation(distances, inverse_outer_scale)
        amplitude = von_karman_variance(
            float(atmosphere.fried_parameter.to_value(u.m)), inverse_outer_scale
        )
        self.atmospheric = (
            correlation
            * amplitude
            * np.sum(atmosphere.layer_weights.value)
            * self.context.scale**2
        )
        if pupil_offsets is not None:
            self.atmospheric = self.atmospheric[pairs]

    def wave_rows(self, rows, columns=slice(None)):
        """Prepare a bounded block of the fixed pupil Fourier projection."""
        waves = (
            2j * np.pi * (self.wave_coordinates[rows] @ self.wave_vectors[:, columns])
        )
        np.exp(waves, out=waves)
        return waves

    @cached_property
    def waves(self):
        return self.wave_rows(slice(None))

    def components(self, xy, *, terms=None):
        """Reduce one direction to its modal covariance and Fourier cross term."""
        context = self.context
        if terms is None:
            terms = context.direction_terms(np.asarray(xy) * ARCSEC)
        phase, _, high = terms
        correction = np.vstack([context.lo, high])
        weighted = np.multiply(correction, context.weights, dtype=phase.dtype)
        correction_cov = (weighted @ correction.conj().T).real * context.scale**2
        correction_cov[:2, :2] += self.tilt_noise
        np.conjugate(weighted, out=weighted)
        weighted *= phase
        mixed_modes = weighted.reshape(context.nmodes, self.layers, -1).sum(1)
        if self.pair_waves:
            half = self.angular // 2
            modes = mixed_modes.reshape(context.nmodes, self.radial, self.angular)
            mixed_modes = (modes[:, :, :half] + modes[:, :, half:].conj()).reshape(
                context.nmodes, -1
            )
        return correction_cov, mixed_modes

    def coherences(self, components):
        """Project a bounded batch using one shared pupil-wave matrix product."""
        coefficients = (
            components[0][1]
            if len(components) == 1
            else np.concatenate([modes for _, modes in components], axis=0)
        )
        projected = (self.waves @ coefficients.T).real
        for (correction_cov, _), projection in zip(
            components, np.split(projected, len(components), axis=1), strict=True
        ):
            yield self._coherence(correction_cov, projection)

    def _coherence(self, correction_cov, projection):
        structure = self._phase_structure(correction_cov, projection)
        return coherence_from_structure(self.geometry, structure)

    def _phase_structure(self, correction_cov, projection):
        basis = self.geometry.pupil.basis
        mixed = projection @ basis * self.context.scale**2
        residual = source_minus_correction_covariance(
            self.atmospheric, basis.T @ correction_cov @ basis, mixed, mixed.T
        )
        variance = residual.diagonal()
        structure = variance[:, None] + variance[None, :] - 2 * residual
        if not np.isfinite(structure).all() or np.min(structure) < -1e-8:
            raise ValueError("Invalid restored science phase structure function")
        return structure

    def tiled_coherences(self, positions, *, tile_nodes=RESTORATION_TILE_NODES):
        """Restore full-pupil coherence for arcsecond positions."""
        coherences, _ = self.tiled_restoration(positions, tile_nodes=tile_nodes)
        return coherences

    def tiled_restoration(
        self, positions, *, lag_weights=None, tile_nodes=RESTORATION_TILE_NODES
    ):
        """Restore coherence and optionally modal zero lag in bounded Fourier tiles."""
        structures, lags = self.tiled_phase_structures(
            positions, lag_weights=lag_weights, tile_nodes=tile_nodes
        )
        return [
            coherence_from_structure(self.geometry, structure)
            for structure in structures
        ], lags

    def tiled_phase_structures(
        self,
        positions,
        *,
        lag_weights=None,
        spectrum=None,
        tile_nodes=RESTORATION_TILE_NODES,
    ):
        """Restore phase structure at the reference wavelength in bounded Fourier tiles.

        Positions are in arcseconds. The modal path takes the weights of the
        native zero-lag Fourier series and returns lags with shape
        (point, 1, mode, mode). The full-pupil path passes no lag weights.
        A compact temporal grid can accumulate spectra into a zero-initialized
        (point, frequency, mode, mode) array during the same restoration pass.
        At fixed AO configuration both outputs scale with inverse wavelength squared.
        """
        context = self.context
        directions = np.asarray(positions, dtype=float) * ARCSEC
        batch = len(directions)
        if batch == 0:
            return [], None
        modes, high_modes = context.nmodes, context.nmodes - 2
        instantaneous = getattr(context, "instantaneous", False)
        compact = getattr(context, "compact", False)
        factored = instantaneous or compact
        if instantaneous and lag_weights is not None:
            raise ValueError("Modal lags require the temporal Fourier grid")
        if spectrum is not None and (
            not compact or spectrum.shape != (batch, context.bin_count, modes, modes)
        ):
            raise ValueError(
                "Tiled spectra require one output per point on a compact temporal grid"
            )
        rings_per_tile = max(1, tile_nodes // self.angular)
        layer_size = self.radial * self.angular
        opposite_waves = self.pair_waves
        wave_columns = self.angular // 2 if opposite_waves else self.angular
        rings = [
            slice(start, min(start + rings_per_tile, self.radial))
            for start in range(0, self.radial, rings_per_tile)
        ]

        def samples_for(layer, nodes):
            offset = layer * layer_size
            return slice(offset + nodes.start, offset + nodes.stop)

        target = np.zeros((batch, high_modes, context.inverse.shape[0]))
        for layer in range(self.layers):
            for ring in rings:
                nodes = slice(ring.start * self.angular, ring.stop * self.angular)
                samples = samples_for(layer, nodes)
                phase = np.exp(2j * np.pi * (directions @ context.spatial[samples].T))
                if factored:
                    weighted_phase = phase * context.weights[samples]
                    high = (
                        weighted_phase[:, None, :] * context.original[None, 2:, nodes]
                    )
                    measured = context.lgs_measurement(samples).view(float)
                else:
                    high = phase[:, None, :] * context.original[None, 2:, samples]
                    measured = (
                        context.weighted_measurement[:, samples].conj().view(float)
                    )
                # Interleaved real/imaginary columns give the real inner
                # product without evaluating or storing the imaginary result.
                cross = high.reshape(batch * high_modes, -1).view(float) @ measured.T
                target += cross.reshape(target.shape)
                del high, measured
        reconstructor = (target @ context.inverse).reshape(batch * high_modes, -1)

        covariance = np.zeros((batch, modes, modes))
        if instantaneous:
            covariance[:, :2, :2] = self._lo_correction_covariance
        modal_covariance = (
            np.zeros_like(covariance) if lag_weights is not None else None
        )
        pupil_nodes = len(self.wave_coordinates) if compact else len(self.waves)
        projection = np.zeros((pupil_nodes, batch * modes))
        for ring in rings:
            ring_count = ring.stop - ring.start
            nodes = slice(ring.start * self.angular, ring.stop * self.angular)
            mixed = np.zeros((batch, modes, ring_count, wave_columns), dtype=complex)
            for layer in range(self.layers):
                samples = samples_for(layer, nodes)
                phase = np.exp(2j * np.pi * (directions @ context.spatial[samples].T))
                common = context.common[:, samples]
                if factored:
                    residual = context.factored_residual(samples, common)
                else:
                    residual = context.residual[:, samples]
                # Reconstruction coefficients are real. Apply them to both
                # residual components together and recover the complex layout.
                high = (
                    (reconstructor @ residual.view(float))
                    .view(complex)
                    .reshape(batch, high_modes, -1)
                )
                del residual
                high += common
                lo = context.lo[:, samples]
                if instantaneous:
                    # LO is field-independent. Keep it separate so the
                    # high-order result can be weighted and reused in place.
                    root = self._root_weights[samples]
                    high *= root
                    lo_weighted = lo * root
                    gram = high.view(float)
                    covariance[:, 2:, 2:] += gram @ gram.swapaxes(-1, -2)
                    cross = gram @ lo_weighted.view(float).T
                    covariance[:, 2:, :2] += cross
                    covariance[:, :2, 2:] += cross.swapaxes(-1, -2)
                    phase_weighted = (phase * root)[:, None, :]
                    np.conjugate(high, out=high)
                    high *= phase_weighted
                    mixed[:, 2:] += high.reshape(
                        batch, high_modes, ring_count, self.angular
                    )
                    low = lo_weighted.conj() * phase_weighted
                    mixed[:, :2] += low.reshape(batch, 2, ring_count, self.angular)
                    del high, gram
                    continue
                correction = np.empty((batch, modes, phase.shape[-1]), dtype=complex)
                correction[:, :2] = lo
                correction[:, 2:] = high
                del high
                if modal_covariance is not None or spectrum is not None:
                    original = context.original[:, nodes if compact else samples]
                    modal = phase[:, None, :] * original[None] - correction
                if modal_covariance is not None:
                    modal_covariance += (modal * lag_weights[samples]).view(
                        float
                    ) @ modal.view(float).swapaxes(-1, -2)
                if spectrum is not None:
                    context.accumulate_spectrum(spectrum, modal, samples)
                weighted = correction * context.weights[samples]
                covariance += weighted.view(float) @ correction.view(float).swapaxes(
                    -1, -2
                )
                np.conjugate(weighted, out=weighted)
                weighted *= phase[:, None, :]
                coefficients = weighted.reshape(batch, modes, ring_count, self.angular)
                if opposite_waves:
                    mixed += coefficients[..., :wave_columns]
                    mixed += coefficients[..., wave_columns:].conj()
                else:
                    mixed += coefficients
                del correction, weighted, coefficients
            columns = slice(ring.start * wave_columns, ring.stop * wave_columns)
            waves = (
                self.wave_rows(slice(None), columns)
                if compact
                else self.waves[:, columns]
            )
            np.conjugate(mixed, out=mixed)
            projection += (
                waves.view(float) @ mixed.reshape(batch * modes, -1).view(float).T
            )
            del mixed

        covariance *= context.scale**2
        if spectrum is not None:
            spectrum *= context.scale**2
            spectrum[:, 0] = spectrum[:, 0].real
        covariance[:, :2, :2] += self.tilt_noise
        structures = [
            self._phase_structure(cov, projected)
            for cov, projected in zip(
                covariance, np.split(projection, batch, axis=1), strict=True
            )
        ]
        if modal_covariance is not None:
            modal_covariance *= context.scale**2
            modal_covariance[:, :2, :2] += self.tilt_noise
            return structures, modal_covariance[:, None]
        return structures, None

    def coherence(self, xy, *, terms=None):
        """Include atmospheric/correction cross terms and static NCPA."""
        if getattr(self.context, "instantaneous", False):
            return self.tiled_coherences([xy])[0]
        return next(self.coherences([self.components(xy, terms=terms)]))
