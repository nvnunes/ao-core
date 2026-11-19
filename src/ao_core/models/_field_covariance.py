"""Modal atmospheric and controlled covariance across field directions."""

from __future__ import annotations

from copy import copy

import numpy as np
from astropy import units as u

from .atmosphere import frozen_flow_frequency, von_karman_spectrum
from .config import Configuration
from .control import polc_transfer
from .pupil import CORRECTION_CUTOFF, ModalPupil, quadrature
from .reconstruction import lo_reconstructor

SETUP_TILE_NODES = 8192


def _required(record, name, *fields):
    if record is None or any(getattr(record, field) is None for field in fields):
        raise ValueError(f"{name} requires {', '.join(fields)}")
    return record


def _positive_wavelength(wavelength):
    metres = float(wavelength.to_value(u.m))
    if not np.isfinite(metres) or metres <= 0:
        raise ValueError("Wavelength must be a positive finite scalar")
    return metres


def _frame_delay(value):
    if value is None or value < 1 or value % 1 != 0:
        raise ValueError("The modal controller requires a positive whole-frame delay")
    return int(value)


def _polar_response(pupil, fx, fy, angular):
    """Evaluate modal Fourier responses on polar rings, pairing opposite nodes."""
    if angular % 2:
        return pupil.response(fx, fy)
    half = angular // 2
    first_x = fx.reshape(-1, angular)[:, :half].ravel()
    first_y = fy.reshape(-1, angular)[:, :half].ravel()
    first = pupil.response(first_x, first_y).reshape(len(pupil.basis), -1, half)
    response = np.empty((*first.shape[:2], angular), dtype=first.dtype)
    response[..., :half] = first
    np.conjugate(first, out=response[..., half:])
    return response.reshape(len(pupil.basis), -1)


def _ideal_modal_lgs_factors(
    fx,
    fy,
    layer_height_m,
    beacon_height_m,
    directions_rad,
    *,
    project_pupil,
    uncontracted_modes,
):
    """Return the higher-order modal amplitudes and beacon phases for one layer.

    The spatial frequencies are in inverse metres. The beacon height and
    layer height are in metres in the supplied propagation geometry. The
    factors have shapes (higher-order mode, sample) and (beacon, sample).
    The pupil projector uses the caller's existing Fourier grid convention.
    """
    cone = 1 - layer_height_m / beacon_height_m
    modes = (uncontracted_modes if cone == 1 else project_pupil(cone * fx, cone * fy))[
        2:
    ]
    phase = np.exp(
        2j
        * np.pi
        * layer_height_m
        * (directions_rad[:, :1] * fx + directions_rad[:, 1:] * fy)
    )
    return modes, phase


def _expand_modal_lgs_response(modes, phase, *, out=None):
    """Form beacon-major WFS channels by multiplying the two layer factors."""
    shape = (len(phase), *modes.shape)
    if out is not None:
        out = out.reshape(shape)
    response = np.multiply(phase[:, None], modes[None], out=out)
    return response.reshape(shape[0] * shape[1], shape[2])


class FieldCovariance:
    """Shared modal atmospheric residual, with field-dependent MOAO correction.

    An instantaneous grid keeps one node per conjugate pair on even polar
    rings and doubles its weight for real zero-lag contractions.
    An optional tilt_noise_prior specifies reconstruction-design covariance in
    reference-wavelength modal phase radians squared. Runtime detector noise is
    propagated separately; omitting the prior preserves the noise-free model.
    Supply exactly four laser guide-star directions and a finite positive
    beacon height projected for the observing geometry. Direction queries
    use radians. Spectra use the modal basis in radians squared at the
    requested wavelength. With common_only, the residual includes only
    the common high-order correction. A temporal grid with dt=None retains
    only the occupied range of frequency bins, bounded by the native
    controller cadence. For common-only or instantaneous calculations,
    period=None prepares covariance
    without generating temporal bins.
    """

    compact = False

    def __init__(
        self,
        configuration: Configuration,
        pupil: ModalPupil,
        wavelength: u.Quantity,
        radial=40,
        angular=48,
        period=40.0,
        dt=0.01,
        *,
        tilt_noise_prior=None,
        sensing_gains=None,
        sensing_ho=None,
        angular_offset=0.0,
        instantaneous=False,
        compact=False,
        common_only=False,
    ):
        self.period, self.dt, self.nmodes = period, dt, len(pupil.basis)
        if self.nmodes < 3:
            raise ValueError("Require two tilt modes and at least one higher mode")
        fx, fy, area = quadrature(radial, angular, CORRECTION_CUTOFF, angular_offset)
        self.instantaneous = instantaneous
        self.compact = compact
        self.common_only = common_only
        factored = instantaneous or compact
        stored_modes = factored or common_only
        if common_only and tilt_noise_prior is not None:
            raise ValueError("A common-only field has no low-order noise prior")
        if period is None and not (common_only or instantaneous):
            raise ValueError("A temporal Fourier period is required for science lags")
        self.paired_temporal = compact and angular % 2 == 0
        self.angular_samples = angular
        if factored and angular % 2 == 0:
            self.angular_samples = angular // 2
            fx, fy, area = (
                values.reshape(radial, angular)[:, : self.angular_samples].ravel()
                for values in (fx, fy, area)
            )
            area *= 2

        def response_on_grid(x, y):
            if self.angular_samples != angular:
                return pupil.response(x, y)
            return _polar_response(pupil, x, y, angular)

        original = response_on_grid(fx, fy)
        atmosphere = _required(
            configuration.atmosphere,
            "Atmosphere",
            "reference_wavelength",
            "fried_parameter",
            "outer_scale",
            "layer_heights",
            "layer_weights",
            "wind_speeds",
            "wind_directions",
        )
        fitting = _required(
            configuration.fitting, "Fitting field", "directions", "weights"
        )
        high_order = _required(
            configuration.high_order_sources, "Beacon geometry", "directions", "height"
        )
        ho_loop = _required(
            configuration.high_order_loop,
            "High-order loop",
            "gain",
            "frame_rate",
            "delay_frames",
        )
        if common_only:
            ngs = np.empty((0, 2))
            lo_loop = None
        else:
            low_order = _required(
                configuration.low_order_sources, "Guide stars", "directions"
            )
            ngs = low_order.directions.to_value(u.rad)
            lo_loop = _required(
                configuration.low_order_loop,
                "Low-order loop",
                "gain",
                "frame_rate",
                "delay_frames",
            )

        fitxy = fitting.directions.to_value(u.rad)
        fitweights = fitting.weights.value
        lgs = high_order.directions.to_value(u.rad)
        if lgs.shape != (4, 2):
            raise ValueError("The retained modal measurement requires four beacons")
        # Heights already refer to the supplied propagation geometry.
        beacon = float(high_order.height.to_value(u.m))
        if beacon <= 0:
            raise ValueError(
                "Modal laser measurements require a finite positive beacon height"
            )
        self.reference_wavelength_m = float(
            atmosphere.reference_wavelength.to_value(u.m)
        )
        self.scale = self.reference_wavelength_m / _positive_wavelength(wavelength)
        ho_interval = float(ho_loop.frame_interval.to_value(u.s))
        ho_delay = _frame_delay(ho_loop.delay_frames)
        density = von_karman_spectrum(
            fx,
            fy,
            float(atmosphere.fried_parameter.to_value(u.m)),
            float(atmosphere.outer_scale.to_value(u.m)),
        )
        h = self.nmodes - 2
        common_target, covariance = np.zeros((h, 4 * h)), np.zeros((4 * h, 4 * h))
        tilt_target, tilt_covariance = (
            np.zeros((2, 2 * len(ngs))),
            np.zeros((2 * len(ngs),) * 2),
        )
        layers = list(
            zip(
                atmosphere.layer_heights.to_value(u.m),
                atmosphere.layer_weights.value,
                atmosphere.wind_speeds.to_value(u.m / u.s),
                atmosphere.wind_directions.to_value(u.rad),
                strict=True,
            )
        )
        samples = len(fx)
        width = SETUP_TILE_NODES if compact else samples
        tiles = [
            slice(start, min(start + width, samples))
            for start in range(0, samples, width)
        ]
        total = samples * len(layers)
        layer_slices = [
            slice(i * samples, (i + 1) * samples) for i in range(len(layers))
        ]
        if stored_modes:
            # The four beacons share the same contracted modal amplitudes.
            # Store the amplitudes and beacon phases once and expand per tile.
            self.lgs_modes = np.empty((h, total), complex)
            self.lgs_phase = np.empty((4, total), complex)
        else:
            # The temporal path reuses this full adjoint in every spectrum.
            self.weighted_measurement = np.empty((4 * h, total), complex)
        self.weights = np.empty(total)
        for selection, (height, fraction, _, _) in zip(
            layer_slices, layers, strict=True
        ):
            weights = self.weights[selection]
            weights[:] = density * fraction * area
            contracted, phase = _ideal_modal_lgs_factors(
                fx,
                fy,
                height,
                beacon,
                lgs,
                project_pupil=response_on_grid,
                uncontracted_modes=original,
            )
            if stored_modes:
                self.lgs_modes[:, selection] = contracted
                self.lgs_phase[:, selection] = phase
            else:
                _expand_modal_lgs_response(
                    contracted, phase, out=self.weighted_measurement[:, selection]
                )
            mean_phase = fitweights @ np.exp(
                2j * np.pi * height * (fitxy[:, :1] * fx + fitxy[:, 1:] * fy)
            )
            for tile in tiles:
                columns = slice(
                    selection.start + tile.start, selection.start + tile.stop
                )
                measured = (
                    self.lgs_measurement(columns)
                    if stored_modes
                    else self.weighted_measurement[:, columns]
                )
                measured_columns = measured.view(float).T
                covariance += (measured * weights[tile]).view(float) @ measured_columns
                common_target += (
                    original[2:, tile] * mean_phase[tile] * weights[tile]
                ).view(float) @ measured_columns
                del measured, measured_columns
            if common_only:
                continue
            ngs_phase = np.exp(
                2j * np.pi * height * (ngs[:, :1] * fx + ngs[:, 1:] * fy)
            )
            tilt = (ngs_phase[:, None] * original[None, :2]).reshape(2 * len(ngs), -1)
            tilt_covariance += ((tilt * weights) @ tilt.conj().T).real
            tilt_target += ((original[:2] * weights) @ tilt.conj().T).real
        values, vectors = np.linalg.eigh(covariance)
        self.inverse = (vectors / np.maximum(values, values[-1] * 1e-8)) @ vectors.T
        self.reconstruction_factor = vectors / np.sqrt(
            np.maximum(values, values[-1] * 1e-8)
        )
        common_reconstructor = common_target @ self.inverse
        # Optional design prior is in the reference-wavelength modal-phase basis.
        prior = None
        if tilt_noise_prior is not None:
            prior = np.asarray(tilt_noise_prior, dtype=float)
            if (
                prior.shape != tilt_covariance.shape
                or not np.isfinite(prior).all()
                or not np.allclose(prior, prior.T)
                or np.linalg.eigvalsh(prior).min() < -1e-12
            ):
                raise ValueError(
                    "Require a finite symmetric PSD modal tilt-noise prior"
                )
        # The native single-NGS path applies the measured tilt directly (R=1).
        tilt_reconstructor = (
            np.empty((2, 0))
            if common_only
            else lo_reconstructor(
                tilt_covariance,
                tilt_target,
                np.zeros_like(tilt_covariance) if prior is None else prior,
                method="pinv",
            )
        )
        gains = (
            np.tile(np.eye(2), (len(ngs), 1, 1))
            if sensing_gains is None
            else np.asarray(sensing_gains, float)
        )
        ho_response = (
            np.zeros((len(ngs), 2, h))
            if sensing_ho is None
            else np.asarray(sensing_ho, float)
        )
        if gains.shape != (len(ngs), 2, 2) or ho_response.shape != (len(ngs), 2, h):
            raise ValueError("Require per-NGS 2x2 sensing gains and 2xHO responses")
        if not np.isfinite(gains).all() or not np.isfinite(ho_response).all():
            raise ValueError("Sensing response must be finite")
        tilt_feedback = tilt_reconstructor @ gains.reshape(-1, 2)
        self.tilt_reconstructor = tilt_reconstructor
        self.tilt_feedback = tilt_feedback
        if factored or common_only:
            self.original = original
            if factored:
                self.delay = np.empty(total, complex)
        else:
            self.original = np.tile(original, len(layers))
            self.residual = np.empty_like(self.weighted_measurement)
        self.common = np.empty((h, total), complex)
        self.lo = np.empty((2, total), complex)
        self.spatial = np.empty((total, 2))
        wave_nodes = np.column_stack((fx, fy))
        gain = None if common_only else float(lo_loop.gain.value)
        lo_interval = (
            None if common_only else float(lo_loop.frame_interval.to_value(u.s))
        )
        lo_delay = None if common_only else _frame_delay(lo_loop.delay_frames)
        if lo_interval is not None and lo_interval != ho_interval:
            raise ValueError("The modal controller uses one common HO and LO cadence")
        frequencies = np.empty(total)
        for selection, (height, _, speed, direction) in zip(
            layer_slices, layers, strict=True
        ):
            frequency = frequencies[selection]
            frequency[:] = frozen_flow_frequency(fx, fy, speed, direction)
            command = self.common[:, selection]
            for tile in tiles:
                columns = slice(
                    selection.start + tile.start, selection.start + tile.stop
                )
                measured = (
                    self.lgs_measurement(columns)
                    if stored_modes
                    else self.weighted_measurement[:, columns]
                )
                command[:, tile].view(float)[:] = common_reconstructor @ measured.view(
                    float
                )
                del measured
            command *= polc_transfer(
                frequency, float(ho_loop.gain.value), ho_interval, ho_delay
            )
            if common_only:
                if factored:
                    self.delay[selection] = np.exp(
                        -2j * np.pi * frequency * ho_interval
                    )
                self.lo[:, selection] = 0
                self.spatial[selection] = height * wave_nodes
                continue
            z = np.exp(-2j * np.pi * frequency * lo_interval)
            left = (1 - z)[:, None, None] * np.eye(2) + gain * z[
                :, None, None
            ] ** lo_delay * tilt_feedback
            right = gain * z[:, None, None] ** lo_delay * tilt_reconstructor
            transfer = np.linalg.solve(left, right)
            if factored:
                self.delay[selection] = z
            else:
                measured = (
                    self.lgs_measurement(selection)
                    if stored_modes
                    else self.weighted_measurement[:, selection]
                )
                residual = self.residual[:, selection].reshape(4, h, samples)
                np.subtract(
                    measured.reshape(4, h, samples), command[None], out=residual
                )
                residual *= z
                np.conjugate(measured, out=measured)
                measured *= self.weights[selection]
                del measured
            ngs_phase = np.exp(
                2j * np.pi * height * (ngs[:, :1] * fx + ngs[:, 1:] * fy)
            )
            ngs_ho = original[None, 2:] * ngs_phase[:, None] - command[None]
            ngs_tilt = ngs_phase[:, None] * original[None, :2]
            sensed = (
                np.einsum("gij,gjf->gif", gains, ngs_tilt)
                + np.einsum("gih,ghf->gif", ho_response, ngs_ho)
            ).reshape(2 * len(ngs), -1)
            self.lo[:, selection] = np.einsum("fij,jf->if", transfer, sensed)
            self.spatial[selection] = height * wave_nodes
        if instantaneous or period is None:
            return
        bins = np.rint(frequencies * period).astype(int)
        self.zero = bins == 0
        if compact:
            self.temporal_bins = bins
        positive = np.flatnonzero(bins > 0)
        self.positive = positive[np.argsort(bins[positive], kind="stable")]
        selected = bins[self.positive]
        self.bin_values, self.starts = np.unique(selected, return_index=True)
        self.ends = np.searchsorted(selected, self.bin_values, side="right")
        self.positive_weights = self.weights[self.positive]
        largest_bin = int(
            np.max(np.abs(bins))
            if self.paired_temporal
            else self.bin_values.max(initial=0)
        )
        if dt is None:
            self.bin_count = largest_bin + 1
            nyquist_bin = (round(period / ho_interval) + 1) // 2
        else:
            self.bin_count = round(period / dt) // 2 + 1
            nyquist_bin = self.bin_count - 1
        if largest_bin >= nyquist_bin:
            raise ValueError("Temporal quadrature exceeds retained Nyquist range")

    def at_wavelength(self, wavelength: u.Quantity):
        """Share the fixed AO grid while applying a new science-phase scale."""
        context = copy(self)
        context.scale = self.reference_wavelength_m / _positive_wavelength(wavelength)
        return context

    def lgs_measurement(self, samples):
        """Expand the four LGS responses for a tile of the factored grid."""
        modes = self.lgs_modes[:, samples]
        phase = self.lgs_phase[:, samples]
        return _expand_modal_lgs_response(modes, phase)

    def factored_residual(self, samples, common):
        """Apply the common correction and controller delay to one LGS tile."""
        residual = self.lgs_measurement(samples)
        residual.reshape(4, self.nmodes - 2, -1)[:] -= common[None]
        residual *= self.delay[samples]
        return residual

    def accumulate_spectrum(self, spectrum, response, samples):
        """Accumulate the same temporal-bin Gram matrices from one Fourier tile."""
        bins = self.temporal_bins[samples]
        weights = self.weights[samples]
        if self.paired_temporal:
            # Each stored sample has the weight of its conjugate pair. Use
            # the positive-frequency member for the one-sided spectrum.
            response = np.where(bins < 0, response.conj(), response)
            weights = weights * np.where(bins == 0, 1.0, 0.5)
            bins = np.abs(bins)
        selected = np.flatnonzero(bins >= 0)
        selected = selected[np.argsort(bins[selected], kind="stable")]
        active, starts = np.unique(bins[selected], return_index=True)
        boundaries = np.r_[starts, len(selected)]
        for index, start, stop in zip(
            active, boundaries[:-1], boundaries[1:], strict=True
        ):
            values = selected[start:stop]
            modal = response[..., values]
            spectrum[:, index] += (modal * weights[values]) @ modal.conj().swapaxes(
                -1, -2
            )

    def _source_response(self, phase):
        if self.common_only or self.compact:
            # Factored grids store one copy of the modal responses for all layers.
            rows = phase.reshape(-1, self.original.shape[-1])
            return (self.original[:, None, :] * rows).reshape(self.nmodes, -1)
        return self.original * phase

    def ngs_ho_covariance(self, theta):
        """Return tilt-removed common-DM NGS covariance in science phase rad².

        The guide-star WFS sees the common HO correction, not the science
        direction's additional MOAO correction. The same physical modal basis
        and layer quadrature are retained; no LO command enters this shape.
        """
        if self.instantaneous:
            raise ValueError("Guide-star covariance requires the sensing grid")
        phase = np.exp(2j * np.pi * (self.spatial @ np.asarray(theta)))
        response = self._source_response(phase)
        response[:2] = 0
        response[2:] -= self.common
        return ((response * self.weights) @ response.conj().T).real * self.scale**2

    def direction_terms(self, theta):
        """Return the field phase, uncorrected modal response and HO correction."""
        if self.instantaneous or self.compact:
            raise ValueError("A factored grid requires tiled science restoration")
        phase = np.exp(2j * np.pi * (self.spatial @ theta))
        response = self._source_response(phase)
        if self.common_only:
            return phase, response, self.common
        target = (response[2:] @ self.weighted_measurement.T).real
        reconstructor = target @ self.inverse
        correction = self.common + reconstructor @ self.residual
        return phase, response, correction

    def zero_lag_weights(self, native_count):
        """Weight Fourier samples as the retained real inverse series at zero lag."""
        if self.instantaneous or self.period is None:
            raise ValueError("Temporal lag weights require a temporal grid")
        if getattr(self, "paired_temporal", False):
            bins = np.abs(self.temporal_bins)
            last = native_count // 2
            factors = np.where(bins <= last, 1.0, 0.0)
            if native_count % 2 == 0:
                factors[bins == last] = 0.5
            return self.weights * factors
        weights = np.zeros_like(self.weights)
        weights[self.zero] = self.weights[self.zero]
        last = native_count // 2
        for index, start, stop in zip(
            self.bin_values, self.starts, self.ends, strict=True
        ):
            if index > last:
                break
            factor = 1 if native_count % 2 == 0 and index == last else 2
            weights[self.positive[start:stop]] = (
                factor * self.positive_weights[start:stop]
            )
        return weights

    def residual_response(self, theta, *, terms=None):
        """Return the science-wavelength response after AO correction."""
        if terms is None:
            _, response, correction = self.direction_terms(theta)
        else:
            _, response, correction = terms
            response = response.copy()
        response[:2] -= self.lo
        response[2:] -= correction
        response *= self.scale
        return response

    def spectrum(self, theta, *, terms=None):
        if self.instantaneous or self.period is None:
            raise ValueError("A temporal spectrum requires a temporal grid")
        response = self.residual_response(theta, terms=terms)
        spectrum = np.zeros((self.bin_count, self.nmodes, self.nmodes), complex)
        selected = response[:, self.zero]
        spectrum[0] = ((selected * self.weights[self.zero]) @ selected.conj().T).real
        selected = response[:, self.positive].T
        # Each temporal bin is a weighted Gram matrix. Contract its Fourier
        # samples directly instead of materializing sample-by-mode-by-mode
        # products before reducing them into the same bins.
        for bin_index, start, stop in zip(
            self.bin_values, self.starts, self.ends, strict=True
        ):
            values = selected[start:stop]
            spectrum[bin_index] = (
                values.T * self.positive_weights[start:stop]
            ) @ values.conj()
        return spectrum
