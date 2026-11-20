"""Finite Shack–Hartmann and Gaussian mirror responses for high-order PSFs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from astropy import units as u
from scipy import ndimage

from ._ho_reconstruction import (
    _command_tilt,
    footprint_shift,
    gaussian_basis,
    gaussian_slopes,
    lenslet_gradient,
    mirror_phase_fit,
)
from ._units import _PhysicalRecord, _quantity
from .config import (
    DeformableMirrorConfig,
    LensletConfig,
    Pupil,
    SourceConfig,
    TelescopeConfig,
)


def tangent_directions(directions):
    """Convert sky offsets in radians to horizontal ray slopes."""
    offset = np.linalg.norm(directions, axis=-1, keepdims=True)
    factor = np.ones_like(offset)
    np.divide(np.tan(offset), offset, out=factor, where=offset != 0)
    return directions * factor


def _same_coordinates(left, right):
    return left.shape == right.shape and np.allclose(
        left.value, right.value, rtol=8 * np.finfo(float).eps, atol=0
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class SlopeLayout(_PhysicalRecord):
    """Guide and lenslet geometry defining the packed raw slope channels.

    Guides follow the source order. For each guide, all x slopes precede
    all y slopes. Within either block, valid lenslets occur in the given
    position order. Lenslet positions exclude the per-guide offsets.
    """

    guide_directions: u.Quantity
    guide_height: u.Quantity
    lenslet_positions: u.Quantity
    lenslet_pitch: u.Quantity
    grid_offsets: u.Quantity

    def __post_init__(self):
        for name, unit, ndim in (
            ("guide_directions", u.arcsec, 2),
            ("guide_height", u.m, 0),
            ("lenslet_positions", u.m, 2),
            ("lenslet_pitch", u.m, 0),
            ("grid_offsets", u.m, 2),
        ):
            value = _quantity(getattr(self, name), unit, name, ndim)
            if ndim == 2 and value.shape[1] != 2:
                raise ValueError(f"{name} must contain x/y pairs")
            object.__setattr__(self, name, value)
        if len(self.grid_offsets) != len(self.guide_directions):
            raise ValueError("grid_offsets must match guide_directions")

    @property
    def size(self) -> int:
        """Number of packed slope channels."""
        return 2 * len(self.guide_directions) * len(self.lenslet_positions)

    def matches(self, other: SlopeLayout) -> bool:
        """Check the geometry and order, allowing roundoff from unit conversion."""
        return all(
            _same_coordinates(getattr(self, name), getattr(other, name))
            for name in (
                "guide_directions",
                "guide_height",
                "lenslet_positions",
                "lenslet_pitch",
                "grid_offsets",
            )
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class CommandLayout(_PhysicalRecord):
    """Ordered active actuator centres in metres and the input slope layout."""

    actuator_positions: u.Quantity
    slopes: SlopeLayout

    def __post_init__(self):
        positions = _quantity(self.actuator_positions, u.m, "actuator_positions", 2)
        if positions.shape[1] != 2:
            raise ValueError("actuator_positions must contain x/y pairs")
        if not isinstance(self.slopes, SlopeLayout):
            raise TypeError("slopes must be a SlopeLayout")
        object.__setattr__(self, "actuator_positions", positions)

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix shape in actuator-by-slope order."""
        return len(self.actuator_positions), self.slopes.size

    def _matches(self, actuator_positions, slopes):
        return self.slopes.matches(slopes) and _same_coordinates(
            self.actuator_positions, actuator_positions
        )


@dataclass(frozen=True, eq=False)
class LayerSlopeResponse:
    """Factored raw slopes from one atmospheric layer.

    The complex gradient has leading x/y axis in inverse metres, per
    radian of input phase. Footprints are guide-ordered x/y translations
    in metres. The cone scale multiplies nominal lenslet coordinates.
    """

    gradient: np.ndarray
    footprints: u.Quantity
    cone_scale: float


@dataclass(frozen=True)
class Lenslets:
    pitch: float
    count: int
    cells: np.ndarray
    indices: np.ndarray
    offsets: np.ndarray

    @property
    def slope_count(self):
        return 2 * len(self.cells) * len(self.offsets)


@dataclass(frozen=True, eq=False)
class PreparedSensing:
    """Resolved sensing geometry and finite-cell Fourier response."""

    _lenslets: Lenslets
    _guides: SourceConfig
    _beacon_height_m: float
    _guide_rays: np.ndarray

    @cached_property
    def layout(self) -> SlopeLayout:
        """The column layout for matrices operating on raw lenslet slopes."""
        return SlopeLayout(
            guide_directions=self._guides.directions,
            guide_height=self._guides.height,
            lenslet_positions=u.Quantity(self._lenslets.cells, u.m, copy=False),
            lenslet_pitch=self._lenslets.pitch * u.m,
            grid_offsets=u.Quantity(self._lenslets.offsets, u.m, copy=False),
        )

    def layer_response(
        self, *, layer_height: u.Quantity, fx: u.Quantity, fy: u.Quantity
    ) -> LayerSlopeResponse:
        """Evaluate the raw response at signed spatial frequencies in cycles/m.

        Use a layer height along the line of sight, in the same convention
        as the zenith-projected beacon height. A beacon at infinity has
        unit cone scale.
        """
        height = layer_height.to_value(u.m)
        if 0 < self._beacon_height_m <= height:
            raise ValueError("HO guide beacons must be above all turbulent layers")
        scale = (
            1.0 if self._beacon_height_m == 0 else 1 - height / self._beacon_height_m
        )
        return LayerSlopeResponse(
            lenslet_gradient(
                fx.to_value(1 / u.m), fy.to_value(1 / u.m), self._lenslets.pitch, scale
            ),
            u.Quantity(
                footprint_shift(
                    height, scale, self._guide_rays, self._lenslets.offsets
                ),
                u.m,
                copy=False,
            ),
            scale,
        )


class ShackHartmannSensing:
    """Noise-free sensing with a common pitch for square lenslets."""

    def prepare(
        self,
        *,
        pupil: Pupil,
        telescope: TelescopeConfig,
        guides: SourceConfig,
        lenslets: LensletConfig,
    ) -> PreparedSensing:
        """Resolve valid lenslet centres from the pupil and nominal geometry."""
        if guides is None or any(
            value is None
            for value in (guides.height, guides.directions, guides.wavelengths)
        ):
            raise ValueError(
                "HO correction requires guide-star height, directions and wavelength"
            )
        if len(guides.wavelengths) != 1:
            raise ValueError("HO correction requires one guide-star wavelength")
        if lenslets is None or lenslets.pitches is None:
            raise ValueError("HO correction requires lenslet pitches")
        sensors = len(guides.directions)
        pitches = lenslets.pitches.to_value(u.m)
        pitch = pitches[0]
        if len(pitches) not in (1, sensors) or not np.allclose(pitches, pitch):
            raise ValueError(
                "HO correction requires equal lenslet pitches for all sensors"
            )
        diameter = telescope.diameter.to_value(u.m)
        count = round(diameter / pitch)
        if not np.isclose(count * pitch, diameter, rtol=1e-10, atol=0) or (
            lenslets.counts is not None
            and (
                len(lenslets.counts) not in (1, sensors)
                or any(n != count for n in lenslets.counts)
            )
        ):
            raise ValueError(
                "The nominal lenslet grid must span the telescope diameter"
            )
        offsets = lenslets.grid_offsets
        offsets = np.zeros((sensors, 2)) if offsets is None else offsets.to_value(u.m)
        if len(offsets) != sensors:
            raise ValueError("HO lenslet offsets must match the guide stars")
        axis = (np.arange(count) - (count - 1) / 2) * pitch
        x, y = np.meshgrid(axis, axis, indexing="ij")
        amplitude = pupil.amplitude
        indices_yx = np.array([y, x]) / pupil.pitch.to_value(u.m) + len(amplitude) / 2
        valid = ndimage.map_coordinates(amplitude, indices_yx, order=0) > 0.5
        if not valid.any():
            raise ValueError("The illuminated pupil contains no valid lenslets")
        angle = telescope.zenith_angle
        airmass = 1.0 if angle is None else 1 / np.cos(angle.to_value(u.rad))
        return PreparedSensing(
            Lenslets(
                pitch,
                count,
                np.column_stack((x[valid], y[valid])),
                np.argwhere(valid),
                offsets,
            ),
            guides,
            guides.height.to_value(u.m) * airmass,
            tangent_directions(guides.directions.to_value(u.rad)),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class MirrorGeometry(_PhysicalRecord):
    """Resolved coordinates for a calibrated deformable mirror.

    Attributes:
        actuator_axis: Shared one-dimensional x/y actuator axis in metres.
        valid_mask: Boolean actuator mask in [y, x] order.
        pupil_x: One-dimensional x coordinates for the tilt-fit pupil in metres.
        pupil_y: One-dimensional y coordinates for the tilt-fit pupil in metres.
        pupil_mask: Pupil weights for calibrating command tilt on the supplied
            axes. If omitted, the model samples the configured pupil there.
        fit_cutoff: Absolute singular-value cutoff for the phase-to-command fit.
    """

    actuator_axis: u.Quantity
    valid_mask: np.ndarray
    pupil_x: u.Quantity
    pupil_y: u.Quantity
    pupil_mask: np.ndarray | None = None
    fit_cutoff: float = 5e-4

    def __post_init__(self) -> None:
        for name in ("actuator_axis", "pupil_x", "pupil_y"):
            axis = _quantity(getattr(self, name), u.m, name, 1)
            if np.any(np.diff(axis.value) <= 0):
                raise ValueError(f"{name} must increase strictly")
            object.__setattr__(self, name, axis)
        mask = np.asarray(self.valid_mask)
        side = len(self.actuator_axis)
        if (
            side < 2
            or mask.dtype != np.bool_
            or mask.shape != (side, side)
            or not mask.any()
        ):
            raise ValueError(
                "valid_mask must mark valid actuators on the square [y, x] grid"
            )
        mask = np.frombuffer(mask.tobytes(), dtype=bool).reshape(mask.shape)
        object.__setattr__(self, "valid_mask", mask)
        if self.pupil_mask is not None:
            weights = np.asarray(self.pupil_mask, dtype=np.float64)
            if (
                weights.shape != (len(self.pupil_y), len(self.pupil_x))
                or not np.isfinite(weights).all()
                or np.any(weights < 0)
                or not np.any(weights > 0)
            ):
                raise ValueError(
                    "pupil_mask must contain finite nonnegative [y, x] weights with light"
                )
            weights = np.frombuffer(weights.tobytes(), dtype=np.float64).reshape(
                weights.shape
            )
            object.__setattr__(self, "pupil_mask", weights)
        if not np.isfinite(self.fit_cutoff) or self.fit_cutoff < 0:
            raise ValueError("fit_cutoff must be finite and nonnegative")


@dataclass(frozen=True, eq=False)
class PreparedDm(_PhysicalRecord):
    """Ground-conjugate actuator geometry and Gaussian phase response."""

    actuator_positions: u.Quantity
    _width: float
    _axis: np.ndarray
    _valid: np.ndarray
    _pupil: Pupil
    _geometry: MirrorGeometry | None

    def phase_response(self, points: u.Quantity) -> np.ndarray:
        """Return (point, actuator) phase per unit command at x/y points."""
        return gaussian_basis(
            points.to_value(u.m), self.actuator_positions.value, self._width
        )

    def slope_response(self, sensing: PreparedSensing) -> np.ndarray:
        """Return the (slope, actuator) interaction matrix in inverse metres."""
        lenslets = sensing._lenslets
        return gaussian_slopes(
            lenslets.cells,
            self.actuator_positions.value,
            self._width,
            lenslets.pitch,
            lenslets.offsets,
        )

    def _projected_phase(self, pupil):
        centres = self.actuator_positions.value
        basis = self.phase_response(u.Quantity(pupil.points, u.m, copy=False))
        if not len(pupil.tilt_fit):
            return basis
        xyz = np.exp(-((self._width * (pupil.fit_axis[:, None, None] - centres)) ** 2))
        moments = np.einsum(
            "ya,iyx,xa->ia", xyz[..., 1], pupil.tilt_fit, xyz[..., 0], optimize=True
        )
        return basis - pupil.ramps.T @ moments

    def _phase_fit(self, sensing):
        amplitude = self._pupil.amplitude
        pitch = self._pupil.pitch.to_value(u.m)
        geometry = self._geometry
        if geometry is None:
            sample_x = sample_y = (
                np.arange(len(amplitude)) - len(amplitude) / 2
            ) * pitch
            weights = amplitude
            cutoff = 5e-4
        else:
            sample_x, sample_y = geometry.pupil_x.value, geometry.pupil_y.value
            weights = geometry.pupil_mask
            if weights is None:
                xy = np.meshgrid(sample_y, sample_x, indexing="ij")
                weights = ndimage.map_coordinates(
                    amplitude, np.array(xy) / pitch + len(amplitude) / 2, order=1
                )
            cutoff = geometry.fit_cutoff
        tilt = _command_tilt(
            self._axis, self._valid, sample_x, sample_y, weights, self._width
        )
        lenslets = sensing._lenslets
        return mirror_phase_fit(
            lenslets.cells,
            lenslets.pitch,
            self.actuator_positions.value,
            self._width,
            tilt,
            cutoff,
        )


class GaussianDm:
    """A Gaussian influence at each active ground-layer actuator."""

    def prepare(
        self,
        *,
        config: DeformableMirrorConfig,
        pupil: Pupil,
        geometry: MirrorGeometry | None = None,
    ) -> PreparedDm:
        """Resolve ordered actuator centres from nominal or calibrated geometry."""
        if config.influence_model != "gaussian" or (
            config.conjugate_height is not None and config.conjugate_height.value != 0
        ):
            raise NotImplementedError(
                "HO correction currently requires a ground-conjugate Gaussian mirror"
            )
        if (
            config.pitch is None
            or config.actuator_count is None
            or config.influence_coupling is None
        ):
            raise ValueError(
                "HO correction requires nominal mirror pitch, count and influence coupling"
            )
        coupling = config.influence_coupling.value
        if config.actuator_count < 2 or not 0 < coupling < 1:
            raise ValueError(
                "The Gaussian mirror requires at least two actuators per axis and coupling in (0, 1)"
            )
        pitch = config.pitch.to_value(u.m)
        if geometry is None:
            count = config.actuator_count
            axis = (np.arange(count) - (count - 1) / 2) * pitch
            y, x = np.meshgrid(axis, axis, indexing="ij")
            amplitude = pupil.amplitude
            coordinates = (
                np.array([y, x]) / pupil.pitch.to_value(u.m) + len(amplitude) / 2
            )
            valid = (
                ndimage.map_coordinates(
                    amplitude, coordinates, order=0, mode="grid-constant"
                )
                > 0
            )
            if not valid.any():
                raise ValueError(
                    "The nominal DM has no actuator centres on the illuminated pupil"
                )
        else:
            axis = geometry.actuator_axis.value
            valid = geometry.valid_mask
            pitch = (axis[-1] - axis[0]) / (len(axis) - 1)
            if not np.allclose(np.diff(axis), pitch, rtol=1e-10, atol=1e-13):
                raise ValueError(
                    "The Gaussian mirror requires a uniformly spaced actuator axis"
                )
        iy, ix = np.nonzero(valid)
        positions = _quantity(
            np.column_stack((axis[ix], axis[iy])) * u.m, u.m, "actuator_positions", 2
        )
        return PreparedDm(
            positions, np.sqrt(-np.log(coupling)) / pitch, axis, valid, pupil, geometry
        )
