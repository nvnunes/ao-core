"""Phase spectra for the finite-cell high-order reconstruction."""

from dataclasses import dataclass

import numpy as np
from astropy import units as u

from ._ho_reconstruction import footprint_shift, lenslet_gradient, piston_filter
from .atmosphere import (
    _project_zenith_atmosphere,
    frozen_flow_frequency,
    von_karman_spectrum,
)
from .control import polc_transfer


@dataclass(frozen=True)
class PhaseGrid:
    period: float
    diameter: float
    lenslet_count: int
    fried: float
    outer_scale: float
    base_side: int
    fx: np.ndarray
    fy: np.ndarray
    kernel_side: int

    @property
    def spacing(self):
        return 1 / self.period


@dataclass(frozen=True)
class PhaseLayer:
    height: float
    cone_scale: float
    weight: float
    power: np.ndarray
    gradient: np.ndarray
    temporal: np.ndarray
    footprints: np.ndarray
    residual_shift: np.ndarray | None = None


def tangent_directions(directions):
    """Convert two-dimensional sky offsets in radians to horizontal ray slopes."""
    offset = np.linalg.norm(directions, axis=-1, keepdims=True)
    factor = np.ones_like(offset)
    np.divide(np.tan(offset), offset, out=factor, where=offset != 0)
    return directions * factor


def science_azimuths(sources):
    """Return azimuths in radians, retaining any supplied on-axis direction."""
    directions = sources.directions.to_value(u.rad)
    azimuths = np.arctan2(directions[:, 1], directions[:, 0])
    on_axis = np.all(directions == 0, axis=1)
    azimuths[on_axis] = (
        0 if sources.azimuth is None else sources.azimuth[on_axis].to_value(u.rad)
    )
    return azimuths


def phase_layers(
    config, period, diameter, pitch, lenslet_count, offsets, wavelength, optical_side
):
    """Build projected atmospheric layers and their sensing responses.

    See Eqs. (1)–(8) for finite-source footprint contraction and
    translation. [Plantet et al. (2022)](https://arxiv.org/pdf/2209.00931)
    """
    atmosphere, airmass = _project_zenith_atmosphere(
        config.atmosphere, config.telescope.zenith_angle
    )
    guides = config.high_order_sources
    if any(
        getattr(atmosphere, name) is None
        for name in (
            "outer_scale",
            "layer_weights",
            "wind_speeds",
            "wind_directions",
        )
    ):
        raise ValueError(
            "HO atmosphere requires an outer scale, layer weights and wind vectors"
        )
    heights = atmosphere.layer_heights.to_value(u.m)
    beacon = guides.height.to_value(u.m) * airmass
    if 0 < beacon <= heights.max():
        raise ValueError("HO guide beacons must be above all turbulent layers")
    contraction = np.ones_like(heights) if beacon == 0 else 1 - heights / beacon
    guide_rays = tangent_directions(guides.directions.to_value(u.rad))
    layer_weights = atmosphere.layer_weights.value
    if layer_weights.sum() <= 0:
        raise ValueError("HO atmospheric layer weights must sum to positive power")
    layer_weights = layer_weights / layer_weights.sum()
    fried = atmosphere.fried_parameter.to_value(u.m) * (
        wavelength / atmosphere.reference_wavelength.to_value(u.m)
    ) ** (6 / 5)
    outer_scale = atmosphere.outer_scale.to_value(u.m)
    base = round(period / pitch)
    if base < 2 or base % 2 or not np.isclose(base * pitch, period, rtol=1e-10, atol=0):
        raise ValueError(
            "The Fourier period must contain an even integer number of lenslets"
        )
    count = 7 * base
    modes = (np.arange(count) - count // 2) / period
    fx, fy = np.meshgrid(modes, modes, indexing="ij")
    power = von_karman_spectrum(fx, fy, fried, outer_scale) / period**2
    power[0] = power[:, 0] = 0
    power[count // 2, count // 2] = 0
    common_loop = config.high_order_loop
    gain = common_loop.gain.value
    interval = common_loop.frame_interval.to_value(u.s)
    delay = int(common_loop.delay_frames)
    layers = []
    residual_loop = config.residual_high_order_loop
    for height, scale, weight, speed, angle in zip(
        heights,
        contraction,
        layer_weights,
        atmosphere.wind_speeds.to_value(u.m / u.s),
        atmosphere.wind_directions.to_value(u.rad),
        strict=True,
    ):
        frequency = frozen_flow_frequency(fx, fy, speed, angle)
        temporal = (
            polc_transfer(frequency, gain, interval, delay)
            if gain
            else np.zeros_like(fx)
        )
        residual_shift = None
        if residual_loop is not None and residual_loop.gain.value:
            residual_shift = (
                speed
                * interval
                * residual_loop.delay_frames
                * np.array([np.cos(angle), np.sin(angle)])
            )
        layers.append(
            PhaseLayer(
                height,
                scale,
                weight,
                weight * power,
                lenslet_gradient(fx, fy, pitch, scale),
                temporal,
                footprint_shift(height, scale, guide_rays, offsets),
                residual_shift,
            )
        )
    kernel_side = 1 << (max(count, 4 * optical_side) - 1).bit_length()
    return PhaseGrid(
        period, diameter, lenslet_count, fried, outer_scale, base, fx, fy, kernel_side
    ), layers


# Mathar (2007), Table I, arranged by the terms of Eq. 7. Wavenumber is cm⁻¹.
_MATHAR = np.array(
    [
        [2.00192e-4, 1.13474e-10, -4.24595e-15, 1.00957e-17, -2.93315e-21, 3.07228e-25],
        [5.88625e-2, -3.85766e-8, 8.88019e-11, -5.67650e-14, 1.66615e-17, -1.74845e-21],
        [-3.01513, 4.06167e-4, -5.14544e-7, 3.43161e-10, -1.01189e-13, 1.06749e-17],
        [
            -1.03945e-8,
            1.36858e-12,
            -1.71039e-15,
            1.12908e-18,
            -3.29925e-22,
            3.44747e-26,
        ],
        [
            5.73256e-13,
            1.86367e-17,
            -2.28150e-20,
            1.50947e-23,
            -4.41214e-27,
            4.61209e-31,
        ],
        [2.67085e-9, 1.35941e-15, 1.35295e-19, 8.18218e-24, -2.22957e-27, 2.49964e-31],
        [
            6.09186e-18,
            5.19024e-24,
            -4.19477e-28,
            4.34120e-31,
            -1.22445e-34,
            1.34816e-38,
        ],
        [4.97859e-5, -6.61752e-9, 8.32034e-12, -5.51793e-15, 1.61899e-18, -1.69901e-22],
        [7.79176e-7, 3.96499e-13, 3.95114e-17, 2.33587e-21, -6.36441e-25, 7.16868e-29],
        [
            -2.06567e-16,
            1.06141e-21,
            -1.49982e-24,
            9.84046e-28,
            -2.88266e-31,
            2.99105e-35,
        ],
    ]
)


def _ciddor_compressibility(temperature_k, pressure_pa, water_mole_fraction):
    """Evaluate Ciddor Eq. (12) using the Appendix A coefficients."""
    temperature_c = temperature_k - 273.15
    pressure_over_temperature = pressure_pa / temperature_k
    x = water_mole_fraction
    return (
        1
        - pressure_over_temperature
        * (
            1.58123e-6
            - 2.9331e-8 * temperature_c
            + 1.1043e-10 * temperature_c**2
            + (5.707e-6 - 2.051e-8 * temperature_c) * x
            + (1.9898e-4 - 2.376e-6 * temperature_c) * x**2
        )
        + pressure_over_temperature**2 * (1.83e-11 - 0.765e-8 * x**2)
    )


def refractivity(
    wavelength, *, temperature_k, pressure_pa, relative_humidity, ciddor_compressibility
):
    """Return n - 1 at a vacuum wavelength in metres and fractional humidity.

    From 0.3 µm to below 1.3 µm, use Ciddor (1996), Eqs. (1) and (3)
    for standard dry air and water vapor, and Section 3 and Appendix A
    for the humidity terms. Combine the components with Eq. (5), using
    densities from Eq. (4) and compressibility from Eq. (12). Set
    `ciddor_compressibility=False` to use Z = 1 for each density ratio.
    https://doi.org/10.1364/AO.35.001566

    From 1.3 to 2.5 µm, use Mathar (2007), Eqs. (6) and (7) with the
    corrected coefficients in Table I. https://arxiv.org/abs/physics/0610256v2
    """
    wavelength_um = wavelength * 1e6
    if 0.3 <= wavelength_um < 1.3:
        # Ciddor Eqs. (1) and (3), with sigma in inverse micrometres.
        sigma_squared = wavelength_um**-2
        standard_dry_air = (
            5792105 / (238.0185 - sigma_squared) + 167917 / (57.362 - sigma_squared)
        ) * 1e-8
        standard_water_vapor = (
            1.022
            * (
                295.235
                + sigma_squared
                * (2.6422 + sigma_squared * (-0.032380 + 0.004028 * sigma_squared))
            )
            * 1e-8
        )

        # Ciddor Section 3 and Appendix A, with the vapor pressure in Pa.
        saturation_pressure = np.exp(
            1.2378847e-5 * temperature_k**2
            - 0.019121316 * temperature_k
            + 33.93711047
            - 6343.1645 / temperature_k
        )
        enhancement = (
            1.00062 + 3.14e-8 * pressure_pa + 5.6e-7 * (temperature_k - 273.15) ** 2
        )
        enhanced_vapor_pressure = enhancement * relative_humidity * saturation_pressure
        water_mole_fraction = enhanced_vapor_pressure / pressure_pa

        # Ciddor Eqs. (4) and (5), with each density divided by its reference.
        dry_air = (
            standard_dry_air
            * (1 - water_mole_fraction)
            * (pressure_pa / 101325)
            * (288.15 / temperature_k)
        )
        water_vapor = (
            standard_water_vapor
            * (enhanced_vapor_pressure / 1333)
            * (293.15 / temperature_k)
        )
        if ciddor_compressibility:
            moist_z = _ciddor_compressibility(
                temperature_k, pressure_pa, water_mole_fraction
            )
            dry_air *= _ciddor_compressibility(288.15, 101325.0, 0.0) / moist_z
            water_vapor *= _ciddor_compressibility(293.15, 1333.0, 1.0) / moist_z
        return dry_air + water_vapor

    if 1.3 <= wavelength_um <= 2.5:
        # Mathar Eq. (7), in the order of the terms in the paper.
        delta_inv_t = 1 / temperature_k - 1 / 290.65
        delta_p = pressure_pa - 75000
        delta_h = 100 * relative_humidity - 10
        state_terms = np.array(
            [
                1,
                delta_inv_t,
                delta_inv_t * delta_inv_t,
                delta_h,
                delta_h * delta_h,
                delta_p,
                delta_p * delta_p,
                delta_inv_t * delta_h,
                delta_inv_t * delta_p,
                delta_h * delta_p,
            ]
        )
        coefficients = state_terms @ _MATHAR
        # Mathar Eq. (6), with the Table I wavenumber origin in cm⁻¹.
        delta_wavenumber = 1e4 / wavelength_um - 1e4 / 2.25
        return np.polynomial.polynomial.polyval(delta_wavenumber, coefficients)
    raise ValueError("HO refractivity requires wavelengths between 0.3 and 2.5 µm")


def refraction_power(fx, fy, separation):
    """Power ratio for the difference of one phase mode and its translation."""
    return 4 * np.sin(np.pi * (fx * separation[0] + fy * separation[1])) ** 2


def chromatic_power(
    grid,
    layers,
    optical_side,
    cutoff,
    refractivity_difference,
    fractional_difference,
    zenith,
    azimuth,
):
    """Return differential refraction and correction chromatism in phase per bin.

    See the derivations of the chromatic correction and
    differential-refraction terms. [Jolissaint & Kendrew (2010)](https://doi.org/10.1051/ao4elt/201005021)
    """
    side = int(2 * cutoff / grid.spacing)
    if side < 1 or side > optical_side:
        raise ValueError("The AO correction bandwidth exceeds the optical grid")
    # The reference embeds its own centred auxiliary square on the native grid.
    shift = (optical_side - side + 1) // 2
    band = slice(shift, shift + side)
    result = np.zeros((2, optical_side, optical_side))
    if refractivity_difference == 0:
        return result
    modes = (np.arange(side) - side // 2) * grid.spacing
    fx, fy = np.meshgrid(modes, modes, indexing="ij")
    radial = np.hypot(fx, fy)
    power = von_karman_spectrum(fx, fy, grid.fried, grid.outer_scale) * grid.spacing**2
    power *= piston_filter(grid.diameter, radial)
    direction = np.array([np.cos(azimuth), np.sin(azimuth)])
    shift_angle = np.tan(refractivity_difference * np.tan(zenith))
    refraction = sum(
        layer.weight
        * refraction_power(
            fx, fy, direction * shift_angle * layer.height / layer.cone_scale
        )
        for layer in layers
    )
    result[0, band, band] = power * refraction * (radial < cutoff)
    result[1, band, band] = power * fractional_difference**2
    return result
