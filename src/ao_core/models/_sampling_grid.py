"""System-level numerical settings for the sampling uncertainty model."""

from configparser import ConfigParser
from dataclasses import dataclass, fields

from ._units import _count

_DEFAULT_SCIENCE_NODES = (
    (1.0, 80, 192),
    (5.0, 160, 192),
    (10.0, 160, 384),
    (20.0, 320, 768),
)


@dataclass(frozen=True, kw_only=True)
class SamplingSettings:
    """Numerical settings shared by Fast, Balanced and Full.

    Pupil sizes are counts per side. The science node counts are lower
    bounds on the radial and azimuthal quadrature for any exposure.

    Attributes:
        modal_pupil_samples: Pupil samples for projecting the phase into modes.
        science_pupil_samples: Pupil samples for the science calculation.
        sensing_pupil_samples: Pupil samples for guide-star optics.
        zernike_radial_order: Maximum retained Zernike radial order.
        sensing_radial_nodes: Radial quadrature nodes for guide-star sensing.
        sensing_angular_nodes: Angular quadrature nodes for guide-star sensing.
        science_min_radial_nodes: Minimum science radial quadrature nodes.
        science_min_angular_nodes: Minimum science angular quadrature nodes.
    """

    modal_pupil_samples: int = 64
    science_pupil_samples: int = 64
    sensing_pupil_samples: int = 96
    zernike_radial_order: int = 11
    sensing_radial_nodes: int = 40
    sensing_angular_nodes: int = 48
    science_min_radial_nodes: int = 80
    science_min_angular_nodes: int = 192

    def __post_init__(self):
        for field in fields(self):
            minimum = 3 if field.name.endswith("_pupil_samples") else 1
            _count(getattr(self, field.name), field.name, minimum)
        if self.modal_pupil_samples > 512:
            raise ValueError(
                "modal_pupil_samples exceeds the 512-sample projection FFT"
            )
        if self.zernike_radial_order not in (5, 7, 9, 11):
            raise ValueError("zernike_radial_order must be 5, 7, 9 or 11")

    @classmethod
    def _from_ini(cls, ini_parser: ConfigParser):
        if not ini_parser.has_section("sampling_uncertainty_model"):
            return cls()
        section = ini_parser["sampling_uncertainty_model"]
        names = {field.name for field in fields(cls)}
        unknown = set(section) - names - ini_parser.defaults().keys()
        if unknown:
            keys = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown [sampling_uncertainty_model] keys: {keys}")
        values = {}
        for name in sorted(names.intersection(section)):
            try:
                values[name] = section.getint(name)
            except ValueError as exc:
                raise ValueError(
                    f"sampling_uncertainty_model.{name} must be an integer"
                ) from exc
        return cls(**values)

    def _science_nodes(self, exposure: float) -> tuple[int, int]:
        for limit, radial, angular in _DEFAULT_SCIENCE_NODES:
            if exposure <= limit:
                return (
                    max(radial, self.science_min_radial_nodes),
                    max(angular, self.science_min_angular_nodes),
                )
        raise ValueError("The science spatial grid schedule covers exposures through 20 s")

    def _science_grid(
        self, exposure: float, *, frame_interval: float, minimum_period_frames: int
    ) -> tuple[int, int, float]:
        radial, angular = self._science_nodes(exposure)
        exposure_frames = round(exposure / frame_interval)
        period_frames = max(2 * exposure_frames, minimum_period_frames)
        return radial, angular, period_frames * frame_interval
