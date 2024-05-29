from dataclasses import dataclass
import math

from .base import BaseCamera


@dataclass(kw_only=True)
class SphereCamera(BaseCamera, name='sphere'):
    radius: float = math.sqrt(2) * 2.0
    h_degree: float = 90 + 55
    v_degree: float = 60

    target: tuple[float, float, float] = (0.5, 0.5, 0.5)

    def __post_init__(self):
        v_radius = math.radians(self.v_degree)
        h_radius = math.radians(self.h_degree)
        self.origin = (
            self.target[0] + self.radius * math.sin(v_radius) * math.cos(h_radius),
            self.target[1] + self.radius * math.cos(v_radius),
            self.target[2] + self.radius * math.sin(v_radius) * math.sin(h_radius),
        )
