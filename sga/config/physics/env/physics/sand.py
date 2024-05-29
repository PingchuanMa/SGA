from dataclasses import dataclass, field
from pathlib import Path
import math
from .base import BasePhysicsConfig


@dataclass(kw_only=True)
class SandPhysicsConfig(BasePhysicsConfig, name='sand'):
    path: str = str(Path(__file__).parent.resolve() / 'templates' / 'sand.py')
    elasticity: str = 'sigma'
    youngs_modulus_log: float = 10.0
    poissons_ratio: float = 0.1
    friction_angle: float = 25.0
    cohesion: float = 0.0
    alpha: float = field(init=False)

    def __post_init__(self):
        sin_phi = math.sin(math.radians(self.friction_angle))
        self.alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
