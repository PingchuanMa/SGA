from dataclasses import dataclass, field
from .base import BaseEnvConfig
from .physics import CorotatedPhysicsConfig
from .shape import CubeShapeConfig
from .vel import MildVelConfig

@dataclass(kw_only=True)
class JellyEnvConfig(BaseEnvConfig, name='jelly'):
    rho: float = 1e3
    clip_bound: float = 0.5

    physics: CorotatedPhysicsConfig = field(default_factory=CorotatedPhysicsConfig)
    shape: CubeShapeConfig = field(default_factory=CubeShapeConfig)
    vel: MildVelConfig = field(default_factory=MildVelConfig)
