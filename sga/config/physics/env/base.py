from dataclasses import dataclass
from sga.utils import Config
from .physics import BasePhysicsConfig
from .shape import BaseShapeConfig
from .vel import BaseVelConfig

@dataclass(kw_only=True)
class BaseEnvConfig(Config):
    physics: BasePhysicsConfig
    shape: BaseShapeConfig
    vel: BaseVelConfig

    rho: float = 1e3
    clip_bound: float = 0.5
