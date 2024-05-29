from dataclasses import dataclass
from .base import BaseVelConfig

@dataclass(kw_only=True)
class MildVelConfig(BaseVelConfig, name='mild'):
    random: bool = False
    lin_vel: tuple[float, float, float] = (1.0, -0.5, -0.75)
    ang_vel: tuple[float, float, float] = (1.0, 3.0, 2.0)
