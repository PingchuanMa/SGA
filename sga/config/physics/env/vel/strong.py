from dataclasses import dataclass
from .base import BaseVelConfig

@dataclass(kw_only=True)
class MildVelConfig(BaseVelConfig, name='mild'):
    random: bool = False
    lin_vel: tuple[float, float, float] = (2.0, -3.0, -3.0)
    ang_vel: tuple[float, float, float] = (1.0, 1.0, 1.0)
