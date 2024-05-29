from dataclasses import dataclass
from sga.utils import Config

@dataclass(kw_only=True)
class BaseShapeConfig(Config):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, float, float] = (1.0, 1.0, 1.0)
