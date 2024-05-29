from dataclasses import dataclass
from .base import BaseShapeConfig

@dataclass(kw_only=True)
class CubeShapeConfig(BaseShapeConfig, name='cube'):
    center: tuple[float, float, float] = (0.5, 0.5, 0.5)
    size: tuple[float, float, float] = (0.5, 0.5, 0.5)
    resolution: int = 10
    key_indices: tuple[int, ...] = (9, 90, 900, 999)
    mode: str = 'uniform'
