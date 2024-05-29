from dataclasses import dataclass
from .cube import CubeShapeConfig

@dataclass(kw_only=True)
class CubeHDShapeConfig(CubeShapeConfig, name='cube_hd'):
    center: tuple[float, float, float] = (0.5, 0.5, 0.5)
    size: tuple[float, float, float] = (0.5, 0.5, 0.5)
    resolution: int = 32
    mode: str = 'random'
