from dataclasses import dataclass
from sga.utils import Config

@dataclass(kw_only=True)
class BaseSimConfig(Config):
    num_steps: int = 1000
    gravity: tuple[float, float, float] = (0.0, -9.8, 0.0)
    bc: str = 'freeslip'
    num_grids: int = 20
    dt: float = 5e-4
    bound: int = 3
    eps: float = 1e-7
    skip_frames: int = 1
