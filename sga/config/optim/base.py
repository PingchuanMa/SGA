from dataclasses import dataclass
from sga.utils import Config

@dataclass(kw_only=True)
class BaseOptimConfig(Config):
    num_epochs: int
    optimizer: str

    alpha_position: float = 1e4
    alpha_velocity: float = 1e1
