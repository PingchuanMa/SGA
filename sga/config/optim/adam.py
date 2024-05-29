from dataclasses import dataclass
from .base import BaseOptimConfig

@dataclass(kw_only=True)
class AdamOptimConfig(BaseOptimConfig, name='adam'):
    num_epochs: int = 30
    optimizer: str = 'adam'
    lr: float = 3e-2
    scheduler: str = 'none'
    num_teacher_steps: int = 100

    alpha_position: float = 1e4
    alpha_velocity: float = 1e1
