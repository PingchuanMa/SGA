from dataclasses import dataclass
from pathlib import Path
from .base import BasePhysicsConfig


@dataclass(kw_only=True)
class WaterPhysicsConfig(BasePhysicsConfig, name='water'):
    path: str = str(Path(__file__).parent.resolve() / 'templates' / 'water.py')
    elasticity: str = 'sigma'
