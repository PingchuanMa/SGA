from dataclasses import dataclass
from pathlib import Path
from .base import BasePhysicsConfig


@dataclass(kw_only=True)
class PlasticinePhysicsConfig(BasePhysicsConfig, name='plasticine'):
    path: str = str(Path(__file__).parent.resolve() / 'templates' / 'plasticine.py')
    elasticity: str = 'sigma'
    youngs_modulus_log: float = 13.0
    poissons_ratio: float = 0.25
    yield_stress: float = 3e4
