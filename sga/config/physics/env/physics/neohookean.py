from dataclasses import dataclass
from pathlib import Path
from .base import BasePhysicsConfig


@dataclass(kw_only=True)
class NeoHookeanPhysicsConfig(BasePhysicsConfig, name='neohookean'):
    path: str = str(Path(__file__).parent.resolve() / 'templates' / 'neohookean.py')
    youngs_modulus_log: float = 13.0
    poissons_ratio_sigmoid: float = 0.0
