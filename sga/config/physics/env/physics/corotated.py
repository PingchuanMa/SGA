from dataclasses import dataclass
from pathlib import Path
from .base import BasePhysicsConfig


@dataclass(kw_only=True)
class CorotatedPhysicsConfig(BasePhysicsConfig, name='corotated'):
    path: str = str(Path(__file__).parent.resolve() / 'templates' / 'corotated.py')
    youngs_modulus_log: float = 10.0
    poissons_ratio_sigmoid: float = -1.0
