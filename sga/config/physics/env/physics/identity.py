from dataclasses import dataclass
from pathlib import Path
from .base import BasePhysicsConfig


@dataclass(kw_only=True)
class IdentityPhysicsConfig(BasePhysicsConfig, name='identity'):
    path: str = str(Path(__file__).parent.resolve() / 'templates' / 'identity.py')
    elasticity: str = 'sigma'
