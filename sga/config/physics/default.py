from dataclasses import dataclass, field
from .base import BasePhysicsConfig
from .env import JellyEnvConfig
from .render import PyVistaRenderConfig
from .sim import LowSimConfig

@dataclass(kw_only=True)
class DefaultPhysicsConfig(BasePhysicsConfig, name='default'):
    env: JellyEnvConfig = field(default_factory=JellyEnvConfig)
    render: PyVistaRenderConfig = field(default_factory=PyVistaRenderConfig)
    sim: LowSimConfig = field(default_factory=LowSimConfig)
