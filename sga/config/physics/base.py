from dataclasses import dataclass
from sga.utils import Config
from .env import BaseEnvConfig
from .render import BaseRenderConfig
from .sim import BaseSimConfig

@dataclass(kw_only=True)
class BasePhysicsConfig(Config):
    env: BaseEnvConfig
    render: BaseRenderConfig
    sim: BaseSimConfig
