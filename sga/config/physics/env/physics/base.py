from dataclasses import dataclass

from sga.utils import Config


@dataclass(kw_only=True)
class BasePhysicsConfig(Config):
    path: str = None
