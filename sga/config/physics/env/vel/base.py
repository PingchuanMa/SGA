from dataclasses import dataclass
from sga.utils import Config

@dataclass(kw_only=True)
class BaseVelConfig(Config):
    random: bool
