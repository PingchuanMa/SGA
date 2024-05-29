from dataclasses import dataclass

from sga.utils import Config


@dataclass(kw_only=True)
class BaseCamera(Config):
    origin: tuple[float, float, float] = (1, 1, 1)
    target: tuple[float, float, float] = (0, 0, 0)
    up: tuple[float, float, float] = (0, 1, 0)
