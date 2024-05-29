from dataclasses import dataclass
from sga.utils import Config

from .camera import BaseCamera


@dataclass(kw_only=True)
class BaseRenderConfig(Config):
    camera: BaseCamera

    width: int = 512
    height: int = 512
    fps: int = 10
    format: str = '%04d.png'

    skip_frames: int = 20
