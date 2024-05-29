from dataclasses import dataclass, field

from .base import BaseRenderConfig
from .camera import SphereCamera


@dataclass(kw_only=True)
class PyVistaRenderConfig(BaseRenderConfig, name='pv'):
    camera: SphereCamera = field(default_factory=SphereCamera)

    background: str = 'grey'
