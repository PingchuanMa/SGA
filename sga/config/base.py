from dataclasses import dataclass
from sga.utils import Config
from .optim import BaseOptimConfig
from .llm import BaseLLMConfig
from .physics import BasePhysicsConfig

@dataclass(kw_only=True)
class BaseConfig(Config):
    path: str

    optim: BaseOptimConfig
    llm: BaseLLMConfig
    physics: BasePhysicsConfig

    tpos: int = 0 # tqdm position
    seed: int = 0
    num_cpus: int = 16
    gpu: int = 0
    overwrite: bool = False
    resume: bool = False
