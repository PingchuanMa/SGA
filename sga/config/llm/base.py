from dataclasses import dataclass
from sga.utils import Config

@dataclass(kw_only=True)
class BaseLLMConfig(Config):
    api_key: str | None
    model: str

    primitives: tuple[str] = ('linear',)
    entry: str = 'elasticity'
    target: float = 0.0
    partition: str = 'gap'
    train: bool = True

    num_iters: int = 5
    batch_size: int = 5
    state_size: int = 10
    randomness: str = 'none'

    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    top_p: float = 1.0

    num_exploit: int = 4
    num_explore: int = 12

    temperature_exploit: float = 0.5
    temperature_explore: float = 1.0
