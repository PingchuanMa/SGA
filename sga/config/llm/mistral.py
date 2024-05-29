from dataclasses import dataclass
from .base import BaseLLMConfig

@dataclass(kw_only=True)
class Mixtral8x7BLLMConfig(BaseLLMConfig, name='mistral-open-mixtral-8x7b'):
    api_key: str | None = None
    model: str = 'open-mixtral-8x7b'
