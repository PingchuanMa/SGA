from dataclasses import dataclass
from .base import BaseLLMConfig

@dataclass(kw_only=True)
class Claude3SonnetLLMConfig(BaseLLMConfig, name='anthropic-claude-3-sonnet-20240229'):
    api_key: str | None = None
    model: str = 'claude-3-sonnet-20240229'
