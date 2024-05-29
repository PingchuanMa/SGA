from dataclasses import dataclass
from .base import BaseLLMConfig

@dataclass(kw_only=True)
class GPT4LLMConfig(BaseLLMConfig, name='openai-gpt-4-1106-preview'):
    api_key: str | None = None
    model: str = 'gpt-4-1106-preview'

@dataclass(kw_only=True)
class GPT35LLMConfig(BaseLLMConfig, name='openai-gpt-3.5-turbo-0125'):
    api_key: str | None = None
    model: str = 'gpt-3.5-turbo-0125'
