from typing import ClassVar, Optional
from dataclasses import dataclass, field
from pathlib import Path
import re
import os
import json
from tqdm import tqdm

import numpy as np

try:
    import openai.types.chat
    from openai import OpenAI
except ImportError as e:
    print('OpenAI API is not available, please install using `pip install openai`.')
    raise e

try:
    import anthropic.types.message
    from anthropic import Anthropic
except ImportError as e:
    print('Anthropic API is not available, please install using `pip install anthropic`.')
    raise e

try:
    import mistralai
    import mistralai.models.chat_completion
    from mistralai.client import MistralClient
except ImportError as e:
    print('Mistral API is not available, please install using `pip install mistralai`.')
    raise e

from sga.config.llm import BaseLLMConfig


@dataclass
class AnalysisData(object):
    analysis: str
    fitness: float = float('inf')


@dataclass
class ChoiceData(object):
    section_pattern: ClassVar[re.Pattern] = re.compile(
        r'^[\s\S]*?(### Analysis\n[\s\S]*?)\n(### Step-by-Step Plan\n[\s\S]*?)\n(### Code\n[\s\S]*?)$')
    code_subsection_pattern: ClassVar[re.Pattern] = re.compile(
        r'```python\n([\s\S]*?\n)```')

    content_raw: str

    # from section_pattern
    analysis_raw: str = field(init=False)
    plan_raw: str = field(init=False)
    code_raw: str = field(init=False)
    code: str = field(init=False) # from code_subsection_pattern

    dump_root: Path = None

    def __post_init__(self) -> None:
        try:
            self.analysis_raw, self.plan_raw, self.code_raw = self.section_pattern.findall(self.content_raw)[0]
        except IndexError:
            self.analysis_raw = ''
            self.plan_raw = ''
            self.code_raw = ''
            self.code = ''
        try:
            self.code = self.code_subsection_pattern.findall(self.code_raw)[-1]
        except IndexError:
            self.code = ''

    def dump(self, dump_root: Path | str, record: bool) -> None:
        dump_root = Path(dump_root)
        dump_root.mkdir(parents=True, exist_ok=True)
        if record:
            self.dump_root = dump_root
        (dump_root / 'content.md').write_text(self.content_raw, 'utf-8')
        (dump_root / 'analysis.md').write_text(self.analysis_raw, 'utf-8')
        (dump_root / 'plan.md').write_text(self.plan_raw, 'utf-8')
        (dump_root / 'code.md').write_text(self.code_raw, 'utf-8')
        (dump_root / 'code.py').write_text(self.code, 'utf-8')


@dataclass
class BaseResponseData(object):
    responses: list[
        anthropic.types.message.Message |
        openai.types.chat.ChatCompletion |
        mistralai.models.chat_completion.ChatCompletionResponse]
    choices: list[ChoiceData] = field(default_factory=list, kw_only=True)


@dataclass
class OpenAIResponseData(BaseResponseData):
    responses: list[
        openai.types.chat.ChatCompletion |
        mistralai.models.chat_completion.ChatCompletionResponse]

    def __post_init__(self) -> None:
        for response in self.responses:
            for choice_raw in response.choices:
                content = choice_raw.message.content
                content = content.replace('import torch.nn as.nn', 'import torch.nn as nn')
                choice = ChoiceData(content)
                self.choices.append(choice)


@dataclass
class AnthropicResponseData(BaseResponseData):
    responses: list[anthropic.types.message.Message]

    def __post_init__(self) -> None:
        for response in self.responses:
            content = response.content[0].text
            content = content.replace('import torch.nn as.nn', 'import torch.nn as nn')
            choice = ChoiceData(content)
            self.choices.append(choice)

@dataclass
class IndividualData(object):
    index: int
    code: str
    feedback: str
    fitness: float
    losses: list[float]
    params: dict[str, list[float]]
    states: dict[str, np.ndarray]
    root: Path | str

    def dump(self) -> None:
        dump_root = Path(self.root) / 'data'
        dump_root.mkdir(parents=True, exist_ok=True)
        (dump_root / 'code.py').write_text(self.code, 'utf-8')
        (dump_root / 'feedback.md').write_text(self.feedback, 'utf-8')
        with (dump_root / 'misc.json').open('w', encoding='utf-8') as f:
            json.dump({'fitness': self.fitness}, f, indent=2)


@dataclass
class PrimitiveData(IndividualData):
    def __post_init__(self) -> None:
        self.dump()


@dataclass
class OffspringData(IndividualData):

    choice: ChoiceData
    analysis_raw: str = field(init=False)
    plan_raw: str = field(init=False)
    code_raw: str = field(init=False)
    code: str = field(init=False)

    def __post_init__(self) -> None:
        self.analysis_raw = self.choice.analysis_raw
        self.plan_raw = self.choice.plan_raw
        self.code_raw = self.choice.code_raw
        self.code = self.choice.code
        self.dump()


class Population(object):
    def __init__(self, cfg: BaseLLMConfig) -> None:
        self.offsprings: list[IndividualData] = []

        self.batch_size = cfg.batch_size
        self.randomness = cfg.randomness

    def add_primitive(
            self,
            code_path: Path | str,
            feedback: str,
            fitness: float,
            losses: list[float],
            params: dict[str, list[float]],
            states: dict[str, np.ndarray],
            ind_root: Optional[Path | str]) -> None:
        code = Path(code_path).read_text('utf-8')
        primitive = PrimitiveData(len(self.offsprings), code, feedback, fitness, losses, params, states, ind_root)
        self.offsprings.append(primitive)

    def add_offspring(
            self,
            choice: ChoiceData,
            feedback: str,
            fitness: float,
            losses: list[float],
            params: dict[str, list[float]],
            states: dict[str, np.ndarray],
            ind_root: Optional[Path | str] = None) -> None:
        offspring = OffspringData(len(self.offsprings), feedback, fitness, losses, params, states, ind_root, choice)
        self.offsprings.append(offspring)

    def sample(self, dump_root: Optional[Path | str]) -> list[int]:
        dump_root = Path(dump_root)
        dump_root.mkdir(parents=True, exist_ok=True)
        indices = [i for i, offspring in enumerate(self.offsprings) if offspring.fitness < float('inf')]
        if len(indices) < self.batch_size:
            indices = list(range(len(self.offsprings)))
        indices = sorted(indices, key=lambda i: self.offsprings[i].fitness)
        with (dump_root / 'all.json').open('w', encoding='utf-8') as f:
            json.dump([{
                'rank': i,
                'fitness': self.offsprings[idx].fitness,
                'root': str(Path(self.offsprings[idx].root))
            } for i, idx in enumerate(indices)], f, indent=2)
        if self.randomness == 'none':
            if len(indices) > self.batch_size:
                indices = indices[:self.batch_size]
        else:
            raise RuntimeError(f'Unknown randomness: {self.randomness}')
        with (dump_root / 'selected.json').open('w', encoding='utf-8') as f:
            json.dump([{
                'rank': i,
                'fitness': self.offsprings[idx].fitness,
                'root': str(Path(self.offsprings[idx].root))
            } for i, idx in enumerate(indices)], f, indent=2)
        return indices


class BasePhysicist(object):

    separator: str = '\n\n'

    def __init__(self, cfg: BaseLLMConfig, seed: int, env_info: Optional[str] = None) -> None:
        self.seed = seed
        self.env_info = env_info if env_info is not None else ''

        self.api_key = cfg.api_key
        self.model = cfg.model
        self.frequency_penalty = cfg.frequency_penalty
        self.presence_penalty = cfg.presence_penalty
        self.top_p = cfg.top_p

        self.num_exploit = cfg.num_exploit
        self.num_explore = cfg.num_explore

        self.temperature_exploit = cfg.temperature_exploit
        self.temperature_explore = cfg.temperature_explore

        if cfg.name.startswith('anthropic'):
            self.client = Anthropic(
                api_key=self.api_key if self.api_key is not None else os.environ.get('ANTHROPIC_API_KEY'), max_retries=5)
        elif cfg.name.startswith('openai'):
            self.client = OpenAI(
                api_key=self.api_key if self.api_key is not None else os.getenv('OPENAI_API_KEY'), max_retries=5)
        elif cfg.name.startswith('mistral'):
            self.client = MistralClient(
                api_key=self.api_key if self.api_key is not None else os.getenv('MISTRAL_API_KEY'), max_retries=5, timeout=5 * 60)
        else:
            raise RuntimeError(f'Unknown LLM name: {cfg.name}')

        self.base_prompt_root = Path(__file__).parent / 'prompts' / 'base'
        self.prompts: dict[str, str] = {path.stem: path.read_text('utf-8') for path in self.base_prompt_root.iterdir() if path.is_file()}

        self.prompt_root = self.base_prompt_root.parent / cfg.entry
        self.prompts |= {path.stem: path.read_text('utf-8') for path in self.prompt_root.iterdir() if path.is_file()}

    def get_msgs(self, population: Population, indices: list[int], dump_root: Optional[Path | str] = None) -> list[dict[str, str]]:
        msgs = [{'role': 'system', 'content': self.prompts['system']},
                {'role': 'user', 'content': self.prompts['prologue']}]

        for i, idx in enumerate(reversed(indices)):
            offspring = population.offsprings[idx]
            if isinstance(offspring, PrimitiveData):
                code = self.prompts['code'].format(code=offspring.code)
                content = code
            elif isinstance(offspring, OffspringData):
                plan = offspring.plan_raw
                code = offspring.code_raw
                content = self.separator.join([plan, code])
            content = self.prompts['iteration'].format(content=content, iter=i + 1)
            msgs.append({'role': 'assistant', 'content': content})

            feedback = offspring.feedback
            if offspring.fitness < float('inf'):
                content = self.prompts['feedback'].format(feedback=feedback)
            else:
                content = self.prompts['error'].format(error=feedback)
            msgs.append({'role': 'user', 'content': content})

        content = self.prompts['epilogue']
        msgs.append({'role': 'user', 'content': content})

        content = self.prompts['iteration'].format(content='', iter=len(indices) + 1).rstrip()
        msgs.append({'role': 'assistant', 'content': content})

        if dump_root is not None:
            dump_root = Path(dump_root)
            dump_root.mkdir(parents=True, exist_ok=True)
            for i_msg, msg in enumerate(msgs):
                role = msg['role']
                content = msg['content']
                dump_path = dump_root / f'{i_msg:04d}-{role}.md'
                dump_path.write_text(content, 'utf-8')

        return msgs

    def send_messages(
            self,
            msgs: list[dict[str, str]],
            num_responses: int,
            temperature: float,
            response_path: Optional[Path | str] = None) -> list[
                anthropic.types.message.Message | openai.types.chat.ChatCompletion]:

        responses = []

        # Anthropic Claude
        if isinstance(self.client, Anthropic):
            sys_msg = msgs[0]['content']
            anthropic_msgs = []
            pre_role = ''
            for msg in msgs[1:]:
                if msg['role'] == pre_role:
                    anthropic_msgs[-1] = {
                        'role': msg['role'],
                        'content': self.separator.join([anthropic_msgs[-1]['content'], msg['content']])
                    }
                else:
                    anthropic_msgs.append(msg)
                    pre_role = msg['role']
            for i in range(num_responses):
                response = self.client.messages.create(
                    model=self.model,
                    system=sys_msg,
                    messages=anthropic_msgs,
                    temperature=temperature,
                    max_tokens=4096,
                    top_p=self.top_p,
                )
                responses.append(response)
                if response_path is not None:
                    with response_path.with_stem(f'{response_path.stem}_{i}').open('w', encoding='utf-8') as f:
                        json.dump(response.model_dump(mode='json'), f, indent=2)

        # Mistral
        elif isinstance(self.client, MistralClient):
            mistral_msgs = []
            for msg in msgs:
                msg = mistralai.models.chat_completion.ChatMessage(role=msg['role'], content=msg['content'])
                mistral_msgs.append(msg)
            if len(mistral_msgs) > 0 and mistral_msgs[-1].role == 'assistant':
                mistral_msgs[-1].role = 'user'
            for i in range(num_responses):
                response = self.client.chat(
                    model=self.model,
                    messages=mistral_msgs,
                    temperature=temperature,
                    max_tokens=6144,
                    top_p=self.top_p,
                    random_seed=self.seed * 42 + i,
                )
                responses.append(response)
                if response_path is not None:
                    with response_path.with_stem(f'{response_path.stem}_{i}').open('w', encoding='utf-8') as f:
                        json.dump(response.model_dump(mode='json'), f, indent=2)

        # OpenAI GPT
        elif isinstance(self.client, OpenAI):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                temperature=temperature,
                n=num_responses,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                top_p=self.top_p,
                seed=self.seed,
            )
            if response_path is not None:
                with response_path.open('w', encoding='utf-8') as f:
                    json.dump(response.model_dump(mode='json'), f, indent=2)
            responses.append(response)

        else:
            raise RuntimeError(f'Unknown LLM client: {self.client}')
        return responses

    def generate(
            self,
            msgs: list[dict[str, str]],
            choice_root: Optional[Path | str] = None,
            response_root: Optional[Path | str] = None) -> BaseResponseData:
        responses = []
        if response_root is not None:
            response_root = Path(response_root)
            response_root.mkdir(parents=True, exist_ok=True)
        if self.num_exploit > 0:
            tqdm.write('Exploiting...')
            new_responses = self.send_messages(
                msgs, self.num_exploit, self.temperature_exploit,
                response_root / 'exploit.json' if response_root is not None else None)
            responses += new_responses
        if self.num_explore > 0:
            tqdm.write('Exploring...')
            new_responses = self.send_messages(msgs, self.num_explore, self.temperature_explore,
                response_root / 'explore.json' if response_root is not None else None)
            responses += new_responses
        tqdm.write('Done.')
        if isinstance(self.client, Anthropic):
            response_data = AnthropicResponseData(responses)
        elif isinstance(self.client, (OpenAI, MistralClient)):
            response_data = OpenAIResponseData(responses)
        else:
            raise RuntimeError(f'Unknown LLM client: {self.client}')
        if choice_root is not None:
            choice_root = Path(choice_root)
            choice_root.mkdir(parents=True, exist_ok=True)
            for i_choice, choice in enumerate(response_data.choices):
                choice.dump(choice_root / f'{i_choice:04d}', True)
        return response_data


class ConstitutivePhysicist(BasePhysicist):

    def __init__(self, cfg: BaseLLMConfig, seed: int, env_info: Optional[str] = None) -> None:
        super().__init__(cfg, seed, env_info)
        self.prompts['prologue'] = self.prompts['prologue'].format(
            context=self.separator.join([self.prompts['context'],self.env_info]),
            task=self.prompts['task'])
        self.prompts['format'] = self.prompts['format'].format(code=self.prompts['code_tmpl'])
        self.prompts['epilogue'] = self.prompts['epilogue'].format(format=self.prompts['format'])


class MolecularPhysicist(BasePhysicist):

    def __init__(self, cfg: BaseLLMConfig, seed: int) -> None:
        super().__init__(cfg, seed, None)
        self.prompts['context'] = self.prompts[f'context_{cfg.partition}']
        self.prompts['prologue'] = self.prompts['prologue'].format(
            context=self.prompts['context'],
            task=self.prompts['task'])
        self.prompts['format'] = self.prompts['format'].format(code=self.prompts['code_tmpl'])
        self.prompts['epilogue'] = self.prompts['epilogue'].format(format=self.prompts['format'])

        self.example_code = (self.prompt_root / 'example.py').read_text('utf-8')
