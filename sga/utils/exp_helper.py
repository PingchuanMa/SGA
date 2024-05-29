from typing import Any
from pathlib import Path
from collections import defaultdict
import importlib.util
import subprocess
from tqdm import tqdm

import torch
from torch import Tensor
import torch.nn.functional as F
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def dict_to_cmds(d: dict) -> list[str]:
    return [f'--{k}={v}' for k, v in d.items()]


def parse_tensorboard(path: Path | str) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    event_acc = EventAccumulator(str(path))
    event_acc.Reload()
    losses = {}
    params = {}
    for tag in event_acc.Tags()['scalars']:
        curve = [scalar.value for scalar in event_acc.Scalars(tag)]
        prefix, name = tag.strip().casefold().split('/', maxsplit=1)
        if prefix == 'loss':
            losses[name] = curve
        elif prefix == 'param':
            params[name] = curve
    return losses, params


def get_class_from_path(path: Path | str, class_name: str) -> Any:
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def loss_fn(x: Tensor, gt_x: Tensor) -> Tensor:
    return F.mse_loss(x, gt_x, reduction='mean')


class StateRecorder(object):
    def __init__(self):
        self.states = defaultdict(list)
        self.hyperstates = {}
    def add_hyper(self, **kwargs):
        self.hyperstates.update(kwargs)
    def add(self, **kwargs):
        for key, val in kwargs.items():
            self.states[key].append(val.detach().clone())
    def save(self, path: Path | str):
        states = {key: torch.stack(val, dim=0) for key, val in self.states.items()}
        states.update(self.hyperstates)
        torch.save(states, path)


def run_exp(base_cmds: list[str], extra_args: dict[str, Any], unknown_args: list[str], env: dict[str, str]) -> str:
    train_cmds = base_cmds + dict_to_cmds(extra_args) + unknown_args
    str_train_cmds = [str(c) for c in train_cmds]
    with subprocess.Popen(str_train_cmds, shell=False, stderr=subprocess.PIPE, env=env) as res:
        _, error = res.communicate()
    if error is not None:
        error = error.decode('utf-8').strip(' \n\r\t\f')
        print(error)
    else:
        error = ''
    return error
