from typing import Any, ClassVar, Optional
from argparse import ArgumentParser
from dataclasses import dataclass, field, is_dataclass


class Candidate(object):
    candidates: ClassVar[dict[str, 'Candidate']] = {}

    def __init_subclass__(cls, name: Optional[str] = None) -> None:
        if name is None:
            cls.candidates = {}
        else:
            if name in cls.candidates:
                raise ValueError(f'Candidate name {name} already exists')
            cls.candidates[name] = cls


@dataclass(kw_only=True)
class Config(Candidate):
    name: str = field(init=False)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        delattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __init_subclass__(cls, name: Optional[str] = None) -> None:
        if name is None:
            cls.candidates = {}
        else:
            if name in cls.candidates:
                raise ValueError(f'Candidate name {name} already exists')
            cls.candidates[name] = cls
            cls.name = name

    def update(self, args: list[str]) -> None:
        kwargs = {}
        for i, arg in enumerate(args):
            if not arg.startswith('--'):
                continue
            if '=' in arg:
                key, val = arg.split('=', maxsplit=1)
            else:
                if i + 1 >= len(args):
                    raise ValueError(f'argument {arg} has no value')
                if args[i + 1].startswith('--'):
                    raise ValueError(f'argument {arg} has no value')
                key = arg
                val = args[i + 1]
            key = key.lstrip('-')
            kwargs[key] = val

        # use sorted to ensure configs are updated before its properties
        for key, val in sorted(kwargs.items()):
            cfg = self
            for level in key.split('.')[:-1]:
                if not level in cfg:
                    raise ValueError(f'unknown argument {key}')
                cfg = cfg[level]
            final_key = key.split('.')[-1]
            final_val = cfg[final_key]
            if is_dataclass(final_val):
                cfg[final_key] = final_val.candidates[val]()
            elif isinstance(final_val, bool):
                cfg[final_key] = val.casefold() in ['true', '1', 't', 'y', 'yes']
            elif isinstance(final_val, tuple):
                cfg[final_key] = tuple(type(cfg_v)(arg_v) for cfg_v, arg_v in zip(final_val, val.strip('()').split(',')))
            else:
                cfg[final_key] = cfg.__dataclass_fields__[final_key].type(val)


def get_script_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('-y', dest='overwrite', action='store_true', help='overwrite')
    parser.add_argument('-r', dest='resume', action='store_true', help='resume')
    parser.add_argument('-g', '--gpu', dest='gpu', type=int, default=0, help='gpu device')
    return parser
