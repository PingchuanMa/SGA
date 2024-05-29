from typing import Any, Optional, Sequence
from collections import OrderedDict

import warp as wp
import warp.sim
from warp.context import Devicelike


ShapeLike = Sequence[int] | int


class Statics(object):

    def init(self, shape: ShapeLike, device: Devicelike = None) -> None:
        raise NotImplementedError

    @staticmethod
    @wp.kernel
    def set_int(x: wp.array(dtype=int), start: int, end: int, value: int) -> None:
        p = wp.tid()
        if start <= p < end:
            x[p] = value

    @staticmethod
    @wp.kernel
    def set_float(x: wp.array(dtype=float), start: int, end: int, value: float) -> None:
        p = wp.tid()
        if start <= p < end:
            x[p] = value


class State(object):
    def __init__(
            self,
            shape: ShapeLike,
            device: Devicelike = None,
            requires_grad: bool = False) -> None:
        self.shape = shape
        self.device = wp.get_device(device)
        self.requires_grad = requires_grad

    def zero_grad(self) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def to_torch(self):
        raise NotImplementedError

    def from_torch(self):
        raise NotImplementedError


class Model(object):

    ConstantType = Any
    StaticsType = Statics
    StateType = State

    def __init__(self, constant: ConstantType, device: Devicelike = None, requires_grad: int = False) -> None:
        self.constant = constant
        self.device = wp.get_device(device)
        self.requires_grad = requires_grad

    def state(self, shape: ShapeLike, requires_grad: Optional[bool] = None) -> StateType:
        if requires_grad is None:
            requires_grad = self.requires_grad
        state = self.StateType(shape=shape, device=self.device, requires_grad=requires_grad)
        return state

    def statics(self, shape: ShapeLike) -> StaticsType:
        statics = self.StaticsType()
        statics.init(shape=shape, device=self.device)
        return statics


class ModelBuilder(object):

    ConstantType = Any
    StateType = State
    ModelType = Model

    def __init__(self) -> None:
        self.config = OrderedDict()

        for name in self.ConstantType.cls.__annotations__.keys(): # pylint: disable=no-member
            self.reserve(name)

    def reserve(self, name: str, init: Optional[Any] = None) -> None:
        if name in self.config:
            raise RuntimeError(f'duplicated key ({name}) reserved in ModelBuilder')
        self.config[name] = init

    @property
    def ready(self) -> bool:
        for k, v in self.config.items():
            if v is None:
                return False
        return True

    def build_constant(self) -> ConstantType:
        return self.ConstantType() # pylint: disable=no-value-for-parameter

    def finalize(self, device: Devicelike = None, requires_grad: bool = False) -> ModelType:
        if not self.ready:
            raise RuntimeError(f'config uninitialized: {self.config}')

        constant = self.build_constant()
        model = self.ModelType(constant, device, requires_grad)
        return model

class StateInitializer(object):

    StateType = State
    ModelType = Model

    def __init__(self, model: ModelType) -> None:
        self.model = model

    def finalize(self, shape: Any, requires_grad: bool = False) -> StateType:
        state = self.model.state(shape=shape, requires_grad=requires_grad)
        return state


class StaticsInitializer(object):

    StaticsType = Any
    ModelType = Model

    def __init__(self, model: ModelType) -> None:
        self.model = model

    def finalize(self, shape: Any) -> StaticsType:
        statics = self.model.statics(shape)
        return statics
