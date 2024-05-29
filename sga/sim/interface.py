import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import Tensor

from sga.warp import Tape
from .mpm import MPMModel, MPMState, MPMStatics


class MPMSimFunction(autograd.Function):

    @staticmethod
    def forward(
            ctx: autograd.function.FunctionCtx,
            model: MPMModel,
            statics: MPMStatics,
            state_curr: MPMState,
            state_next: MPMState,
            x: Tensor,
            v: Tensor,
            C: Tensor,
            F: Tensor,
            stress: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        tape: Tape = Tape()
        state_curr.from_torch(x=x, v=v, C=C, F=F, stress=stress)
        model.forward(statics, state_curr, state_next, tape)

        x_next, v_next, C_next, F_next, _ = state_next.to_torch()

        ctx.model = model
        ctx.tape = tape
        ctx.statics = statics
        ctx.state_curr = state_curr
        ctx.state_next = state_next

        return x_next, v_next, C_next, F_next

    @staticmethod
    def backward(
            ctx: autograd.function.FunctionCtx,
            grad_x_next: Tensor,
            grad_v_next: Tensor,
            grad_C_next: Tensor,
            grad_F_next: Tensor) -> tuple[None, None, None, None, Tensor, Tensor, Tensor, Tensor, Tensor]:

        model: MPMModel = ctx.model
        tape: Tape = ctx.tape
        statics: MPMStatics = ctx.statics
        state_curr: MPMState = ctx.state_curr
        state_next: MPMState = ctx.state_next

        state_next.from_torch_grad(
            grad_x=grad_x_next,
            grad_v=grad_v_next,
            grad_C=grad_C_next,
            grad_F=grad_F_next)

        model.backward(statics, state_curr, state_next, tape)

        grad_x, grad_v, grad_C, grad_F, grad_stress = state_curr.to_torch_grad()

        return None, None, None, None, grad_x, grad_v, grad_C, grad_F, grad_stress


class MPMSim(nn.Module):

    def __init__(self, model: MPMModel, statics: MPMStatics) -> None:
        super().__init__()
        self.model = model
        self.statics = statics


class MPMDiffSim(MPMSim):

    def forward(self, x: Tensor, v: Tensor, C: Tensor, F: Tensor, stress: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        shape = x.size(0)
        state_curr: MPMState = self.model.state(shape)
        state_next: MPMState = self.model.state(shape)
        return MPMSimFunction.apply(self.model, self.statics, state_curr, state_next, x, v, C, F, stress)


class MPMCacheDiffSim(MPMSim):

    def __init__(self, model: MPMModel, statics: MPMStatics, num_steps: int) -> None:
        super().__init__(model, statics)
        self.curr_states = [None for _ in range(num_steps)]
        self.next_states = [None for _ in range(num_steps)]

    def forward(self, step: int, x: Tensor, v: Tensor, C: Tensor, F: Tensor, stress: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        shape = x.size(0)
        if self.curr_states[step] is None:
            self.curr_states[step] = self.model.state(shape)
        if self.next_states[step] is None:
            self.next_states[step] = self.model.state(shape)
        state_curr = self.curr_states[step]
        state_next = self.next_states[step]
        return MPMSimFunction.apply(self.model, self.statics, state_curr, state_next, x, v, C, F, stress)


class MPMForwardSim(MPMSim):

    def forward(self, state: MPMState) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        model = self.model
        model.forward(self.statics, state, state, None)
        x_next, v_next, C_next, F_next, _ = state.to_torch()
        return x_next, v_next, C_next, F_next
