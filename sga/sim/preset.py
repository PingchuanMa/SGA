import torch
import torch.nn as nn
from torch import Tensor


class VolumeElasticity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.log_E = nn.Parameter(torch.Tensor([1e5]).log())
        self.register_buffer('nu', torch.Tensor([0.3]))

    def forward(self, F: Tensor) -> Tensor:
        E = self.log_E.exp()
        nu = self.nu

        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        J = torch.det(F).view(-1, 1, 1)
        I = torch.eye(3, dtype=F.dtype, device=F.device).unsqueeze(0)

        stress = la * J * (J - 1) * I

        return stress


class SigmaElasticity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.log_E = nn.Parameter(torch.Tensor([1e5]).log())
        self.register_buffer('nu', torch.Tensor([0.2]))

    def forward(self, F: Tensor) -> Tensor:
        E = self.log_E.exp()
        nu = self.nu

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        U, sigma, Vh = torch.linalg.svd(F) # pylint: disable=not-callable
        sigma = torch.clamp_min(sigma, 1e-5)

        epsilon = sigma.log()
        trace = epsilon.sum(dim=1, keepdim=True)
        tau = 2 * mu * epsilon + la * trace

        stress = torch.matmul(torch.matmul(U, torch.diag_embed(tau)), U.transpose(1, 2))

        return stress
