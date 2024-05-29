import torch
import torch.nn as nn


class Physics(nn.Module):

    def __init__(self, youngs_modulus_log: float = {youngs_modulus_log}, poissons_ratio: float = {poissons_ratio}, yield_stress: float = {yield_stress}):
        """
        Define trainable continuous physical parameters for differentiable optimization.
        Tentatively initialize the parameters with the default values in args.

        Args:
            youngs_modulus_log (float): log of Young's modulus.
            poissons_ratio (float): Poisson's ratio.
            yield_stress (float): yield stress.
        """

        super().__init__()
        self.youngs_modulus_log = nn.Parameter(torch.tensor(youngs_modulus_log))
        self.poissons_ratio = nn.Parameter(torch.tensor(poissons_ratio))
        self.yield_stress = nn.Parameter(torch.tensor(yield_stress))

    def forward(self, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute corrected deformation gradient from deformation gradient tensor.

        Args:
            F (torch.Tensor): deformation gradient tensor (B, 3, 3).

        Returns:
            F_corrected (torch.Tensor): corrected deformation gradient tensor (B, 3, 3).
        """

        youngs_modulus = self.youngs_modulus_log.exp()
        poissons_ratio = self.poissons_ratio
        yield_stress = self.yield_stress

        mu = youngs_modulus / (2 * (1 + poissons_ratio))

        U, sigma, Vh = torch.linalg.svd(F) # (B, 3, 3), (B, 3), (B, 3, 3)

        threshold = 0.01
        sigma = torch.clamp_min(sigma, threshold) # (B, 3)

        epsilon = torch.log(sigma) # (B, 3)
        epsilon_trace = epsilon.sum(dim=1, keepdim=True) # (B, 1)
        epsilon_bar = epsilon - epsilon_trace / 3 # (B, 3)
        epsilon_bar_norm = epsilon_bar.norm(dim=1, keepdim=True) + 1e-5 # (B, 1)

        delta_gamma = epsilon_bar_norm - yield_stress / (2 * mu) # (B, 1)

        yield_epsilon = epsilon - (delta_gamma / epsilon_bar_norm) * epsilon_bar # (B, 3)
        yield_sigma = torch.exp(yield_epsilon) # (B, 3)

        sigma = torch.where((delta_gamma > 0).view(-1, 1), yield_sigma, sigma) # (B, 3)
        F_corrected = torch.matmul(U, torch.matmul(torch.diag_embed(sigma), Vh)) # (B, 3, 3)

        return F_corrected