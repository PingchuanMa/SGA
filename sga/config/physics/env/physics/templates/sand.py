import torch
import torch.nn as nn


class Physics(nn.Module):

    def __init__(self, youngs_modulus_log: float = {youngs_modulus_log}, poissons_ratio: float = {poissons_ratio}, alpha: float = {alpha}, cohesion: float = {cohesion}):
        """
        Define trainable continuous physical parameters for differentiable optimization.
        Tentatively initialize the parameters with the default values in args.

        Args:
            youngs_modulus_log (float): log of Young's modulus.
            poissons_ratio (float): Poisson's ratio.
            alpha (float): alpha.
            cohesion (float): cohesion.
        """

        super().__init__()
        self.youngs_modulus_log = nn.Parameter(torch.tensor(youngs_modulus_log))
        self.poissons_ratio = nn.Parameter(torch.tensor(poissons_ratio))
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.cohesion = nn.Parameter(torch.tensor(cohesion))

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
        alpha = self.alpha
        cohesion = self.cohesion

        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))

        U, sigma, Vh = torch.linalg.svd(F) # (B, 3, 3), (B, 3), (B, 3, 3)

        # prevent NaN
        thredhold = 0.05
        sigma = torch.clamp_min(sigma, thredhold)

        epsilon = torch.log(sigma)
        trace = epsilon.sum(dim=1, keepdim=True)
        epsilon_hat = epsilon - trace / 3
        epsilon_hat_norm = torch.linalg.norm(epsilon_hat, dim=1, keepdim=True)

        expand_epsilon = torch.ones_like(epsilon) * cohesion

        shifted_trace = trace - cohesion * 3
        cond_yield = (shifted_trace < 0).view(-1, 1)

        delta_gamma = epsilon_hat_norm + (3 * la + 2 * mu) / (2 * mu) * shifted_trace * alpha
        compress_epsilon = epsilon - (torch.clamp_min(delta_gamma, 0.0) / epsilon_hat_norm) * epsilon_hat

        epsilon = torch.where(cond_yield, compress_epsilon, expand_epsilon)

        F_corrected = torch.matmul(torch.matmul(U, torch.diag_embed(epsilon.exp())), Vh)

        return F_corrected