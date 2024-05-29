import torch
import torch.nn as nn


class Physics(nn.Module):

    def __init__(self, youngs_modulus_log: float = {youngs_modulus_log}, poissons_ratio: float = {poissons_ratio}, yield_stress: float = {yield_stress}, alpha: float = {alpha}, cohesion: float = {cohesion}):
        """
        Define trainable continuous physical parameters for differentiable optimization.
        Tentatively initialize the parameters with the default values in args.

        Args:
            youngs_modulus_log (float): log of Young's modulus.
            poissons_ratio (float): Poisson's ratio.
            yield_stress (float): yield stress.
            alpha (float): alpha.
            cohesion (float): cohesion.
        """

        super().__init__()
        self.youngs_modulus_log = nn.Parameter(torch.tensor(youngs_modulus_log))
        self.poissons_ratio = nn.Parameter(torch.tensor(poissons_ratio))
        self.yield_stress = nn.Parameter(torch.tensor(yield_stress))
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.cohesion = nn.Parameter(torch.tensor(cohesion))

    def plasticine(self, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def sand(self, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def water(self, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute corrected deformation gradient from deformation gradient tensor.

        Args:
            F (torch.Tensor): deformation gradient tensor (B, 3, 3).

        Returns:
            F_corrected (torch.Tensor): corrected deformation gradient tensor (B, 3, 3).
        """

        J = torch.det(F).view(-1, 1, 1) # (B, 1, 1)
        Je_1_3 = torch.pow(J, 1 / 3) # (B, 1, 1)
        sigma_corrected = Je_1_3.view(-1, 1).expand(-1, 3) # (B, 3)

        F_corrected = torch.diag_embed(sigma_corrected) # (B, 3, 3)

        return F_corrected

    def forward(self, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute corrected deformation gradient from deformation gradient tensor.

        Args:
            F (torch.Tensor): deformation gradient tensor (B, 3, 3).

        Returns:
            F_corrected (torch.Tensor): corrected deformation gradient tensor (B, 3, 3).
        """

        plasticine_F_corrected = self.plasticine(F)
        sand_F_corrected = self.sand(F)
        water_F_corrected = self.water(F)

        F_corrected = 0.5 * plasticine_F_corrected + 0.3 * sand_F_corrected + 0.2 * water_F_corrected

        return F_corrected