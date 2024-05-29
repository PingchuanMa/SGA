import torch
import torch.nn as nn


class Physics(nn.Module):

    def __init__(self, youngs_modulus_log: float = {youngs_modulus_log}, poissons_ratio_sigmoid: float = {poissons_ratio_sigmoid}):
        """
        Define trainable continuous physical parameters for differentiable optimization.
        Tentatively initialize the parameters with the default values in args.

        Args:
            youngs_modulus_log (float): log of Young's modulus.
            poissons_ratio_sigmoid (float): Poisson's ratio before sigmoid.
        """

        super().__init__()
        self.youngs_modulus_log = nn.Parameter(torch.tensor(youngs_modulus_log))
        self.poissons_ratio_sigmoid = nn.Parameter(torch.tensor(poissons_ratio_sigmoid))

    def forward(self, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute updated Kirchhoff stress tensor from deformation gradient tensor.

        Args:
            F (torch.Tensor): deformation gradient tensor (B, 3, 3).

        Returns:
            kirchhoff_stress (torch.Tensor): Kirchhoff stress tensor (B, 3, 3).
        """

        youngs_modulus = self.youngs_modulus_log.exp()
        poissons_ratio = self.poissons_ratio_sigmoid.sigmoid() * 0.49

        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))

        I = torch.eye(3, dtype=F.dtype, device=F.device).unsqueeze(0) # (1, 3, 3)

        Ft = F.transpose(1, 2) # (B, 3, 3)
        FFt = torch.matmul(F, Ft) # (B, 3, 3)
        J = torch.linalg.det(F).view(-1, 1, 1) # (B, 1, 1)

        kirchhoff_stress = mu * (FFt - I) + la * torch.log(J) * I # (B, 3, 3)

        return kirchhoff_stress