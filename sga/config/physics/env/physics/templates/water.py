import torch
import torch.nn as nn


class Physics(nn.Module):

    def __init__(self):
        """
        Define trainable continuous physical parameters for differentiable optimization.
        Tentatively initialize the parameters with the default values in args.
        """

        super().__init__()

    def forward(self, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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