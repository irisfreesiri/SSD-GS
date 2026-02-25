import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScatterMLP(nn.Module):
    """
    input: lat code per gaussian point, light direction, view direction
    output: scattering
    """
    def __init__(self, neural_material_size):
        super(ScatterMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3 * 27 + 3 + neural_material_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 3)
        ).cuda()

    def freeze(self):
        for name, param in self.network.named_parameters():
            param.requires_grad = False

    def unfreeze(self):
        for name, param in self.network.named_parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Predict per-point reduced scattering parameters and evaluate
        the scalar standard dipole BSSRDF radial profile.
        """

        out = self.network(x)
        # effective diffusion distance parameter
        # Typical reference range: 0.5–3.0 (skin, wax, marble etc.)
        effective_r = torch.sigmoid(out[:, 0]) * 3.0 + 0.1
        # Reduced scattering coefficient
        # Typical reference range: 0.2–2.0 (higher for low-absorption materials like wax, lower for skin).
        sigma_s_prime = torch.sigmoid(out[:, 1]).unsqueeze(-1) * 2.0 + 0.05
        # Absorption coefficient
        # Typical reference range: 0.1–2.0 (higher for highly absorbing materials like marble).
        sigma_a = torch.sigmoid(out[:, 2]).unsqueeze(-1) * 2.0 + 0.05

        scatter = self.dipole_scatter_profile(effective_r, sigma_s_prime, sigma_a)
        return scatter

    def dipole_scatter_profile(self, r, sigma_s_prime, sigma_a, eta=1.3):
        """
        Single-channel Standard Dipole BSSRDF profile.
        This implementation follows Jensen et al. (2001) diffusion theory

        Args:
            r: Distance between the incident point and outgoing point
            sigma_s_prime: Scattering coefficient for each Gaussian point.
            sigma_a: Absorption coefficient for each Gaussian point.
            eta: Relative index of refraction (IOR). Typically a global constant (e.g., 1.3 for skin or water).

        Returns:
            Tensor: Scalar BSSRDF profile value for each point at distance r.
                    Should be multiplied with the multi-channel ksss to get the final per-channel scatter contribution.
        """
        sigma_t_prime = sigma_s_prime + sigma_a  # [N, 1]
        sigma_tr = torch.sqrt(3 * sigma_a * sigma_t_prime)
        albedo_prime = sigma_s_prime / sigma_t_prime  # [N, 1]

        D = 1.0 / (3.0 * sigma_t_prime)  # [N, 1]
        Fdr = (-1.440 / eta ** 2 + 0.710 / eta + 0.668 + 0.0636 * eta)
        A = (1 + Fdr) / (1 - Fdr)

        zr = 1.0 / sigma_t_prime  # [N, 1]
        zv = zr + 4.0 * A * D  # [N, 1]

        r_expand = r.unsqueeze(-1)  # [N, 1]
        dr = torch.sqrt(r_expand ** 2 + zr ** 2)  # [N, 1]
        dv = torch.sqrt(r_expand ** 2 + zv ** 2)  # [N, 1]

        term_r = zr * (sigma_tr * dr + 1) * torch.exp(-sigma_tr * dr) / dr.pow(3)
        term_v = zr * zv * (sigma_tr * dv + 1) * torch.exp(-sigma_tr * dv) / dv.pow(3)
        profile = albedo_prime * (term_r + term_v) / (4.0 * math.pi)  # [N, 1]

        return profile

