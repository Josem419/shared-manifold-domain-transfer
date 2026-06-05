"""
RunwayMFlow — Injective normalizing flow for manifold learning.

Implements the key ideas from Brehmer & Cranmer, NeurIPS 2020 in pure PyTorch
avoiding using an external manifold learning library

Architecture
------------
  Encoder MLP:  R^1280 → R^64  (manifold coordinates z)
  Noise head:   R^1280 → R^1216 (off-manifold noise e)
  Manifold flow: RealNVP bijection on R^64 (exact log-likelihood)
  Decoder MLP:  R^64  → R^1280 (ambient reconstruction)
  Pose encoder: R^6   → R^64  (injected as z += pose_enc)

Training losses (all on I-JEPA embeddings, not pixels):
  L_recon    = ||decode(encode(x)) - x||^2
  L_nll      = -log p_flow(z)               (exact NLL under RealNVP)
  L_geometry = ||e||^2                       (push noise toward zero)
  L_total    = L_recon + λ1·L_nll + λ2·L_geometry

At inference: set e = 0, condition on pose, use flow inverse for sampling.
"""

from __future__ import annotations

import sys
import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# RealNVP coupling layer
class AffineCouplingLayer(nn.Module):
    """
    Split x into (x1, x2); transform x2 conditioned on x1.
    Alternating mask between layers ensures all dimensions are transformed.
    """

    def __init__(self, dim: int, hidden_dim: int, mask_first_half: bool = True, cond_dim: int = 0) -> None:
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        half = dim // 2
        in_dim  = (half if mask_first_half else (dim - half)) + cond_dim
        out_dim = (dim - half) if mask_first_half else half
        self.mask_first_half = mask_first_half

        # Scale and translate network
        self.st_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim * 2),
        )
        # Initialise last layer near zero so flow starts near identity
        nn.init.zeros_(self.st_net[-1].weight)
        nn.init.zeros_(self.st_net[-1].bias)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (y, log_det_jacobian)."""
        half = self.dim // 2
        if self.mask_first_half:
            x1, x2 = x[:, :half], x[:, half:]
        else:
            x1, x2 = x[:, half:], x[:, :half]

        inp = torch.cat([x1, cond], dim=-1) if cond is not None else x1
        st = self.st_net(inp)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 2.0    # bound scale to [-2, 2] for stability

        y2 = x2 * torch.exp(s) + t
        log_det = s.sum(dim=-1)    # (B,)

        if self.mask_first_half:
            y = torch.cat([x1, y2], dim=-1)
        else:
            y = torch.cat([y2, x1], dim=-1)

        return y, log_det

    def inverse(self, y: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Inverse pass (no log-det needed)."""
        half = self.dim // 2
        if self.mask_first_half:
            y1, y2 = y[:, :half], y[:, half:]
        else:
            y1, y2 = y[:, half:], y[:, :half]

        inp = torch.cat([y1, cond], dim=-1) if cond is not None else y1
        st = self.st_net(inp)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 2.0

        x2 = (y2 - t) * torch.exp(-s)

        if self.mask_first_half:
            return torch.cat([y1, x2], dim=-1)
        else:
            return torch.cat([x2, y1], dim=-1)


class RealNVP(nn.Module):
    """
    Stack of alternating affine coupling layers.

    Forward:  z → u (base space), returns (u, log_det_sum)
    Inverse:  u → z (manifold space)
    Log-prob: standard Gaussian in u space + log_det
    """

    def __init__(self, dim: int, n_layers: int = 8, hidden_dim: int = 128, cond_dim: int = 0) -> None:
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(AffineCouplingLayer(dim, hidden_dim, mask_first_half=(i % 2 == 0), cond_dim=cond_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, z: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """z → u (base), returns (u, log_det_sum (B,))"""
        log_det = torch.zeros(z.shape[0], device=z.device)
        for layer in self.layers:
            z, ld = layer(z, cond)
            log_det = log_det + ld
        return z, log_det   # u, log_det

    def inverse(self, u: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """u → z (manifold coords)"""
        for layer in reversed(self.layers):
            u = layer.inverse(u, cond)
        return u

    def log_prob(self, z: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Exact log-probability under standard Gaussian + change-of-variables. (B,)"""
        u, log_det = self.forward(z, cond)
        log_pz = -0.5 * (u ** 2 + math.log(2 * math.pi)).sum(dim=-1)
        return log_pz + log_det


# Main M-Flow model
def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 3) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d, hidden_dim), nn.GELU()]
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class RunwayMFlow(nn.Module):
    """
    Injective normalizing flow for the runway approach image manifold.

    Args:
        ambient_dim:   I-JEPA embedding dimension (default 1280)
        manifold_dim:  Intrinsic manifold dimension (default 64)
        n_coupling_layers: RealNVP depth (default 8)
        hidden_dim:    Hidden dimension for encoder/decoder MLPs (default 256)
        pose_dim:      6-DOF pose input dimension
        pose_hidden_dim: Pose encoder hidden dimension
    """

    def __init__(
        self,
        ambient_dim:        int = 1280,
        manifold_dim:       int = 64,
        n_coupling_layers:  int = 8,
        hidden_dim:         int = 256,
        pose_dim:           int = 6,
        pose_hidden_dim:    int = 64,
        cond_flow:          bool = False,
    ) -> None:
        super().__init__()
        self.ambient_dim   = ambient_dim
        self.manifold_dim  = manifold_dim
        self.noise_dim     = ambient_dim - manifold_dim
        self.cond_flow     = cond_flow

        # Encoder: ambient → manifold coordinates
        self.encoder = _mlp(ambient_dim, hidden_dim, manifold_dim, n_layers=4)

        # Noise head: ambient → off-manifold noise
        self.noise_head = _mlp(ambient_dim, hidden_dim, self.noise_dim, n_layers=3)

        # Pose encoder: 6-DOF → manifold_dim (used for additive shift + optional flow conditioning)
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, pose_hidden_dim),
            nn.GELU(),
            nn.Linear(pose_hidden_dim, pose_hidden_dim),
            nn.GELU(),
            nn.Linear(pose_hidden_dim, manifold_dim),
        )

        # Bijective flow on the manifold (exact likelihood)
        # When cond_flow=True the coupling MLPs are conditioned on the pose embedding
        flow_cond_dim = manifold_dim if cond_flow else 0
        self.manifold_flow = RealNVP(
            dim=manifold_dim,
            n_layers=n_coupling_layers,
            hidden_dim=max(64, manifold_dim * 2),
            cond_dim=flow_cond_dim,
        )

        # Decoder: manifold coordinates → ambient
        self.decoder = _mlp(manifold_dim, hidden_dim, ambient_dim, n_layers=4)

        # Pose regressor: z → pose  (used for lambda_pose supervision loss)
        self.pose_regressor = _mlp(manifold_dim, pose_hidden_dim, pose_dim, n_layers=3)

    # Forward / encode / decode
    def encode(
        self,
        x: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (B, ambient_dim) — I-JEPA embeddings
            pose: (B, 6)           — optional 6-DOF pose

        Returns:
            z:        (B, manifold_dim)  — on-manifold coordinates
            e:        (B, noise_dim)     — off-manifold noise
            log_prob: (B,)               — exact log-probability on manifold
        """
        z = self.encoder(x)           # (B, manifold_dim)
        e = self.noise_head(x)        # (B, noise_dim)

        pose_emb = self.pose_encoder(pose) if pose is not None else None
        if pose_emb is not None:
            z = z + pose_emb          # additive shift keeps existing behaviour

        cond = pose_emb if self.cond_flow else None

        # Exact log-probability via RealNVP
        u, log_det = self.manifold_flow(z, cond)
        log_p_base = -0.5 * (u ** 2 + math.log(2 * math.pi)).sum(dim=-1)
        log_prob = log_p_base + log_det    # (B,)

        return z, e, log_prob

    def decode(
        self,
        z: torch.Tensor,
        e: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z: (B, manifold_dim)
            e: (B, noise_dim) — if None, zeros are used (inference mode)

        Returns:
            x_hat: (B, ambient_dim)
        """
        return self.decoder(z)   # noise e is not used in decoder (off-manifold)

    def forward(
        self,
        x: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.

        Returns:
            z:        (B, manifold_dim)
            e:        (B, noise_dim)
            log_prob: (B,)
            x_hat:    (B, ambient_dim)  — reconstructed embedding
        """
        z, e, log_prob = self.encode(x, pose)
        x_hat = self.decode(z)
        return z, e, log_prob, x_hat

    def sample(
        self,
        n: int,
        device: torch.device,
        pose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample n manifold vectors from the learned manifold distribution.

        Returns:
            z_samples: (n, manifold_dim)
        """
        # Sample from base distribution, then invert the flow
        pose_emb = self.pose_encoder(pose) if pose is not None else None
        cond = pose_emb if self.cond_flow else None
        u = torch.randn(n, self.manifold_dim, device=device)
        z = self.manifold_flow.inverse(u, cond)
        if pose_emb is not None:
            z = z + pose_emb
        return z

    # Loss computation
    def compute_loss(
        self,
        x: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
        lambda_likelihood: float = 0.1,
        lambda_geometry: float = 0.01,
        lambda_pose: float = 0.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined training loss.

        Returns:
            total_loss: scalar
            loss_dict:  {'recon', 'nll', 'geometry', 'pose', 'total'}
        """
        z, e, log_prob, x_hat = self.forward(x, pose)

        l_recon    = F.mse_loss(x_hat, x)
        l_nll      = -log_prob.mean()
        l_geometry = (e ** 2).mean()

        l_pose = torch.zeros(1, device=x.device)[0]
        if lambda_pose > 0.0 and pose is not None:
            pose_pred = self.pose_regressor(z)
            l_pose = F.mse_loss(pose_pred, pose)

        total = l_recon + lambda_likelihood * l_nll + lambda_geometry * l_geometry + lambda_pose * l_pose

        return total, {
            "recon":    l_recon.item(),
            "nll":      l_nll.item(),
            "geometry": l_geometry.item(),
            "pose":     l_pose.item(),
            "total":    total.item(),
        }


# Smoke test
if __name__ == "__main__":

    print("=== RunwayMFlow smoke test ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RunwayMFlow(
        ambient_dim=1280, manifold_dim=64, n_coupling_layers=4
    ).to(device)

    B = 4
    x    = torch.randn(B, 1280, device=device)
    pose = torch.randn(B, 6, device=device)

    z, e, log_prob, x_hat = model(x, pose)
    print(f"  z shape:       {z.shape}")        # (4, 64)
    print(f"  e shape:       {e.shape}")        # (4, 1216)
    print(f"  log_prob:      {log_prob.shape}") # (4,)
    print(f"  x_hat shape:   {x_hat.shape}")    # (4, 1280)

    assert z.shape    == (B, 64)
    assert e.shape    == (B, 1216)
    assert x_hat.shape == (B, 1280)

    loss, ld = model.compute_loss(x, pose)
    print(f"  loss_dict:     {ld}")
    loss.backward()
    print("  [PASS] Backward pass succeeded")

    # Sampling
    samples = model.sample(3, device)
    assert samples.shape == (3, 64)
    print(f"  [PASS] Sample shape: {samples.shape}")

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {n_params:,}")

    print("\nAll smoke tests passed.")
    sys.exit(0)
