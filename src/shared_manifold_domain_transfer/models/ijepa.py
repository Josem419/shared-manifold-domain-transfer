"""
I-JEPA Context Encoder wrapper.

Loads the pretrained ViT-H/14 context encoder from a Facebook Research
checkpoint (IN22K-vit.h.14-900e.pth.tar) and exposes a clean inference API.

The predictor and target encoder are stripped after loading — only the context
encoder is kept. All parameters are frozen.

Forward:
    images (B, 3, 224, 224) → embeddings (B, 1280)

Optional:
    return_patches=True  → (B, N_patches, 1280)
    get_runway_embedding(images, corners) → (B, 1280) runway-patch mean
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ViT-H/14 constants
VIT_H14_EMBED_DIM  = 1280
VIT_H14_PATCH_SIZE = 14
VIT_H14_DEPTH      = 32
VIT_H14_NUM_HEADS  = 16

PRETRAINED_URL = "https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar"


# Lightweight ViT-H/14 implementation (matches I-JEPA checkpoint layout)
class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        # multi headed self-attention with optional bias for qkv projections (timm ViT default is True)
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, act_layer=nn.GELU) -> None:
        super().__init__()
        # GELU activation, 2 layer MLP with hidden dimension = 4x input dim (timm default for ViTs)
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        # layer norm, attention, layer norm, and MLP
        # this is our building block for the ViT, matching the I-JEPA context encoder architecture
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Minimal ViT implementation matching I-JEPA context encoder layout."""

    def __init__(
        self,
        img_size:    int = 224,
        patch_size:  int = 14,
        embed_dim:   int = 1280,
        depth:       int = 32,
        num_heads:   int = 16,
        mlp_ratio:   float = 4.0,
        in_chans:    int = 3,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim  = embed_dim
        n_patches = (img_size // patch_size) ** 2
        self.n_patches  = n_patches

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_patches: bool = False,
    ) -> torch.Tensor:
        B = x.shape[0]
        # Patchify
        x = self.patch_embed(x)                             # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)                   # (B, N, D)
        x = x + self.pos_embed
        # Transformer
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if return_patches:
            return x                 # (B, N, D)

        return x.mean(1)             # (B, D) — mean pool over patches


# Public wrapper
class IJEPAEncoder(nn.Module):
    """
    Frozen I-JEPA context encoder.

    Args:
        weights_path: local path to IN22K-vit.h.14-900e.pth.tar.
                      If None, attempts to download from Facebook CDN.
        device: 'cuda' | 'cpu'
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.encoder = VisionTransformer(
            img_size=224,
            patch_size=VIT_H14_PATCH_SIZE,
            embed_dim=VIT_H14_EMBED_DIM,
            depth=VIT_H14_DEPTH,
            num_heads=VIT_H14_NUM_HEADS,
        )
        if weights_path is not None:
            self._load_weights(weights_path)
        else:
            log.warning(
                "No weights_path provided. Encoder weights are random. "
                f"Download from: {PRETRAINED_URL}"
            )
        self._freeze()
        self.to(device)
        self._device = device

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        return_patches: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 224, 224) normalised to ImageNet mean/std
            return_patches: if True return (B, N_patches, 1280)
        Returns:
            embeddings: (B, 1280) or (B, N_patches, 1280)
        """
        return self.encoder(images, return_patches=return_patches)

    @torch.no_grad()
    def get_runway_embedding(
        self,
        images: torch.Tensor,
        corners_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return mean of patch embeddings that overlap with the runway AABB.

        Args:
            images:       (B, 3, 224, 224)
            corners_norm: (B, 4, 2) corner coordinates normalised to [0,1]
        Returns:
            embeddings: (B, 1280)
        """
        B = images.shape[0]
        patches = self.encoder(images, return_patches=True)   # (B, N, 1280)

        img_size   = 224
        patch_size = VIT_H14_PATCH_SIZE
        n_side     = img_size // patch_size                    # 16

        embeddings = []
        for b in range(B):
            # AABB of runway in pixel space
            cx = corners_norm[b, :, 0] * img_size  # (4,)
            cy = corners_norm[b, :, 1] * img_size  # (4,)
            x_min, x_max = cx.min().item(), cx.max().item()
            y_min, y_max = cy.min().item(), cy.max().item()

            # Which patch columns/rows overlap the AABB?
            col_min = max(0,      int(x_min // patch_size))
            col_max = min(n_side, int(math.ceil(x_max / patch_size)))
            row_min = max(0,      int(y_min // patch_size))
            row_max = min(n_side, int(math.ceil(y_max / patch_size)))

            # Patch indices: patches are stored row-major
            patch_indices = []
            for row in range(row_min, row_max):
                for col in range(col_min, col_max):
                    patch_indices.append(row * n_side + col)

            if not patch_indices:
                # Fallback to global mean
                emb = patches[b].mean(0)
            else:
                idx = torch.tensor(patch_indices, device=patches.device)
                emb = patches[b][idx].mean(0)

            embeddings.append(emb)

        return torch.stack(embeddings, dim=0)  # (B, 1280)

    def _load_weights(self, weights_path: str) -> None:
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Weights file not found: {path}\n"
                f"Download from: {PRETRAINED_URL}"
            )
        log.info(f"Loading I-JEPA weights from {path}")
        ckpt = torch.load(path, map_location="cpu")

        # The checkpoint may be stored under different keys
        if "encoder" in ckpt:
            state_dict = ckpt["encoder"]
        elif "target_encoder" in ckpt:
            state_dict = ckpt["target_encoder"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # Strip 'module.' prefix (DataParallel)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # timm-style patch_embed uses proj sub-layer; remap to our flat Conv2d
        state_dict = {
            k.replace("patch_embed.proj.", "patch_embed."): v
            for k, v in state_dict.items()
        }

        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            log.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        log.info("I-JEPA weights loaded successfully.")

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        assert not any(p.requires_grad for p in self.parameters()), \
            "All I-JEPA parameters must be frozen."
        log.info("I-JEPA encoder frozen (no gradients).")


# Sanity check (python -m shared_manifold_domain_transfer.models.ijepa)
if __name__ == "__main__":
    
    print("=== IJEPAEncoder sanity check ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    enc = IJEPAEncoder(weights_path=None, device=device)  # random weights for shape check

    # Shape check
    dummy = torch.randn(2, 3, 224, 224, device=device)
    emb = enc(dummy)
    assert emb.shape == (2, VIT_H14_EMBED_DIM), f"Expected (2, 1280), got {emb.shape}"
    print(f"  [PASS] Forward shape: {emb.shape}")

    # Patch shape check
    patches = enc(dummy, return_patches=True)
    n_patches = (224 // VIT_H14_PATCH_SIZE) ** 2
    assert patches.shape == (2, n_patches, VIT_H14_EMBED_DIM), \
        f"Expected (2, {n_patches}, 1280), got {patches.shape}"
    print(f"  [PASS] Patch shape: {patches.shape}")

    # Runway embedding
    corners = torch.tensor([[
        [0.3, 0.4], [0.7, 0.4], [0.7, 0.7], [0.3, 0.7]
    ]] * 2, dtype=torch.float32, device=device)
    runway_emb = enc.get_runway_embedding(dummy, corners)
    assert runway_emb.shape == (2, VIT_H14_EMBED_DIM)
    print(f"  [PASS] Runway embedding shape: {runway_emb.shape}")

    # Gradient check
    assert not any(p.requires_grad for p in enc.parameters())
    print("  [PASS] No gradients in encoder")

    # Embedding norm
    norms = emb.norm(dim=-1)
    print(f"  Embedding norms: {norms.tolist()}")

    print("\nAll checks passed.")
    sys.exit(0)
