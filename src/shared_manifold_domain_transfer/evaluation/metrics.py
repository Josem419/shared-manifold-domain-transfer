"""
Evaluation metrics.

All metrics are computed on runway crop only (not full image).

pixel_metrics_mse_and_ssim(gen, gt)
    Pixel-level reconstruction quality: MSE and structural similarity (SSIM).

semantic_fidelity_ijepa_cosine(encoder, gen_crop, gt_crop)
    I-JEPA embedding cosine similarity between generated and ground-truth runway crops.
    Measures whether the generated image is semantically correct, not just pixel-close.

pose_distance_to_nearest_training_sample(target_poses, train_poses)
    Minimum Euclidean distance from each holdout pose to the training set.
    Used to study how fidelity degrades as poses move away from the training distribution.

frechet_distance_between_embedding_sets(emb_a, emb_b)
    Fréchet Distance (FD) between two sets of embeddings.
    Fits a Gaussian to each set (mean + covariance) and computes the Bures metric.
    More reliable than silhouette at high class imbalance; our primary domain-gap metric.

maximum_mean_discrepancy_rbf_kernel(emb_a, emb_b)
    Unbiased MMD estimate using an RBF (Gaussian) kernel.
    sigma defaults to the median heuristic: median pairwise distance of a subsample,
    which avoids kernel saturation in high-dimensional embedding spaces.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity


def pixel_metrics_mse_and_ssim(
    generated: torch.Tensor,
    ground_truth: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute Mean-Squared Error (MSE) and Structural Similarity Index (SSIM) 
    on runway crop tensors.

    Args:
        generated:    (B, 3, H, W) or (3, H, W), values in [-1, 1] or [0, 1]
        ground_truth: same shape

    Returns:
        {'mse': float, 'ssim': float}
    """
    if generated.dim() == 3:
        generated    = generated.unsqueeze(0)
        ground_truth = ground_truth.unsqueeze(0)

    # Normalise to [0, 1] if needed
    def _to_unit(t: torch.Tensor) -> torch.Tensor:
        if t.min() < -0.1:
            return (t + 1.0) / 2.0
        return t

    g = _to_unit(generated).float()
    gt = _to_unit(ground_truth).float()

    mse = F.mse_loss(g, gt).item()

    # SSIM: computed per image, averaged over batch
    g_np  = g.cpu().permute(0, 2, 3, 1).numpy()    # (B, H, W, 3)
    gt_np = gt.cpu().permute(0, 2, 3, 1).numpy()

    ssim_vals = []
    for i in range(g_np.shape[0]):
        s = structural_similarity(
            g_np[i], gt_np[i],
            data_range=1.0,
            channel_axis=-1,
        )
        ssim_vals.append(s)

    return {"mse": mse, "ssim": float(np.mean(ssim_vals))}


def semantic_fidelity_ijepa_cosine(
    encoder,
    generated_crop: torch.Tensor,
    gt_crop:        torch.Tensor,
) -> float:
    """
    I-JEPA cosine similarity between generated and ground-truth runway crops.

    Args:
        encoder:        IJEPAEncoder (frozen, eval mode)
        generated_crop: (B, 3, H, W) — runway region of generated image
        gt_crop:        (B, 3, H, W) — runway region of ground truth image

    Returns:
        mean cosine similarity (float)
    """
    device = next(encoder.parameters()).device

    with torch.no_grad():
        gen_emb = encoder(generated_crop.to(device))  # (B, 1280)
        gt_emb  = encoder(gt_crop.to(device))         # (B, 1280)

    cos_sim = F.cosine_similarity(gen_emb, gt_emb, dim=-1)  # (B,)
    return cos_sim.mean().item()


def pose_distance_to_nearest_training_sample(
    target_poses: torch.Tensor,
    train_poses:  torch.Tensor,
) -> torch.Tensor:
    """
    Minimum euclidean distance from each target pose to the training set.

    Args:
        target_poses: (M, 6)
        train_poses:  (N, 6)

    Returns:
        distances: (M,) minimum distance to any training pose
    """
    if target_poses.dim() == 1:
        target_poses = target_poses.unsqueeze(0)

    diff = target_poses[:, None, :] - train_poses[None, :, :]  # (M, N, 6)
    dists = diff.norm(dim=-1)                                   # (M, N)
    return dists.min(dim=1).values                              # (M,)


def frechet_distance_between_embedding_sets(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
) -> float:
    """
    Approximate Fréchet distance between two embedding sets.
    Uses mean and covariance (Fréchet Inception Distance style).

    Args:
        emb_a, emb_b: (N, D) and (M, D) float arrays

    Returns:
        fd: float
    """

    mu_a = emb_a.mean(0)
    mu_b = emb_b.mean(0)
    sigma_a = np.cov(emb_a, rowvar=False)
    sigma_b = np.cov(emb_b, rowvar=False)

    diff = mu_a - mu_b
    cov_mean, _ = sqrtm(sigma_a @ sigma_b, disp=False)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    fd = float(diff @ diff + np.trace(sigma_a + sigma_b - 2.0 * cov_mean))
    return fd


def maximum_mean_discrepancy_rbf_kernel(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    sigma: float | None = None,
) -> float:
    """
    Unbiased Maximum Mean Discrepancy (MMD) estimate using a 
    Radial Basis Function (Gaussian) kernel.

    The RBF kernel measures similarity between two vectors by how close they
    are: k(x,y) = exp(-‖x−y‖² / 2σ²). MMD is zero when both sets are drawn
    from the same distribution and positive otherwise.

    sigma defaults to the median heuristic (median pairwise distance over a
    random subsample of up to 500 points from each set). This avoids kernel
    saturation in high-dimensional spaces where a fixed σ=1.0 causes all
    pairwise kernel values to collapse near zero.

    Args:
        emb_a, emb_b: (N, D) and (M, D) float arrays
        sigma: RBF bandwidth. If None, computed via median heuristic.

    Returns:
        mmd: float (≥ 0)
    """
    a = torch.tensor(emb_a, dtype=torch.float32)
    b = torch.tensor(emb_b, dtype=torch.float32)

    if sigma is None:
        # Median heuristic: subsample up to 500 points from each set,
        # compute pairwise distances, use median as bandwidth.
        sub_a = a[:500]
        sub_b = b[:500]
        sample = torch.cat([sub_a, sub_b], dim=0)
        dists = torch.cdist(sample, sample)  # (N+M, N+M)
        sigma = float(dists.median().item()) + 1e-6

    def rbf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x[:, None, :] - y[None, :, :]     # (N, M, D)
        sq   = (diff ** 2).sum(-1)                # (N, M)
        return torch.exp(-sq / (2.0 * sigma ** 2))

    kaa = rbf(a, a)  # (N, N) within-set kernel for domain A — measures self-similarity
    kbb = rbf(b, b)  # (M, M) within-set kernel for domain B — measures self-similarity
    kab = rbf(a, b)  # (N, M) cross-set kernel — measures similarity between domains

    n, m = a.shape[0], b.shape[0]
    # Unbiased estimator: subtract diagonal (self-similarity, k(x,x)=1) before averaging,
    # so same-set terms are not inflated by the trivially perfect self-kernel values.
    kaa_off = (kaa.sum() - kaa.trace()) / (n * (n - 1))  # mean off-diagonal of kaa
    kbb_off = (kbb.sum() - kbb.trace()) / (m * (m - 1))  # mean off-diagonal of kbb
    kab_mean = kab.mean()  # mean cross-kernel (no diagonal to exclude; all pairs are cross-domain)

    return float(kaa_off + kbb_off - 2.0 * kab_mean)
