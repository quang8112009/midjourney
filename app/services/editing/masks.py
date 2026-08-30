"""Mask utilities shared by the region-aware editing path.

Masks are soft: float tensors in [0, 1] shaped (1, 1, H, W). 1 means "this is the
region the prompt is allowed to change", 0 means "preserve the source here".
Soft edges matter - a hard binary mask produces visible seams when the latent is
blended every denoise step.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def as_soft_mask(mask: torch.Tensor) -> torch.Tensor:
    """Normalize any (H,W) / (1,H,W) / (1,1,H,W) mask to a (1,1,H,W) float in [0,1]."""
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[None]
    elif mask.ndim != 4:
        raise ValueError(f"mask must be 2D, 3D or 4D, got {mask.ndim}D")
    mask = mask.to(dtype=torch.float32)
    if mask.shape[1] != 1:
        mask = mask.mean(dim=1, keepdim=True)
    return mask.clamp(0.0, 1.0)


def resize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resample a mask onto the latent grid. Bilinear keeps the soft edge soft."""
    mask = as_soft_mask(mask)
    if mask.shape[-2:] == (height, width):
        return mask
    return F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False).clamp(
        0.0, 1.0
    )


def _gaussian_kernel1d(radius: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def feather(mask: torch.Tensor, radius: int = 2, sigma: float | None = None) -> torch.Tensor:
    """Blur the mask edge so blended latents do not show a seam at the boundary."""
    mask = as_soft_mask(mask)
    if radius <= 0:
        return mask
    sigma = sigma if sigma is not None else max(radius / 2.0, 1e-3)
    kernel = _gaussian_kernel1d(radius, sigma, mask.device, mask.dtype)
    # Separable blur: rows then columns, replicate padding so edges do not darken.
    padded = F.pad(mask, (radius, radius, 0, 0), mode="replicate")
    mask = F.conv2d(padded, kernel.view(1, 1, 1, -1))
    padded = F.pad(mask, (0, 0, radius, radius), mode="replicate")
    mask = F.conv2d(padded, kernel.view(1, 1, -1, 1))
    return mask.clamp(0.0, 1.0)


def dilate(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Grow the mask by `radius`, so an edit has room for its own shadow/edge."""
    mask = as_soft_mask(mask)
    if radius <= 0:
        return mask
    size = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=size, stride=1, padding=radius)


def area_ratio(mask: torch.Tensor) -> float:
    """Fraction of the frame the mask covers, in [0,1]. Soft mass, not a pixel count."""
    mask = as_soft_mask(mask)
    return float(mask.sum().item() / mask.numel())


def iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Intersection-over-union of two masks after thresholding."""
    pred_b = as_soft_mask(pred) >= threshold
    target_b = resize_mask(as_soft_mask(target), *pred_b.shape[-2:]) >= threshold
    union = (pred_b | target_b).sum().item()
    if union == 0:
        return 1.0 if (pred_b.sum().item() == 0 and target_b.sum().item() == 0) else 0.0
    return float((pred_b & target_b).sum().item() / union)


def bounding_box(mask: torch.Tensor, threshold: float = 0.5) -> tuple[int, int, int, int] | None:
    """Tight (top, left, bottom, right) box around the mask, or None if it is empty."""
    binary = as_soft_mask(mask)[0, 0] >= threshold
    rows = torch.any(binary, dim=1).nonzero().flatten()
    cols = torch.any(binary, dim=0).nonzero().flatten()
    if rows.numel() == 0 or cols.numel() == 0:
        return None
    return (int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1)
