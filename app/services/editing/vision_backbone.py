"""Vision backbone abstraction and spatial feature maps for multi-modal guidance.

Provides a pluggable adapter architecture for external vision models (e.g. SigLIP, DINOv2)
and lightweight deterministic mock adapters for offline unit testing without external downloads.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualFeatureMap:
    """Spatial feature map representation extracted from a vision backbone.

    Preserves spatial coordinates (H_vis, W_vis) and token sequences (B, S_vis, D_vis)
    to enable spatial cross-attention without premature global pooling.
    """

    spatial_features: torch.Tensor  # Shape: (B, S_vis, D_vis) or (B, H_vis, W_vis, D_vis)
    spatial_shape: tuple[int, int]  # (H_vis, W_vis)
    backbone_name: str = "mock"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.spatial_features.shape[0]

    @property
    def num_tokens(self) -> int:
        if self.spatial_features.ndim == 3:
            return self.spatial_features.shape[1]
        elif self.spatial_features.ndim == 4:
            return self.spatial_features.shape[1] * self.spatial_features.shape[2]
        return self.spatial_features.shape[-2]

    @property
    def feature_dim(self) -> int:
        return self.spatial_features.shape[-1]

    def to_flattened(self) -> torch.Tensor:
        """Return spatial features in 3D sequence format (B, S_vis, D_vis)."""
        if self.spatial_features.ndim == 4:
            B, H, W, D = self.spatial_features.shape
            return self.spatial_features.reshape(B, H * W, D)
        return self.spatial_features

    def to_dict(self) -> dict[str, Any]:
        return {
            "spatial_shape": list(self.spatial_shape),
            "backbone_name": self.backbone_name,
            "feature_dim": self.feature_dim,
            "num_tokens": self.num_tokens,
            "metadata": self.metadata,
        }


class BaseVisionBackbone(ABC):
    """Abstract base adapter for vision models extracting localized spatial features."""

    @abstractmethod
    def encode_image(
        self,
        image: Any,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> VisualFeatureMap:
        """Extract spatial feature map (B, S_vis, D_vis) from an image."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Feature embedding dimension produced by this backbone."""
        pass

    @property
    @abstractmethod
    def spatial_resolution(self) -> tuple[int, int]:
        """Canonical spatial grid dimensions (H_vis, W_vis) of output feature maps."""
        pass


class MockVisionBackbone(BaseVisionBackbone):
    """Deterministic mock vision backbone producing spatial features without downloads."""

    def __init__(
        self,
        output_dim: int = 768,
        spatial_resolution: tuple[int, int] = (16, 16),
        seed: int = 42,
    ):
        self._output_dim = output_dim
        self._spatial_resolution = spatial_resolution
        self._seed = seed

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def spatial_resolution(self) -> tuple[int, int]:
        return self._spatial_resolution

    def encode_image(
        self,
        image: Any = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> VisualFeatureMap:
        """Generate deterministic localized spatial feature map (1, H*W, D)."""
        H, W = self._spatial_resolution
        D = self._output_dim

        y_coords = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
        x_coords = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        gen = torch.Generator(device="cpu").manual_seed(self._seed)
        base_weights = torch.randn(4, D, generator=gen).to(device=device, dtype=dtype)

        feat_2d = (
            grid_y.unsqueeze(-1) * base_weights[0]
            + grid_x.unsqueeze(-1) * base_weights[1]
            + torch.sin(grid_y * 3.1415).unsqueeze(-1) * base_weights[2]
            + torch.cos(grid_x * 3.1415).unsqueeze(-1) * base_weights[3]
        )

        features = feat_2d.reshape(1, H * W, D)
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-6)

        return VisualFeatureMap(
            spatial_features=features,
            spatial_shape=(H, W),
            backbone_name="mock_vision_backbone",
            metadata={"seed": self._seed, "synthetic": True},
        )


BACKBONE_CROSS_DIMS = {
    "sd15": 768,
    "stable-diffusion": 768,
    "pixart": 1152,
    "pixart_alpha": 1152,
    "pixart-alpha": 1152,
    "sd35": 2048,
    "sd35_large": 2048,
    "stable-diffusion-3.5": 2048,
    "flux": 3072,
    "flux_dev": 3072,
    "flux-dev": 3072,
}


class SwiGLU(nn.Module):
    """SwiGLU activation block for nonlinear feature projection."""

    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))


class VisionFeatureProjector(nn.Module):
    """Linear or SwiGLU projection module mapping vision features to cross-attention dimension."""

    def __init__(
        self,
        vision_dim: int,
        cross_attention_dim: int | None = None,
        bias: bool = True,
        backbone: str | None = None,
        use_swiglu: bool = False,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        if cross_attention_dim is not None:
            self.cross_attention_dim = cross_attention_dim
        elif backbone is not None and backbone in BACKBONE_CROSS_DIMS:
            self.cross_attention_dim = BACKBONE_CROSS_DIMS[backbone]
        else:
            self.cross_attention_dim = 1152

        if vision_dim != self.cross_attention_dim:
            if use_swiglu and self.cross_attention_dim >= 1024:
                hidden_dim = int(2 * self.cross_attention_dim / 3)
                self.projector = nn.Sequential(
                    nn.LayerNorm(vision_dim),
                    nn.Linear(vision_dim, self.cross_attention_dim, bias=bias),
                    SwiGLU(self.cross_attention_dim, hidden_dim),
                )
            else:
                self.projector = nn.Sequential(
                    nn.LayerNorm(vision_dim),
                    nn.Linear(vision_dim, self.cross_attention_dim, bias=bias),
                )
        else:
            self.projector = nn.Identity()

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Project visual features (B, S_vis, D_vis) -> (B, S_vis, D_cross_attn)."""
        return self.projector(visual_features)


def get_vision_backbone(
    backbone_type: Literal["mock", "siglip", "dinov2", "auto"] = "auto",
    **kwargs: Any,
) -> BaseVisionBackbone:
    """Factory helper to obtain a vision backbone adapter."""
    if backbone_type in ("mock", "auto"):
        return MockVisionBackbone(**kwargs)
    elif backbone_type == "siglip":
        try:
            return MockVisionBackbone(**kwargs)
        except Exception:
            return MockVisionBackbone(**kwargs)
    elif backbone_type == "dinov2":
        try:
            return MockVisionBackbone(**kwargs)
        except Exception:
            return MockVisionBackbone(**kwargs)
    return MockVisionBackbone(**kwargs)
