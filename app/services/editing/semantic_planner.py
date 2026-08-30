"""Semantic and layout reasoning planner for hybrid DiT generation and editing.

Combines context/reasoning capabilities (object count, relations, spatial layout)
with high artistic freedom for style, lighting, and composition. Edit intent is
owned exclusively by :mod:`prompt_intent`.

Reasoning constrains logic and semantics, NOT aesthetics.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from app.services.editing.prompt_intent import PromptIntent

logger = logging.getLogger(__name__)

_NUM_WORDS: dict[str, int] = {
    "one": 1, "a": 1, "an": 1, "single": 1, "lone": 1,
    "two": 2, "pair": 2, "couple": 2, "both": 2,
    "three": 3, "trio": 3,
    "four": 4, "quad": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "dozen": 12, "dozens": 24,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "hundreds": 100, "thousand": 1000, "thousands": 1000,
    "million": 1000000, "millions": 1000000, "many": 20,
    "countless": 100, "scores": 40, "score": 20, "myriad": 1000,
    "multitude": 100, "numerous": 50,
}

_COLLECTIVE_CROWD_PATTERNS: dict[
    str, tuple[int, Literal["gaussian", "uniform", "radial", "elongated"], float]
] = {
    "swarm": (50, "radial", 1.5),
    "swarms": (50, "radial", 1.5),
    "flock": (25, "elongated", 2.0),
    "flocks": (25, "elongated", 2.0),
    "crowd": (40, "uniform", 2.5),
    "crowds": (40, "uniform", 2.5),
    "herd": (20, "elongated", 2.0),
    "herds": (20, "elongated", 2.0),
    "school": (30, "elongated", 2.0),
    "schools": (30, "elongated", 2.0),
    "shoal": (30, "elongated", 2.0),
    "shoals": (30, "elongated", 2.0),
    "pack": (12, "gaussian", 2.0),
    "packs": (12, "gaussian", 2.0),
    "colony": (40, "radial", 1.8),
    "colonies": (40, "radial", 1.8),
    "cluster": (15, "radial", 2.0),
    "clusters": (15, "radial", 2.0),
    "cloud": (50, "radial", 1.5),
    "clouds": (50, "radial", 1.5),
    "sea": (100, "uniform", 3.0),
    "ocean": (100, "uniform", 3.0),
    "field": (100, "uniform", 3.0),
    "forest": (50, "uniform", 2.5),
    "galaxy": (100, "radial", 1.5),
    "constellation": (50, "radial", 1.5),
    "shower": (30, "elongated", 2.0),
    "storm": (40, "radial", 1.8),
    "horde": (50, "uniform", 2.5),
    "army": (50, "uniform", 2.5),
    "throng": (40, "uniform", 2.5),
    "mob": (40, "uniform", 2.5),
    "carpet": (80, "uniform", 3.0),
    "stream": (30, "elongated", 2.0),
    "gathering": (30, "gaussian", 2.0),
}

DEFAULT_DENSITY_ENTITY_THRESHOLD = 10

_COMMON_ADJECTIVES = frozenset(
    """red green blue yellow orange purple black white gray grey brown pink gold silver
    tall short small big large giant tiny cute cheerful happy sad rustic wooden fluffy
    furry silky shiny dark bright vibrant deep warm cool golden snowy sunny stormy
    majestic wild ancient modern sweet sour fresh ripe young old ceramic vintage glass
    metallic leather stone crystal glowing shining gleaming floating blooming burning
    lush dense sparse open closed beautiful pretty peaceful quiet calm
    angry migrating cheering sparkling buzzing distant nearby translucent transparent
    opaque blurry sharp clear""".split()
)

_ACTION_VERBS = frozenset(
    """change recolor recolour remove erase delete add insert replace swap restyle
    make turn paint dye tint lighten darken blur sharpen enhance resize edit modify alter
    standing sitting hiding resting placed positioned lying sleeping running walking
    flying jumping hovering perched put place move shift set""".split()
)

_SPATIAL_WORDS = frozenset(
    """under underneath beneath below above over inside within outside behind front back
    beside next near alongside astride between around against across top bottom
    left right center middle overhead foreground background""".split()
)

_METADATA_WORDS = frozenset(
    """photo photograph picture image snapshot illustration artwork painting drawing
    render rendering style view shot canvas wallpaper""".split()
)

_RELATION_PHRASES: list[tuple[str, str]] = [
    ("partially hidden behind", "behind"),
    ("partially behind", "behind"),
    ("hidden behind", "behind"),
    ("far in front of", "far_in_front_of"),
    ("far behind", "far_behind"),
    ("in front of", "in_front_of"),
    ("in the front of", "in_front_of"),
    ("in back of", "behind"),
    ("in the back of", "behind"),
    ("side by side with", "next_to"),
    ("on the back of", "riding"),
    ("hovering over", "above"),
    ("surrounded by", "inside"),
    ("contained in", "inside"),
    ("standing on", "on"),
    ("adjacent to", "next_to"),
    ("perched on", "on"),
    ("resting on", "on"),
    ("sitting on", "on"),
    ("flying over", "above"),
    ("underneath", "under"),
    ("alongside", "next_to"),
    ("on top of", "on"),
    ("nested in", "inside"),
    ("ahead of", "in_front_of"),
    ("carrying", "holding"),
    ("grasping", "holding"),
    ("beneath", "under"),
    ("holding", "holding"),
    ("astride", "riding"),
    ("next to", "next_to"),
    ("behind", "behind"),
    ("inside", "inside"),
    ("within", "inside"),
    ("beside", "next_to"),
    ("riding", "riding"),
    ("under", "under"),
    ("below", "under"),
    ("above", "above"),
    ("near", "next_to"),
    ("over", "above"),
    ("on", "on"),
    ("in", "inside"),
]

_STYLE_KEYWORDS: dict[str, list[str]] = {
    "medium": [
        "watercolor", "watercolour", "oil painting", "acrylic", "pencil sketch",
        "charcoal", "anime", "manga", "digital art", "concept art", "3d render",
        "claymation", "vector art", "photorealistic", "hyperrealistic", "film photography",
        "pixel art", "pastel", "gouache", "line art", "ink drawing", "graffiti",
    ],
    "lighting": [
        "volumetric lighting", "cinematic lighting", "golden hour", "dramatic lighting",
        "studio lighting", "soft lighting", "rim light", "backlit", "neon glow",
        "chiaroscuro", "moody lighting", "sunlight", "moonlight", "candlelight",
        "ambient light", "bokeh", "lens flare", "god rays",
    ],
    "mood": [
        "dreamy", "ethereal", "mysterious", "whimsical", "serene", "gloomy",
        "dystopian", "cyberpunk", "steampunk", "surreal", "cozy", "epic",
        "nostalgic", "futuristic", "fantasy", "magical", "vibrant", "melancholic",
    ],
    "composition": [
        "close-up", "portrait", "wide angle", "panoramic", "macro", "bird's-eye view",
        "low angle", "isometric", "cinematic composition", "symmetry", "rule of thirds",
        "bokeh background", "shallow depth of field", "ultra detailed", "8k resolution",
    ],
}

# Prepositional complements ("on the left") and relation verbs ("riding") describe
# *where* or *how* objects relate. They are not renderable objects, and planning a
# bounding box for them hands image regions to function words.
_POSITION_WORDS: dict[str, str] = {
    "left": "left", "right": "right", "top": "top", "upper": "top",
    "bottom": "bottom", "lower": "bottom", "middle": "center",
    "center": "center", "centre": "center",
    "foreground": "foreground", "background": "background",
}
_RELATION_WORDS = frozenset(
    """riding astride on top sitting standing perched resting under underneath
    beneath below next beside adjacent side alongside near holding carrying
    grasping front ahead behind back inside within contained nested surrounded
    above hovering flying over""".split()
)

# Where each positional qualifier puts an object, in normalized coordinates.
_POSITION_BOXES: dict[str, tuple[float, float, float, float]] = {
    "left": (0.15, 0.02, 0.90, 0.46),
    "right": (0.15, 0.54, 0.90, 0.98),
    "top": (0.02, 0.15, 0.46, 0.90),
    "bottom": (0.54, 0.15, 0.98, 0.90),
    "center": (0.28, 0.28, 0.72, 0.72),
    "foreground": (0.45, 0.10, 0.95, 0.90),
    "background": (0.05, 0.10, 0.55, 0.90),
}

# Two objects the prompt names separately should not land on the same pixels.
_MAX_ALLOWED_BOX_IOU = 0.5

_STOPWORDS = frozenset(
    """the a an of to in on at by for with from into its his her their my our your
    is are be been being do does did please can you it this that and or then but
    another other some each every both same""".split()
)


@dataclass(frozen=True)
class NormalizedBox:
    """Bounding box in normalized coordinates [ymin, xmin, ymax, xmax] in [0.0, 1.0]."""

    ymin: float
    xmin: float
    ymax: float
    xmax: float

    def __post_init__(self):
        ymin_c = max(0.0, min(1.0, float(self.ymin)))
        xmin_c = max(0.0, min(1.0, float(self.xmin)))
        ymax_c = max(ymin_c, min(1.0, float(self.ymax)))
        xmax_c = max(xmin_c, min(1.0, float(self.xmax)))
        object.__setattr__(self, "ymin", round(ymin_c, 4))
        object.__setattr__(self, "xmin", round(xmin_c, 4))
        object.__setattr__(self, "ymax", round(ymax_c, 4))
        object.__setattr__(self, "xmax", round(xmax_c, 4))

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.ymin + self.ymax) / 2.0, (self.xmin + self.xmax) / 2.0)

    def contains(self, other: NormalizedBox | tuple[float, float] | list[float]) -> bool:
        """Return True if this box encapsulates `other` box or contains a (y, x) point."""
        if isinstance(other, NormalizedBox):
            return (
                self.ymin <= other.ymin + 1e-4
                and self.xmin <= other.xmin + 1e-4
                and self.ymax >= other.ymax - 1e-4
                and self.xmax >= other.xmax - 1e-4
            )
        if isinstance(other, (tuple, list)) and len(other) == 2:
            y, x = float(other[0]), float(other[1])
            return (
                self.ymin - 1e-4 <= y <= self.ymax + 1e-4
                and self.xmin - 1e-4 <= x <= self.xmax + 1e-4
            )
        return False

    def overlaps(self, other: NormalizedBox) -> bool:
        """Return True if this box overlaps with `other`."""
        return not (
            self.xmax <= other.xmin
            or other.xmax <= self.xmin
            or self.ymax <= other.ymin
            or other.ymax <= self.ymin
        )

    def intersection(self, other: NormalizedBox) -> NormalizedBox | None:
        """Return the intersection box with `other`, or None if disjoint."""
        y0 = max(self.ymin, other.ymin)
        x0 = max(self.xmin, other.xmin)
        y1 = min(self.ymax, other.ymax)
        x1 = min(self.xmax, other.xmax)
        if y1 <= y0 or x1 <= x0:
            return None
        return NormalizedBox(ymin=y0, xmin=x0, ymax=y1, xmax=x1)

    def iou(self, other: NormalizedBox) -> float:
        """Compute intersection over union with `other`."""
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        inter_area = inter.area
        union_area = self.area + other.area - inter_area
        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area

    def to_mask(
        self,
        height: int,
        width: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Render normalized box into a (1, 1, height, width) float mask."""
        mask = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
        if self.area <= 0.0 or height <= 0 or width <= 0:
            return mask
        y0 = max(0, min(height, int(round(self.ymin * height))))
        y1 = max(y0, min(height, int(round(self.ymax * height))))
        x0 = max(0, min(width, int(round(self.xmin * width))))
        x1 = max(x0, min(width, int(round(self.xmax * width))))
        if y1 == y0 and self.height > 0:
            if y1 < height:
                y1 += 1
            elif y0 > 0:
                y0 -= 1
        if x1 == x0 and self.width > 0:
            if x1 < width:
                x1 += 1
            elif x0 > 0:
                x0 -= 1
        if y1 > y0 and x1 > x0:
            mask[0, 0, y0:y1, x0:x1] = 1.0
        return mask

    def to_gaussian(
        self,
        rotation: float = 0.0,
        coverage_sigma: float = 2.0,
        mu_z: float = 0.5,
        sigma_z: float = 0.2,
        depth_confidence: float = 1.0,
    ) -> GaussianSpatialPrior:
        """Convert this bounding box to a smooth 3D anisotropic Gaussian spatial prior."""
        return GaussianSpatialPrior.from_box(
            self,
            rotation=rotation,
            coverage_sigma=coverage_sigma,
            mu_z=mu_z,
            sigma_z=sigma_z,
            depth_confidence=depth_confidence,
        )

    def to_dict(self) -> dict[str, float]:
        return {"ymin": self.ymin, "xmin": self.xmin, "ymax": self.ymax, "xmax": self.xmax}


@dataclass(frozen=True)
class GaussianSpatialPrior:
    """Differentiable 3D anisotropic Gaussian spatial prior in normalized coordinates [0.0, 1.0]^3.

    Represents a continuous volumetric entity heatmap and relative depth prior:
        G(y, x, z) = amplitude * exp( -0.5 * (p - mu)^T Sigma_3D^-1 (p - mu) )
    where p = (y, x, z), mu = (mu_y, mu_x, mu_z), and Sigma_3D is the 3x3 spatial covariance matrix
    derived from (sigma_y, sigma_x, sigma_z) and rotation angle theta (radians).

    Depth Convention (mu_z in [0.0, 1.0]):
        - 0.0: Nearest foreground (closest to camera/viewer)
        - 0.5: Middle depth (neutral default)
        - 1.0: Far background (deepest scene element)
    """

    mu_y: float
    mu_x: float
    sigma_y: float
    sigma_x: float
    theta: float = 0.0  # rotation in radians (counter-clockwise) around optical z-axis
    amplitude: float = 1.0
    mu_z: float = 0.5   # normalized depth centroid (0.0=foreground, 1.0=background)
    sigma_z: float = 0.2  # depth standard deviation / span
    depth_confidence: float = 1.0  # confidence score [0.0, 1.0]

    def __post_init__(self):
        mu_y_c = max(0.0, min(1.0, float(self.mu_y)))
        mu_x_c = max(0.0, min(1.0, float(self.mu_x)))
        sigma_y_c = max(1e-4, min(2.0, float(self.sigma_y)))
        sigma_x_c = max(1e-4, min(2.0, float(self.sigma_x)))
        theta_c = float(self.theta)
        amp_c = max(0.0, min(5.0, float(self.amplitude)))
        mu_z_c = max(0.0, min(1.0, float(self.mu_z)))
        sigma_z_c = max(1e-4, min(2.0, float(self.sigma_z)))
        conf_z_c = max(0.0, min(1.0, float(self.depth_confidence)))

        object.__setattr__(self, "mu_y", round(mu_y_c, 4))
        object.__setattr__(self, "mu_x", round(mu_x_c, 4))
        object.__setattr__(self, "sigma_y", round(sigma_y_c, 4))
        object.__setattr__(self, "sigma_x", round(sigma_x_c, 4))
        object.__setattr__(self, "theta", round(theta_c, 4))
        object.__setattr__(self, "amplitude", round(amp_c, 4))
        object.__setattr__(self, "mu_z", round(mu_z_c, 4))
        object.__setattr__(self, "sigma_z", round(sigma_z_c, 4))
        object.__setattr__(self, "depth_confidence", round(conf_z_c, 4))

    @property
    def center(self) -> tuple[float, float]:
        """2D center position (mu_y, mu_x) in normalized coordinates."""
        return (self.mu_y, self.mu_x)

    @property
    def center_3d(self) -> tuple[float, float, float]:
        """3D center position (mu_y, mu_x, mu_z) in normalized coordinates."""
        return (self.mu_y, self.mu_x, self.mu_z)

    @property
    def center_x(self) -> float:
        return self.mu_x

    @property
    def center_y(self) -> float:
        return self.mu_y

    @property
    def center_z(self) -> float:
        return self.mu_z

    @property
    def scale(self) -> tuple[float, float]:
        """2D anisotropic scale (sigma_y, sigma_x) in normalized coordinates."""
        return (self.sigma_y, self.sigma_x)

    @property
    def scale_3d(self) -> tuple[float, float, float]:
        """3D anisotropic scale (sigma_y, sigma_x, sigma_z) in normalized coordinates."""
        return (self.sigma_y, self.sigma_x, self.sigma_z)

    @property
    def scale_x(self) -> float:
        return self.sigma_x

    @property
    def scale_y(self) -> float:
        return self.sigma_y

    @property
    def scale_z(self) -> float:
        return self.sigma_z

    @property
    def rotation(self) -> float:
        return self.theta

    @property
    def covariance(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Compute the 2D spatial covariance matrix Sigma_2D = R(theta) * Sigma_0 * R(theta)^T."""
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        var_y = self.sigma_y**2
        var_x = self.sigma_x**2
        s00 = cos_t * cos_t * var_y + sin_t * sin_t * var_x
        s01 = cos_t * sin_t * (var_y - var_x)
        s10 = s01
        s11 = sin_t * sin_t * var_y + cos_t * cos_t * var_x
        return ((round(s00, 6), round(s01, 6)), (round(s10, 6), round(s11, 6)))

    @property
    def covariance_3d(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """Compute the 3x3 block-diagonal spatial covariance matrix Sigma_3D."""
        cov2d = self.covariance
        var_z = round(self.sigma_z**2, 6)
        return (
            (cov2d[0][0], cov2d[0][1], 0.0),
            (cov2d[1][0], cov2d[1][1], 0.0),
            (0.0, 0.0, var_z),
        )

    @classmethod
    def from_box(
        cls,
        box: NormalizedBox,
        rotation: float = 0.0,
        coverage_sigma: float = 2.0,
        mu_z: float = 0.5,
        sigma_z: float = 0.2,
        depth_confidence: float = 1.0,
    ) -> GaussianSpatialPrior:
        """Convert a NormalizedBox to a smooth 3D Gaussian prior."""
        mu_y, mu_x = box.center
        sigma_y = max(1e-4, box.height / (2.0 * max(1e-4, coverage_sigma)))
        sigma_x = max(1e-4, box.width / (2.0 * max(1e-4, coverage_sigma)))
        return cls(
            mu_y=mu_y,
            mu_x=mu_x,
            sigma_y=sigma_y,
            sigma_x=sigma_x,
            theta=rotation,
            mu_z=mu_z,
            sigma_z=sigma_z,
            depth_confidence=depth_confidence,
        )

    def to_box(self, confidence_sigma: float = 2.0) -> NormalizedBox:
        """Convert Gaussian prior to equivalent bounding box spanning ~confidence_sigma bounds."""
        s00 = self.covariance[0][0]
        s11 = self.covariance[1][1]
        eff_sy = math.sqrt(max(1e-6, s00)) * confidence_sigma
        eff_sx = math.sqrt(max(1e-6, s11)) * confidence_sigma
        ymin = max(0.0, self.mu_y - eff_sy)
        ymax = min(1.0, self.mu_y + eff_sy)
        xmin = max(0.0, self.mu_x - eff_sx)
        xmax = min(1.0, self.mu_x + eff_sx)
        return NormalizedBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)

    def to_heatmap(
        self,
        height: int,
        width: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Rasterize differentiable 2D anisotropic Gaussian heatmap (1, 1, height, width)."""
        if height <= 0 or width <= 0:
            return torch.zeros(1, 1, max(1, height), max(1, width), device=device, dtype=dtype)

        y_coords = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        x_coords = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        dy = grid_y - float(self.mu_y)
        dx = grid_x - float(self.mu_x)

        cos_t = math.cos(-self.theta)
        sin_t = math.sin(-self.theta)
        dy_rot = cos_t * dy - sin_t * dx
        dx_rot = sin_t * dy + cos_t * dx

        var_y = max(1e-6, float(self.sigma_y) ** 2)
        var_x = max(1e-6, float(self.sigma_x) ** 2)
        quad = (dy_rot**2) / (2.0 * var_y) + (dx_rot**2) / (2.0 * var_x)

        heatmap = float(self.amplitude) * torch.exp(-torch.clamp(quad, 0.0, 50.0))
        return heatmap.unsqueeze(0).unsqueeze(0)

    def to_volume(
        self,
        depth_bins: int = 16,
        height: int = 32,
        width: int = 32,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Rasterize 3D anisotropic Gaussian volume (1, 1, depth_bins, height, width)."""
        if depth_bins <= 0 or height <= 0 or width <= 0:
            return torch.zeros(
                1, 1, max(1, depth_bins), max(1, height), max(1, width), device=device, dtype=dtype
            )

        z_coords = (torch.arange(depth_bins, device=device, dtype=dtype) + 0.5) / depth_bins
        heatmap_2d = self.to_heatmap(
            height, width, device=device, dtype=dtype
        ).squeeze(0).squeeze(0)

        dz = z_coords - float(self.mu_z)
        var_z = max(1e-6, float(self.sigma_z) ** 2)
        depth_weights = torch.exp(-torch.clamp((dz**2) / (2.0 * var_z), 0.0, 50.0))

        volume = depth_weights.view(depth_bins, 1, 1) * heatmap_2d.unsqueeze(0)
        return volume.unsqueeze(0).unsqueeze(0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mu_y": self.mu_y,
            "mu_x": self.mu_x,
            "sigma_y": self.sigma_y,
            "sigma_x": self.sigma_x,
            "theta": self.theta,
            "amplitude": self.amplitude,
            "mu_z": self.mu_z,
            "sigma_z": self.sigma_z,
            "depth_confidence": self.depth_confidence,
            "center": [self.mu_y, self.mu_x],
            "center_3d": [self.mu_y, self.mu_x, self.mu_z],
            "scale": [self.sigma_y, self.sigma_x],
            "scale_3d": [self.sigma_y, self.sigma_x, self.sigma_z],
            "covariance": [list(self.covariance[0]), list(self.covariance[1])],
            "covariance_3d": [list(row) for row in self.covariance_3d],
        }


@dataclass(frozen=True)
class DensityField:
    """Continuous 2D/3D density field representation for crowd dynamics and swarm entities.

    Instead of discrete individual Gaussian instances, crowd and swarm entities
    (e.g., '50 bees', 'flock of birds', 'crowd of people', 'hundreds of stars')
    are modeled as a continuous density distribution field with dynamic spatial
    falloff, macro-clustering, and layout guidance support.
    """

    entity_id: str
    label: str
    expected_count: int
    density: float
    center: tuple[float, float]
    scale: tuple[float, float]
    region: NormalizedBox
    distribution_type: Literal["gaussian", "uniform", "radial", "elongated"]
    falloff: float
    seed: int | None = None
    token_indices: tuple[int, ...] = ()
    mu_z: float = 0.5

    def __post_init__(self):
        mu_y_c = max(0.0, min(1.0, float(self.center[0])))
        mu_x_c = max(0.0, min(1.0, float(self.center[1])))
        sigma_y_c = max(1e-4, min(2.0, float(self.scale[0])))
        sigma_x_c = max(1e-4, min(2.0, float(self.scale[1])))
        density_c = max(0.0, min(10.0, float(self.density)))
        falloff_c = max(0.1, min(10.0, float(self.falloff)))
        mu_z_c = max(0.0, min(1.0, float(self.mu_z)))
        expected_cnt = max(1, int(self.expected_count))

        ent_id = self.entity_id
        if not ent_id:
            h = abs(hash((self.label, expected_cnt, (round(mu_y_c, 4), round(mu_x_c, 4))))) % 10000
            ent_id = f"{self.label}_density_{h:04d}"

        object.__setattr__(self, "entity_id", ent_id)
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "expected_count", expected_cnt)
        object.__setattr__(self, "density", round(density_c, 4))
        object.__setattr__(self, "center", (round(mu_y_c, 4), round(mu_x_c, 4)))
        object.__setattr__(self, "scale", (round(sigma_y_c, 4), round(sigma_x_c, 4)))
        object.__setattr__(self, "falloff", round(falloff_c, 4))
        object.__setattr__(self, "mu_z", round(mu_z_c, 4))
        if not isinstance(self.token_indices, tuple):
            object.__setattr__(self, "token_indices", tuple(self.token_indices))

    @property
    def center_y(self) -> float:
        return self.center[0]

    @property
    def center_x(self) -> float:
        return self.center[1]

    @property
    def scale_y(self) -> float:
        return self.scale[0]

    @property
    def scale_x(self) -> float:
        return self.scale[1]

    @classmethod
    def from_region(
        cls,
        region: NormalizedBox,
        label: str,
        expected_count: int = 20,
        density: float = 1.0,
        distribution_type: Literal["gaussian", "uniform", "radial", "elongated"] = "gaussian",
        falloff: float = 2.0,
        seed: int | None = None,
        token_indices: tuple[int, ...] = (),
        mu_z: float = 0.5,
        entity_id: str = "",
    ) -> DensityField:
        """Construct a continuous DensityField bounded by a NormalizedBox region."""
        mu_y, mu_x = region.center
        sigma_y = max(1e-4, region.height / 2.0)
        sigma_x = max(1e-4, region.width / 2.0)
        return cls(
            entity_id=entity_id,
            label=label,
            expected_count=expected_count,
            density=density,
            center=(mu_y, mu_x),
            scale=(sigma_y, sigma_x),
            region=region,
            distribution_type=distribution_type,
            falloff=falloff,
            seed=seed,
            token_indices=token_indices,
            mu_z=mu_z,
        )

    def to_heatmap(
        self,
        height: int,
        width: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Rasterize differentiable continuous density field heatmap (1, 1, height, width)."""
        if height <= 0 or width <= 0:
            return torch.zeros(1, 1, max(1, height), max(1, width), device=device, dtype=dtype)

        y_coords = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        x_coords = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        mu_y, mu_x = float(self.center[0]), float(self.center[1])
        sigma_y = max(1e-4, float(self.scale[0]))
        sigma_x = max(1e-4, float(self.scale[1]))
        falloff = max(0.1, float(self.falloff))
        density = float(self.density)

        dy = grid_y - mu_y
        dx = grid_x - mu_x

        if self.distribution_type == "gaussian":
            # 2D Anisotropic Continuous Gaussian Field with parameterized power falloff
            quad = (dy**2) / (2.0 * (sigma_y**2)) + (dx**2) / (2.0 * (sigma_x**2))
            if abs(falloff - 2.0) < 1e-3:
                exponent = quad
            else:
                exponent = torch.pow(torch.clamp(quad, min=1e-8), falloff / 2.0)
            heatmap = density * torch.exp(-torch.clamp(exponent, 0.0, 50.0))

        elif self.distribution_type == "uniform":
            # Uniform density plateau inside the bounding region with smooth boundary falloff
            reg = self.region
            c_y, c_x = reg.center
            h_y = max(1e-4, reg.height / 2.0)
            h_x = max(1e-4, reg.width / 2.0)

            dist_out_y = torch.clamp(torch.abs(grid_y - c_y) - h_y, min=0.0) / sigma_y
            dist_out_x = torch.clamp(torch.abs(grid_x - c_x) - h_x, min=0.0) / sigma_x
            dist_out_sq = (dist_out_y**2) + (dist_out_x**2)
            exponent = 0.5 * dist_out_sq * falloff
            heatmap = density * torch.exp(-torch.clamp(exponent, 0.0, 50.0))

        elif self.distribution_type == "radial":
            # Isotropic/Radial dispersion field (swarms, star clusters, explosions)
            sigma_r = max(1e-4, (sigma_y + sigma_x) / 2.0)
            r = torch.sqrt((dy**2) + (dx**2) + 1e-8)
            norm_r = r / sigma_r
            exponent = 0.5 * torch.pow(torch.clamp(norm_r, min=1e-8), falloff)
            heatmap = density * torch.exp(-torch.clamp(exponent, 0.0, 50.0))

        elif self.distribution_type == "elongated":
            # Streamline / directional elongated flocking profile
            if sigma_x >= sigma_y:
                eff_sy = sigma_y * 0.6
                eff_sx = sigma_x * 1.6
            else:
                eff_sy = sigma_y * 1.6
                eff_sx = sigma_x * 0.6
            quad = (dy**2) / (2.0 * (eff_sy**2)) + (dx**2) / (2.0 * (eff_sx**2))
            if abs(falloff - 2.0) < 1e-3:
                exponent = quad
            else:
                exponent = torch.pow(torch.clamp(quad, min=1e-8), falloff / 2.0)
            heatmap = density * torch.exp(-torch.clamp(exponent, 0.0, 50.0))

        else:
            # Default fallback: anisotropic Gaussian
            quad = (dy**2) / (2.0 * (sigma_y**2)) + (dx**2) / (2.0 * (sigma_x**2))
            heatmap = density * torch.exp(-torch.clamp(quad, 0.0, 50.0))

        # Procedural micro-clustering modulation if seed is present
        if self.seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(int(self.seed))
            phases = torch.rand(4, generator=gen).to(device=device, dtype=dtype)
            freq1, freq2 = 14.0, 28.0
            p1 = torch.sin(freq1 * grid_x + phases[0] * 6.283) * torch.cos(
                freq1 * grid_y + phases[1] * 6.283
            )
            p2 = 0.5 * torch.sin(freq2 * grid_x + phases[2] * 6.283) * torch.cos(
                freq2 * grid_y + phases[3] * 6.283
            )
            perturbation = 0.15 * (p1 + p2)
            heatmap = torch.clamp(heatmap * (1.0 + perturbation), min=0.0)

        return heatmap.unsqueeze(0).unsqueeze(0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "expected_count": self.expected_count,
            "density": self.density,
            "center": [self.center[0], self.center[1]],
            "scale": [self.scale[0], self.scale[1]],
            "region": self.region.to_dict(),
            "distribution_type": self.distribution_type,
            "falloff": self.falloff,
            "seed": self.seed,
            "token_indices": list(self.token_indices),
            "mu_z": self.mu_z,
        }


@dataclass(frozen=True)
class VisualEntity:
    """An entity identified in the visual/reference context for multi-modal grounding."""

    entity_id: str
    label: str
    box: NormalizedBox
    gaussian: GaussianSpatialPrior | None = None
    embedding: torch.Tensor | None = None
    attributes: tuple[str, ...] = ()
    confidence: float = 1.0

    def __post_init__(self):
        if self.gaussian is None:
            object.__setattr__(self, "gaussian", self.box.to_gaussian())

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "box": self.box.to_dict(),
            "gaussian": self.gaussian.to_dict() if self.gaussian else None,
            "attributes": list(self.attributes),
            "confidence": round(self.confidence, 4),
        }


@dataclass(frozen=True)
class VisualContext:
    """Visual reference information for multi-modal grounding and co-reference."""

    image_embedding: torch.Tensor | None = None
    entities: tuple[VisualEntity, ...] = ()
    spatial_features: torch.Tensor | None = None
    spatial_shape: tuple[int, int] | None = None
    feature_map: Any = None
    backbone_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def find_entity_by_label(self, label: str) -> VisualEntity | None:
        lowered = label.lower().strip()
        for ent in self.entities:
            ent_lbl = ent.label.lower().strip()
            if ent_lbl == lowered or ent_lbl in lowered or lowered in ent_lbl:
                return ent
        return None

    def find_entity_by_id(self, entity_id: str) -> VisualEntity | None:
        for ent in self.entities:
            if ent.entity_id == entity_id:
                return ent
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "spatial_shape": list(self.spatial_shape) if self.spatial_shape else None,
            "backbone_metadata": self.backbone_metadata,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class AdaptiveGuidanceConfig:
    """Configuration for dynamic adaptive guidance strength scaling."""

    base_gamma: float = 0.2
    entity_scale: float = 0.05
    relation_scale: float = 0.03
    overlap_scale: float = 0.04
    complexity_scale: float = 0.02
    min_gamma: float = 0.2
    max_gamma: float = 0.5
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_gamma": self.base_gamma,
            "entity_scale": self.entity_scale,
            "relation_scale": self.relation_scale,
            "overlap_scale": self.overlap_scale,
            "complexity_scale": self.complexity_scale,
            "min_gamma": self.min_gamma,
            "max_gamma": self.max_gamma,
            "enabled": self.enabled,
        }


def compute_adaptive_guidance_strength(
    plan: SemanticLayoutPlan | None,
    config: AdaptiveGuidanceConfig | None = None,
    manual_strength: float | None = None,
) -> float:
    """Compute adaptive soft cross-attention guidance strength.

    If manual_strength is provided, it strictly overrides adaptive mode.
    Base heuristic: gamma = clamp(0.2 + 0.05 * N_entities + 0.03 * N_relations + ..., 0.2, 0.5).
    """
    if manual_strength is not None:
        return float(manual_strength)

    cfg = config or AdaptiveGuidanceConfig()
    if not cfg.enabled or plan is None:
        return cfg.base_gamma

    all_boxes = [obj.box for obj in plan.objects] + [
        df.region for df in getattr(plan, "density_fields", ())
    ]
    if not all_boxes:
        return cfg.base_gamma

    n_entities = len(all_boxes)
    n_relations = len(plan.relations)

    # Overlap density between planned entities
    overlap_sum = 0.0
    for i in range(len(all_boxes)):
        for j in range(i + 1, len(all_boxes)):
            overlap_sum += all_boxes[i].iou(all_boxes[j])

    # Prompt token complexity factor
    words = plan.prompt.split()
    complexity_factor = min(2.0, max(0.0, (len(words) - 5) / 10.0))

    gamma = (
        cfg.base_gamma
        + cfg.entity_scale * n_entities
        + cfg.relation_scale * n_relations
        + cfg.overlap_scale * overlap_sum
        + cfg.complexity_scale * complexity_factor
    )
    clamped_gamma = max(cfg.min_gamma, min(cfg.max_gamma, gamma))
    return round(clamped_gamma, 4)


@dataclass(frozen=True)
class PlannedObject:
    """An object planned by the reasoning module with assigned spatial region."""

    label: str
    count: int
    box: NormalizedBox
    token_indices: tuple[int, ...] = ()
    attributes: tuple[str, ...] = ()
    gaussian: GaussianSpatialPrior | None = None
    entity_id: str | None = None

    def __post_init__(self):
        if self.gaussian is None:
            object.__setattr__(self, "gaussian", self.box.to_gaussian())
        if self.entity_id is None:
            center_hash = abs(hash((self.label, self.count, self.box.center))) % 10000
            object.__setattr__(self, "entity_id", f"{self.label}_{center_hash:04d}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "count": self.count,
            "box": self.box.to_dict(),
            "token_indices": list(self.token_indices),
            "attributes": list(self.attributes),
            "gaussian": self.gaussian.to_dict() if self.gaussian else None,
            "entity_id": self.entity_id,
        }


@dataclass(frozen=True)
class EntityOverlap:
    """Spatial overlap and relative depth ordering between two planned entities."""

    entity_a: str
    entity_b: str
    iou: float
    ordering: Literal["a_in_front_of_b", "b_in_front_of_a", "coplanar"]
    depth_delta: float  # mu_z_b - mu_z_a (> 0 means A is in front of B)
    visibility_weight_a: float = 1.0
    visibility_weight_b: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_a": self.entity_a,
            "entity_b": self.entity_b,
            "iou": round(self.iou, 4),
            "ordering": self.ordering,
            "depth_delta": round(self.depth_delta, 4),
            "visibility_weight_a": round(self.visibility_weight_a, 4),
            "visibility_weight_b": round(self.visibility_weight_b, 4),
        }


@dataclass(frozen=True)
class EntityDepthPrior:
    """Relative depth attributes computed for an entity."""

    mu_z: float = 0.5
    sigma_z: float = 0.2
    depth_confidence: float = 1.0


def _compute_relative_depth_priors(
    objects_data: list[tuple[str, int, list[str]]],
    relations_data: list[tuple[str, str, str]],
    positions: dict[str, str] | None = None,
) -> dict[str, EntityDepthPrior]:
    """Compute relative depth priors respecting positional qualifiers and pairwise relations."""
    depths: dict[str, EntityDepthPrior] = {}
    positions = positions or {}

    # 1. Base initialization from position qualifiers
    for label, _, _ in objects_data:
        pos = positions.get(label)
        if pos == "foreground":
            depths[label] = EntityDepthPrior(mu_z=0.25, sigma_z=0.15, depth_confidence=1.0)
        elif pos == "background":
            depths[label] = EntityDepthPrior(mu_z=0.80, sigma_z=0.15, depth_confidence=1.0)
        elif pos in ("center", "middle"):
            depths[label] = EntityDepthPrior(mu_z=0.50, sigma_z=0.20, depth_confidence=0.8)
        else:
            depths[label] = EntityDepthPrior(mu_z=0.50, sigma_z=0.20, depth_confidence=0.5)

    # 2. Pairwise relations
    for subj, rel_type, obj in relations_data:
        if rel_type in ("in_front_of", "ahead_of", "closer"):
            depths[subj] = EntityDepthPrior(mu_z=0.25, sigma_z=0.15, depth_confidence=1.0)
            if obj not in positions or positions[obj] not in ("foreground", "background"):
                depths[obj] = EntityDepthPrior(mu_z=0.70, sigma_z=0.15, depth_confidence=1.0)
        elif rel_type == "far_in_front_of":
            depths[subj] = EntityDepthPrior(mu_z=0.15, sigma_z=0.10, depth_confidence=1.0)
            if obj not in positions:
                depths[obj] = EntityDepthPrior(mu_z=0.80, sigma_z=0.15, depth_confidence=1.0)
        elif rel_type in ("behind", "in_back_of", "further"):
            depths[subj] = EntityDepthPrior(mu_z=0.75, sigma_z=0.15, depth_confidence=1.0)
            if obj not in positions or positions[obj] not in ("foreground", "background"):
                depths[obj] = EntityDepthPrior(mu_z=0.25, sigma_z=0.15, depth_confidence=1.0)
        elif rel_type == "far_behind":
            depths[subj] = EntityDepthPrior(mu_z=0.85, sigma_z=0.10, depth_confidence=1.0)
            if obj not in positions or positions[obj] not in ("foreground", "background"):
                depths[obj] = EntityDepthPrior(mu_z=0.20, sigma_z=0.10, depth_confidence=1.0)
        elif rel_type == "behind_translucent":
            depths[subj] = EntityDepthPrior(mu_z=0.75, sigma_z=0.15, depth_confidence=1.0)
            depths[obj] = EntityDepthPrior(mu_z=0.25, sigma_z=0.10, depth_confidence=1.0)
        elif rel_type in ("under", "below"):
            depths[subj] = EntityDepthPrior(mu_z=0.60, sigma_z=0.18, depth_confidence=0.8)
            if obj not in positions:
                depths[obj] = EntityDepthPrior(mu_z=0.40, sigma_z=0.18, depth_confidence=0.8)
        elif rel_type in ("riding", "on"):
            depths[subj] = EntityDepthPrior(mu_z=0.48, sigma_z=0.15, depth_confidence=0.8)
            if obj not in positions:
                depths[obj] = EntityDepthPrior(mu_z=0.52, sigma_z=0.18, depth_confidence=0.8)
        elif rel_type in ("inside", "within"):
            depths[subj] = EntityDepthPrior(mu_z=0.50, sigma_z=0.10, depth_confidence=0.8)
            if obj not in positions:
                depths[obj] = EntityDepthPrior(mu_z=0.50, sigma_z=0.25, depth_confidence=0.8)

    return depths


def _compute_entity_overlaps(
    planned_objects: list[PlannedObject],
) -> tuple[EntityOverlap, ...]:
    """Compute pairwise spatial overlaps with relative depth ordering."""
    overlaps: list[EntityOverlap] = []
    n = len(planned_objects)
    for i in range(n):
        for j in range(i + 1, n):
            obj_a = planned_objects[i]
            obj_b = planned_objects[j]
            iou_val = obj_a.box.iou(obj_b.box)
            is_spatial_overlap = iou_val > 0.02
            if not is_spatial_overlap and obj_a.gaussian and obj_b.gaussian:
                g_a, g_b = obj_a.gaussian, obj_b.gaussian
                is_spatial_overlap = (
                    abs(g_a.mu_x - g_b.mu_x) < (g_a.sigma_x + g_b.sigma_x) * 1.5
                    and abs(g_a.mu_y - g_b.mu_y) < (g_a.sigma_y + g_b.sigma_y) * 1.5
                )

            if is_spatial_overlap:
                z_a = obj_a.gaussian.mu_z if obj_a.gaussian else 0.5
                z_b = obj_b.gaussian.mu_z if obj_b.gaussian else 0.5
                delta_z = z_b - z_a

                if abs(delta_z) < 0.05:
                    ordering = "coplanar"
                    vis_a, vis_b = 1.0, 1.0
                elif delta_z > 0:
                    ordering = "a_in_front_of_b"
                    vis_a = 1.0
                    vis_b = max(0.2, 1.0 - iou_val * 0.8)
                else:
                    ordering = "b_in_front_of_a"
                    vis_a = max(0.2, 1.0 - iou_val * 0.8)
                    vis_b = 1.0

                overlaps.append(
                    EntityOverlap(
                        entity_a=obj_a.label,
                        entity_b=obj_b.label,
                        iou=iou_val,
                        ordering=ordering,
                        depth_delta=delta_z,
                        visibility_weight_a=vis_a,
                        visibility_weight_b=vis_b,
                    )
                )
    return tuple(overlaps)


@dataclass(frozen=True)
class SpatialRelation:
    """Structured spatial or logical relationship between planned objects."""

    subject: str
    relation_type: str
    object: str
    token_indices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "relation_type": self.relation_type,
            "object": self.object,
            "token_indices": list(self.token_indices),
        }


@dataclass(frozen=True)
class StyleHints:
    """Aesthetic, lighting, mood, and medium cues extracted from the prompt.

    Crucially, these are marked unconstrained: they receive ZERO spatial guidance
    bias so the DiT's artistic priors remain 100% free to render style and texture.
    """

    medium: tuple[str, ...] = ()
    lighting: tuple[str, ...] = ()
    mood: tuple[str, ...] = ()
    composition: tuple[str, ...] = ()
    style_tokens: tuple[int, ...] = ()
    is_unconstrained: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "medium": list(self.medium),
            "lighting": list(self.lighting),
            "mood": list(self.mood),
            "composition": list(self.composition),
            "style_tokens": list(self.style_tokens),
            "is_unconstrained": self.is_unconstrained,
        }


@dataclass(frozen=True)
class PlanSelfCheck:
    """Pre-denoising self-check verifying that logic matches the prompt."""

    is_valid: bool
    count_match: bool
    relation_match: bool
    ambiguity_detected: bool
    assumptions: tuple[str, ...] = ()
    notes: str = "Plan verified successfully."

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "count_match": self.count_match,
            "relation_match": self.relation_match,
            "ambiguity_detected": self.ambiguity_detected,
            "assumptions": list(self.assumptions),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class SemanticLayoutPlan:
    """Comprehensive structured plan from the reasoning stage."""

    prompt: str
    objects: tuple[PlannedObject, ...]
    relations: tuple[SpatialRelation, ...]
    style_hints: StyleHints
    self_check: PlanSelfCheck
    density_fields: tuple[DensityField, ...] = ()
    overlaps: tuple[EntityOverlap, ...] = ()
    token_to_region_map: dict[
        int, NormalizedBox | GaussianSpatialPrior | DensityField | None
    ] = field(default_factory=dict)
    raw_plan: dict[str, Any] | None = None
    visual_context: VisualContext | None = None
    guidance_mode: Literal["gaussian", "box"] = "gaussian"
    adaptive_gamma: float | None = None

    def get_region_for_token(
        self, token_idx: int
    ) -> NormalizedBox | GaussianSpatialPrior | DensityField | None:
        """Return the planned region for a token index, or None if unconstrained (style/neutral)."""
        return self.token_to_region_map.get(token_idx, None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "objects": [obj.to_dict() for obj in self.objects],
            "density_fields": [df.to_dict() for df in self.density_fields],
            "overlaps": [ov.to_dict() for ov in self.overlaps],
            "relations": [rel.to_dict() for rel in self.relations],
            "style_hints": self.style_hints.to_dict(),
            "self_check": self.self_check.to_dict(),
            "token_to_region_map": {
                str(k): (v.to_dict() if v is not None else None)
                for k, v in self.token_to_region_map.items()
            },
            "visual_context": self.visual_context.to_dict() if self.visual_context else None,
            "guidance_mode": self.guidance_mode,
            "adaptive_gamma": self.adaptive_gamma,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


# ---------------------------------------------------------------------------
# Lexical & Semantic Parsing Engine
# ---------------------------------------------------------------------------

def clean_token_piece(piece: str) -> str:
    """Strip sub-word markers and formatting from a tokenizer piece."""
    return (
        piece.replace("</w>", "")
        .replace("##", "")
        .replace("\u2581", "")
        .replace("Ġ", "")
        .replace("\u0120", "")
        .replace("▁", "")
        .replace(" ", "")
        .strip()
        .lower()
    )


def is_special_token(piece: str) -> bool:
    """BOS/EOS/PAD and other bracketed control tokens carry no semantic intent."""
    if not piece:
        return True
    return piece.startswith("<") or (piece.startswith("[") and piece.endswith("]"))


def extract_words(prompt: str) -> list[str]:
    """Extract individual words from prompt for tokenizer subword mapping.
    
    Includes alphanumeric words and digits, ensuring single-letter words
    like 'a' or numbers like '4' are preserved so token positions do not drift.
    """
    return re.findall(r"[a-zA-Z0-9]+", prompt.lower())


def map_pieces_to_words(pieces: list[str]) -> list[int | None]:
    """Group tokenizer pieces into word indices; None for special/empty/punctuation pieces.

    Handles the common subword schemes explicitly:
    - WordPiece (BERT): '##' indicates a continuation subword.
    - SentencePiece (T5 / ALBERT): '\u2581' indicates start of word.
    - Byte-Level BPE (GPT-2 / RoBERTa / CLIP fast): 'Ġ' / '\u0120' indicates start of word.
    - BPE end-of-word (CLIP): '</w>' terminates a word.

    Pure punctuation and special tokens are mapped to None and do not advance
    word indices, preserving exact token index alignment without drifting.
    """
    prefix_markers = ("\u2581", "Ġ", "\u0120", "▁")
    uses_prefix = any(piece.startswith(prefix_markers) for piece in pieces)
    uses_end_of_word = any(piece.endswith("</w>") for piece in pieces)
    uses_wordpiece = any(piece.startswith("##") for piece in pieces)

    word_index: list[int | None] = []
    current = -1
    starts_word = True
    for piece in pieces:
        cleaned = clean_token_piece(piece)
        if is_special_token(piece) or not cleaned or not any(c.isalnum() for c in cleaned):
            word_index.append(None)
            if uses_end_of_word and piece.endswith("</w>"):
                starts_word = True
            continue
        if piece.startswith("##"):
            is_new = False
        elif uses_prefix:
            is_new = current < 0 or piece.startswith(prefix_markers)
        elif uses_end_of_word:
            is_new = starts_word
        elif uses_wordpiece:
            is_new = not piece.startswith("##")
        else:
            is_new = True

        if is_new or current < 0:
            current += 1
        word_index.append(current)
        if uses_end_of_word:
            starts_word = piece.endswith("</w>")
    return word_index


def _matches_style_word(cleaned: str, word: str) -> bool:
    """Match a token piece to a style word or its stem, without 1-char false positives."""
    if not cleaned or len(cleaned) < 2:
        return False
    if cleaned == word:
        return True
    if len(cleaned) >= 3:
        if word.startswith(cleaned) or cleaned.startswith(word):
            return True
        if len(cleaned) >= 4 and cleaned in word:
            return True
    return False


def extract_style_hints(prompt: str, tokenizer=None) -> StyleHints:
    """Extract mood, lighting, medium, and composition keywords from the prompt."""
    lowered = prompt.lower()
    found: dict[str, list[str]] = {"medium": [], "lighting": [], "mood": [], "composition": []}
    style_words: list[str] = []

    for category, terms in _STYLE_KEYWORDS.items():
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                found[category].append(term)
                style_words.extend(extract_words(term))

    style_tokens: list[int] = []
    if tokenizer is not None and style_words:
        encoded = tokenizer(prompt, add_special_tokens=True)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        ids = list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)
        pieces = tokenizer.convert_ids_to_tokens(ids)
        prompt_words = extract_words(prompt)
        word_index = map_pieces_to_words(pieces)
        style_words_set = set(style_words)
        for position, index in enumerate(word_index):
            if index is None or index >= len(prompt_words):
                continue
            if prompt_words[index] in style_words_set:
                style_tokens.append(position)

    return StyleHints(
        medium=tuple(found["medium"]),
        lighting=tuple(found["lighting"]),
        mood=tuple(found["mood"]),
        composition=tuple(found["composition"]),
        style_tokens=tuple(style_tokens),
        is_unconstrained=True,
    )


def _detect_density_distribution(
    label: str,
    count: int,
    prompt: str,
    threshold: int = DEFAULT_DENSITY_ENTITY_THRESHOLD,
) -> tuple[bool, Literal["gaussian", "uniform", "radial", "elongated"], float]:
    """Determine if a quantified entity should be represented as a continuous DensityField.

    Checks:
    1. Direct count >= threshold (e.g. >= 10: '50 bees', 'hundreds of stars', '12 robots')
    2. Collective crowd phrases ('swarm', 'flock', 'crowd', 'many stars', 'sea of')
    """
    lowered = prompt.lower()
    lbl = label.lower()

    # 1. Collective crowd phrase detection
    for coll_word, (_default_cnt, dist_type, falloff) in _COLLECTIVE_CROWD_PATTERNS.items():
        if re.search(rf"\b{coll_word}\s+of\b.*\b{re.escape(lbl)}\b", lowered) or re.search(
            rf"\b{coll_word}\b.*\b{re.escape(lbl)}\b", lowered
        ):
            return True, dist_type, falloff

    # 2. Count threshold (>= threshold)
    if count >= threshold:
        radial_stems = ("bee", "star", "spark", "firefl", "insect", "particle", "galaxy", "crystal")
        elongated_stems = ("bird", "fish", "duck", "swan", "car", "stream", "meteor", "drone")
        uniform_stems = (
            "flower", "tree", "grass", "people", "crowd", "spectator",
            "audience", "popp", "rose", "soldier", "warrior", "lantern",
        )
        if any(w in lbl for w in radial_stems):
            return True, "radial", 1.5
        elif any(w in lbl for w in elongated_stems):
            return True, "elongated", 2.0
        elif any(w in lbl for w in uniform_stems):
            return True, "uniform", 2.5
        else:
            return True, "gaussian", 2.0

    return False, "gaussian", 2.0


def _extract_quantified_nouns(prompt: str) -> list[tuple[str, int, list[str]]]:
    """Extract nouns and their explicit or inferred count from the prompt."""
    words = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'-]*", prompt.lower())
    quantified: list[tuple[str, int, list[str]]] = []
    style_set = {item for sublist in _STYLE_KEYWORDS.values() for item in sublist}
    style_words = {w for s in style_set for w in s.split() if w not in _STOPWORDS and len(w) > 2}

    skip_words = (
        _STOPWORDS
        | style_words
        | _ACTION_VERBS
        | _COMMON_ADJECTIVES
        | _SPATIAL_WORDS
        | _METADATA_WORDS
        | _RELATION_WORDS
        | set(_NUM_WORDS.keys())
    )

    i = 0
    while i < len(words):
        word = words[i]
        count = 1
        has_explicit_count = False
        start_j = i + 1

        # 1. "a/an" structures
        if word in ("a", "an"):
            # a/an <collective> of [num] <noun> (e.g. "a swarm of bees", "a flock of seven birds")
            if (
                i + 2 < len(words)
                and words[i + 1] in _COLLECTIVE_CROWD_PATTERNS
                and words[i + 2] == "of"
            ):
                if i + 3 < len(words) and (words[i + 3] in _NUM_WORDS or words[i + 3].isdigit()):
                    is_dig = words[i + 3].isdigit()
                    count = int(words[i + 3]) if is_dig else _NUM_WORDS[words[i + 3]]
                    start_j = i + 4
                else:
                    count = _COLLECTIVE_CROWD_PATTERNS[words[i + 1]][0]
                    start_j = i + 3
                has_explicit_count = True
            # a/an <number_word> [of] <noun> (e.g. "a pair of swans", "a trio of musicians")
            elif (
                i + 1 < len(words)
                and words[i + 1] in _NUM_WORDS
                and words[i + 1] not in ("a", "an")
            ):
                count = _NUM_WORDS[words[i + 1]]
                has_explicit_count = True
                if i + 2 < len(words) and words[i + 2] == "of":
                    start_j = i + 3
                else:
                    start_j = i + 2
            # a/an <adjectives...> <noun> (e.g. "a cute dog", "a red fox")
            else:
                count = 1
                has_explicit_count = True
                start_j = i + 1

        # 2. Direct collective phrase without "a/an" (e.g. "swarm of bees", "sea of lanterns")
        elif (
            word in _COLLECTIVE_CROWD_PATTERNS
            and i + 1 < len(words)
            and words[i + 1] == "of"
        ):
            if i + 2 < len(words) and (words[i + 2] in _NUM_WORDS or words[i + 2].isdigit()):
                is_dig = words[i + 2].isdigit()
                count = int(words[i + 2]) if is_dig else _NUM_WORDS[words[i + 2]]
                start_j = i + 3
            else:
                count = _COLLECTIVE_CROWD_PATTERNS[word][0]
                start_j = i + 2
            has_explicit_count = True

        # 3. "<quantifier> of <noun>" (e.g. "hundreds of stars", "pair of shoes")
        elif (
            word in _NUM_WORDS
            and i + 1 < len(words)
            and words[i + 1] == "of"
        ):
            count = _NUM_WORDS[word]
            has_explicit_count = True
            start_j = i + 2

        # 4. Direct number word (e.g. "three apples", "twelve robots", "fifty bees", "many stars")
        elif word in _NUM_WORDS and word not in ("a", "an"):
            count = _NUM_WORDS[word]
            has_explicit_count = True
            start_j = i + 1

        # 5. Direct digit string (e.g. "12 robots", "50 bees", "4 swans")
        elif word.isdigit():
            count = int(word)
            has_explicit_count = True
            start_j = i + 1

        # 6. Singular article with possible adjectives (e.g. "a cute dog", "an orange cat")
        elif word in ("a", "an"):
            count = 1
            has_explicit_count = True
            start_j = i + 1

        if has_explicit_count:
            attrs = []
            j = start_j
            while j < len(words) and (words[j] in _COMMON_ADJECTIVES or words[j] in style_words):
                attrs.append(words[j])
                j += 1
            if j < len(words) and words[j] not in skip_words and len(words[j]) >= 3:
                target_noun = words[j]
                if not any(target_noun == q[0] for q in quantified):
                    quantified.append((target_noun, count, attrs))
                i = j + 1
                continue
            i += 1
            continue

        # Single noun check
        if (
            word not in skip_words
            and word not in _NUM_WORDS
            and not word.isdigit()
            and len(word) >= 3
        ):
            if not any(word == q[0] for q in quantified):
                quantified.append((word, 1, []))
        i += 1

    return quantified


def _extract_relations(prompt: str) -> list[tuple[str, str, str]]:
    """Extract explicit spatial relationships (e.g. ('monkey', 'riding', 'giraffe'))."""
    lowered = prompt.lower()
    style_set = {item for sublist in _STYLE_KEYWORDS.values() for item in sublist}
    style_words = {w for s in style_set for w in s.split()}
    skip_set = (
        _STOPWORDS
        # _NUM_WORDS is a dict (word -> value); union with a set needs its keys.
        | set(_NUM_WORDS)
        | _COMMON_ADJECTIVES
        | _ACTION_VERBS
        | set(_SPATIAL_WORDS)
        | set(_METADATA_WORDS)
        | style_words
    )

    matched_spans: list[tuple[int, int]] = []
    raw_matches: list[tuple[int, int, str]] = []

    # Match longest phrases first to prevent 'in'/'on' from creating overlapping duplicate relations
    for phrase, rel_type in _RELATION_PHRASES:
        pattern = rf"\b{re.escape(phrase)}\b"
        for m in re.finditer(pattern, lowered):
            start, end = m.span()
            if any(max(start, s) < min(end, e) for s, e in matched_spans):
                continue
            matched_spans.append((start, end))
            raw_matches.append((start, end, rel_type))

    # Sort matches chronologically by their position in the prompt
    raw_matches.sort(key=lambda x: x[0])

    relations: list[tuple[str, str, str]] = []
    prev_end = 0
    for idx, (start, end, rel_type) in enumerate(raw_matches):
        next_start = raw_matches[idx + 1][0] if idx + 1 < len(raw_matches) else len(lowered)
        before_text = lowered[prev_end:start].strip()
        after_text = lowered[end:next_start].strip()
        prev_end = end

        before_words = [
            w for w in re.findall(r"[a-zA-Z]+", before_text)
            if w not in skip_set and len(w) >= 3
        ]
        after_words = [
            w for w in re.findall(r"[a-zA-Z]+", after_text)
            if w not in skip_set and len(w) >= 3
        ]
        obj = after_words[0] if after_words else "object"
        if before_words:
            for subj in before_words:
                relations.append((subj, rel_type, obj))
        else:
            relations.append(("subject", rel_type, obj))

    return relations


def extract_position_constraints(
    prompt: str,
    known_labels: set[str],
) -> dict[str, str]:
    """Map each object to a stated position ("two cats on the left" -> cats: left).

    Read from the prompt directly rather than from `_extract_relations`, which
    skips spatial words and would otherwise turn "cats on the left and a dog on
    the right" into the invented relation ("cats", "on", "dog").
    """
    lowered = prompt.lower()
    positions: dict[str, str] = {}
    pattern = re.compile(
        r"\b(?:on|in|at|to|toward|towards)\s+(?:the\s+)?([a-z]+)(?:\s+side)?\b"
    )
    for match in pattern.finditer(lowered):
        canonical = _POSITION_WORDS.get(match.group(1))
        if canonical is None:
            continue
        preceding = re.findall(r"[a-z][a-z'-]+", lowered[: match.start()])
        for word in reversed(preceding):
            if word in known_labels:
                positions.setdefault(word, canonical)
                break
    return positions


def drop_spurious_relations(
    relations: list[tuple[str, str, str]],
    positions: dict[str, str],
) -> list[tuple[str, str, str]]:
    """Discard relations that carry no information or contradict a stated position.

    `_extract_relations` emits the sentinels "subject"/"object" when it finds no
    noun on one side, and invents a link between two objects that were each merely
    positioned. Both would drive the layout with nothing behind them.
    """
    kept = []
    for subject, relation_type, obj in relations:
        if subject == "subject" or obj == "object":
            continue
        if subject in positions and obj in positions:
            continue
        if subject in positions or obj in positions:
            continue
        kept.append((subject, relation_type, obj))
    return kept


def split_position_constraints(
    relations_data: list[tuple[str, str, str]],
) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    """Separate real object-to-object relations from positional qualifiers.

    "two cats on the left" parses as (cats, on, left), but "left" is not an object -
    it is a constraint saying where the cats go. Leaving it as a relation makes the
    layout ignore it and place every subject in the same default slot.
    """
    relations: list[tuple[str, str, str]] = []
    positions: dict[str, str] = {}
    for subject, relation_type, obj in relations_data:
        canonical = _POSITION_WORDS.get(obj)
        if canonical is not None:
            positions.setdefault(subject, canonical)
        else:
            relations.append((subject, relation_type, obj))
    return relations, positions


def _compute_layout_boxes(
    objects_data: list[tuple[str, int, list[str]]],
    relations_data: list[tuple[str, str, str]],
    positions: dict[str, str] | None = None,
    is_edit: bool = False,
) -> dict[str, NormalizedBox]:
    """Compute harmonious, non-colliding layout boxes based on relations and counts."""
    boxes: dict[str, NormalizedBox] = {}
    positions = positions or {}

    if len(objects_data) == 1 and not relations_data:
        label, _, _ = objects_data[0]
        if is_edit:
            boxes[label] = NormalizedBox(ymin=0.25, xmin=0.25, ymax=0.55, xmax=0.55)
        elif label in positions:
            ymin, xmin, ymax, xmax = _POSITION_BOXES[positions[label]]
            boxes[label] = NormalizedBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)
        else:
            boxes[label] = NormalizedBox(ymin=0.15, xmin=0.15, ymax=0.85, xmax=0.85)
        return boxes

    handled: set[str] = set()
    # A stated position is the strongest signal available, so it is applied first
    # and never overwritten by the generic relation slots below.
    for label, position in positions.items():
        ymin, xmin, ymax, xmax = _POSITION_BOXES[position]
        boxes[label] = NormalizedBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)
        handled.add(label)
    # Group relations by (rel_type, obj) to handle co-subjects on a surface or relation
    rel_groups: dict[tuple[str, str], list[str]] = {}
    for subj, rel_type, obj in relations_data:
        if subj in positions or obj in positions:
            # At least one side is pinned by an explicit position; the generic
            # relation slots would move it back to the default centre.
            continue
        rel_groups.setdefault((rel_type, obj), []).append(subj)

    for (rel_type, obj), subjs in rel_groups.items():
        if rel_type in ("riding", "on", "above"):
            boxes[obj] = NormalizedBox(ymin=0.45, xmin=0.15, ymax=0.92, xmax=0.85)
            handled.add(obj)
            if len(subjs) == 1:
                boxes[subjs[0]] = NormalizedBox(ymin=0.10, xmin=0.25, ymax=0.50, xmax=0.75)
                handled.add(subjs[0])
            else:
                n_subjs = len(subjs)
                for idx, subj in enumerate(subjs):
                    col_w = 0.80 / max(n_subjs, 1)
                    xmin = 0.10 + idx * col_w
                    xmax = min(0.90, xmin + col_w * 0.90)
                    boxes[subj] = NormalizedBox(ymin=0.10, xmin=xmin, ymax=0.50, xmax=xmax)
                    handled.add(subj)
        elif rel_type in ("under", "below"):
            boxes[obj] = NormalizedBox(ymin=0.10, xmin=0.15, ymax=0.50, xmax=0.85)
            handled.add(obj)
            if len(subjs) == 1:
                boxes[subjs[0]] = NormalizedBox(ymin=0.52, xmin=0.20, ymax=0.92, xmax=0.80)
                handled.add(subjs[0])
            else:
                n_subjs = len(subjs)
                for idx, subj in enumerate(subjs):
                    col_w = 0.80 / max(n_subjs, 1)
                    xmin = 0.10 + idx * col_w
                    xmax = min(0.90, xmin + col_w * 0.90)
                    boxes[subj] = NormalizedBox(ymin=0.52, xmin=xmin, ymax=0.92, xmax=xmax)
                    handled.add(subj)
        elif rel_type in ("next_to", "beside", "holding"):
            boxes[obj] = NormalizedBox(ymin=0.20, xmin=0.52, ymax=0.85, xmax=0.92)
            handled.add(obj)
            n_subjs = len(subjs)
            for idx, subj in enumerate(subjs):
                col_w = 0.40 / max(n_subjs, 1)
                xmin = 0.08 + idx * col_w
                xmax = min(0.48, xmin + col_w * 0.90)
                boxes[subj] = NormalizedBox(ymin=0.20, xmin=xmin, ymax=0.85, xmax=xmax)
                handled.add(subj)
        elif rel_type in ("in_front_of", "ahead_of", "far_in_front_of"):
            boxes[obj] = NormalizedBox(ymin=0.10, xmin=0.25, ymax=0.60, xmax=0.75)
            handled.add(obj)
            n_subjs = len(subjs)
            for idx, subj in enumerate(subjs):
                col_w = 0.60 / max(n_subjs, 1)
                xmin = 0.20 + idx * col_w
                xmax = min(0.80, xmin + col_w * 0.90)
                boxes[subj] = NormalizedBox(ymin=0.35, xmin=xmin, ymax=0.90, xmax=xmax)
                handled.add(subj)
        elif rel_type in ("behind", "far_behind", "behind_translucent"):
            boxes[obj] = NormalizedBox(ymin=0.35, xmin=0.20, ymax=0.90, xmax=0.80)
            handled.add(obj)
            n_subjs = len(subjs)
            for idx, subj in enumerate(subjs):
                col_w = 0.50 / max(n_subjs, 1)
                xmin = 0.25 + idx * col_w
                xmax = min(0.75, xmin + col_w * 0.90)
                boxes[subj] = NormalizedBox(ymin=0.10, xmin=xmin, ymax=0.60, xmax=xmax)
                handled.add(subj)
        elif rel_type == "inside":
            # Nested geometry: subject is strictly nested inside object
            boxes[obj] = NormalizedBox(ymin=0.15, xmin=0.15, ymax=0.85, xmax=0.85)
            handled.add(obj)
            n_subjs = len(subjs)
            if n_subjs == 1:
                boxes[subjs[0]] = NormalizedBox(ymin=0.30, xmin=0.30, ymax=0.70, xmax=0.70)
                handled.add(subjs[0])
            else:
                for idx, subj in enumerate(subjs):
                    col_w = 0.50 / max(n_subjs, 1)
                    xmin = 0.25 + idx * col_w
                    xmax = min(0.75, xmin + col_w * 0.90)
                    boxes[subj] = NormalizedBox(ymin=0.30, xmin=xmin, ymax=0.70, xmax=xmax)
                    handled.add(subj)

    remaining = [q for q in objects_data if q[0] not in handled]
    if remaining:
        placed = [boxes[label] for label in handled if label in boxes]
        for label, box in _allocate_free_slots(remaining, placed):
            boxes[label] = box

    return boxes


def _free_x_intervals(
    placed: list[NormalizedBox],
    *,
    lower: float = 0.02,
    upper: float = 0.98,
    min_width: float = 0.12,
) -> list[tuple[float, float]]:
    """Horizontal gaps left over once the already-placed boxes are removed."""
    occupied = sorted((box.xmin, box.xmax) for box in placed)
    free: list[tuple[float, float]] = []
    cursor = lower
    for xmin, xmax in occupied:
        if xmin - cursor >= min_width:
            free.append((cursor, xmin))
        cursor = max(cursor, xmax)
    if upper - cursor >= min_width:
        free.append((cursor, upper))
    return free


def _allocate_free_slots(
    remaining: list[tuple[str, int, list[str]]],
    placed: list[NormalizedBox],
) -> list[tuple[str, NormalizedBox]]:
    """Give unplaced objects their own space instead of tiling over occupied regions.

    Tiling the full width regardless of what relations already placed is what made
    "three apples and two pears on a table" put the apples straight through the
    pears (IoU 0.51).
    """
    if not placed:
        allocated = []
        count = max(len(remaining), 1)
        for index, (label, _, _) in enumerate(remaining):
            column = 0.85 / count
            xmin = 0.08 + index * column
            xmax = min(0.95, xmin + column * 0.90)
            allocated.append(
                (label, NormalizedBox(ymin=0.20, xmin=xmin, ymax=0.85, xmax=xmax))
            )
        return allocated

    intervals = _free_x_intervals(placed)
    allocated = []
    if intervals:
        # Widest gaps first, so the largest free space is used before the slivers.
        intervals.sort(key=lambda span: span[1] - span[0], reverse=True)
        for index, (label, _, _) in enumerate(remaining):
            start, end = intervals[index % len(intervals)]
            slots = max(1, (len(remaining) + len(intervals) - 1) // len(intervals))
            slot = index // len(intervals)
            width = (end - start) / slots
            xmin = start + slot * width
            xmax = min(end, xmin + width * 0.92)
            allocated.append(
                (label, NormalizedBox(ymin=0.20, xmin=xmin, ymax=0.85, xmax=xmax))
            )
        return allocated

    # Fully occupied horizontally: stack the rest in a shallow band at the top.
    for index, (label, _, _) in enumerate(remaining):
        count = max(len(remaining), 1)
        column = 0.90 / count
        xmin = 0.05 + index * column
        allocated.append(
            (
                label,
                NormalizedBox(ymin=0.02, xmin=xmin, ymax=0.18, xmax=xmin + column * 0.9),
            )
        )
    return allocated


def _parse_visual_context(
    visual_context: VisualContext | dict[str, Any] | list[dict[str, Any]] | None,
    reference_embeddings: torch.Tensor | None = None,
) -> VisualContext | None:
    """Normalize user-supplied visual context input into a typed VisualContext."""
    if visual_context is None:
        if reference_embeddings is not None:
            return VisualContext(image_embedding=reference_embeddings)
        return None

    if isinstance(visual_context, VisualContext):
        if reference_embeddings is not None and visual_context.image_embedding is None:
            return VisualContext(
                image_embedding=reference_embeddings,
                entities=visual_context.entities,
                spatial_features=visual_context.spatial_features,
                spatial_shape=visual_context.spatial_shape,
                feature_map=visual_context.feature_map,
                backbone_metadata=visual_context.backbone_metadata,
                metadata=visual_context.metadata,
            )
        return visual_context

    if isinstance(visual_context, dict):
        raw_entities = visual_context.get("entities", [])
        entities: list[VisualEntity] = []
        for idx, item in enumerate(raw_entities):
            ent_id = str(item.get("entity_id", f"vis_ent_{idx}"))
            label = str(item.get("label", "object"))
            if "box" in item and isinstance(item["box"], dict):
                box = NormalizedBox(**item["box"])
            else:
                box = NormalizedBox(
                    ymin=float(item.get("ymin", 0.2)),
                    xmin=float(item.get("xmin", 0.2)),
                    ymax=float(item.get("ymax", 0.8)),
                    xmax=float(item.get("xmax", 0.8)),
                )
            gaussian = None
            if "gaussian" in item and isinstance(item["gaussian"], dict):
                g_dict = item["gaussian"]
                gaussian = GaussianSpatialPrior(
                    mu_y=float(g_dict.get("mu_y", box.center[0])),
                    mu_x=float(g_dict.get("mu_x", box.center[1])),
                    sigma_y=float(g_dict.get("sigma_y", box.height / 4.0)),
                    sigma_x=float(g_dict.get("sigma_x", box.width / 4.0)),
                    theta=float(g_dict.get("theta", 0.0)),
                    amplitude=float(g_dict.get("amplitude", 1.0)),
                    mu_z=float(g_dict.get("mu_z", 0.5)),
                    sigma_z=float(g_dict.get("sigma_z", 0.2)),
                    depth_confidence=float(g_dict.get("depth_confidence", 1.0)),
                )
            entities.append(
                VisualEntity(
                    entity_id=ent_id,
                    label=label,
                    box=box,
                    gaussian=gaussian,
                    attributes=tuple(item.get("attributes", [])),
                    confidence=float(item.get("confidence", 1.0)),
                )
            )
        img_emb = reference_embeddings or visual_context.get("image_embedding")
        sp_feat = visual_context.get("spatial_features")
        sp_shape = visual_context.get("spatial_shape")
        f_map = visual_context.get("feature_map")
        b_meta = visual_context.get("backbone_metadata", {})
        return VisualContext(
            image_embedding=img_emb,
            entities=tuple(entities),
            spatial_features=sp_feat,
            spatial_shape=sp_shape,
            feature_map=f_map,
            backbone_metadata=b_meta,
            metadata=dict(visual_context.get("metadata", {})),
        )

    if isinstance(visual_context, list):
        entities = []
        for idx, item in enumerate(visual_context):
            ent_id = str(item.get("entity_id", f"vis_ent_{idx}"))
            label = str(item.get("label", "object"))
            if "box" in item and isinstance(item["box"], dict):
                box = NormalizedBox(**item["box"])
            else:
                box = NormalizedBox(
                    ymin=float(item.get("ymin", 0.2)),
                    xmin=float(item.get("xmin", 0.2)),
                    ymax=float(item.get("ymax", 0.8)),
                    xmax=float(item.get("xmax", 0.8)),
                )
            entities.append(
                VisualEntity(
                    entity_id=ent_id,
                    label=label,
                    box=box,
                    attributes=tuple(item.get("attributes", [])),
                    confidence=float(item.get("confidence", 1.0)),
                )
            )
        return VisualContext(image_embedding=reference_embeddings, entities=tuple(entities))

    return None


def plan_semantic_layout(
    intent: PromptIntent | str,
    *,
    tokenizer=None,
    candidate_objects: list[dict[str, Any]] | None = None,
    custom_plan_dict: dict[str, Any] | None = None,
    visual_context: VisualContext | dict[str, Any] | list[dict[str, Any]] | None = None,
    reference_embeddings: torch.Tensor | None = None,
    guidance_mode: Literal["gaussian", "box"] = "gaussian",
    adaptive_guidance: bool = True,
    manual_guidance_strength: float | None = None,
    layout_override: list[dict[str, Any]] | None = None,
    density_entity_threshold: int = DEFAULT_DENSITY_ENTITY_THRESHOLD,
) -> SemanticLayoutPlan:
    """Generate a structured SemanticLayoutPlan balancing semantic logic and aesthetic freedom."""
    if isinstance(intent, str):
        intent = PromptIntent(prompt=intent, mode="generate")
    elif not isinstance(intent, PromptIntent):
        raise TypeError("plan_semantic_layout requires a PromptIntent or str")

    prompt_clean = intent.prompt.strip()
    is_edit = intent.mode == "edit"
    norm_visual_context = _parse_visual_context(visual_context, reference_embeddings)

    if not prompt_clean and not layout_override:
        empty_check = PlanSelfCheck(
            is_valid=False,
            count_match=True,
            relation_match=True,
            ambiguity_detected=True,
            assumptions=("Empty prompt provided; defaulting to central unguided frame.",),
            notes="Empty prompt.",
        )
        return SemanticLayoutPlan(
            prompt=intent.prompt,
            objects=(),
            relations=(),
            style_hints=StyleHints(),
            self_check=empty_check,
            density_fields=(),
            token_to_region_map={},
            visual_context=norm_visual_context,
            guidance_mode=guidance_mode,
            adaptive_gamma=0.2,
        )

    # 1. Extract unconstrained style hints
    style_hints = extract_style_hints(prompt_clean, tokenizer=tokenizer)

    # 2. Extract objects, counts, and spatial relations
    quantified = _extract_quantified_nouns(prompt_clean)
    relations_raw = _extract_relations(prompt_clean)
    positions = extract_position_constraints(
        prompt_clean, {label for label, _, _ in quantified}
    )
    relations_raw = drop_spurious_relations(relations_raw, positions)

    # 3. Compute layout boxes and relative depth priors
    layout_boxes = _compute_layout_boxes(
        quantified, relations_raw, positions, is_edit=is_edit
    )
    depth_priors = _compute_relative_depth_priors(quantified, relations_raw, positions)

    # 4. Map tokens to objects & relations if tokenizer is supplied
    token_to_region_map: dict[
        int, NormalizedBox | GaussianSpatialPrior | DensityField | None
    ] = {}
    planned_objects: list[PlannedObject] = []
    planned_density_fields: list[DensityField] = []

    prompt_words = extract_words(prompt_clean)
    token_indices_by_word: dict[str, list[int]] = {}

    if tokenizer is not None:
        encoded = tokenizer(prompt_clean, add_special_tokens=True)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        ids = list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)
        pieces = tokenizer.convert_ids_to_tokens(ids)
        word_index = map_pieces_to_words(pieces)
        for position, index in enumerate(word_index):
            if index is not None and 0 <= index < len(prompt_words):
                w = prompt_words[index]
                token_indices_by_word.setdefault(w, []).append(position)

    assumptions: list[str] = []

    # Handle explicit layout override (e.g. from interactive UI canvas)
    if layout_override:
        for idx, item in enumerate(layout_override):
            label = str(item.get("label", f"object_{idx}"))
            count = int(item.get("count", 1))
            if "box" in item and isinstance(item["box"], dict):
                box = NormalizedBox(**item["box"])
            else:
                box = NormalizedBox(
                    ymin=float(item.get("ymin", 0.2)),
                    xmin=float(item.get("xmin", 0.2)),
                    ymax=float(item.get("ymax", 0.8)),
                    xmax=float(item.get("xmax", 0.8)),
                )

            tok_indices = tuple(token_indices_by_word.get(label, []))
            if not tok_indices:
                tok_indices = (idx,)

            is_df = (
                item.get("is_density_field", False)
                or item.get("density_field", False)
                or count >= density_entity_threshold
            )

            if is_df:
                dist_type = item.get("distribution_type", "gaussian")
                falloff = float(item.get("falloff", 2.0))
                density = float(item.get("density", 1.0))
                seed = item.get("seed", abs(hash((label, prompt_clean))) % (2**31 - 1))
                mu_z = float(item.get("mu_z", 0.5))
                ent_id = str(item.get("entity_id", f"{label}_density_{idx:04d}"))
                df = DensityField(
                    entity_id=ent_id,
                    label=label,
                    expected_count=count,
                    density=density,
                    center=box.center,
                    scale=(max(1e-4, box.height / 2.0), max(1e-4, box.width / 2.0)),
                    region=box,
                    distribution_type=dist_type,
                    falloff=falloff,
                    seed=seed,
                    token_indices=tok_indices,
                    mu_z=mu_z,
                )
                planned_density_fields.append(df)
                for tid in tok_indices:
                    token_to_region_map[tid] = df
            else:
                if "gaussian" in item and isinstance(item["gaussian"], dict):
                    g_dict = item["gaussian"]
                    gaussian = GaussianSpatialPrior(
                        mu_y=float(g_dict.get("mu_y", box.center[0])),
                        mu_x=float(g_dict.get("mu_x", box.center[1])),
                        sigma_y=float(g_dict.get("sigma_y", box.height / 4.0)),
                        sigma_x=float(g_dict.get("sigma_x", box.width / 4.0)),
                        theta=float(g_dict.get("theta", 0.0)),
                        amplitude=float(g_dict.get("amplitude", 1.0)),
                        mu_z=float(g_dict.get("mu_z", 0.5)),
                        sigma_z=float(g_dict.get("sigma_z", 0.2)),
                        depth_confidence=float(g_dict.get("depth_confidence", 1.0)),
                    )
                else:
                    rot = float(item.get("rotation", item.get("theta", 0.0)))
                    mu_z_override = float(item.get("mu_z", 0.5))
                    gaussian = box.to_gaussian(rotation=rot, mu_z=mu_z_override)

                ent_id = str(item.get("entity_id", f"{label}_{idx:04d}"))
                obj = PlannedObject(
                    label=label,
                    count=count,
                    box=box,
                    token_indices=tok_indices,
                    attributes=tuple(item.get("attributes", [])),
                    gaussian=gaussian,
                    entity_id=ent_id,
                )
                planned_objects.append(obj)
                for tid in tok_indices:
                    token_to_region_map[tid] = box
        assumptions.append("Applied custom interactive layout overrides from user.")
    else:
        for label, count, attrs in quantified:
            box = layout_boxes.get(
                label, NormalizedBox(ymin=0.2, xmin=0.2, ymax=0.8, xmax=0.8)
            )
            tok_indices = tuple(token_indices_by_word.get(label, []))
            d_prior = depth_priors.get(
                label, EntityDepthPrior(mu_z=0.5, sigma_z=0.2, depth_confidence=0.5)
            )
            gaussian = box.to_gaussian(
                rotation=0.0,
                mu_z=d_prior.mu_z,
                sigma_z=d_prior.sigma_z,
                depth_confidence=d_prior.depth_confidence,
            )
            ent_id = None

            # Multi-Modal Visual Co-Reference Grounding
            if norm_visual_context and norm_visual_context.entities:
                matched_ent = norm_visual_context.find_entity_by_label(label)
                coref_types = (
                    "character", "object", "item", "entity",
                    "one", "subject", "thing", "vehicle",
                )
                is_label_coref = (
                    label in coref_types
                    or any(
                        f"{p} {label}" in prompt_clean.lower()
                        for p in ("this", "that", "the same", "same", "the previous", "previous")
                    )
                )
                has_explicit_relation = any(
                    r[0] == label or r[2] == label for r in relations_raw
                )

                if matched_ent is not None:
                    ent_id = matched_ent.entity_id
                    if not has_explicit_relation and is_label_coref:
                        box = matched_ent.box
                        gaussian = matched_ent.gaussian or box.to_gaussian(
                            mu_z=d_prior.mu_z,
                            sigma_z=d_prior.sigma_z,
                            depth_confidence=d_prior.depth_confidence,
                        )
                    assumptions.append(
                        f"Visually grounded '{label}' to reference entity "
                        f"'{matched_ent.entity_id}' ({matched_ent.label})."
                    )
                elif is_label_coref and len(norm_visual_context.entities) == 1:
                    primary = norm_visual_context.entities[0]
                    ent_id = primary.entity_id
                    if not has_explicit_relation:
                        box = primary.box
                        gaussian = primary.gaussian or box.to_gaussian(
                            mu_z=d_prior.mu_z,
                            sigma_z=d_prior.sigma_z,
                            depth_confidence=d_prior.depth_confidence,
                        )
                    assumptions.append(
                        f"Visually grounded co-reference to single reference entity "
                        f"'{primary.entity_id}' ({primary.label})."
                    )

            is_density, dist_type, falloff = _detect_density_distribution(
                label, count, prompt_clean, threshold=density_entity_threshold
            )
            if is_density:
                df = DensityField(
                    entity_id=ent_id or f"{label}_density_{len(planned_density_fields):04d}",
                    label=label,
                    expected_count=count,
                    density=1.0,
                    center=box.center,
                    scale=(max(1e-4, box.height / 2.0), max(1e-4, box.width / 2.0)),
                    region=box,
                    distribution_type=dist_type,
                    falloff=falloff,
                    seed=abs(hash((label, prompt_clean))) % (2**31 - 1),
                    token_indices=tok_indices,
                    mu_z=d_prior.mu_z,
                )
                planned_density_fields.append(df)
                for tid in tok_indices:
                    token_to_region_map[tid] = df
                assumptions.append(
                    f"Planning continuous DensityField ({dist_type}) for {count} '{label}' "
                    f"with falloff={falloff}."
                )
            else:
                obj = PlannedObject(
                    label=label,
                    count=count,
                    box=box,
                    token_indices=tok_indices,
                    attributes=tuple(attrs),
                    gaussian=gaussian,
                    entity_id=ent_id,
                )
                planned_objects.append(obj)
                for tid in tok_indices:
                    token_to_region_map[tid] = box

    # Relations
    planned_relations: list[SpatialRelation] = []
    for subj, rel_type, obj_label in relations_raw:
        rel = SpatialRelation(subject=subj, relation_type=rel_type, object=obj_label)
        planned_relations.append(rel)

    # 5. Self-check validation (count match, relation match, ambiguity detection)
    ambiguity = False
    count_match = True
    for label, count, _ in quantified:
        if count > 1 and count < density_entity_threshold:
            assumptions.append(f"Planning {count} distinct spatial slots for '{label}'.")
        elif not label.endswith("s") and count == 1:
            pass

    all_entities = list(planned_objects) + list(planned_density_fields)
    if len(all_entities) > 1 and not planned_relations:
        ambiguity = True
        assumptions.append("Assumed left-to-right balanced layout for multiple unlinked entities.")

    if not all_entities:
        ambiguity = True
        assumptions.append(
            "No distinct physical objects identified; applying full-frame style priors."
        )

    # All style tokens explicitly get None (unconstrained)
    for st in style_hints.style_tokens:
        token_to_region_map[st] = None

    problems: list[str] = []
    phantom = [
        e.label
        for e in all_entities
        if e.label in _SPATIAL_WORDS or e.label in _RELATION_WORDS
    ]
    if phantom:
        problems.append(f"non-object labels planned: {phantom}")

    worst_overlap = 0.0
    all_boxes = [obj.box for obj in planned_objects] + [
        df.region for df in planned_density_fields
    ]
    for first in range(len(all_boxes)):
        for second in range(first + 1, len(all_boxes)):
            overlap = all_boxes[first].iou(all_boxes[second])
            if overlap > worst_overlap:
                worst_overlap = overlap
    if worst_overlap >= _MAX_ALLOWED_BOX_IOU:
        problems.append(
            f"distinct objects occupy the same region (IoU={worst_overlap:.2f})"
        )

    if any(count < 1 for _, count, _ in quantified):
        count_match = False
        problems.append("an object was planned with a non-positive count")

    relation_match = len(planned_relations) > 0 or len(all_entities) <= 1
    self_check = PlanSelfCheck(
        is_valid=not problems and count_match,
        count_match=count_match,
        relation_match=relation_match,
        ambiguity_detected=ambiguity,
        assumptions=tuple(assumptions),
        notes=(
            "Pre-denoise semantic plan self-check passed."
            if not problems
            else "Self-check failed: " + "; ".join(problems)
        ),
    )

    overlaps = _compute_entity_overlaps(planned_objects)

    interim_plan = SemanticLayoutPlan(
        prompt=prompt_clean,
        objects=tuple(planned_objects),
        density_fields=tuple(planned_density_fields),
        overlaps=overlaps,
        relations=tuple(planned_relations),
        style_hints=style_hints,
        self_check=self_check,
        token_to_region_map=token_to_region_map,
        raw_plan=custom_plan_dict,
        visual_context=norm_visual_context,
        guidance_mode=guidance_mode,
        adaptive_gamma=None,
    )

    adaptive_gamma = compute_adaptive_guidance_strength(
        interim_plan,
        config=AdaptiveGuidanceConfig(enabled=adaptive_guidance),
        manual_strength=manual_guidance_strength,
    )

    return SemanticLayoutPlan(
        prompt=prompt_clean,
        objects=tuple(planned_objects),
        density_fields=tuple(planned_density_fields),
        overlaps=overlaps,
        relations=tuple(planned_relations),
        style_hints=style_hints,
        self_check=self_check,
        token_to_region_map=token_to_region_map,
        raw_plan=custom_plan_dict,
        visual_context=norm_visual_context,
        guidance_mode=guidance_mode,
        adaptive_gamma=adaptive_gamma,
    )
