"""Application settings and startup directory validation."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    APP_NAME: str = "AI Image Generation API"
    HOST: str = "0.0.0.0"
    PORT: int = Field(8000, ge=1, le=65535)
    DEBUG: bool = True

    ALLOWED_ORIGINS: list[str] = ["*"]

    MODEL_ID: str = "runwayml/stable-diffusion-v1-5"
    PIXART_MODEL_ID: str = "PixArt-alpha/PixArt-XL-2-512x512"
    SD35_MODEL_ID: str = "stabilityai/stable-diffusion-3.5-large"
    FLUX_MODEL_ID: str = "black-forest-labs/FLUX.1-dev"
    DEFAULT_MODEL_BACKBONE: str = "pixart-alpha"
    MODEL_CACHE_DIR: str = "./models/cache"
    DEVICE: Literal["auto", "cpu", "cuda"] = "auto"
    DTYPE: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    MODEL_CPU_OFFLOAD: bool = True

    DEFAULT_WIDTH: int = Field(512, ge=256, le=2048, multiple_of=8)
    DEFAULT_HEIGHT: int = Field(512, ge=256, le=2048, multiple_of=8)
    DEFAULT_STEPS: int = Field(20, ge=1, le=100)
    DEFAULT_GUIDANCE_SCALE: float = Field(7.5, ge=1.0, le=20.0)
    PIXART_DEFAULT_STEPS: int = Field(20, ge=1, le=100)
    PIXART_DEFAULT_GUIDANCE_SCALE: float = Field(4.5, ge=1.0, le=20.0)
    SD35_DEFAULT_STEPS: int = Field(28, ge=1, le=100)
    SD35_DEFAULT_GUIDANCE_SCALE: float = Field(4.5, ge=1.0, le=20.0)
    SD35_DEFAULT_WIDTH: int = Field(1024, ge=256, le=2048, multiple_of=8)
    SD35_DEFAULT_HEIGHT: int = Field(1024, ge=256, le=2048, multiple_of=8)

    # Inference-Time Quality & Tier Management
    DEFAULT_QUALITY_TIER: Literal["preview", "final", "ultra"] = "final"
    CFG_RESCALE: float = Field(0.70, ge=0.0, le=1.0)
    REFINER_ENABLED: bool = False
    REFINER_STRENGTH: float = Field(0.25, ge=0.05, le=0.60)
    REFINER_STEPS: int = Field(8, ge=1, le=50)
    GUIDANCE_SWITCH_THRESHOLD: float = Field(0.55, ge=0.0, le=1.0)
    GUIDANCE_SCHEDULE_TYPE: Literal["two_phase", "depth_aware", "cosine", "linear"] = "two_phase"

    MAX_BATCH_PIXELS: int = Field(1_048_576, ge=65_536)
    MAX_GENERATION_WORK_UNITS: int = Field(52_428_800, ge=65_536)
    MAX_ASPECT_RATIO: float = Field(2.0, ge=1.0, le=8.0)
    MAX_PROMPT_LENGTH: int = Field(2_000, ge=1, le=20_000)
    MAX_NEGATIVE_PROMPT_LENGTH: int = Field(2_000, ge=1, le=20_000)
    MAX_EDIT_UPLOAD_BYTES: int = Field(10_485_760, ge=1)

    OUTPUT_DIR: str = "./outputs"
    MAX_OUTPUT_FILES: int = Field(1_000, ge=4)
    MAX_OUTPUT_BYTES: int = Field(5_368_709_120, ge=1_048_576)
    MAX_OUTPUT_AGE_SECONDS: int = Field(604_800, ge=0)

    RATE_LIMIT_PER_MINUTE: int = Field(10, ge=1)
    LAZY_LOAD_MODEL: bool = True

    # Spatial, Depth & Multi-Modal Guidance Configuration
    DEPTH_GUIDANCE_ENABLED: bool = True
    DEPTH_GUIDANCE_STRENGTH: float = Field(0.3, ge=0.0, le=20.0)  # Legacy fallback & global default

    # Per-relation guidance strengths
    LATERAL_GUIDANCE_STRENGTH: float = Field(6.0, ge=0.0, le=20.0)
    DEPTH_RELATION_GUIDANCE_STRENGTH: float = Field(0.0, ge=0.0, le=20.0)  # Disabled by default
    VERTICAL_ON_GUIDANCE_STRENGTH: float = Field(0.0, ge=0.0, le=20.0)     # Disabled by default
    VERTICAL_UNDER_GUIDANCE_STRENGTH: float = Field(0.3, ge=0.0, le=20.0)  # Preserved default

    RELATION_GUIDANCE_STRENGTHS: dict[str, float] = Field(
        default_factory=lambda: {
            "lateral": 6.0,
            "depth": 0.0,
            "vertical_on": 0.0,
            "vertical_under": 0.3,
            "default": 0.3,
        }
    )

    def get_relation_guidance_strength(self, relation_type: str | None) -> float:
        """Resolve guidance strength for a specific relation type with legacy fallback."""
        if not relation_type:
            return self.DEPTH_GUIDANCE_STRENGTH
        rel = relation_type.lower().strip().replace(" ", "_").replace("-", "_")
        if rel in {
            "left_of", "right_of", "beside", "next_to", "side_by_side", "adjacent_to", "lateral"
        }:
            return self.LATERAL_GUIDANCE_STRENGTH
        elif rel in {
            "in_front_of", "behind", "far_in_front_of", "far_behind", "front", "back", "depth"
        }:
            return self.DEPTH_RELATION_GUIDANCE_STRENGTH
        elif rel in {
            "on", "on_top_of", "resting_on", "perched_on", "standing_on",
            "sitting_on", "atop", "riding", "above", "over", "vertical_on"
        }:
            return self.VERTICAL_ON_GUIDANCE_STRENGTH
        elif rel in {"under", "below", "underneath", "beneath", "vertical_under"}:
            return self.VERTICAL_UNDER_GUIDANCE_STRENGTH
        return self.RELATION_GUIDANCE_STRENGTHS.get(
            rel,
            self.RELATION_GUIDANCE_STRENGTHS.get("default", self.DEPTH_GUIDANCE_STRENGTH),
        )

    SELF_ATTENTION_DEPTH_BIAS_ENABLED: bool = True
    DENSITY_FIELD_ENABLED: bool = True
    DENSITY_ENTITY_THRESHOLD: int = Field(10, ge=2, le=500)
    VISUAL_CROSS_ATTN_ENABLED: bool = True
    VISUAL_FEATURE_STRENGTH: float = Field(0.25, ge=0.0, le=2.0)
    VISION_BACKBONE: Literal["auto", "mock", "siglip", "dinov2"] = "auto"
    ROTATION_EDITING_ENABLED: bool = True

    CHAT_PROVIDER_TYPE: Literal["openai", "anthropic", "unconfigured"] = "openai"
    CHAT_API_BASE_URL: str = "https://api.openai.com/v1"
    CHAT_API_KEY: SecretStr = SecretStr("")
    CHAT_MODEL: str = ""
    CHAT_TWO_PASS_ENABLED: bool = True
    CHAT_REASONING_TIMEOUT_SECONDS: float = Field(4.0, gt=0, le=120)
    CHAT_RESPONSE_TIMEOUT_SECONDS: float = Field(12.0, gt=0, le=120)
    CHAT_MAX_REASONING_TOKENS: int = Field(500, ge=64, le=4096)
    CHAT_MAX_OUTPUT_TOKENS: int = Field(1_200, ge=128, le=16_384)
    CHAT_REASONING_MAX_TURNS: int = Field(8, ge=1, le=50)
    CHAT_REASONING_TEMPERATURE: float = Field(0.1, ge=0.0, le=2.0)
    CHAT_RESPONSE_TEMPERATURE: float = Field(0.7, ge=0.0, le=2.0)
    CHAT_MAX_MESSAGE_LENGTH: int = Field(4_000, ge=1, le=20_000)
    CHAT_MAX_SESSIONS: int = Field(1_000, ge=1)
    CHAT_MAX_HISTORY_MESSAGES: int = Field(12, ge=2, le=200)
    CHAT_MAX_HISTORY_CHARS: int = Field(16_000, ge=1_000, le=1_000_000)
    CHAT_SESSION_TTL_SECONDS: int = Field(3_600, ge=60)
    CHAT_RATE_LIMIT_PER_MINUTE: int = Field(30, ge=1)
    CHAT_LOG_DECISION_CONTENT: bool = False

    @model_validator(mode="after")
    def validate_default_pixel_budget(self):
        if self.DEFAULT_WIDTH * self.DEFAULT_HEIGHT > self.MAX_BATCH_PIXELS:
            raise ValueError("DEFAULT_WIDTH * DEFAULT_HEIGHT must not exceed MAX_BATCH_PIXELS")
        if (
            self.DEFAULT_WIDTH * self.DEFAULT_HEIGHT * self.DEFAULT_STEPS
            > self.MAX_GENERATION_WORK_UNITS
        ):
            raise ValueError("Default dimensions and steps exceed MAX_GENERATION_WORK_UNITS")
        if self.CHAT_MAX_HISTORY_CHARS < self.CHAT_MAX_MESSAGE_LENGTH + 8_500:
            raise ValueError("CHAT_MAX_HISTORY_CHARS must fit one maximum user/assistant exchange")
        return self


settings = Settings()

os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
