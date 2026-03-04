"""Application configuration for YT-Short-Clipper."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    """Runtime configuration loaded from environment variables."""

    openai_api_key: str
    output_dir: Path = Path("output")
    cache_dir: Path = Path(".cache")
    model_size: str = "small"
    clip_min_seconds: int = 15
    clip_max_seconds: int = 40
    max_clips: int = 15
    target_width: int = 1080
    target_height: int = 1920
    ffmpeg_preset: str = "veryfast"
    batch_workers: int = 2


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


def load_config() -> AppConfig:
    """Load application configuration from environment variables."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ConfigError(
            "OPENAI_API_KEY is missing. Set it in your shell or .env file before running."
        )

    output = Path(os.getenv("YT_SC_OUTPUT_DIR", "output"))
    output.mkdir(parents=True, exist_ok=True)

    cache = Path(os.getenv("YT_SC_CACHE_DIR", ".cache"))
    cache.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        openai_api_key=key,
        output_dir=output,
        cache_dir=cache,
        model_size=os.getenv("WHISPER_MODEL_SIZE", "small"),
        clip_min_seconds=int(os.getenv("CLIP_MIN_SECONDS", "15")),
        clip_max_seconds=int(os.getenv("CLIP_MAX_SECONDS", "40")),
        max_clips=int(os.getenv("MAX_CLIPS", "15")),
        target_width=int(os.getenv("TARGET_WIDTH", "1080")),
        target_height=int(os.getenv("TARGET_HEIGHT", "1920")),
        ffmpeg_preset=os.getenv("FFMPEG_PRESET", "veryfast"),
        batch_workers=max(1, int(os.getenv("BATCH_WORKERS", "2"))),
    )
