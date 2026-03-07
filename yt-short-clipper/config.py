"""Application configuration for YT-Short-Clipper."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LLM_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
ENABLE_SUBTITLES = os.getenv("ENABLE_SUBTITLES", "true").lower() == "true"


@dataclass(slots=True)
class AppConfig:
    """Runtime configuration loaded from environment variables."""

    llm_provider: str = DEFAULT_LLM_PROVIDER
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_api_key: str = ""
    output_dir: Path = Path("output")
    cache_dir: Path = Path(".cache")
    model_size: str = "tiny"
    clip_min_seconds: int = 15
    clip_max_seconds: int = 40
    max_clips: int = 15
    target_width: int = 1080
    target_height: int = 1920
    ffmpeg_preset: str = "veryfast"
    skip_scene_detection: bool = True
    max_parallel_clips: int = 2
    auto_upload: bool = False
    upload_provider: str = "gdrive"
    upload_path: str = "shorts"
    delete_zip_after_upload: bool = False


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


def load_config() -> AppConfig:
    """Load application configuration from environment variables."""
    global ENABLE_SUBTITLES
    ENABLE_SUBTITLES = os.getenv("ENABLE_SUBTITLES", "true").lower() == "true"

    llm_provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
    if llm_provider not in {"gemini", "mock"}:
        raise ConfigError("LLM_PROVIDER must be one of: gemini, mock.")

    gemini_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if llm_provider == "gemini" and not gemini_api_key:
        raise ConfigError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini.")

    output = Path(os.getenv("YT_SC_OUTPUT_DIR", "output"))
    output.mkdir(parents=True, exist_ok=True)

    cache = Path(os.getenv("YT_SC_CACHE_DIR", ".cache"))
    cache.mkdir(parents=True, exist_ok=True)

    model_size = os.getenv("WHISPER_MODEL_SIZE", "tiny").strip().lower()
    if model_size not in {"tiny", "base"}:
        raise ConfigError(
            "WHISPER_MODEL_SIZE must be one of: tiny, base (CPU-optimized)."
        )

    cpu_default_workers = min(4, max(1, os.cpu_count() or 1))

    auto_upload = os.getenv("AUTO_UPLOAD", "false").strip().lower() in {"1", "true", "yes", "on"}
    upload_provider = os.getenv("UPLOAD_PROVIDER", "gdrive").strip()
    upload_path = os.getenv("UPLOAD_PATH", "shorts").strip()
    delete_zip_after_upload = os.getenv("DELETE_ZIP_AFTER_UPLOAD", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    return AppConfig(
        llm_provider=llm_provider,
        gemini_model=gemini_model,
        gemini_api_key=gemini_api_key,
        output_dir=output,
        cache_dir=cache,
        model_size=model_size,
        clip_min_seconds=int(os.getenv("CLIP_MIN_SECONDS", "15")),
        clip_max_seconds=int(os.getenv("CLIP_MAX_SECONDS", "40")),
        max_clips=int(os.getenv("MAX_CLIPS", "15")),
        target_width=int(os.getenv("TARGET_WIDTH", "1080")),
        target_height=int(os.getenv("TARGET_HEIGHT", "1920")),
        ffmpeg_preset=os.getenv("FFMPEG_PRESET", "veryfast"),
        skip_scene_detection=os.getenv("SKIP_SCENE_DETECTION", "true").strip().lower() in {"1", "true", "yes", "on"},
        max_parallel_clips=max(
            1, int(os.getenv("MAX_PARALLEL_CLIPS", str(cpu_default_workers)))
        ),
        auto_upload=auto_upload,
        upload_provider=upload_provider,
        upload_path=upload_path,
        delete_zip_after_upload=delete_zip_after_upload,
    )
