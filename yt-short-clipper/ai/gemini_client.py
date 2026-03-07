"""Gemini API client with retry/backoff and lightweight timing logs."""

from __future__ import annotations

import logging
import time

from google import genai

from config import load_config

logger = logging.getLogger(__name__)


def _is_retryable_error(exc: Exception) -> bool:
    """Return True for transient/rate-limit errors."""
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    return (
        "resourceexhausted" in name
        or "serviceunavailable" in name
        or "toomanyrequests" in name
        or "429" in message
        or "rate" in message
        or "timeout" in message
        or "tempor" in message
        or "unavailable" in message
        or "connection" in message
        or "network" in message
    )


def generate(prompt: str, max_retries: int = 3) -> str:
    """Generate text from Gemini with exponential backoff."""
    cfg = load_config()
    provider = cfg.llm_provider.lower()
    if provider == "mock":
        return '{"clips": []}'
    if provider != "gemini":
        raise RuntimeError(f"Unsupported LLM_PROVIDER='{cfg.llm_provider}'. Supported: gemini, mock.")

    client = genai.Client(api_key=cfg.gemini_api_key)

    delay = 1.0
    for attempt in range(max_retries):
        try:
            started = time.perf_counter()
            response = client.models.generate_content(
                model=cfg.gemini_model,
                contents=prompt,
            )
            elapsed = time.perf_counter() - started
            logger.info("gemini_request_time=%.3fs", elapsed)
            text = (getattr(response, "text", "") or "").strip()
            if text:
                return text
            # Fallback for SDK responses that keep text in candidates/parts.
            candidates = getattr(response, "candidates", None) or []
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) or []
                for part in parts:
                    part_text = (getattr(part, "text", "") or "").strip()
                    if part_text:
                        return part_text
            return ""
        except Exception as exc:  # noqa: BLE001
            retryable = _is_retryable_error(exc)
            if attempt == max_retries - 1 or not retryable:
                raise RuntimeError(f"Gemini request failed: {exc}") from exc
            logger.warning(
                "Gemini transient error (attempt %s/%s): %s",
                attempt + 1,
                max_retries,
                exc,
            )
            time.sleep(delay)
            delay *= 2

    raise RuntimeError("Gemini request failed after retries.")
