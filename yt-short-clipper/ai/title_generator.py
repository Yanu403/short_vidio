"""Generate social metadata for produced clips."""

from __future__ import annotations

from utils.json_utils import safe_json_parse


class TitleGenerator:
    """Generates title/description/hashtags for each clip."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _extract_precomputed(clip_context: str) -> dict:
        marker = "PRECOMPUTED_CLIP_JSON:"
        if marker not in clip_context:
            return {}
        payload_text = clip_context.split(marker, 1)[1].strip()
        parsed = safe_json_parse(payload_text)
        if isinstance(parsed, dict):
            return parsed
        return {}

    def generate(self, clip_context: str) -> dict:
        """Return metadata dictionary from precomputed clip payload."""
        precomputed = self._extract_precomputed(clip_context)
        hashtags = precomputed.get("hashtags", ["#shorts", "#viral"])
        if not isinstance(hashtags, list):
            hashtags = ["#shorts", "#viral"]
        return {
            "title": str(precomputed.get("title", "")).strip(),
            "description": str(precomputed.get("description", "")).strip(),
            "hashtags": hashtags or ["#shorts", "#viral"],
        }
