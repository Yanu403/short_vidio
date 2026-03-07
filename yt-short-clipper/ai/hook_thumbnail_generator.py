"""Generate hooks and thumbnail overlay text for clips."""

from __future__ import annotations

from utils.json_utils import safe_json_parse


class HookThumbnailGenerator:
    """Creates attention hooks and thumbnail overlays from clip context."""

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

    def generate(self, clip_context: str, fallback_hook: str) -> dict[str, str]:
        """Return hook and thumbnail text from precomputed clip metadata."""
        precomputed = self._extract_precomputed(clip_context)
        hook_text = str(precomputed.get("hook", "")).strip() or fallback_hook
        thumbnail_text = str(precomputed.get("thumbnail_text", "")).strip() or hook_text
        return {"hook_text": hook_text, "thumbnail_text": thumbnail_text}
