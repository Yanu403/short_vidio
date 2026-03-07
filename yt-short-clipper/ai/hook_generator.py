"""Generate short hook text for clips using Gemini."""

from __future__ import annotations

from utils.json_utils import safe_json_parse


class HookGenerator:
    """Creates concise hook text from clip context."""

    def generate(self, clip_context: str, fallback_hook: str = "Wait for this...") -> str:
        marker = "PRECOMPUTED_CLIP_JSON:"
        if marker in clip_context:
            payload_text = clip_context.split(marker, 1)[1].strip()
            parsed = safe_json_parse(payload_text)
            if isinstance(parsed, dict):
                hook = str(parsed.get("hook", "")).strip()
                if hook:
                    return hook
        return fallback_hook
