"""Generate hooks and thumbnail overlay text for clips."""

from __future__ import annotations

import json

from openai import OpenAI


class HookThumbnailGenerator:
    """Creates attention hooks and thumbnail overlays from clip context."""

    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key)

    def generate(self, clip_context: str, fallback_hook: str) -> dict[str, str]:
        """Return hook and thumbnail text as strict JSON."""
        prompt = f"""
Generate short-form social creative text.
Return strict JSON with keys: hook_text, thumbnail_text.
Rules:
- hook_text: <= 8 words, high curiosity, to show in first 2 seconds.
- thumbnail_text: <= 5 words, bold and readable.
Context:
{clip_context}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You return concise JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        payload = json.loads(response.choices[0].message.content)
        hook_text = payload.get("hook_text", "").strip() or fallback_hook
        thumbnail_text = payload.get("thumbnail_text", "").strip() or hook_text
        return {"hook_text": hook_text, "thumbnail_text": thumbnail_text}
