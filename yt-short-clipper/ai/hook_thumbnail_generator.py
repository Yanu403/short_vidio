"""Generate hooks and thumbnail overlay text for clips."""

from __future__ import annotations

import json
import time

from openai import OpenAI


class HookThumbnailGenerator:
    """Creates attention hooks and thumbnail overlays from clip context."""

    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key)

    def _request(self, prompt: str, max_retries: int = 3) -> dict:
        """Call OpenAI with bounded retries and exponential backoff."""
        delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.7,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You return concise JSON."},
                        {"role": "user", "content": prompt},
                    ],
                )
                return json.loads(response.choices[0].message.content)
            except Exception:  # noqa: BLE001
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        return {}

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
        payload = self._request(prompt)
        hook_text = payload.get("hook_text", "").strip() or fallback_hook
        thumbnail_text = payload.get("thumbnail_text", "").strip() or hook_text
        return {"hook_text": hook_text, "thumbnail_text": thumbnail_text}
