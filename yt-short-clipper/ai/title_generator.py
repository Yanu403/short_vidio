"""Generate social metadata for produced clips."""

from __future__ import annotations

import json
import time

from openai import OpenAI


class TitleGenerator:
    """Generates title/description/hashtags for each clip."""

    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key)

    def _request(self, prompt: str, max_retries: int = 3) -> dict:
        """Call OpenAI with bounded retries and exponential backoff."""
        delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.6,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You generate concise social metadata."},
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

    def generate(self, clip_context: str) -> dict:
        """Return metadata dictionary for one clip."""
        prompt = f"""
Generate metadata for a short-form clip.
Return strict JSON with keys title, description, hashtags.
hashtags must be an array of hashtag strings.
Context:
{clip_context}
"""
        payload = self._request(prompt)
        payload.setdefault("hashtags", ["#shorts", "#viral"])
        return payload
