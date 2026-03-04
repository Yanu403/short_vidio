"""Generate social metadata for produced clips."""

from __future__ import annotations

import json

from openai import OpenAI


class TitleGenerator:
    """Generates title/description/hashtags for each clip."""

    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key)

    def generate(self, clip_context: str) -> dict:
        """Return metadata dictionary for one clip."""
        prompt = f"""
Generate metadata for a short-form clip.
Return strict JSON with keys title, description, hashtags.
hashtags must be an array of hashtag strings.
Context:
{clip_context}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.6,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You generate concise social metadata."},
                {"role": "user", "content": prompt},
            ],
        )
        payload = json.loads(response.choices[0].message.content)
        payload.setdefault("hashtags", ["#shorts", "#viral"])
        return payload
