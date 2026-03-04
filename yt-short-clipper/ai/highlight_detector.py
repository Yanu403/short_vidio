"""Detect viral highlight moments from transcript using OpenAI."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass(slots=True)
class ViralScore:
    """Per-dimension viral scoring and aggregate."""

    curiosity: int
    emotional_intensity: int
    educational_value: int
    shock_factor: int
    total: int


@dataclass(slots=True)
class HighlightSegment:
    """Candidate clip segment in seconds."""

    start: float
    end: float
    hook: str
    rationale: str
    viral_score: ViralScore


class HighlightDetector:
    """Uses LLM semantic reasoning to find high-retention moments."""

    def __init__(self, api_key: str, min_seconds: int, max_seconds: int) -> None:
        self.client = OpenAI(api_key=api_key)
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

    def _request(self, prompt: str, max_retries: int = 3) -> dict:
        """Call OpenAI with bounded retries and exponential backoff."""
        delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.3,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You produce valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                )
                return json.loads(response.choices[0].message.content)
            except Exception:  # noqa: BLE001
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        return {"clips": []}

    def detect(self, transcript_text: str, max_clips: int = 15) -> list[HighlightSegment]:
        """Return virality-focused highlight segments as structured JSON."""
        prompt = f"""
You are a social media shorts editor.
From this timestamped transcript, select up to {max_clips} viral clip ideas.
Each clip must be between {self.min_seconds} and {self.max_seconds} seconds.
For each clip provide an attention-grabbing hook for first 2 seconds.
Score each clip from 1-100 for curiosity, emotional_intensity, educational_value, shock_factor.
Also return total as rounded average of the 4 dimensions.
Return strict JSON only in this format:
{{"clips":[{{"start":12.3,"end":36.8,"hook":"...","rationale":"...","viral_score":{{"curiosity":78,"emotional_intensity":81,"educational_value":65,"shock_factor":72,"total":74}}}}]}}
Transcript:
{transcript_text}
"""

        payload = self._request(prompt)

        clips: list[HighlightSegment] = []
        for item in payload.get("clips", []):
            start = float(item["start"])
            end = float(item["end"])
            if end <= start:
                continue
            duration = end - start
            if duration < self.min_seconds or duration > self.max_seconds:
                continue
            raw_score = item.get("viral_score", {})
            score = ViralScore(
                curiosity=max(1, min(100, int(raw_score.get("curiosity", 50)))),
                emotional_intensity=max(
                    1, min(100, int(raw_score.get("emotional_intensity", 50)))
                ),
                educational_value=max(
                    1, min(100, int(raw_score.get("educational_value", 50)))
                ),
                shock_factor=max(1, min(100, int(raw_score.get("shock_factor", 50)))),
                total=max(1, min(100, int(raw_score.get("total", 50)))),
            )
            clips.append(
                HighlightSegment(
                    start=start,
                    end=end,
                    hook=item.get("hook", "Wait for this..."),
                    rationale=item.get("rationale", "High engagement moment"),
                    viral_score=score,
                )
            )
        clips.sort(key=lambda c: c.viral_score.total, reverse=True)
        return clips[:max_clips]
