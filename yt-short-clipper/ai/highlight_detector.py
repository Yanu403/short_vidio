"""Detect viral highlight moments and clip metadata using Gemini."""

from __future__ import annotations

import logging
import json
import time
from dataclasses import dataclass, field

from ai.gemini_client import generate
from transcription.whisper_transcriber import TranscriptSegment
from utils.json_utils import safe_json_parse

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ViralScore:
    """Single virality score."""

    total: float


@dataclass(slots=True)
class HighlightSegment:
    """Candidate clip segment in seconds."""

    start: float
    end: float
    original_hook: str
    viral_hook: str
    viral_score: ViralScore
    retention_score: float = 0.0
    total_score: float = 0.0
    hook_offset_seconds: float | None = None
    has_dramatic_pause: bool = False
    has_emotional_keyword: bool = False
    has_payoff_near_end: bool = False
    title: str = ""
    description: str = ""
    hashtags: list[str] = field(default_factory=list)
    thumbnail_text: str = ""
    rationale: str = ""


class HighlightDetector:
    """Uses LLM semantic reasoning to find high-retention moments."""

    PREFERRED_MIN_SECONDS = 18.0
    PREFERRED_MAX_SECONDS = 32.0
    HOOK_WINDOW_SECONDS = 3.0
    EARLY_HOOK_LIMIT_SECONDS = 5.0
    PAYOFF_WINDOW_SECONDS = 5.0
    DRAMATIC_PAUSE_GAP_SECONDS = 0.6
    EMOTIONAL_KEYWORDS = {
        "amazing",
        "angry",
        "anxious",
        "awesome",
        "crazy",
        "devastating",
        "disaster",
        "emotional",
        "epic",
        "fear",
        "furious",
        "hate",
        "heartbreaking",
        "incredible",
        "insane",
        "love",
        "panic",
        "powerful",
        "regret",
        "sad",
        "scared",
        "shocking",
        "surprise",
        "terrifying",
        "unbelievable",
        "wow",
    }

    def __init__(self, min_seconds: int, max_seconds: int) -> None:
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

    @staticmethod
    def _valid_clip_list(value: object) -> bool:
        if not isinstance(value, list):
            return False
        for item in value:
            if not isinstance(item, dict):
                return False
            if not {"start", "end", "original_hook", "viral_hook", "title", "viral_score"}.issubset(
                item.keys()
            ):
                return False
        return True

    @staticmethod
    def _request(prompt: str) -> dict:
        """Call Gemini and parse JSON safely."""
        raw = generate(prompt)
        parsed = safe_json_parse(raw)
        if isinstance(parsed, dict):
            if "clips" in parsed and HighlightDetector._valid_clip_list(parsed["clips"]):
                return {"clips": parsed["clips"]}
            # Some models return a single clip object instead of {"clips":[...]}.
            if {"start", "end", "original_hook", "viral_hook", "title", "viral_score"}.issubset(
                parsed.keys()
            ):
                return {"clips": [parsed]}
            return parsed
        if isinstance(parsed, list):
            if HighlightDetector._valid_clip_list(parsed):
                return {"clips": parsed}
            return {"clips": []}
        return {"clips": []}

    @staticmethod
    def _parse_time(value: object) -> float:
        """Parse numeric seconds or HH:MM:SS(.ms) strings into seconds."""
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return 0.0
        if ":" not in text:
            try:
                return float(text)
            except ValueError:
                return 0.0
        parts = text.split(":")
        try:
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            if len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s)
        except ValueError:
            return 0.0
        return 0.0

    @staticmethod
    def _transcript_payload(transcript_segments: list[TranscriptSegment]) -> str:
        segments: list[dict[str, object]] = []
        previous_end: float | None = None
        for seg in transcript_segments:
            if previous_end is not None and seg.start - previous_end > HighlightDetector.DRAMATIC_PAUSE_GAP_SECONDS:
                segments.append(
                    {
                        "start": round(previous_end, 3),
                        "end": round(seg.start, 3),
                        "text": " ... ",
                    }
                )
            segments.append(
                {
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text,
                }
            )
            previous_end = seg.end
        return json.dumps({"segments": segments}, ensure_ascii=True)

    @staticmethod
    def _contains_emotional_keyword(text: str) -> bool:
        lower_text = text.lower()
        return any(keyword in lower_text for keyword in HighlightDetector.EMOTIONAL_KEYWORDS)

    @staticmethod
    def _is_strong_sentence(text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return False
        if "?" in cleaned or "!" in cleaned:
            return True
        if HighlightDetector._contains_emotional_keyword(cleaned):
            return True
        return len(cleaned.split()) >= 7

    @staticmethod
    def _clip_segments(
        transcript_segments: list[TranscriptSegment], start: float, end: float
    ) -> list[TranscriptSegment]:
        return [
            seg
            for seg in transcript_segments
            if seg.end > start and seg.start < end and seg.text.strip()
        ]

    @staticmethod
    def _retention_signals(
        transcript_segments: list[TranscriptSegment],
        start: float,
        end: float,
    ) -> dict[str, object]:
        clip_segments = HighlightDetector._clip_segments(transcript_segments, start, end)
        first_hook_at: float | None = None
        has_payoff_near_end = False
        has_dramatic_pause = False
        has_emotional_keyword = False
        combined_text_parts: list[str] = []
        previous_end: float | None = None
        payoff_start = max(start, end - HighlightDetector.PAYOFF_WINDOW_SECONDS)

        for seg in clip_segments:
            relative_start = max(start, seg.start) - start
            combined_text_parts.append(seg.text)
            if first_hook_at is None and HighlightDetector._is_strong_sentence(seg.text):
                first_hook_at = relative_start
            if seg.end > payoff_start and HighlightDetector._is_strong_sentence(seg.text):
                has_payoff_near_end = True
            if previous_end is not None and seg.start - previous_end > HighlightDetector.DRAMATIC_PAUSE_GAP_SECONDS:
                has_dramatic_pause = True
            previous_end = seg.end

        has_emotional_keyword = HighlightDetector._contains_emotional_keyword(" ".join(combined_text_parts))
        hook_within_first_three = first_hook_at is not None and first_hook_at <= HighlightDetector.HOOK_WINDOW_SECONDS

        retention_score = 0.0
        if hook_within_first_three:
            retention_score += 2.0
        if has_payoff_near_end:
            retention_score += 2.0
        if has_dramatic_pause:
            retention_score += 1.0
        if has_emotional_keyword:
            retention_score += 1.0

        return {
            "retention_score": retention_score,
            "hook_offset_seconds": first_hook_at,
            "has_dramatic_pause": has_dramatic_pause,
            "has_emotional_keyword": has_emotional_keyword,
            "has_payoff_near_end": has_payoff_near_end,
        }

    def detect(self, transcript_segments: list[TranscriptSegment], max_clips: int = 15) -> list[HighlightSegment]:
        """Return virality-focused highlight segments as structured JSON."""
        started = time.perf_counter()
        requested_clips = min(8, max_clips)
        transcript_payload = self._transcript_payload(transcript_segments)
        prompt = f"""
You are a viral content strategist for YouTube Shorts and TikTok.

Your job is NOT to summarize the video.

Your job is to detect moments that will make viewers STOP scrolling.

You will receive transcript segments with timestamps.

You must identify moments that trigger strong viewer psychology.

Prioritize moments that contain:

• a shocking or controversial statement
• a strong claim or bold opinion
• a surprising fact or unexpected reveal
• conflict between ideas
• curiosity gaps (something the viewer wants to know next)
• emotional reactions (anger, excitement, disbelief)
• punchlines or dramatic realizations

Avoid moments that are:
• introductions
• slow explanations
• context without payoff
• filler conversation

A good Shorts clip should feel like the viewer joined in the middle of something intense.

IMPORTANT:
The clip must start slightly BEFORE the most interesting sentence so the buildup is preserved.

Return ONLY valid JSON using this schema:

{{
 "clips":[
  {{
   "start":0.0,
   "end":0.0,
   "original_hook":"exact spoken moment summarized",
   "viral_hook":"rewritten hook optimized for Shorts subtitles",
   "title":"high curiosity YouTube Shorts title",
   "viral_score":0
  }}
 ]
}}

Rules:
- Keep each clip between {self.min_seconds} and {self.max_seconds} seconds
- start/end must align with transcript segment timestamps
- viral_hook should feel like a bold subtitle someone would read instantly
- titles should create curiosity, not summaries
- viral_score must be an integer from 1–10 based on scroll-stopping potential

Scoring guide:
10 = extremely shocking / highly controversial
8–9 = strong emotional or curiosity hook
6–7 = interesting but less intense
<6 = avoid selecting

Transcript JSON:
{transcript_payload}
"""

        payload = self._request(prompt)

        clips: list[HighlightSegment] = []
        for item in payload.get("clips", []):
            if not isinstance(item, dict):
                continue
            start = self._parse_time(item.get("start", 0))
            end = self._parse_time(item.get("end", 0))
            if end <= start:
                continue
            duration = end - start
            if duration > self.PREFERRED_MAX_SECONDS:
                end = start + self.PREFERRED_MAX_SECONDS
                duration = end - start
            if duration < 3.0:
                continue
            try:
                score_value = float(item.get("viral_score", 0))
            except (TypeError, ValueError):
                score_value = 0.0
            if score_value < 7:
                continue
            score = ViralScore(total=max(1.0, min(10.0, score_value)))
            original_hook = str(item.get("original_hook", "")).strip()
            viral_hook = str(item.get("viral_hook", "")).strip()
            signals = self._retention_signals(transcript_segments, start, end)
            retention_score = float(signals["retention_score"])
            total_score = score.total + retention_score
            clips.append(
                HighlightSegment(
                    start=start,
                    end=end,
                    original_hook=original_hook or "Wait for this...",
                    viral_hook=viral_hook or original_hook or "Wait for this...",
                    retention_score=retention_score,
                    total_score=total_score,
                    hook_offset_seconds=signals["hook_offset_seconds"],  # type: ignore[arg-type]
                    has_dramatic_pause=bool(signals["has_dramatic_pause"]),
                    has_emotional_keyword=bool(signals["has_emotional_keyword"]),
                    has_payoff_near_end=bool(signals["has_payoff_near_end"]),
                    title=str(item.get("title", "")).strip() or (viral_hook or "Untitled short"),
                    description="",
                    hashtags=["#shorts", "#viral"],
                    thumbnail_text=(viral_hook or original_hook or "Wait for this...")[:42],
                    rationale="High engagement moment",
                    viral_score=score,
                )
            )
        clips.sort(key=lambda c: c.total_score, reverse=True)
        logger.info("clip_detection_time=%.3fs", time.perf_counter() - started)
        return clips[:max_clips]
