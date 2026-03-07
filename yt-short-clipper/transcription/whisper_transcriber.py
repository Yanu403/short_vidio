"""Speech-to-text utilities using faster-whisper."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from faster_whisper import WhisperModel


@dataclass(slots=True)
class TranscriptSegment:
    """Timed text segment from transcription."""

    start: float
    end: float
    text: str


class WhisperTranscriber:
    """Wraps faster-whisper model loading and transcription with caching."""

    def __init__(self, model_size: str = "tiny") -> None:
        cpu_threads = max(1, os.cpu_count() or 1)
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
            num_workers=1,
        )

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        """Transcribe an audio file into timestamped segments."""
        segments, _ = self.model.transcribe(
            str(audio_path),
            vad_filter=True,
            beam_size=1,
            best_of=1,
            temperature=0.0,
        )
        return [
            TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
            for s in segments
            if s.text.strip()
        ]

    @staticmethod
    def load_cached(cache_path: Path) -> list[TranscriptSegment] | None:
        """Load transcript cache if available."""
        if not cache_path.exists():
            return None
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_segments = payload.get("segments", [])
        elif isinstance(payload, list):
            # Backward compatibility with previous cache format.
            raw_segments = payload
        else:
            raw_segments = []
        return [
            TranscriptSegment(
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                text=str(item.get("text", "")).strip(),
            )
            for item in raw_segments
            if isinstance(item, dict) and str(item.get("text", "")).strip()
        ]

    @staticmethod
    def save_cached(cache_path: Path, segments: list[TranscriptSegment]) -> None:
        """Save transcript cache to disk."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"segments": [asdict(seg) for seg in segments]}, indent=2),
            encoding="utf-8",
        )


def transcript_to_text(segments: Iterable[TranscriptSegment]) -> str:
    """Convert transcript segments into plain text prompt input."""
    return "\n".join(
        f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text}" for seg in segments
    )
