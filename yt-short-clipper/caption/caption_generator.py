"""Subtitle generation from transcript segments."""

from __future__ import annotations

from pathlib import Path

from transcription.whisper_transcriber import TranscriptSegment


class CaptionGenerator:
    """Builds SRT caption files for clip ranges."""

    @staticmethod
    def _fmt(seconds: float) -> str:
        millis = int(round(seconds * 1000))
        s = millis // 1000
        ms = millis % 1000
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02}:{m:02}:{sec:02},{ms:03}"

    def write_srt(
        self,
        transcript: list[TranscriptSegment],
        clip_start: float,
        clip_end: float,
        output_path: Path,
        first_line: str = "",
    ) -> None:
        """Write subtitles trimmed to a clip timeline."""
        lines: list[str] = []
        idx = 1
        duration = max(0.0, clip_end - clip_start)

        first_line = first_line.strip()
        if first_line and duration > 0.0:
            hook_end = min(2.0, duration)
            lines.extend(
                [
                    str(idx),
                    f"{self._fmt(0.0)} --> {self._fmt(hook_end)}",
                    first_line,
                    "",
                ]
            )
            idx += 1

        for seg in transcript:
            if seg.end < clip_start or seg.start > clip_end:
                continue
            start = max(seg.start, clip_start) - clip_start
            end = min(seg.end, clip_end) - clip_start
            if end <= start:
                continue

            lines.extend(
                [
                    str(idx),
                    f"{self._fmt(start)} --> {self._fmt(end)}",
                    seg.text,
                    "",
                ]
            )
            idx += 1

        output_path.write_text("\n".join(lines), encoding="utf-8")
