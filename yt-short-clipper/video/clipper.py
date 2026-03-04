"""Video clipping and transcoding with FFmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path


class FFmpegClipper:
    """Cuts and converts highlight ranges into platform-ready clips."""

    def __init__(self, preset: str = "veryfast") -> None:
        self.preset = preset

    def extract_clip(
        self,
        source_video: Path,
        output_path: Path,
        start: float,
        end: float,
        vf_filter: str,
    ) -> None:
        """Create a clip from source using accelerated H.264 settings."""
        duration = max(0.1, end - start)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source_video),
            "-t",
            f"{duration:.3f}",
            "-vf",
            vf_filter,
            "-c:v",
            "libx264",
            "-preset",
            self.preset,
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
