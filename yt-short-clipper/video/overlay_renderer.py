"""Overlay rendering for hooks, thumbnails and caption burn-in."""

from __future__ import annotations

import subprocess
from pathlib import Path


class OverlayRenderer:
    """Renders hook text, subtitles, and thumbnail overlays."""

    @staticmethod
    def _escape_drawtext(text: str) -> str:
        return (
            text.replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
            .replace("%", "\\%")
        )

    def add_hook_and_captions(
        self,
        input_clip: Path,
        output_clip: Path,
        hook_text: str,
        subtitle_file: Path,
    ) -> None:
        """Render first-2-second hook plus dynamic subtitles."""
        hook = self._escape_drawtext(hook_text)
        vf = (
            "drawbox=x=0:y=0:w=iw:h=180:color=black@0.45:t=fill,"
            f"drawtext=text='{hook}':x=(w-text_w)/2:y=60:fontsize=64:fontcolor=white:"
            "enable='between(t,0,2)',"
            f"subtitles='{subtitle_file.as_posix()}'"
        )

        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_clip),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "copy",
            str(output_clip),
        ]
        subprocess.run(cmd, check=True)

    def generate_thumbnail(self, input_clip: Path, output_image: Path, text: str) -> None:
        """Extract first frame and render thumbnail text overlay."""
        escaped = self._escape_drawtext(text)
        vf = (
            "drawbox=x=0:y=h-260:w=iw:h=260:color=black@0.45:t=fill,"
            f"drawtext=text='{escaped}':x=(w-text_w)/2:y=h-170:fontsize=72:fontcolor=white"
        )
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_clip),
            "-vf",
            vf,
            "-frames:v",
            "1",
            str(output_image),
        ]
        subprocess.run(cmd, check=True)
