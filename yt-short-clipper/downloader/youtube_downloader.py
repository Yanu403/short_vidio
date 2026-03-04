"""YouTube downloader based on yt-dlp."""

from __future__ import annotations

import time
from pathlib import Path

from yt_dlp import YoutubeDL


class YouTubeDownloader:
    """Downloads YouTube videos into a local working directory."""

    def __init__(self, download_dir: Path) -> None:
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str, max_retries: int = 3) -> Path:
        """Download video from URL and return local file path."""
        opts = {
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "outtmpl": str(self.download_dir / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": False,
            "socket_timeout": 30,
            "retries": 10,
        }

        delay = 1.0
        for attempt in range(max_retries):
            try:
                with YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    downloaded = Path(ydl.prepare_filename(info))
                break
            except Exception:  # noqa: BLE001
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        else:
            raise RuntimeError("Download failed")

        if downloaded.suffix != ".mp4":
            mp4_path = downloaded.with_suffix(".mp4")
            if mp4_path.exists():
                return mp4_path

        return downloaded
