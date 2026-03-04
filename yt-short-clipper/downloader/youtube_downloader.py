"""YouTube downloader based on yt-dlp."""

from __future__ import annotations

from pathlib import Path

from yt_dlp import YoutubeDL


class YouTubeDownloader:
    """Downloads YouTube videos into a local working directory."""

    def __init__(self, download_dir: Path) -> None:
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str) -> Path:
        """Download video from URL and return local file path."""
        opts = {
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "outtmpl": str(self.download_dir / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": False,

    # bypass beberapa limit youtube
            "nocheckcertificate": True,
            "ignoreerrors": False,

    # cookies untuk bypass bot check
            "cookiefile": "cookies.txt",

    # supaya lebih stabil
            "retries": 10,
            "fragment_retries": 10,

    # pakai client android (lebih jarang diblokir)
            "extractor_args": {
                "youtube": {
                    "player_client": ["android"," web"]
                }
            }
        }

        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded = Path(ydl.prepare_filename(info))

        if downloaded.suffix != ".mp4":
            mp4_path = downloaded.with_suffix(".mp4")
            if mp4_path.exists():
                return mp4_path

        return downloaded
