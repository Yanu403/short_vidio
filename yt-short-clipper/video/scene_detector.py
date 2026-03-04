"""Scene detection and clip boundary alignment."""

from __future__ import annotations

from pathlib import Path

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector


class SceneDetector:
    """Detects scene boundaries and aligns suggested ranges to nearest cut points."""

    def detect_boundaries(self, video_path: Path) -> list[tuple[float, float]]:
        """Return a list of (start_sec, end_sec) scene boundaries."""
        video = open_video(str(video_path))
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=27.0))
        manager.detect_scenes(video)

        scenes = manager.get_scene_list()
        return [
            (start.get_seconds(), end.get_seconds())
            for start, end in scenes
            if end.get_seconds() > start.get_seconds()
        ]

    @staticmethod
    def align_range(start: float, end: float, scenes: list[tuple[float, float]]) -> tuple[float, float]:
        """Snap clip start/end to closest scene boundaries."""
        if not scenes:
            return start, end

        boundaries = sorted({t for pair in scenes for t in pair})
        aligned_start = min(boundaries, key=lambda t: abs(t - start))
        aligned_end = min(boundaries, key=lambda t: abs(t - end))

        if aligned_end <= aligned_start:
            aligned_end = end
        return aligned_start, aligned_end
