from __future__ import annotations

from pathlib import Path

from video.face_crop import FaceCropTracker


class AutoReframer:
    """Detect subject horizontal position to help vertical cropping."""

    def __init__(self) -> None:
        self.tracker = FaceCropTracker(sample_stride=8, max_samples=360)
        self._last_crop_x_expr = (
            "max(0,min(in_w-min(in_w\\,in_h),0.500000*in_w-min(in_w\\,in_h)/2))"
        )

    def detect_subject_x_ratio(
        self,
        video_path: Path,
        sample_stride: int = 10,
        max_seconds: float = 15.0,
        max_frames: int = 30,
        cache_path: Path | None = None,
    ) -> float:
        """Return normalized X center for detected faces with bounded sampling."""
        _ = max_seconds
        _ = max_frames
        result = self.tracker.analyze(video_path, cache_path=cache_path)
        self._last_crop_x_expr = result.crop_x_expr
        return result.subject_x

    def crop_filter(self, subject_x: float, target_width: int, target_height: int) -> str:
        """Return safe cinematic 9:16 filter graph for landscape compatibility."""
        _ = target_width
        _ = target_height
        clamped_subject_x = max(0.0, min(1.0, subject_x))
        crop_x_expr = (
            "max(0,min(in_w-min(in_w\\,in_h),"
            f"{clamped_subject_x:.6f}*in_w-min(in_w\\,in_h)/2))"
        )
        return (
            "[0:v]scale='if(gte(a,9/16),-2,1080)':'if(gte(a,9/16),1920,-2)',"
            "crop=1080:1920,setsar=1,boxblur=20[bg];"
            f"[0:v]crop='min(in_w,in_h)':'min(in_w,in_h)':"
            f"'{crop_x_expr}':'(in_h-min(in_w,in_h))/2'[fg];"
            "[fg]scale=1080:1080[fg2];"
            "[bg][fg2]overlay=(W-w)/2:(H-h)/2,"
            "zoompan=z='min(zoom+0.0015,1.06)':d=1:s=1080x1920[v]"
        )
