"""Dynamic face-tracked crop helpers for vertical clips."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2

try:
    import mediapipe as mp
except Exception:  # noqa: BLE001
    mp = None


@dataclass(slots=True)
class FaceCropResult:
    """Face tracking output used to build FFmpeg crop filters."""

    subject_x: float
    crop_x_expr: str


class FaceCropTracker:
    """Tracks face center across time and builds dynamic crop X expression."""

    def __init__(self, sample_stride: int = 10, max_samples: int = 300) -> None:
        self.sample_stride = max(1, sample_stride)
        self.max_samples = max(1, max_samples)

    @staticmethod
    def _clamp_ratio(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _default_expr(subject_x: float = 0.5) -> str:
        subject_x = FaceCropTracker._clamp_ratio(subject_x)
        return (
            f"max(0,min(in_w-min(in_w\\,in_h),"
            f"{subject_x:.6f}*in_w-min(in_w\\,in_h)/2))"
        )

    @staticmethod
    def _smooth(points: list[tuple[float, float]], window: int = 2) -> list[tuple[float, float]]:
        if not points:
            return []
        smoothed: list[tuple[float, float]] = []
        for idx, (t, _) in enumerate(points):
            lo = max(0, idx - window)
            hi = min(len(points), idx + window + 1)
            avg = sum(points[j][1] for j in range(lo, hi)) / (hi - lo)
            smoothed.append((t, avg))
        return smoothed

    @staticmethod
    def _piecewise_expr(points: list[tuple[float, float]]) -> str:
        if not points:
            return FaceCropTracker._default_expr(0.5)
        terms = []
        for _, ratio in points:
            ratio = FaceCropTracker._clamp_ratio(ratio)
            terms.append(
                f"max(0,min(in_w-min(in_w\\,in_h),{ratio:.6f}*in_w-min(in_w\\,in_h)/2))"
            )
        expr = terms[-1]
        for idx in range(len(points) - 2, -1, -1):
            t = max(0.0, points[idx + 1][0])
            expr = f"if(lt(t\\,{t:.3f}),{terms[idx]},{expr})"
        return expr

    def _detect_points(self, video_path: Path) -> list[tuple[float, float]]:
        if mp is None:
            return []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        points: list[tuple[float, float]] = []
        frame_idx = 0
        with mp.solutions.face_detection.FaceDetection(  # type: ignore[attr-defined]
            model_selection=0,
            min_detection_confidence=0.5,
        ) as detector:
            while cap.isOpened() and len(points) < self.max_samples:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % self.sample_stride != 0:
                    frame_idx += 1
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.process(rgb)
                if result.detections:
                    box = result.detections[0].location_data.relative_bounding_box
                    cx = box.xmin + (box.width / 2.0)
                    points.append((frame_idx / fps, self._clamp_ratio(float(cx))))
                frame_idx += 1
        cap.release()
        return self._smooth(points, window=2)

    def analyze(self, video_path: Path, cache_path: Path | None = None) -> FaceCropResult:
        """Analyze face positions and return dynamic crop expression."""
        if cache_path and cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            crop_expr = str(payload.get("crop_x_expr", "")).strip()
            subject_x = float(payload.get("subject_x", 0.5))
            if crop_expr:
                return FaceCropResult(subject_x=self._clamp_ratio(subject_x), crop_x_expr=crop_expr)
            return FaceCropResult(
                subject_x=self._clamp_ratio(subject_x),
                crop_x_expr=self._default_expr(subject_x),
            )

        points = self._detect_points(video_path)
        if points:
            subject_x = sum(x for _, x in points) / len(points)
            crop_expr = self._piecewise_expr(points)
        else:
            subject_x = 0.5
            crop_expr = self._default_expr(subject_x)

        result = FaceCropResult(subject_x=self._clamp_ratio(subject_x), crop_x_expr=crop_expr)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "subject_x": result.subject_x,
                        "crop_x_expr": result.crop_x_expr,
                    }
                ),
                encoding="utf-8",
            )
        return result

