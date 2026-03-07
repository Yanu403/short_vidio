"""Lightweight face center tracking helpers."""

from __future__ import annotations

from pathlib import Path

import cv2

try:
    import mediapipe as mp
except Exception:  # noqa: BLE001
    mp = None


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, value))


def detect_face_center(video_path: Path, sample_rate: int = 10) -> float | None:
    """Return average face center X ratio from sampled frames, or None if unavailable."""
    if mp is None:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    sample_stride = max(1, int(sample_rate))
    centers: list[float] = []
    frame_idx = 0

    with mp.solutions.face_detection.FaceDetection(  # type: ignore[attr-defined]
        model_selection=0,
        min_detection_confidence=0.5,
    ) as detector:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_stride != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)
            if result.detections:
                box = max(
                    (
                        detection.location_data.relative_bounding_box
                        for detection in result.detections
                    ),
                    key=lambda bbox: float(bbox.width * bbox.height),
                )
                face_center_x = float(box.xmin + (box.width / 2.0))
                centers.append(_clamp_ratio(face_center_x))
            frame_idx += 1

    cap.release()
    if not centers:
        return None
    return sum(centers) / len(centers)
