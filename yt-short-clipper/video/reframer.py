"""Auto-reframing utilities for converting horizontal content to vertical."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp


class AutoReframer:
    """Detects dominant face X-position and computes FFmpeg crop coordinates."""

    def __init__(self) -> None:
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

    def detect_subject_x_ratio(self, video_path: Path, sample_stride: int = 15) -> float:
        """Return the normalized X center for most visible face across sampled frames."""
        cap = cv2.VideoCapture(str(video_path))
        centers: list[float] = []
        idx = 0

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if idx % sample_stride != 0:
                idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_detection.process(rgb)
            if result.detections:
                detection = result.detections[0]
                bbox = detection.location_data.relative_bounding_box
                centers.append(max(0.0, min(1.0, bbox.xmin + bbox.width / 2)))
            idx += 1

        cap.release()

        if not centers:
            return 0.5
        return sum(centers) / len(centers)

    @staticmethod
    def crop_filter(subject_x_ratio: float, target_w: int, target_h: int) -> str:
        """Build FFmpeg crop+scale filter for vertical framing."""
        crop_expr = (
            f"crop='min(iw,ih*{target_w}/{target_h})':ih:"
            f"max(0,min(iw-min(iw,ih*{target_w}/{target_h}),"
            f"{subject_x_ratio}*iw-min(iw,ih*{target_w}/{target_h})/2))':0"
        )
        return f"{crop_expr},scale={target_w}:{target_h}"
