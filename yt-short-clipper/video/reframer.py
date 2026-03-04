from __future__ import annotations

from pathlib import Path
import cv2


class AutoReframer:
    """Detect subject horizontal position to help vertical cropping."""

    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_subject_x_ratio(self, video_path: Path, sample_stride: int = 10) -> float:
        """Return normalized X center for detected faces."""
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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                center_x = x + w / 2
                centers.append(center_x / frame.shape[1])

            idx += 1

        cap.release()

        if not centers:
            return 0.5

        return sum(centers) / len(centers)

    def crop_filter(self, subject_x: float, target_width: int, target_height: int) -> str:
        """Return ffmpeg crop filter centered around subject."""
        x = f"(in_w-{target_width})*{subject_x}"
        return f"crop={target_width}:{target_height}:{x}:0"
