"""Video clipping and transcoding with FFmpeg."""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path

import config
from utils.hook_optimizer import optimize_hook, sanitize_ffmpeg_text
from video.face_tracker import detect_face_center

logger = logging.getLogger(__name__)


class FFmpegClipper:
    """Cuts and converts highlight ranges into platform-ready clips."""

    def __init__(self, preset: str = "veryfast") -> None:
        self.preset = preset

    @staticmethod
    def _probe_dimensions(video_path: Path) -> tuple[int, int] | None:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(video_path),
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            raw = result.stdout.strip()
            if "x" not in raw:
                return None
            width_text, height_text = raw.split("x", 1)
            return int(width_text), int(height_text)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def compute_face_crop_expression(face_ratio: float | None) -> str:
        """Build FFmpeg-safe crop x-expression from normalized face ratio."""
        clamped_ratio = 0.5 if face_ratio is None else max(0.0, min(1.0, float(face_ratio)))
        return (
            "max(0,min((in_w-min(in_w\\,in_h)),"
            f"{clamped_ratio:.6f}*(in_w-min(in_w\\,in_h))))"
        )

    @staticmethod
    def _cinematic_filter_for_dims(width: int | None, height: int | None, face_ratio: float | None) -> str:
        _ = width
        _ = height
        crop_x_expr = FFmpegClipper.compute_face_crop_expression(face_ratio)
        return (
            "[0:v]scale='if(gte(a,9/16),-2,1080)':'if(gte(a,9/16),1920,-2)',"
            "crop=1080:1920,setsar=1,boxblur=20[bg];"
            "[0:v]crop='min(in_w,in_h)':'min(in_w,in_h)':"
            f"'{crop_x_expr}':'(in_h-min(in_w,in_h))/2'[fg];"
            "[fg]scale=1080:1080[fg2];"
            "[bg][fg2]overlay=(W-w)/2:(H-h)/2,"
            "zoompan=z='min(zoom+0.0015,1.06)':d=1:s=1080x1920[v]"
        )

    @staticmethod
    def _summarize_ffmpeg_reason(stderr: str) -> str:
        low = (stderr or "").lower()
        if "invalid" in low and "filter" in low:
            return "invalid filter syntax"
        if "no such file or directory" in low:
            return "missing input/output file"
        if "permission denied" in low:
            return "permission denied"
        if "resource temporarily unavailable" in low or "temporarily unavailable" in low:
            return "temporary I/O resource contention"
        if "i/o error" in low or "input/output error" in low:
            return "temporary I/O error"
        if "device or resource busy" in low:
            return "temporary device busy"
        return "ffmpeg command failed"

    @staticmethod
    def _is_temporary_io_error(stderr: str) -> bool:
        low = (stderr or "").lower()
        temporary_markers = (
            "resource temporarily unavailable",
            "temporarily unavailable",
            "i/o error",
            "input/output error",
            "device or resource busy",
        )
        return any(marker in low for marker in temporary_markers)

    @staticmethod
    def _enhance_hook_text(text: str) -> str:
        prefix = ""
        if "?" in text:
            prefix += "🤯"
        if "!" in text:
            prefix += "🔥"
        if not prefix:
            return text
        return f"{prefix} {text}"

    @staticmethod
    def _with_dynamic_zoom(filtergraph: str, hook_boost: bool) -> str:
        dynamic_zoom = "1.02+0.02*sin(on/10)"
        zoom_expr = f"if(lt(on,15),1.08,{dynamic_zoom})" if hook_boost else dynamic_zoom
        if "zoompan=z='" in filtergraph:
            return re.sub(
                r"zoompan=z='[^']*':d=1(?P<size>:s=[^,;\]]+)?",
                lambda match: f"zoompan=z='{zoom_expr}':d=1{match.group('size') or ''}",
                filtergraph,
                count=1,
            )
        if filtergraph.rstrip().endswith("[v]"):
            return re.sub(
                r"\[v\]\s*$",
                f",zoompan=z='{zoom_expr}':d=1:s=1080x1920[v]",
                filtergraph,
                count=1,
            )
        return f"{filtergraph},zoompan=z='{zoom_expr}':d=1:s=1080x1920[v]"

    def _run_ffmpeg(self, cmd: list[str], output_name: str) -> None:
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                return
            except subprocess.CalledProcessError as exc:
                reason = self._summarize_ffmpeg_reason(exc.stderr)
                logger.error("FFmpeg failed for %s", output_name)
                logger.error("reason: %s", reason)
                if attempt >= max_retries or not self._is_temporary_io_error(exc.stderr):
                    raise
                logger.warning(
                    "Temporary I/O failure for %s (attempt %s/%s); retrying",
                    output_name,
                    attempt + 1,
                    max_retries + 1,
                )

    def extract_clip(
        self,
        source_video: Path,
        output_path: Path,
        subtitle_file: Path | None,
        hook_text: str,
        viral_hook_text: str,
        start: float,
        end: float,
        vf_filter: str,
        segments: list[tuple[float, float]] | None = None,
    ) -> None:
        """Create a clip and render overlays in a single encode pass."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        hook_text = optimize_hook(hook_text)
        viral_hook_text = optimize_hook(viral_hook_text)
        effective_hook = sanitize_ffmpeg_text(hook_text)
        effective_viral_hook = sanitize_ffmpeg_text(viral_hook_text)
        escaped_hook = effective_hook.replace("%", "\\%")
        hook_color = "yellow" if effective_hook.strip() == effective_viral_hook.strip() else "white"
        overlay_filter = (
            "drawbox=x=0:y=0:w=iw:h=220:color=black@0.55:t=fill,"
            f"drawtext=text='{escaped_hook}':"
            "x=max(40,(w-text_w)/2):"
            "y=h*0.08:"
            "fontsize=72:"
            "line_spacing=8:"
            f"fontcolor={hook_color}:"
            "borderw=4:"
            "bordercolor=black:"
            "shadowx=3:"
            "shadowy=3:"
            "alpha='if(lt(t,0.4),t*2.5,1)':"
            "enable='between(t,0,3)'"
        )
        subtitle_path: Path | None = subtitle_file if config.ENABLE_SUBTITLES else None
        if subtitle_path is not None:
            escaped_subtitle_path = subtitle_path.as_posix().replace("\\", "\\\\").replace("'", "\\'")
            overlay_filter += f",subtitles='{escaped_subtitle_path}'"

        face_ratio = detect_face_center(source_video)
        probed = self._probe_dimensions(source_video)
        cinematic_vf = self._cinematic_filter_for_dims(*(probed or (None, None)), face_ratio=face_ratio)
        base_vf = vf_filter or cinematic_vf
        if not (base_vf.startswith("[") or ";" in base_vf):
            base_vf = f"[0:v]{base_vf}[v]"
        logger.info("render_mode=cinematic_vertical")
        logger.info("aspect_fix=enabled")

        render_segments = segments or [(start, end)]
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="clip_segments_") as temp_dir:
            temp_root = Path(temp_dir)
            segment_outputs: list[Path] = []

            for segment_idx, (seg_start, seg_end) in enumerate(render_segments, start=1):
                segment_duration = max(0.1, seg_end - seg_start)
                segment_output = temp_root / f"segment_{segment_idx:03d}.mp4"
                segment_outputs.append(segment_output)
                segment_filter = self._with_dynamic_zoom(base_vf, hook_boost=segment_idx == 1)
                segment_cmd = [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{seg_start:.3f}",
                    "-i",
                    str(source_video),
                    "-t",
                    f"{segment_duration:.3f}",
                    "-filter_complex",
                    segment_filter,
                    "-map",
                    "[v]",
                    "-map",
                    "0:a?",
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
                    str(segment_output),
                ]
                self._run_ffmpeg(segment_cmd, segment_output.name)

            concat_list = temp_root / "segments.txt"
            concat_list.write_text(
                "\n".join(f"file '{segment_path.as_posix()}'" for segment_path in segment_outputs),
                encoding="utf-8",
            )
            stitched_output = temp_root / "stitched.mp4"
            concat_cmd = [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
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
                str(stitched_output),
            ]
            self._run_ffmpeg(concat_cmd, stitched_output.name)

            final_cmd = [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(stitched_output),
                "-vf",
                overlay_filter,
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
            self._run_ffmpeg(final_cmd, output_path.name)

        elapsed = time.perf_counter() - started
        logger.info("render_time clip=%s seconds=%.3f", output_path.name, elapsed)
