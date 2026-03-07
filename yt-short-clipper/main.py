"""CLI entrypoint for YT-Short-Clipper."""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from ai.highlight_detector import HighlightDetector, HighlightSegment
from ai.hook_thumbnail_generator import HookThumbnailGenerator
from ai.title_generator import TitleGenerator
from caption.caption_generator import CaptionGenerator
from config import AppConfig, ConfigError, load_config
from downloader.youtube_downloader import YouTubeDownloader
from transcription.whisper_transcriber import TranscriptSegment, WhisperTranscriber, transcript_to_text
from utils.archive import archive_output
from utils.uploader import upload_file
from video.clipper import FFmpegClipper
from video.overlay_renderer import OverlayRenderer
from video.reframer import AutoReframer
from video.scene_detector import SceneDetector
from utils.logger import error, header, info, stage, success

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised when pipeline validation fails."""


def safe_unlink(path: Path) -> None:
    """Delete path if it exists."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def cleanup_workdir(path: Path) -> None:
    """Remove temporary media/artifacts from workdir while preserving the directory."""
    for pattern in ("*.mp4", "*.wav", "*.json"):
        for file_path in path.glob(pattern):
            if file_path.is_file():
                safe_unlink(file_path)


def validate_video_url(url: str) -> None:
    """Apply minimal URL validation for YouTube input."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if "youtube.com" not in domain and "youtu.be" not in domain:
        raise ValueError("Unsupported video URL")


class ClipPaths:
    """Helper for deterministic clip file naming."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def final(self, idx: int) -> Path:
        return self.output_dir / f"clip_{idx}.mp4"

    def srt(self, idx: int) -> Path:
        return self.output_dir / f"clip_{idx}.srt"

    def thumbnail(self, idx: int) -> Path:
        return self.output_dir / f"clip_{idx}_thumbnail.jpg"


@dataclass(slots=True)
class PlannedClip:
    """Planned clip boundaries and source highlight."""

    idx: int
    start: float
    end: float
    highlight: HighlightSegment
    segments: list[tuple[float, float]] = field(default_factory=list)


@dataclass(slots=True)
class PipelineContext:
    """Shared pipeline values passed between stages."""

    url: str
    video_path: Path | None = None
    audio_path: Path | None = None
    transcript: list[TranscriptSegment] | None = None
    transcript_text: str = ""
    highlights: list[HighlightSegment] = field(default_factory=list)
    planned_clips: list[PlannedClip] = field(default_factory=list)
    vf_filter: str = ""
    metadata_entries: list[dict] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class DownloadStage:
    """Stage 1: Download source video."""

    name = "Download Video"
    metric_key = "download_time"

    def __init__(self, downloader: YouTubeDownloader) -> None:
        self.downloader = downloader

    def run(self, ctx: PipelineContext) -> PipelineContext:
        validate_video_url(ctx.url)
        ctx.video_path = self.downloader.download(ctx.url)
        return ctx


class AudioExtractionStage:
    """Stage 2: Extract WAV audio for transcription."""

    name = "Extract Audio"

    @staticmethod
    def _extract_audio(video_path: Path, output_audio_path: Path) -> None:
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_audio_path),
        ]
        subprocess.run(cmd, check=True)

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir

    def run(self, ctx: PipelineContext) -> PipelineContext:
        assert ctx.video_path is not None
        audio_path = self.work_dir / f"{ctx.video_path.stem}.wav"
        self._extract_audio(ctx.video_path, audio_path)
        ctx.audio_path = audio_path
        return ctx


class TranscriptionStage:
    """Stage 3: Transcribe extracted audio with CPU-optimized whisper."""

    name = "Transcription"
    metric_key = "transcription_time"

    def __init__(self, transcriber: WhisperTranscriber, cache_dir: Path, model_size: str) -> None:
        self.transcriber = transcriber
        self.cache_dir = cache_dir
        self.model_size = model_size

    def run(self, ctx: PipelineContext) -> PipelineContext:
        assert ctx.video_path is not None
        assert ctx.audio_path is not None
        cache_path = self.cache_dir / "transcripts" / f"{ctx.video_path.stem}_{self.model_size}.json"
        transcript = self.transcriber.load_cached(cache_path)
        if transcript is None:
            transcript = self.transcriber.transcribe(ctx.audio_path)
            self.transcriber.save_cached(cache_path, transcript)
        ctx.transcript = transcript
        ctx.transcript_text = transcript_to_text(transcript)
        if len(ctx.transcript_text) < 50:
            raise PipelineError("Transcript too short")
        return ctx


class HighlightDetectionStage:
    """Stage 4: Detect and rank viral highlights."""

    name = "Highlight Detection"
    metric_key = "highlight_detection_time"

    def __init__(self, detector: HighlightDetector, max_clips: int) -> None:
        self.detector = detector
        self.max_clips = max_clips

    def run(self, ctx: PipelineContext) -> PipelineContext:
        assert ctx.transcript is not None
        ctx.highlights = self.detector.detect(ctx.transcript, max_clips=self.max_clips)
        return ctx


class ClipPlanningStage:
    """Stage 5: Build clip plan with optional scene alignment and cached reframing."""

    name = "Clip Planning"
    HOOK_WINDOW_SECONDS = 3.0
    BUILDUP_MIN_SECONDS = 10.0
    BUILDUP_MAX_SECONDS = 15.0
    PAYOFF_MIN_SECONDS = 5.0
    PAYOFF_MAX_SECONDS = 10.0
    PREFERRED_MIN_SECONDS = 18.0
    PREFERRED_MAX_SECONDS = 32.0
    EARLY_HOOK_LIMIT_SECONDS = 5.0
    EMOTIONAL_KEYWORDS = {
        "amazing",
        "angry",
        "anxious",
        "awesome",
        "crazy",
        "devastating",
        "disaster",
        "emotional",
        "epic",
        "fear",
        "furious",
        "hate",
        "heartbreaking",
        "incredible",
        "insane",
        "love",
        "panic",
        "powerful",
        "regret",
        "sad",
        "scared",
        "shocking",
        "surprise",
        "terrifying",
        "unbelievable",
        "wow",
    }

    def __init__(
        self,
        cfg: AppConfig,
        detector: SceneDetector,
        reframer: AutoReframer,
    ) -> None:
        self.cfg = cfg
        self.detector = detector
        self.reframer = reframer

    @staticmethod
    def _build_micro_segments(
        start: float,
        end: float,
        scenes: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Build 3-5 second micro segments preferring scene cut boundaries."""
        if end <= start:
            return [(start, start + 0.1)]
        if end - start <= 5.0:
            return [(start, end)]

        scene_boundaries = sorted(
            {
                boundary
                for scene_start, scene_end in scenes
                for boundary in (scene_start, scene_end)
                if start < boundary < end
            }
        )

        segments: list[tuple[float, float]] = []
        current = start
        while end - current > 0.01:
            remaining = end - current
            if remaining <= 5.0:
                if remaining < 3.0 and segments:
                    prev_start, _ = segments[-1]
                    segments[-1] = (prev_start, end)
                else:
                    segments.append((current, end))
                break

            min_cut = current + 3.0
            max_cut = min(current + 5.0, end)
            candidates = [b for b in scene_boundaries if min_cut <= b <= max_cut]
            if candidates:
                target = current + 4.0
                cut = min(candidates, key=lambda value: abs(value - target))
            else:
                cut = min(current + 4.0, max_cut)

            if cut <= current + 0.05:
                cut = max_cut

            segments.append((current, cut))
            current = cut

        return [(seg_start, max(seg_start + 0.1, seg_end)) for seg_start, seg_end in segments]

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _structured_duration(self, highlight_duration: float) -> float:
        if highlight_duration > self.PREFERRED_MAX_SECONDS:
            return self.PREFERRED_MAX_SECONDS

        target_total = self._clamp(
            highlight_duration,
            self.PREFERRED_MIN_SECONDS,
            self.PREFERRED_MAX_SECONDS,
        )
        payoff = self._clamp(target_total * 0.3, self.PAYOFF_MIN_SECONDS, self.PAYOFF_MAX_SECONDS)
        buildup = target_total - self.HOOK_WINDOW_SECONDS - payoff
        buildup = self._clamp(buildup, self.BUILDUP_MIN_SECONDS, self.BUILDUP_MAX_SECONDS)
        target_total = self.HOOK_WINDOW_SECONDS + buildup + payoff

        if target_total < self.PREFERRED_MIN_SECONDS:
            deficit = self.PREFERRED_MIN_SECONDS - target_total
            expansion = min(deficit, self.BUILDUP_MAX_SECONDS - buildup)
            buildup += expansion
            deficit -= expansion
            payoff += min(deficit, self.PAYOFF_MAX_SECONDS - payoff)
            target_total = self.HOOK_WINDOW_SECONDS + buildup + payoff

        if target_total > self.PREFERRED_MAX_SECONDS:
            overflow = target_total - self.PREFERRED_MAX_SECONDS
            reduction = min(overflow, buildup - self.BUILDUP_MIN_SECONDS)
            buildup -= reduction
            overflow -= reduction
            payoff -= min(overflow, payoff - self.PAYOFF_MIN_SECONDS)
            target_total = self.HOOK_WINDOW_SECONDS + buildup + payoff

        return self._clamp(target_total, self.PREFERRED_MIN_SECONDS, self.PREFERRED_MAX_SECONDS)

    @classmethod
    def _contains_emotional_keyword(cls, text: str) -> bool:
        lower_text = text.lower()
        return any(keyword in lower_text for keyword in cls.EMOTIONAL_KEYWORDS)

    @classmethod
    def _is_strong_sentence(cls, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return False
        if "?" in cleaned or "!" in cleaned:
            return True
        if cls._contains_emotional_keyword(cleaned):
            return True
        return len(cleaned.split()) >= 7

    def _first_strong_sentence_offset(
        self,
        transcript: list[TranscriptSegment],
        clip_start: float,
        clip_end: float,
    ) -> float | None:
        for seg in transcript:
            if seg.end <= clip_start or seg.start >= clip_end:
                continue
            if self._is_strong_sentence(seg.text):
                return max(0.0, seg.start - clip_start)
        return None

    @staticmethod
    def _bounded_window(start: float, duration: float, video_duration: float) -> tuple[float, float]:
        start = max(0.0, start)
        end = start + max(0.1, duration)
        if video_duration > 0.0 and end > video_duration:
            end = video_duration
            start = max(0.0, end - duration)
        return start, max(start + 0.1, end)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        assert ctx.video_path is not None
        assert ctx.transcript is not None

        scenes: list[tuple[float, float]] = []
        if not self.cfg.skip_scene_detection:
            scenes = self.detector.detect_boundaries(ctx.video_path)

        reframe_cache = (
            self.cfg.cache_dir
            / "reframe"
            / f"{ctx.video_path.stem}_{self.cfg.target_width}x{self.cfg.target_height}.json"
        )
        try:
            subject_x = self.reframer.detect_subject_x_ratio(
                ctx.video_path,
                sample_stride=10,
                max_seconds=15.0,
                max_frames=40,
                cache_path=reframe_cache,
            )
        except Exception:
            subject_x = 0.5

        ctx.vf_filter = self.reframer.crop_filter(subject_x, self.cfg.target_width, self.cfg.target_height)

        planned: list[PlannedClip] = []
        video_duration = max((edge for scene in scenes for edge in scene), default=0.0)
        for idx, highlight in enumerate(ctx.highlights, start=1):
            target_duration = self._structured_duration(max(0.1, highlight.end - highlight.start))
            start = max(0.0, highlight.start - random.uniform(0.4, 0.8))
            end = start + target_duration
            if scenes:
                start, end = SceneDetector.align_range(start, end, scenes)
            start, end = self._bounded_window(start, target_duration, video_duration)

            hook_offset = self._first_strong_sentence_offset(ctx.transcript, start, end)
            if hook_offset is not None and hook_offset > self.EARLY_HOOK_LIMIT_SECONDS:
                adjusted_start = max(0.0, start + (hook_offset - self.HOOK_WINDOW_SECONDS))
                start, end = self._bounded_window(adjusted_start, target_duration, video_duration)

            micro_segments = self._build_micro_segments(start, end, scenes)
            planned.append(
                PlannedClip(
                    idx=idx,
                    start=start,
                    end=end,
                    highlight=highlight,
                    segments=micro_segments,
                )
            )

        ctx.planned_clips = planned
        return ctx


class ClipProcessor:
    """Process one planned clip into final assets and metadata."""

    def __init__(
        self,
        cfg: AppConfig,
        paths: ClipPaths,
        clipper: FFmpegClipper,
        captioner: CaptionGenerator,
        renderer: OverlayRenderer,
        creative_gen: HookThumbnailGenerator,
        title_gen: TitleGenerator,
    ) -> None:
        self.cfg = cfg
        self.paths = paths
        self.clipper = clipper
        self.captioner = captioner
        self.renderer = renderer
        self.creative_gen = creative_gen
        self.title_gen = title_gen

    def process_clip(
        self,
        planned: PlannedClip,
        video_path: Path,
        transcript: list[TranscriptSegment],
        transcript_text: str,
        vf: str,
    ) -> dict:
        highlight = planned.highlight
        final_clip = self.paths.final(planned.idx)
        thumb_path = self.paths.thumbnail(planned.idx)
        srt_path = self.paths.srt(planned.idx)

        self.captioner.write_srt(
            transcript,
            planned.start,
            planned.end,
            srt_path,
            first_line=highlight.viral_hook,
        )
        precomputed_payload = {
            "start": round(planned.start, 3),
            "end": round(planned.end, 3),
            "hook": highlight.viral_hook,
            "original_hook": highlight.original_hook,
            "viral_hook": highlight.viral_hook,
            "thumbnail_text": highlight.thumbnail_text,
            "title": highlight.title,
            "description": highlight.description,
            "hashtags": highlight.hashtags,
            "rationale": highlight.rationale,
        }
        precomputed_context = (
            "PRECOMPUTED_CLIP_JSON:\n"
            f"{json.dumps(precomputed_payload, ensure_ascii=True)}"
        )
        creative = self.creative_gen.generate(
            clip_context=precomputed_context,
            fallback_hook=highlight.viral_hook,
        )

        self.clipper.extract_clip(
            source_video=video_path,
            output_path=final_clip,
            subtitle_file=srt_path,
            hook_text=creative["hook_text"],
            viral_hook_text=highlight.viral_hook,
            start=planned.start,
            end=planned.end,
            vf_filter=vf,
            segments=planned.segments,
        )
        self.renderer.generate_thumbnail(final_clip, thumb_path, creative["thumbnail_text"])

        metadata = self.title_gen.generate(
            precomputed_context
        )

        return {
            "clip": final_clip.name,
            "thumbnail": thumb_path.name,
            "start": round(planned.start, 3),
            "end": round(planned.end, 3),
            "hook": creative["hook_text"],
            "thumbnail_text": creative["thumbnail_text"],
            "viral_score": {
                "total": highlight.viral_score.total,
            },
            "retention_score": {
                "total": round(highlight.retention_score, 3),
                "hook_within_first_3s": bool(
                    highlight.hook_offset_seconds is not None and highlight.hook_offset_seconds <= 3.0
                ),
                "payoff_in_final_5s": highlight.has_payoff_near_end,
                "dramatic_pause_detected": highlight.has_dramatic_pause,
                "emotional_keyword_present": highlight.has_emotional_keyword,
            },
            "total_score": round(highlight.total_score, 3),
            **metadata,
        }


class RenderStage:
    """Stage 6: Render clips in parallel and write metadata."""

    name = "Rendering Clips"
    metric_key = "clip_render_time"

    def __init__(self, cfg: AppConfig, paths: ClipPaths) -> None:
        self.cfg = cfg
        self.paths = paths

    def _create_processor(self) -> ClipProcessor:
        return ClipProcessor(
            cfg=self.cfg,
            paths=self.paths,
            clipper=FFmpegClipper(self.cfg.ffmpeg_preset),
            captioner=CaptionGenerator(),
            renderer=OverlayRenderer(),
            creative_gen=HookThumbnailGenerator(),
            title_gen=TitleGenerator(),
        )

    def run(self, ctx: PipelineContext) -> PipelineContext:
        assert ctx.video_path is not None
        assert ctx.transcript is not None
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

        if not ctx.planned_clips:
            info("No clips planned; skipping rendering")
            ctx.metadata_entries = []
            return ctx

        metadata_entries: list[dict] = []
        with ThreadPoolExecutor(max_workers=self.cfg.max_parallel_clips) as executor:
            futures = [
                executor.submit(
                    self._create_processor().process_clip,
                    planned,
                    ctx.video_path,
                    ctx.transcript,
                    ctx.transcript_text,
                    ctx.vf_filter,
                )
                for planned in ctx.planned_clips
            ]
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
            ) as progress:
                task = progress.add_task("Rendering clips...", total=len(futures))
                for future in as_completed(futures):
                    try:
                        metadata_entries.append(future.result())
                    except Exception as exc:  # noqa: BLE001
                        error(f"Clip failed: {exc}")
                    finally:
                        progress.update(task, advance=1)

        metadata_entries.sort(
            key=lambda item: float(item.get("total_score", item["viral_score"]["total"])),
            reverse=True,
        )
        metadata_path = self.cfg.output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata_entries, indent=2), encoding="utf-8")
        ctx.metadata_entries = metadata_entries
        return ctx


class PipelineRunner:
    """CPU-optimized stage-oriented clip generation pipeline."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.work_dir = cfg.output_dir / "work"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.paths = ClipPaths(cfg.output_dir)

        self.stages = [
            DownloadStage(YouTubeDownloader(self.work_dir)),
            AudioExtractionStage(self.work_dir),
            TranscriptionStage(
                transcriber=WhisperTranscriber(cfg.model_size),
                cache_dir=cfg.cache_dir,
                model_size=cfg.model_size,
            ),
            HighlightDetectionStage(
                detector=HighlightDetector(
                    min_seconds=cfg.clip_min_seconds,
                    max_seconds=cfg.clip_max_seconds,
                ),
                max_clips=cfg.max_clips,
            ),
            ClipPlanningStage(
                cfg=cfg,
                detector=SceneDetector(),
                reframer=AutoReframer(),
            ),
            RenderStage(cfg, self.paths),
        ]

    def run(self, url: str) -> None:
        """Execute the optimized 6-stage pipeline."""
        ctx = PipelineContext(url=url)
        header("YouTube Auto Clip Pipeline")
        try:
            for pipeline_stage in self.stages:
                stage_start = time.perf_counter()
                stage_name = getattr(pipeline_stage, "name", pipeline_stage.__class__.__name__)
                stage(stage_name)
                ctx = pipeline_stage.run(ctx)
                elapsed = time.perf_counter() - stage_start
                metric_key = getattr(pipeline_stage, "metric_key", None)
                if metric_key:
                    ctx.metrics[metric_key] = round(elapsed, 3)
                if stage_name == "Highlight Detection":
                    success(f"Found {len(ctx.highlights)} clips")
                elif stage_name == "Rendering Clips":
                    success(f"Completed ({elapsed:.1f}s)")
                elif stage_name == "Transcription":
                    success("Completed")
                else:
                    success(f"Completed ({elapsed:.1f}s)")
        finally:
            if ctx.audio_path is not None:
                safe_unlink(ctx.audio_path)

        timing_summary = {
            "download_time": float(ctx.metrics.get("download_time", 0.0)),
            "transcription_time": float(ctx.metrics.get("transcription_time", 0.0)),
            "highlight_detection_time": float(ctx.metrics.get("highlight_detection_time", 0.0)),
            "clip_render_time": float(ctx.metrics.get("clip_render_time", 0.0)),
        }
        generated_clips = len(ctx.metadata_entries)
        if generated_clips > 0:
            cleanup_workdir(self.work_dir)
            info("🧹 Cleaned temporary files")
            zip_path = archive_output(self.cfg.output_dir)
            if zip_path is not None and self.cfg.auto_upload:
                uploaded = False
                try:
                    uploaded = upload_file(zip_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Archive upload failed: %s", exc)
                if uploaded and self.cfg.delete_zip_after_upload and zip_path.exists():
                    safe_unlink(zip_path)
                    info("🗑 Deleted local archive after upload")
        info(f"Timings: {json.dumps(timing_summary, sort_keys=True)}")
        success("Done")
        info(f"Generated {generated_clips} clips in {self.cfg.output_dir}")


def run_pipeline(url: str, cfg: AppConfig) -> None:
    """Backward-compatible pipeline entrypoint."""
    PipelineRunner(cfg).run(url)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single URL or batch mode."""
    parser = argparse.ArgumentParser(description="Convert YouTube videos into vertical shorts.")
    parser.add_argument("url", nargs="?", help="Single YouTube URL")
    parser.add_argument("--input-file", type=Path, help="Text file with one YouTube URL per line")
    return parser.parse_args()


def main() -> None:
    """Command-line wrapper."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.ERROR)
    load_dotenv()
    args = parse_args()

    urls: list[str] = []
    if args.url:
        urls.append(args.url)
    if args.input_file:
        urls.extend(
            [line.strip() for line in args.input_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        )

    if not urls:
        print('Usage: python main.py "YOUTUBE_URL" OR python main.py --input-file urls.txt')
        raise SystemExit(1)

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"Config error: {exc}")
        raise SystemExit(1) from exc

    for idx, url in enumerate(urls, start=1):
        if len(urls) > 1:
            stage(f"Video {idx}/{len(urls)}")
        run_pipeline(url, cfg)


if __name__ == "__main__":
    main()
