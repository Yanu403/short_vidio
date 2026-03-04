"""CLI entrypoint for YT-Short-Clipper."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from tqdm import tqdm

from ai.highlight_detector import HighlightDetector, HighlightSegment
from ai.hook_thumbnail_generator import HookThumbnailGenerator
from ai.title_generator import TitleGenerator
from caption.caption_generator import CaptionGenerator
from config import AppConfig, ConfigError, load_config
from downloader.youtube_downloader import YouTubeDownloader
from transcription.whisper_transcriber import TranscriptSegment, WhisperTranscriber, transcript_to_text
from video.clipper import FFmpegClipper
from video.overlay_renderer import OverlayRenderer
from video.reframer import AutoReframer
from video.scene_detector import SceneDetector

logger = logging.getLogger(__name__)


def safe_unlink(path: Path) -> None:
    """Delete path if it exists."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


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

    def base(self, idx: int) -> Path:
        return self.output_dir / f"clip_{idx}_base.mp4"

    def final(self, idx: int) -> Path:
        return self.output_dir / f"clip_{idx}.mp4"

    def srt(self, idx: int) -> Path:
        return self.output_dir / f"clip_{idx}.srt"

    def thumbnail(self, idx: int) -> Path:
        return self.output_dir / f"clip_{idx}_thumbnail.jpg"


@dataclass(slots=True)
class PipelineContext:
    """Shared pipeline values passed between stages."""

    url: str
    video_path: Path | None = None
    audio_path: Path | None = None
    transcript: list[TranscriptSegment] | None = None
    transcript_text: str = ""
    highlights: list[HighlightSegment] | None = None
    scenes: list[tuple[float, float]] | None = None
    vf_filter: str = ""


class DownloadStage:
    """Download source video."""

    def __init__(self, downloader: YouTubeDownloader) -> None:
        self.downloader = downloader

    def run(self, ctx: PipelineContext) -> PipelineContext:
        validate_video_url(ctx.url)
        ctx.video_path = self.downloader.download(ctx.url)
        return ctx


class AudioExtractionStage:
    """Extract WAV audio for transcription."""

    @staticmethod
    def _extract_audio(video_path: Path, output_audio_path: Path) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
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
    """Transcribe audio with caching."""

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
        return ctx


class HighlightDetectionStage:
    """Detect and rank viral highlights."""

    def __init__(self, detector: HighlightDetector, max_clips: int) -> None:
        self.detector = detector
        self.max_clips = max_clips

    def run(self, ctx: PipelineContext) -> PipelineContext:
        ctx.highlights = self.detector.detect(ctx.transcript_text, max_clips=self.max_clips)
        return ctx


class SceneDetectionStage:
    """Detect scenes and compute crop filter."""

    def __init__(self, detector: SceneDetector, reframer: AutoReframer, target_width: int, target_height: int) -> None:
        self.detector = detector
        self.reframer = reframer
        self.target_width = target_width
        self.target_height = target_height

    def run(self, ctx: PipelineContext) -> PipelineContext:
        assert ctx.video_path is not None
        ctx.scenes = self.detector.detect_boundaries(ctx.video_path)
        subject_x = self.reframer.detect_subject_x_ratio(ctx.video_path)
        ctx.vf_filter = self.reframer.crop_filter(subject_x, self.target_width, self.target_height)
        return ctx


class ClipProcessor:
    """Process one highlight into final clip assets and metadata."""

    def __init__(
        self,
        cfg: AppConfig,
        paths: ClipPaths,
        clipper: FFmpegClipper,
        captioner: CaptionGenerator,
        renderer: OverlayRenderer,
    ) -> None:
        self.cfg = cfg
        self.paths = paths
        self.clipper = clipper
        self.captioner = captioner
        self.renderer = renderer

    def _align(self, highlight: HighlightSegment, scenes: list[tuple[float, float]]) -> tuple[float, float]:
        return SceneDetector.align_range(highlight.start, highlight.end, scenes)

    def _extract_clip(self, video_path: Path, base_clip: Path, start: float, end: float, vf: str) -> None:
        self.clipper.extract_clip(video_path, base_clip, start, end, vf)

    def _write_captions(
        self,
        transcript: list[TranscriptSegment],
        srt_path: Path,
        start: float,
        end: float,
    ) -> None:
        self.captioner.write_srt(transcript, start, end, srt_path)

    def _generate_creative(self, highlight: HighlightSegment) -> dict[str, str]:
        # Thread-local AI clients for thread safety.
        generator = HookThumbnailGenerator(self.cfg.openai_api_key)
        return generator.generate(
            clip_context=f"{highlight.rationale}\n{highlight.hook}", fallback_hook=highlight.hook
        )

    def _render_assets(
        self,
        base_clip: Path,
        final_clip: Path,
        srt_path: Path,
        thumb_path: Path,
        hook_text: str,
        thumbnail_text: str,
    ) -> None:
        self.renderer.add_hook_and_captions(base_clip, final_clip, hook_text, srt_path)
        self.renderer.generate_thumbnail(final_clip, thumb_path, thumbnail_text)

    def _generate_metadata(
        self,
        transcript_text: str,
        rationale: str,
        hook_text: str,
    ) -> dict:
        # Thread-local AI clients for thread safety.
        title_gen = TitleGenerator(self.cfg.openai_api_key)
        return title_gen.generate(
            f"Hook: {hook_text}\nRationale: {rationale}\nTranscript:\n{transcript_text}"
        )

    def process_clip(
        self,
        idx: int,
        highlight: HighlightSegment,
        video_path: Path,
        transcript: list[TranscriptSegment],
        transcript_text: str,
        scenes: list[tuple[float, float]],
        vf: str,
    ) -> dict:
        start, end = self._align(highlight, scenes)
        base_clip = self.paths.base(idx)
        final_clip = self.paths.final(idx)
        thumb_path = self.paths.thumbnail(idx)
        srt_path = self.paths.srt(idx)

        try:
            self._extract_clip(video_path, base_clip, start, end, vf)
            self._write_captions(transcript, srt_path, start, end)
            creative = self._generate_creative(highlight)
            self._render_assets(
                base_clip,
                final_clip,
                srt_path,
                thumb_path,
                creative["hook_text"],
                creative["thumbnail_text"],
            )
            metadata = self._generate_metadata(
                transcript_text,
                highlight.rationale,
                creative["hook_text"],
            )
        finally:
            safe_unlink(base_clip)

        return {
            "clip": final_clip.name,
            "thumbnail": thumb_path.name,
            "start": round(start, 3),
            "end": round(end, 3),
            "hook": creative["hook_text"],
            "thumbnail_text": creative["thumbnail_text"],
            "viral_score": {
                "curiosity": highlight.viral_score.curiosity,
                "emotional_intensity": highlight.viral_score.emotional_intensity,
                "educational_value": highlight.viral_score.educational_value,
                "shock_factor": highlight.viral_score.shock_factor,
                "total": highlight.viral_score.total,
            },
            **metadata,
        }


class ClipProcessingStage:
    """Render clips in parallel and aggregate metadata entries."""

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
        )

    def run(self, ctx: PipelineContext) -> list[dict]:
        assert ctx.video_path is not None
        assert ctx.transcript is not None
        assert ctx.highlights is not None
        assert ctx.scenes is not None

        metadata_entries: list[dict] = []
        with ThreadPoolExecutor(max_workers=self.cfg.batch_workers) as executor:
            futures = [
                executor.submit(
                    self._create_processor().process_clip,
                    idx,
                    highlight,
                    ctx.video_path,
                    ctx.transcript,
                    ctx.transcript_text,
                    ctx.scenes,
                    ctx.vf_filter,
                )
                for idx, highlight in enumerate(ctx.highlights, start=1)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering clips", unit="clip"):
                try:
                    metadata_entries.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    logger.error("Clip processing failed: %s", exc)
                    continue

        return metadata_entries


class MetadataStage:
    """Finalize and write metadata output."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def run(self, entries: list[dict]) -> list[dict]:
        entries.sort(key=lambda item: item["viral_score"]["total"], reverse=True)
        metadata_path = self.output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        return entries


class Pipeline:
    """Stage-oriented clip generation pipeline."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.work_dir = cfg.output_dir / "work"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.paths = ClipPaths(cfg.output_dir)

        self.download_stage = DownloadStage(YouTubeDownloader(self.work_dir))
        self.audio_stage = AudioExtractionStage(self.work_dir)
        self.transcription_stage = TranscriptionStage(
            transcriber=WhisperTranscriber(cfg.model_size),
            cache_dir=cfg.cache_dir,
            model_size=cfg.model_size,
        )
        self.highlight_stage = HighlightDetectionStage(
            detector=HighlightDetector(
                cfg.openai_api_key,
                min_seconds=cfg.clip_min_seconds,
                max_seconds=cfg.clip_max_seconds,
            ),
            max_clips=cfg.max_clips,
        )
        self.scene_stage = SceneDetectionStage(
            detector=SceneDetector(),
            reframer=AutoReframer(),
            target_width=cfg.target_width,
            target_height=cfg.target_height,
        )
        self.clip_stage = ClipProcessingStage(cfg, self.paths)
        self.metadata_stage = MetadataStage(cfg.output_dir)

    def run(self, url: str) -> None:
        """Execute all pipeline stages in original order."""
        ctx = PipelineContext(url=url)
        steps = tqdm(total=7, desc="Pipeline", unit="step")
        try:
            ctx = self.download_stage.run(ctx)
            steps.update(1)

            ctx = self.audio_stage.run(ctx)
            steps.update(1)

            ctx = self.transcription_stage.run(ctx)
            steps.update(1)

            ctx = self.highlight_stage.run(ctx)
            steps.update(1)

            ctx = self.scene_stage.run(ctx)
            steps.update(1)

            metadata_entries = self.clip_stage.run(ctx)
            steps.update(1)

            metadata_entries = self.metadata_stage.run(metadata_entries)
            steps.update(1)
        finally:
            steps.close()
            if ctx.audio_path is not None:
                safe_unlink(ctx.audio_path)

        print(f"Done. Generated {len(metadata_entries)} clips in {self.cfg.output_dir}")


def run_pipeline(url: str, cfg: AppConfig) -> None:
    """Backward-compatible pipeline entrypoint."""
    Pipeline(cfg).run(url)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single URL or batch mode."""
    parser = argparse.ArgumentParser(description="Convert YouTube videos into vertical shorts.")
    parser.add_argument("url", nargs="?", help="Single YouTube URL")
    parser.add_argument("--input-file", type=Path, help="Text file with one YouTube URL per line")
    return parser.parse_args()


def main() -> None:
    """Command-line wrapper."""
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

    for url in tqdm(urls, desc="Videos", unit="video"):
        run_pipeline(url, cfg)


if __name__ == "__main__":
    main()
