"""CLI entrypoint for YT-Short-Clipper."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from ai.highlight_detector import HighlightDetector
from ai.hook_thumbnail_generator import HookThumbnailGenerator
from ai.title_generator import TitleGenerator
from caption.caption_generator import CaptionGenerator
from config import AppConfig, ConfigError, load_config
from downloader.youtube_downloader import YouTubeDownloader
from transcription.whisper_transcriber import WhisperTranscriber, transcript_to_text
from video.clipper import FFmpegClipper
from video.overlay_renderer import OverlayRenderer
from video.reframer import AutoReframer
from video.scene_detector import SceneDetector


def extract_audio(video_path: Path, output_audio_path: Path) -> None:
    """Extract mono 16k wav audio for speech recognition."""
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


def _process_clip(
    idx: int,
    highlight,
    cfg: AppConfig,
    video_path: Path,
    transcript,
    transcript_text: str,
    scenes,
    vf: str,
    title_gen: TitleGenerator,
    hook_thumb_gen: HookThumbnailGenerator,
) -> dict:
    start, end = SceneDetector.align_range(highlight.start, highlight.end, scenes)
    base_clip = cfg.output_dir / f"clip_{idx}_base.mp4"
    final_clip = cfg.output_dir / f"clip_{idx}.mp4"
    thumb_path = cfg.output_dir / f"clip_{idx}_thumbnail.jpg"
    srt_path = cfg.output_dir / f"clip_{idx}.srt"

    clipper = FFmpegClipper(cfg.ffmpeg_preset)
    captioner = CaptionGenerator()
    renderer = OverlayRenderer()

    clipper.extract_clip(video_path, base_clip, start, end, vf)
    captioner.write_srt(transcript, start, end, srt_path)

    creative = hook_thumb_gen.generate(
        clip_context=f"{highlight.rationale}\n{highlight.hook}", fallback_hook=highlight.hook
    )
    renderer.add_hook_and_captions(base_clip, final_clip, creative["hook_text"], srt_path)
    renderer.generate_thumbnail(final_clip, thumb_path, creative["thumbnail_text"])

    metadata = title_gen.generate(
        f"Hook: {creative['hook_text']}\nRationale: {highlight.rationale}\nTranscript:\n{transcript_text}"
    )

    if base_clip.exists():
        base_clip.unlink()

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


def run_pipeline(url: str, cfg: AppConfig) -> None:
    """Execute end-to-end clipping pipeline for one URL."""
    work_dir = cfg.output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    steps = tqdm(total=7, desc="Pipeline", unit="step")

    downloader = YouTubeDownloader(work_dir)
    video_path = downloader.download(url)
    steps.update(1)

    audio_path = work_dir / f"{video_path.stem}.wav"
    extract_audio(video_path, audio_path)
    steps.update(1)

    transcriber = WhisperTranscriber(cfg.model_size)
    cache_path = cfg.cache_dir / "transcripts" / f"{video_path.stem}_{cfg.model_size}.json"
    transcript = transcriber.load_cached(cache_path)
    if transcript is None:
        transcript = transcriber.transcribe(audio_path)
        transcriber.save_cached(cache_path, transcript)
    transcript_text = transcript_to_text(transcript)
    steps.update(1)

    highlights = HighlightDetector(
        cfg.openai_api_key,
        min_seconds=cfg.clip_min_seconds,
        max_seconds=cfg.clip_max_seconds,
    ).detect(transcript_text, max_clips=cfg.max_clips)
    steps.update(1)

    scenes = SceneDetector().detect_boundaries(video_path)
    reframer = AutoReframer()
    subject_x = reframer.detect_subject_x_ratio(video_path)
    vf = reframer.crop_filter(subject_x, cfg.target_width, cfg.target_height)
    steps.update(1)

    title_gen = TitleGenerator(cfg.openai_api_key)
    hook_thumb_gen = HookThumbnailGenerator(cfg.openai_api_key)

    metadata_entries: list[dict] = []
    with ThreadPoolExecutor(max_workers=cfg.batch_workers) as executor:
        futures = [
            executor.submit(
                _process_clip,
                idx,
                h,
                cfg,
                video_path,
                transcript,
                transcript_text,
                scenes,
                vf,
                title_gen,
                hook_thumb_gen,
            )
            for idx, h in enumerate(highlights, start=1)
        ]
        for future in tqdm(futures, desc="Rendering clips", unit="clip"):
            metadata_entries.append(future.result())
    steps.update(1)

    metadata_entries.sort(key=lambda item: item["viral_score"]["total"], reverse=True)
    metadata_path = cfg.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata_entries, indent=2), encoding="utf-8")
    steps.update(1)
    steps.close()

    print(f"Done. Generated {len(metadata_entries)} clips in {cfg.output_dir}")


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
