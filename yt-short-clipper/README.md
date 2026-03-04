# YT-Short-Clipper

Production-ready Python pipeline that converts long YouTube videos into vertical short clips for YouTube Shorts, TikTok, and Instagram Reels.

## Features

- Download source video via `yt-dlp`
- Transcribe with `faster-whisper`
- **Transcript caching** (`.cache/transcripts`) to skip re-transcription on repeated runs
- Detect viral highlights with OpenAI semantic analysis
- **Viral score system (1–100)**: curiosity, emotional intensity, educational value, shock factor, plus total score
- **Multi-clip strategy up to 15 clips** per source video
- Align highlights with scene boundaries via `PySceneDetect`
- Auto-reframe speaker for 9:16 using OpenCV + MediaPipe face detection
- **AI hook generator** for first 2 seconds of each clip
- Generate dynamic captions and burn them into clips
- **Auto thumbnail generator** (`clip_n_thumbnail.jpg`)
- Generate title/description/hashtags metadata for each clip
- CLI **progress bars** for videos, pipeline steps, and clip rendering
- Batch mode from URL file + threaded clip rendering for faster processing

## Project Structure

```text
yt-short-clipper/
├── main.py
├── config.py
├── requirements.txt
├── .env.example
├── downloader/
│   └── youtube_downloader.py
├── transcription/
│   └── whisper_transcriber.py
├── ai/
│   ├── highlight_detector.py
│   ├── hook_thumbnail_generator.py
│   └── title_generator.py
├── video/
│   ├── scene_detector.py
│   ├── clipper.py
│   ├── reframer.py
│   └── overlay_renderer.py
├── caption/
│   └── caption_generator.py
├── utils/
│   └── time_utils.py
└── output/
```

## Requirements

- Python 3.11+
- FFmpeg installed and available in PATH

## Installation

```bash
cd yt-short-clipper
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Set your OpenAI API key in `.env` (or export as env var).

## Usage

Single video:

```bash
export OPENAI_API_KEY="sk-..."
python main.py "https://www.youtube.com/watch?v=..."
```

Batch processing:

```bash
python main.py --input-file urls.txt
```

`urls.txt` should contain one URL per line.

## Output

The pipeline writes files to `output/`:

- `clip_1.mp4`, ..., `clip_15.mp4` (depending on highlights found)
- `clip_1_thumbnail.jpg`, ...
- `metadata.json`

`metadata.json` contains entries like:

```json
[
  {
    "clip": "clip_1.mp4",
    "thumbnail": "clip_1_thumbnail.jpg",
    "start": 12.3,
    "end": 38.9,
    "hook": "You won't believe this",
    "thumbnail_text": "Big reveal",
    "viral_score": {
      "curiosity": 84,
      "emotional_intensity": 78,
      "educational_value": 62,
      "shock_factor": 75,
      "total": 75
    },
    "title": "...",
    "description": "...",
    "hashtags": ["#shorts", "#viral"]
  }
]
```

## Notes for Production

- For GPU transcription, update `WhisperModel(... device='cuda' ...)`.
- For speed, FFmpeg uses `-preset veryfast` and threaded clip rendering (`BATCH_WORKERS`).
- Ensure legal rights for downloaded video reuse.
