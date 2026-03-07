# YouTube Shorts Auto-Clip Generator

A Python pipeline that converts long YouTube videos into short, vertical clips optimized for Shorts/Reels-style publishing.

# Features

- AI highlight detection from transcript context
- Whisper-based transcription workflow
- Automatic vertical reframing for portrait output
- Dynamic zoom styling for clip energy
- Subtitle generation and burn-in support
- Hook text overlays for stronger retention
- Automatic ZIP archive creation of outputs
- Optional cloud upload (Google Drive)

# Architecture

```text
YouTube Video
↓
Download
↓
Audio Extraction
↓
Whisper Transcription
↓
Gemini Highlight Detection
↓
Clip Planning
↓
Cinematic Reframing
↓
Rendering
↓
Archive + Upload
```

# Installation

```bash
git clone <REPO_URL>
cd yt-short-clipper
pip install -r requirements.txt
```

# Configuration

Create a `.env` file (or copy from `.env.example`) and set required values, including API credentials and runtime options.

Example:

```env
GEMINI_API_KEY=your_key_here
WHISPER_MODEL_SIZE=tiny
MAX_CLIPS=15
FFMPEG_PRESET=veryfast
ENABLE_SUBTITLES=true
AUTO_UPLOAD=false
```

# Usage

```bash
python main.py "YOUTUBE_URL"
```

# Output

Generated artifacts are written to the output directory.

Example:

```text
clips/
clip_1.mp4
clip_1_thumbnail.jpg
metadata.json
shorts_archive.zip
```

# Requirements

- Python 3.11+
- FFmpeg installed and available in `PATH`

# License

MIT (placeholder)
