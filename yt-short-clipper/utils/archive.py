"""Archive helpers for packaging generated clip outputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import uuid
from zipfile import ZIP_DEFLATED, ZipFile

from utils.logger import info

ARCHIVE_EXTENSIONS = {".mp4", ".srt", ".jpg", ".json"}


def _unique_archive_name() -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"shorts_{timestamp}_{rand}.zip"


def archive_output(output_dir: Path) -> Path | None:
    """Zip generated outputs and remove original packaged files.

    Returns the created archive path, or None if nothing was archived.
    """
    output_path = Path(output_dir)
    files_to_archive = [
        file_path
        for file_path in output_path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in ARCHIVE_EXTENSIONS
    ]

    # Only archive when clips exist.
    if not any(file_path.suffix.lower() == ".mp4" for file_path in files_to_archive):
        return None

    archive_path = output_path / _unique_archive_name()
    while archive_path.exists():
        archive_path = output_path / _unique_archive_name()

    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for file_path in files_to_archive:
            archive.write(file_path, arcname=file_path.name)

    for file_path in files_to_archive:
        file_path.unlink()

    info(f"📦 Created archive: {archive_path.name}")
    info("🧹 Cleaned original files")
    return archive_path
