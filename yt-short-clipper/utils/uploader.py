"""Cloud upload helpers for generated archives."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from utils.logger import info


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def upload_file(path: Path) -> bool:
    """Upload a file with rclone using env-configured remote settings."""
    auto_upload = _is_true(os.getenv("AUTO_UPLOAD", "false"))
    if not auto_upload:
        return False

    provider = os.getenv("UPLOAD_PROVIDER", "").strip()
    remote_folder = os.getenv("UPLOAD_PATH", "").strip()
    if not provider or not remote_folder:
        raise ValueError("AUTO_UPLOAD=true requires UPLOAD_PROVIDER and UPLOAD_PATH.")

    remote = f"{provider}:{remote_folder}"
    cmd = ["rclone", "copy", str(path), remote]
    subprocess.run(cmd, check=True)
    info(f"☁ Uploaded archive to {remote}")
    return True

