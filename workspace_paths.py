"""
Persistent data paths on the /workspace network volume (RunPod).

Override any path via environment variable. All directories are created on demand.
"""

import os
from pathlib import Path

WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE_ROOT", "/workspace"))

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", str(WORKSPACE_ROOT / "uploads"))
OUTPUT_2D = os.environ.get("OUTPUT_2D", str(WORKSPACE_ROOT / "unlabeled"))
OUTPUT_3D = os.environ.get("OUTPUT_3D", str(WORKSPACE_ROOT / "3d_outputs"))
WORKSPACE_TMP = os.environ.get("WORKSPACE_TMP", str(WORKSPACE_ROOT / "tmp"))
WORKSPACE_BLOG_DIR = os.environ.get("WORKSPACE_BLOG_DIR", str(WORKSPACE_ROOT / "blog"))


def default_database_url():
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    db_file = Path(WORKSPACE_BLOG_DIR) / "blog.db"
    return f"sqlite:////{db_file.as_posix().lstrip('/')}"


def ensure_workspace_dirs():
    for path in (WORKSPACE_ROOT, UPLOAD_DIR, OUTPUT_2D, OUTPUT_3D, WORKSPACE_TMP, WORKSPACE_BLOG_DIR):
        Path(path).mkdir(parents=True, exist_ok=True)


def tmp_path(filename):
    """Path for transient processing files (still on /workspace by default)."""
    ensure_workspace_dirs()
    return str(Path(WORKSPACE_TMP) / filename)
