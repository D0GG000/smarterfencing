#!/usr/bin/env python3
"""One-time setup for ODTrack (clone vendor + download weights + pip deps)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

FENCING_DIR = Path(__file__).resolve().parent
ODTRACK_ROOT = FENCING_DIR / "vendor" / "odtrack"
MODELS_DIR = FENCING_DIR / "models" / "odtrack"
CHECKPOINT_NAME = "ODTrack_ep0300.pth.tar"
GDRIVE_FOLDER = "https://drive.google.com/drive/folders/17LacrfRO01R75bxU4bgA87eo1b_rX5Gj"
MIN_CHECKPOINT_BYTES = 300_000_000


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(">", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def clone_odtrack() -> None:
    if ODTRACK_ROOT.is_dir():
        print(f"ODTrack already at {ODTRACK_ROOT}", flush=True)
        return
    ODTRACK_ROOT.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/GXNU-ZhongLab/ODTrack.git",
            str(ODTRACK_ROOT),
        ]
    )


def _patch_vendor() -> None:
    """Patches for modern PyTorch / optional jpeg4py."""
    loader_py = ODTRACK_ROOT / "lib" / "train" / "data" / "loader.py"
    if loader_py.is_file():
        text = loader_py.read_text(encoding="utf-8")
        if "torch._six" in text and "except ImportError" not in text:
            text = text.replace(
                "from torch._six import string_classes",
                "try:\n    from torch._six import string_classes\nexcept ImportError:\n    string_classes = str",
            )
            text = text.replace(
                "    from torch._six import int_classes",
                "    try:\n        from torch._six import int_classes\n    except ImportError:\n        int_classes = int",
            )
            loader_py.write_text(text, encoding="utf-8")

    img_loader = ODTRACK_ROOT / "lib" / "train" / "data" / "image_loader.py"
    if img_loader.is_file():
        text = img_loader.read_text(encoding="utf-8")
        if "jpeg4py = None" not in text:
            text = text.replace(
                "import jpeg4py",
                "try:\n    import jpeg4py\nexcept ImportError:\n    jpeg4py = None  # type: ignore",
                1,
            )
            text = text.replace(
                'def jpeg4py_loader(path):\n    """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""\n    try:',
                'def jpeg4py_loader(path):\n    """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""\n    if jpeg4py is None:\n        return opencv_loader(path)\n    try:',
                1,
            )
            img_loader.write_text(text, encoding="utf-8")


def write_local_py() -> None:
    local_py = ODTRACK_ROOT / "lib" / "test" / "evaluation" / "local.py"
    prj = str(ODTRACK_ROOT).replace("\\", "/")
    save = str(FENCING_DIR / "models" / "odtrack").replace("\\", "/")
    local_py.parent.mkdir(parents=True, exist_ok=True)
    local_py.write_text(
        "from lib.test.evaluation.environment import EnvSettings\n\n"
        "def local_env_settings():\n"
        "    settings = EnvSettings()\n"
        f"    settings.prj_dir = r'{prj}'\n"
        f"    settings.save_dir = r'{save}'\n"
        "    return settings\n",
        encoding="utf-8",
    )


def pip_deps() -> None:
    pkgs = ["timm", "easydict", "PyYAML", "pycocotools", "tensorboardX"]
    run([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def _valid_ckpt(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= MIN_CHECKPOINT_BYTES


def download_checkpoint() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / CHECKPOINT_NAME
    if _valid_ckpt(dest):
        print(f"Checkpoint ready: {dest}", flush=True)
        return dest

    try:
        import gdown  # type: ignore
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "gdown>=4.7"])
        import gdown  # type: ignore

    dl_dir = FENCING_DIR / "models" / "_odtrack_dl"
    candidates = list(dl_dir.rglob(CHECKPOINT_NAME)) if dl_dir.is_dir() else []
    if candidates:
        preferred = [p for p in candidates if "Base-Fulldata" in str(p)]
        pick = preferred[0] if preferred else candidates[0]
        if _valid_ckpt(pick):
            shutil.copy2(pick, dest)
            print(f"Checkpoint saved from cache: {dest}", flush=True)
            return dest

    if dl_dir.is_dir():
        shutil.rmtree(dl_dir, ignore_errors=True)
    dl_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading ODTrack weights from Google Drive (~370MB, one-time)...", flush=True)
    gdown.download_folder(GDRIVE_FOLDER, output=str(dl_dir), quiet=False, use_cookies=False)

    candidates = list(dl_dir.rglob(CHECKPOINT_NAME))
    preferred = [p for p in candidates if "Base-Fulldata" in str(p) or "baseline" in str(p).lower()]
    pick = preferred[0] if preferred else (candidates[0] if candidates else None)
    if pick is None or not _valid_ckpt(pick):
        raise FileNotFoundError(
            f"ODTrack checkpoint not found after download. Expected {CHECKPOINT_NAME} under {dl_dir}"
        )

    shutil.copy2(pick, dest)
    print(f"Checkpoint saved: {dest}", flush=True)
    return dest


def main() -> None:
    clone_odtrack()
    _patch_vendor()
    write_local_py()
    pip_deps()
    ckpt = download_checkpoint()
    print("\nODTrack setup complete.", flush=True)
    print(f"  Vendor: {ODTRACK_ROOT}", flush=True)
    print(f"  Weights: {ckpt}", flush=True)


if __name__ == "__main__":
    main()
