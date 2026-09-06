"""Fetch the CubiCasa5k pretrained weights for the segmentation provider.

The CubiCasa5k baseline model (hg_furukawa architecture) is published by the
CubiCasa5k authors; the pretrained checkpoint `model_best_val_loss_var.pkl`
is distributed via the repository's Google Drive link.

Steps this script automates where possible:
  1. pip install the 'ai' extra (torch) if missing.
  2. Download the checkpoint with gdown when available.
  3. Point configs/default.yaml `ai.segmentation_checkpoint` at the file.

Manual fallback if the download fails:
  - Visit https://github.com/CubiCasa/CubiCasa5k and follow the README link to
    the pretrained weights.
  - Save the checkpoint to assets/models/cubicasa5k/model_best_val_loss_var.pkl
  - Set ai.segmentation_checkpoint to that path in configs/default.yaml.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CHECKPOINT_DIR = Path("assets/models/cubicasa5k")
CHECKPOINT = CHECKPOINT_DIR / "model_best_val_loss_var.pkl"
GDRIVE_FILE_ID = "1gRB7ez1e2-1KYFVSAdw5eArAdRZAX-Vw"  # from the CubiCasa5k README


def main() -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch is not installed; run: pip install -e .[ai]")
        return 1
    if CHECKPOINT.exists():
        print(f"checkpoint already present: {CHECKPOINT}")
        return 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import gdown  # type: ignore[import-not-found]
    except ImportError:
        print("gdown is not installed; run: pip install gdown, or download manually (see module docstring)")
        return 1
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print(f"downloading CubiCasa5k checkpoint from {url}")
    gdown.download(url, str(CHECKPOINT), quiet=False)
    if not CHECKPOINT.exists():
        print("download failed; follow the manual steps in the module docstring")
        return 1
    print(f"saved: {CHECKPOINT}")
    print("set ai.segmentation_checkpoint in configs/default.yaml to enable the provider")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _self_check() -> None:  # pragma: no cover - convenience for CI linting
    subprocess.run([sys.executable, "-c", "import architecture_walkthrough"], check=False)
