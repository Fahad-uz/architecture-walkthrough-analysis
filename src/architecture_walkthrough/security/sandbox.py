from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def require_executable(executable: str, purpose: str) -> str:
    resolved = shutil.which(executable)
    if not resolved:
        raise RuntimeError(f"{purpose} executable not found on PATH: {executable}")
    return resolved


def run_subprocess(args: list[str], timeout_seconds: int, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    if not args:
        raise ValueError("subprocess args cannot be empty")
    return subprocess.run(
        args,
        cwd=cwd,
        timeout=timeout_seconds,
        check=True,
        text=True,
        capture_output=True,
        shell=False,
    )
