from __future__ import annotations

from pathlib import Path

from architecture_walkthrough.security.sandbox import require_executable, run_subprocess


def blender_background_command(blender_executable: str, script_path: Path) -> list[str]:
    return [blender_executable, "--background", "--python", str(script_path)]


def run_blender_script(blender_executable: str, script_path: Path, timeout_seconds: int) -> None:
    executable = require_executable(blender_executable, "Blender")
    run_subprocess(blender_background_command(executable, script_path), timeout_seconds=timeout_seconds)
