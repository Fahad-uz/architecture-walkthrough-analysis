from __future__ import annotations

import json
from pathlib import Path

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.security.sandbox import require_executable, run_subprocess


def ffmpeg_frames_to_mp4_command(ffmpeg_executable: str, frames_pattern: Path, output_mp4: Path, fps: int) -> list[str]:
    return [
        ffmpeg_executable,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_pattern),
        "-pix_fmt",
        "yuv420p",
        str(output_mp4),
    ]


def encode_frames_to_mp4(frames_pattern: Path, output_mp4: Path, config: AppConfig) -> Path:
    executable = require_executable(str(config.paths.ffmpeg_executable), "FFmpeg")
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    progress = output_mp4.with_suffix(".progress.json")
    progress.write_text(json.dumps({"stage": "encoding", "output": str(output_mp4)}), encoding="utf-8")
    run_subprocess(
        ffmpeg_frames_to_mp4_command(executable, frames_pattern, output_mp4, config.render.preview_fps),
        timeout_seconds=config.limits.subprocess_timeout_seconds,
    )
    progress.write_text(json.dumps({"stage": "complete", "output": str(output_mp4)}), encoding="utf-8")
    return output_mp4
