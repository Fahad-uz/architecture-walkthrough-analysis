from __future__ import annotations

from pathlib import Path

from architecture_walkthrough.scene.blender_runner import blender_background_command
from architecture_walkthrough.walkthrough.render_video import ffmpeg_frames_to_mp4_command


def test_blender_command_uses_argument_list() -> None:
    assert blender_background_command("blender", Path("script.py")) == ["blender", "--background", "--python", "script.py"]


def test_ffmpeg_command_uses_argument_list() -> None:
    command = ffmpeg_frames_to_mp4_command("ffmpeg", Path("frames/frame_%04d.png"), Path("out.mp4"), 24)
    assert command[0] == "ffmpeg"
    assert "out.mp4" == command[-1]
    assert "-i" in command
