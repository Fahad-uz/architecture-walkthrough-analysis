from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceLimits:
    timeout_seconds: int
    max_frames: int
    render_width: int
    render_height: int

    def validate_frame_count(self, frame_count: int) -> None:
        if frame_count <= 0 or frame_count > self.max_frames:
            raise ValueError("requested frame count exceeds configured limit")
