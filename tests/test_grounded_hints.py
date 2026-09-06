from __future__ import annotations

import numpy as np

from architecture_walkthrough.ai.floorplan_vision import (
    AIWallHint,
    FloorPlanVisionHints,
    NormalizedPoint,
)
from architecture_walkthrough.vision.grounded_hints import ground_ai_wall_hints


def _hint(x1: float, y1: float, x2: float, y2: float) -> AIWallHint:
    return AIWallHint(
        start=NormalizedPoint(x=x1, y=y1),
        end=NormalizedPoint(x=x2, y=y2),
        confidence=0.9,
    )


def test_ai_wall_proposals_require_local_ink_and_snap_to_wall_center() -> None:
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[39:41, 10:103] = 255
    mask[43:45, 10:103] = 255
    hints = FloorPlanVisionHints(
        walls=[
            _hint(0.08, 0.42, 0.86, 0.42),
            _hint(0.50, 0.58, 0.50, 0.95),  # hallucinated: no vertical ink
        ]
    )

    result = ground_ai_wall_hints(
        hints,
        mask,
        120,
        100,
        internal_thickness_m=0.12,
        external_thickness_m=0.20,
        wall_height_m=3.0,
        corridor_ratio=0.04,
    )

    assert len(result.walls) == 1
    assert result.rejected_low_support == 1
    assert result.walls[0].evidence_source == "ai_proposal+local_structural_ink"
    assert 40.0 <= result.walls[0].start.y <= 44.0
    assert result.walls[0].start.y == result.walls[0].end.y


def test_external_ai_wall_must_lie_on_local_ink_envelope() -> None:
    mask = np.zeros((120, 160), dtype=np.uint8)
    mask[10:13, 10:151] = 255
    mask[55:58, 10:151] = 255  # hatch/detail line, not an exterior wall
    mask[105:108, 10:151] = 255
    hints = FloorPlanVisionHints(
        walls=[
            AIWallHint(
                start=NormalizedPoint(x=0.08, y=0.47),
                end=NormalizedPoint(x=0.92, y=0.47),
                confidence=0.95,
                external=True,
            )
        ]
    )

    result = ground_ai_wall_hints(
        hints,
        mask,
        160,
        120,
        internal_thickness_m=0.12,
        external_thickness_m=0.20,
        wall_height_m=3.0,
        corridor_ratio=0.02,
        external_envelope_tolerance_ratio=0.08,
    )

    assert result.walls == []
    assert result.rejected_off_envelope == 1
