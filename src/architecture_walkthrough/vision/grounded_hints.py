from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from architecture_walkthrough.geometry.models import Point2D, WallSegment


@dataclass(frozen=True)
class GroundedWallHints:
    """Vision-model wall proposals that were verified against local pixels."""

    walls: list[WallSegment]
    rejected_low_support: int = 0
    rejected_non_axis: int = 0
    rejected_off_envelope: int = 0
    debug_overlay_path: Path | None = None


def _axis_orientation(start: Point2D, end: Point2D, angle_tolerance_deg: float) -> str | None:
    dx = abs(end.x - start.x)
    dy = abs(end.y - start.y)
    if max(dx, dy) < 3.0:
        return None
    tangent = float(np.tan(np.deg2rad(angle_tolerance_deg)))
    if dy <= max(2.0, dx * tangent):
        return "h"
    if dx <= max(2.0, dy * tangent):
        return "v"
    return None


def _nearest_supported(target: int, supported: np.ndarray, lower: int, upper: int) -> int:
    candidates = supported[(supported >= lower) & (supported <= upper)]
    if not len(candidates):
        return target
    return int(candidates[np.argmin(np.abs(candidates - target))])


def _ground_horizontal(
    mask: np.ndarray,
    start: Point2D,
    end: Point2D,
    corridor_px: int,
) -> tuple[Point2D, Point2D, float] | None:
    height, width = mask.shape[:2]
    x0, x1 = sorted((int(round(start.x)), int(round(end.x))))
    y = int(round((start.y + end.y) / 2))
    x0 = max(0, min(width - 1, x0))
    x1 = max(0, min(width - 1, x1))
    if x1 - x0 < 3:
        return None
    y0, y1 = max(0, y - corridor_px), min(height - 1, y + corridor_px)
    roi = mask[y0 : y1 + 1, x0 : x1 + 1] > 0
    support_by_x = roi.any(axis=0)
    support = float(np.mean(support_by_x))
    row_scores = roi.sum(axis=1)
    if not np.any(row_scores):
        return None
    # Hollow CAD walls produce two strong parallel strokes.  A weighted mean
    # lands on their centerline; a filled wall naturally lands in its middle.
    rows = np.arange(y0, y1 + 1, dtype=float)
    snapped_y = float(np.average(rows, weights=np.maximum(row_scores, 1)))
    supported_x = np.flatnonzero(support_by_x) + x0
    snap_radius = corridor_px * 2
    snapped_x0 = _nearest_supported(x0, supported_x, max(0, x0 - snap_radius), min(width - 1, x0 + snap_radius))
    snapped_x1 = _nearest_supported(x1, supported_x, max(0, x1 - snap_radius), min(width - 1, x1 + snap_radius))
    if snapped_x1 <= snapped_x0:
        snapped_x0, snapped_x1 = x0, x1
    return Point2D(x=float(snapped_x0), y=snapped_y), Point2D(x=float(snapped_x1), y=snapped_y), support


def _ground_vertical(
    mask: np.ndarray,
    start: Point2D,
    end: Point2D,
    corridor_px: int,
) -> tuple[Point2D, Point2D, float] | None:
    height, width = mask.shape[:2]
    y0, y1 = sorted((int(round(start.y)), int(round(end.y))))
    x = int(round((start.x + end.x) / 2))
    y0 = max(0, min(height - 1, y0))
    y1 = max(0, min(height - 1, y1))
    if y1 - y0 < 3:
        return None
    x0, x1 = max(0, x - corridor_px), min(width - 1, x + corridor_px)
    roi = mask[y0 : y1 + 1, x0 : x1 + 1] > 0
    support_by_y = roi.any(axis=1)
    support = float(np.mean(support_by_y))
    column_scores = roi.sum(axis=0)
    if not np.any(column_scores):
        return None
    columns = np.arange(x0, x1 + 1, dtype=float)
    snapped_x = float(np.average(columns, weights=np.maximum(column_scores, 1)))
    supported_y = np.flatnonzero(support_by_y) + y0
    snap_radius = corridor_px * 2
    snapped_y0 = _nearest_supported(y0, supported_y, max(0, y0 - snap_radius), min(height - 1, y0 + snap_radius))
    snapped_y1 = _nearest_supported(y1, supported_y, max(0, y1 - snap_radius), min(height - 1, y1 + snap_radius))
    if snapped_y1 <= snapped_y0:
        snapped_y0, snapped_y1 = y0, y1
    return Point2D(x=snapped_x, y=float(snapped_y0)), Point2D(x=snapped_x, y=float(snapped_y1)), support


def ground_ai_wall_hints(
    hints: Any,
    structural_mask: np.ndarray,
    image_width_px: int,
    image_height_px: int,
    *,
    internal_thickness_m: float,
    external_thickness_m: float,
    wall_height_m: float,
    min_hint_confidence: float = 0.55,
    min_ink_support: float = 0.48,
    angle_tolerance_deg: float = 10.0,
    corridor_ratio: float = 0.018,
    external_envelope_tolerance_ratio: float = 0.045,
    debug_dir: Path | None = None,
) -> GroundedWallHints:
    """Accept AI wall proposals only when local structural ink supports them.

    The model proposes *where to look*; the accepted centerline is snapped to
    actual image pixels.  This recovers fragmented thin CAD walls without
    allowing an LLM coordinate to become unverified building geometry.
    """

    if hints is None or structural_mask is None:
        return GroundedWallHints(walls=[])
    if structural_mask.ndim != 2:
        raise ValueError("structural mask must be a single-channel image")

    corridor_px = max(4, int(round(max(image_width_px, image_height_px) * corridor_ratio)))
    accepted: list[WallSegment] = []
    rejected_low_support = 0
    rejected_non_axis = 0
    rejected_off_envelope = 0
    rejected_lines: list[tuple[Point2D, Point2D]] = []
    ink_y, ink_x = np.nonzero(structural_mask)
    ink_bounds = (
        (float(ink_x.min()), float(ink_y.min()), float(ink_x.max()), float(ink_y.max()))
        if len(ink_x)
        else None
    )
    envelope_tolerance = max(6.0, max(image_width_px, image_height_px) * external_envelope_tolerance_ratio)

    for hint in getattr(hints, "walls", []):
        if float(getattr(hint, "confidence", 0.0)) < min_hint_confidence:
            continue
        start = Point2D(x=hint.start.x * image_width_px, y=hint.start.y * image_height_px)
        end = Point2D(x=hint.end.x * image_width_px, y=hint.end.y * image_height_px)
        orientation = _axis_orientation(start, end, angle_tolerance_deg)
        if orientation is None:
            rejected_non_axis += 1
            rejected_lines.append((start, end))
            continue
        grounded = (
            _ground_horizontal(structural_mask, start, end, corridor_px)
            if orientation == "h"
            else _ground_vertical(structural_mask, start, end, corridor_px)
        )
        if grounded is None or grounded[2] < min_ink_support:
            rejected_low_support += 1
            rejected_lines.append((start, end))
            continue
        grounded_start, grounded_end, support = grounded
        external = bool(getattr(hint, "external", False))
        if external and ink_bounds is not None:
            ink_min_x, ink_min_y, ink_max_x, ink_max_y = ink_bounds
            if orientation == "h":
                coordinate = (grounded_start.y + grounded_end.y) / 2.0
                on_envelope = min(abs(coordinate - ink_min_y), abs(coordinate - ink_max_y)) <= envelope_tolerance
            else:
                coordinate = (grounded_start.x + grounded_end.x) / 2.0
                on_envelope = min(abs(coordinate - ink_min_x), abs(coordinate - ink_max_x)) <= envelope_tolerance
            if not on_envelope:
                rejected_off_envelope += 1
                rejected_lines.append((start, end))
                continue
        hint_confidence = float(hint.confidence)
        accepted.append(
            WallSegment(
                id=f"grounded_{len(accepted):03d}",
                start=grounded_start,
                end=grounded_end,
                thickness_m=external_thickness_m if external else internal_thickness_m,
                height_m=wall_height_m,
                external=external,
                wall_type="external" if external else "internal",
                confidence=max(0.1, min(1.0, (hint_confidence + support) / 2)),
                evidence_source="ai_proposal+local_structural_ink",
            )
        )

    overlay_path: Path | None = None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        overlay = cv2.cvtColor(structural_mask, cv2.COLOR_GRAY2BGR)
        for start, end in rejected_lines:
            cv2.line(
                overlay,
                (int(round(start.x)), int(round(start.y))),
                (int(round(end.x)), int(round(end.y))),
                (40, 40, 180),
                1,
            )
        for wall in accepted:
            cv2.line(
                overlay,
                (int(round(wall.start.x)), int(round(wall.start.y))),
                (int(round(wall.end.x)), int(round(wall.end.y))),
                (40, 210, 40),
                2,
            )
        overlay_path = debug_dir / "grounded_wall_hints.png"
        cv2.imwrite(str(overlay_path), overlay)

    return GroundedWallHints(
        walls=accepted,
        rejected_low_support=rejected_low_support,
        rejected_non_axis=rejected_non_axis,
        rejected_off_envelope=rejected_off_envelope,
        debug_overlay_path=overlay_path,
    )
