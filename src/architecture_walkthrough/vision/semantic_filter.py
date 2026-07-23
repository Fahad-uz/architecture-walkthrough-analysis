from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from architecture_walkthrough.geometry.constraints import is_horizontal, wall_length
from architecture_walkthrough.geometry.models import Point2D, WallSegment
from architecture_walkthrough.vision.ocr import OCRText
from architecture_walkthrough.vision.wall_detection import RepetitiveDetailRegion


NON_STRUCTURAL_TOKENS = (
    "appliance",
    "balcony",
    "bed",
    "cabinet",
    "chair",
    "counter",
    "fixture",
    "island",
    "plant",
    "railing",
    "rug",
    "shelf",
    "sink",
    "sofa",
    "stair",
    "stove",
    "table",
    "toilet",
    "tv",
    "wardrobe",
)


@dataclass(frozen=True)
class SemanticExclusion:
    category: str
    center: Point2D
    width_px: float
    depth_px: float
    rotation_deg: float

    def contains(self, point: Point2D, margin_px: float) -> bool:
        angle = math.radians(-self.rotation_deg)
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        local_x = dx * math.cos(angle) - dy * math.sin(angle)
        local_y = dx * math.sin(angle) + dy * math.cos(angle)
        return (
            abs(local_x) <= self.width_px / 2 + margin_px
            and abs(local_y) <= self.depth_px / 2 + margin_px
        )


@dataclass(frozen=True)
class SemanticWallFilterResult:
    walls: list[WallSegment]
    rejected: list[WallSegment]


def filter_walls_near_colored_objects(
    walls: list[WallSegment],
    colored_object_mask: np.ndarray | None,
    *,
    coverage_threshold: float = 0.52,
    dilation_ratio: float = 0.0125,
    preserve_long_external: bool = True,
) -> SemanticWallFilterResult:
    """Remove wall candidates traced from local colored object outlines.

    The mask is intentionally dilated a little because the colored fill and
    its black/blue outline are adjacent, not coincident.  A wall crossing an
    object only briefly remains intact; a line following most of a sofa,
    counter, bed or hatch is rejected.  Long envelope walls are protected even
    when a balcony or fitted counter touches them.
    """

    if colored_object_mask is None or colored_object_mask.ndim != 2 or not walls:
        return SemanticWallFilterResult(walls=list(walls), rejected=[])
    height, width = colored_object_mask.shape[:2]
    basis = max(width, height)
    dilation_px = max(3, int(round(basis * dilation_ratio)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilation_px * 2 + 1, dilation_px * 2 + 1),
    )
    dilated = cv2.dilate(colored_object_mask, kernel) > 0
    sample_count = 81
    kept: list[WallSegment] = []
    rejected: list[WallSegment] = []
    for wall in walls:
        if preserve_long_external and wall.external and wall_length(wall) >= basis * 0.24:
            kept.append(wall)
            continue
        covered = 0
        for amount in np.linspace(0.0, 1.0, sample_count):
            x = int(round(wall.start.x + (wall.end.x - wall.start.x) * float(amount)))
            y = int(round(wall.start.y + (wall.end.y - wall.start.y) * float(amount)))
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            covered += int(dilated[y, x])
        if covered / sample_count >= coverage_threshold:
            rejected.append(wall)
        else:
            kept.append(wall)
    return SemanticWallFilterResult(walls=kept, rejected=rejected)


def filter_walls_near_text_regions(
    walls: list[WallSegment],
    text_regions: list[OCRText],
    *,
    min_confidence: float = 0.35,
    coverage_threshold: float = 0.55,
) -> SemanticWallFilterResult:
    """Reject short wall candidates traced from OCR glyphs or label plaques.

    The preprocessing text mask handles most isolated glyphs, but outlined
    labels and text embedded in a hatch can stay connected to surrounding
    linework. Only short candidates substantially contained by a measured OCR
    box are removed; envelope walls and lines merely crossing a label survive.
    """

    boxes: list[tuple[float, float, float, float]] = []
    for item in text_regions:
        if item.confidence < min_confidence or len(item.polygon) < 3:
            continue
        xs = [point[0] for point in item.polygon]
        ys = [point[1] for point in item.polygon]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        padding = max(3.0, min(max(width, 1.0), max(height, 1.0)) * 0.45)
        boxes.append(
            (
                min(xs) - padding,
                min(ys) - padding,
                max(xs) + padding,
                max(ys) + padding,
            )
        )
    if not boxes:
        return SemanticWallFilterResult(walls=list(walls), rejected=[])

    samples = np.linspace(0.0, 1.0, 31)
    kept: list[WallSegment] = []
    rejected: list[WallSegment] = []
    for wall in walls:
        if wall.external:
            kept.append(wall)
            continue
        reject = False
        for x0, y0, x1, y1 in boxes:
            box_span = max(x1 - x0, y1 - y0)
            if wall_length(wall) > box_span * 1.8:
                continue
            inside = 0
            for amount in samples:
                x = wall.start.x + (wall.end.x - wall.start.x) * float(amount)
                y = wall.start.y + (wall.end.y - wall.start.y) * float(amount)
                inside += int(x0 <= x <= x1 and y0 <= y <= y1)
            if inside / len(samples) >= coverage_threshold:
                reject = True
                break
        if reject:
            rejected.append(wall)
        else:
            kept.append(wall)
    return SemanticWallFilterResult(walls=kept, rejected=rejected)


def filter_walls_crossing_repetitive_details(
    walls: list[WallSegment],
    regions: list[RepetitiveDetailRegion],
    *,
    min_overlap_ratio: float = 0.25,
) -> SemanticWallFilterResult:
    """Reject supplemental walls whose evidence is a stair/hatch interior."""

    if not regions:
        return SemanticWallFilterResult(walls=list(walls), rejected=[])
    kept: list[WallSegment] = []
    rejected: list[WallSegment] = []
    for wall in walls:
        horizontal = is_horizontal(wall)
        wall_span = max(wall_length(wall), 1.0)
        crosses_detail = False
        for region in regions:
            if (horizontal and region.orientation == "h") or (not horizontal and region.orientation == "v"):
                continue
            x, y, width, height = region.rect
            margin = max(3.0, region.spacing_px * 0.35)
            if horizontal:
                coordinate = wall.start.y
                start, end = sorted((wall.start.x, wall.end.x))
                overlap = max(0.0, min(end, x + width) - max(start, x))
                interior = y + margin < coordinate < y + height - margin
            else:
                coordinate = wall.start.x
                start, end = sorted((wall.start.y, wall.end.y))
                overlap = max(0.0, min(end, y + height) - max(start, y))
                interior = x + margin < coordinate < x + width - margin
            if interior and overlap / wall_span >= min_overlap_ratio:
                crosses_detail = True
                break
        if crosses_detail:
            rejected.append(wall)
        else:
            kept.append(wall)
    return SemanticWallFilterResult(walls=kept, rejected=rejected)


def semantic_exclusions_from_hints(
    hints: Any,
    image_width_px: int,
    image_height_px: int,
    min_confidence: float,
) -> list[SemanticExclusion]:
    if hints is None:
        return []
    exclusions: list[SemanticExclusion] = []
    for item in getattr(hints, "furniture", []):
        category = str(getattr(item, "category", "")).strip().lower()
        if float(getattr(item, "confidence", 0.0)) < min_confidence:
            continue
        if not any(token in category for token in NON_STRUCTURAL_TOKENS):
            continue
        exclusions.append(
            SemanticExclusion(
                category=category,
                center=Point2D(
                    x=float(item.center.x) * image_width_px,
                    y=float(item.center.y) * image_height_px,
                ),
                width_px=max(2.0, float(item.width) * image_width_px),
                depth_px=max(2.0, float(item.depth) * image_height_px),
                rotation_deg=float(getattr(item, "rotation_deg", 0.0)),
            )
        )
    return exclusions


def semantic_exclusions_from_local_footprints(
    footprints: list[Any],
) -> list[SemanticExclusion]:
    return [
        SemanticExclusion(
            category=str(footprint.category),
            center=footprint.center,
            width_px=float(footprint.width_px),
            depth_px=float(footprint.depth_px),
            rotation_deg=float(footprint.rotation_deg),
        )
        for footprint in footprints
    ]


def filter_walls_inside_semantic_objects(
    walls: list[WallSegment],
    exclusions: list[SemanticExclusion],
    *,
    coverage_threshold: float = 0.74,
    margin_px: float = 3.0,
    preserve_grounded_external: bool = False,
    preserve_external: bool = False,
) -> SemanticWallFilterResult:
    """Remove linework that lies inside a known furniture/special footprint."""

    if not exclusions:
        return SemanticWallFilterResult(walls=list(walls), rejected=[])
    kept: list[WallSegment] = []
    rejected: list[WallSegment] = []
    samples = np.linspace(0.0, 1.0, 21)
    for wall in walls:
        if preserve_external and wall.external:
            kept.append(wall)
            continue
        if preserve_grounded_external and wall.external and "ai_proposal" in wall.evidence_source:
            kept.append(wall)
            continue
        covered = False
        for exclusion in exclusions:
            inside = 0
            for amount in samples:
                point = Point2D(
                    x=wall.start.x + (wall.end.x - wall.start.x) * float(amount),
                    y=wall.start.y + (wall.end.y - wall.start.y) * float(amount),
                )
                if exclusion.contains(point, margin_px):
                    inside += 1
            if inside / len(samples) >= coverage_threshold:
                covered = True
                break
        if covered:
            rejected.append(wall)
        else:
            kept.append(wall)
    return SemanticWallFilterResult(walls=kept, rejected=rejected)
