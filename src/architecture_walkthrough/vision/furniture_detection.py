from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from architecture_walkthrough.geometry.models import FurniturePlacement, Point2D
from architecture_walkthrough.vision.preprocessing import load_image, resize_preserving_aspect

if TYPE_CHECKING:
    from architecture_walkthrough.vision.ocr import OCRText


@dataclass(frozen=True)
class LocalObjectFootprint:
    category: str
    center: Point2D
    width_px: float
    depth_px: float
    rotation_deg: float


@dataclass(frozen=True)
class GroundedFurnitureSemantics:
    """Local footprints with optional AI category labels attached.

    Geometry always comes from local pixels.  AI hints can rename a nearby
    footprint, but an unmatched hint never becomes furniture by itself.
    """

    furniture: list[FurniturePlacement]
    matched_hint_count: int = 0
    rejected_hint_count: int = 0


def _mask_for_colored_objects(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Rendered plans usually draw floors and walls in low-saturation whites,
    # creams, and greys. Furniture, rugs, plants, wood, and counters carry more
    # chroma, so this isolates semantic objects without relying on text labels.
    colored = (saturation > 28) & (value > 35) & (value < 250)
    plants = (hue >= 35) & (hue <= 95) & (saturation > 40) & (value > 35)
    mask: np.ndarray = np.where(colored | plants, 255, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)


def colored_object_mask_from_image(image_path: Path, max_side: int = 1600) -> np.ndarray:
    """Return the precise local color footprint used by furniture detection.

    Keeping this as a pixel mask avoids turning connected L-shaped counters or
    balcony patterns into a large rectangular exclusion that could swallow a
    nearby room wall.
    """

    image = resize_preserving_aspect(load_image(image_path), max_side=max_side)
    return _mask_for_colored_objects(image)


def _saturated_contour_pixels(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Return HSV pixels inside a contour, excluding pale background corners."""

    x, y, w, h = cv2.boundingRect(contour)
    patch = image[y : y + h, x : x + w]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    local_contour = contour.astype(np.int32).copy()
    local_contour[:, 0, 0] -= x
    local_contour[:, 0, 1] -= y
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_mask, [local_contour], -1, 255, thickness=-1)
    pixels = hsv[contour_mask > 0]
    if not len(pixels):
        return pixels
    saturated = pixels[(pixels[:, 1] >= 35) & (pixels[:, 2] >= 30) & (pixels[:, 2] <= 250)]
    return saturated if len(saturated) else pixels


def _category_for_patch(image: np.ndarray, contour: np.ndarray, width: int, height: int) -> str:
    _x, _y, w, h = cv2.boundingRect(contour)
    pixels = _saturated_contour_pixels(image, contour)
    aspect = max(w, h) / max(min(w, h), 1)
    area_ratio = (w * h) / max(width * height, 1)
    if not len(pixels):
        return "rug" if area_ratio > 0.008 and aspect > 1.5 else "furniture"

    hues = pixels[:, 0]
    saturation = float(np.median(pixels[:, 1]))
    hue = float(np.median(hues))
    red_brown_fraction = float(np.mean((hues <= 25) | (hues >= 170)))
    green_fraction = float(np.mean((hues >= 35) & (hues <= 95)))

    if green_fraction >= 0.55 and saturation > 55 and area_ratio < 0.025:
        return "plant"
    # Counters in rendered plans are commonly dark red/brown bars.  Use only
    # pixels inside the contour: a rotated narrow bar has mostly white pixels
    # in its axis-aligned bounding box, which made the old median turn grey.
    if red_brown_fraction >= 0.55 and saturation > 45 and aspect >= 2.2:
        return "kitchen_counter"
    if 25 <= hue <= 95 and saturation > 45:
        # Large near-square olive footprints are beds. Long, moderately
        # slender footprints are dining tables; very slender ones are sofas.
        # Using the contour's saturated pixels keeps pale room backgrounds
        # from shifting these categories.
        if area_ratio > 0.01 and aspect <= 1.55:
            return "bed"
        if area_ratio > 0.009 and 1.70 <= aspect <= 2.80:
            return "dining_table"
        if area_ratio > 0.002 and aspect >= 1.65:
            return "sofa"
        if area_ratio > 0.004:
            return "sofa"
        return "chair"
    if red_brown_fraction >= 0.45 and saturation > 35:
        return "dining_table" if area_ratio > 0.004 else "side_table"
    if area_ratio > 0.008 and aspect > 1.5:
        return "rug"
    return "furniture"


def _is_giant_connected_region(
    contour: np.ndarray,
    contour_area: float,
    image_width: int,
    image_height: int,
) -> bool:
    """Reject page-scale fills, decorative bands, and connected color washes."""

    _x, _y, width, height = cv2.boundingRect(contour)
    page_area = max(image_width * image_height, 1)
    bbox_ratio = (width * height) / page_area
    width_ratio = width / max(image_width, 1)
    height_ratio = height / max(image_height, 1)
    contour_ratio = contour_area / page_area
    if contour_ratio > 0.08 or bbox_ratio > 0.08:
        return True
    if width_ratio > 0.72 and height_ratio > 0.10:
        return True
    return height_ratio > 0.72 and width_ratio > 0.10


def _counter_color_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    warm = ((hue <= 25) | (hue >= 170)) & (saturation >= 50) & (value >= 30) & (value <= 245)
    mask = np.where(warm, 255, 0).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)


def _counter_bar_footprints(
    image: np.ndarray,
    min_area: int,
    text_regions: list[OCRText] | None,
) -> tuple[list[LocalObjectFootprint], np.ndarray]:
    """Decompose connected red/brown L- and U-shaped counters into bars."""

    height, width = image.shape[:2]
    page_area = max(width * height, 1)
    basis = max(width, height)
    # The opening kernel must be longer than a typical counter's thickness;
    # otherwise both arms of an L survive both orientation passes and remain
    # one square bounding box instead of two useful bars.
    min_length = max(24, int(round(basis * 0.055)))
    max_thickness = max(10, int(round(basis * 0.075)))
    warm = _counter_color_mask(image)
    footprints: list[LocalObjectFootprint] = []
    coverage = np.zeros_like(warm)

    for orientation in ("horizontal", "vertical"):
        kernel = (
            cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 3))
            if orientation == "horizontal"
            else cv2.getStructuringElement(cv2.MORPH_RECT, (3, min_length))
        )
        opened = cv2.morphologyEx(warm, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, rect_width, rect_height = cv2.boundingRect(contour)
            long_side = rect_width if orientation == "horizontal" else rect_height
            short_side = rect_height if orientation == "horizontal" else rect_width
            aspect = long_side / max(short_side, 1)
            area = float(cv2.countNonZero(opened[y : y + rect_height, x : x + rect_width]))
            if (
                area < min_area
                or aspect < 2.4
                or long_side < min_length
                or long_side > basis * 0.58
                or short_side > max_thickness
                or (rect_width * rect_height) / page_area > 0.06
                or _contains_semantic_text(contour, text_regions)
            ):
                continue
            footprint = LocalObjectFootprint(
                category="kitchen_counter",
                center=Point2D(x=x + rect_width / 2, y=y + rect_height / 2),
                width_px=float(long_side),
                depth_px=float(short_side),
                rotation_deg=0.0 if orientation == "horizontal" else 90.0,
            )
            duplicate = any(
                abs(footprint.rotation_deg - existing.rotation_deg) < 1.0
                and footprint.center.distance_to(existing.center) < max(8.0, long_side * 0.18)
                for existing in footprints
            )
            if duplicate:
                continue
            footprints.append(footprint)
            cv2.rectangle(coverage, (x, y), (x + rect_width - 1, y + rect_height - 1), 255, -1)
    return footprints, coverage


def _contour_mask_overlap(contour: np.ndarray, mask: np.ndarray) -> float:
    x, y, width, height = cv2.boundingRect(contour)
    local = np.zeros((height, width), dtype=np.uint8)
    shifted = contour.astype(np.int32).copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(local, [shifted], -1, 255, thickness=-1)
    contour_pixels = int(cv2.countNonZero(local))
    if contour_pixels == 0:
        return 0.0
    overlap = cv2.bitwise_and(local, mask[y : y + height, x : x + width])
    return float(cv2.countNonZero(overlap)) / contour_pixels


def _contains_semantic_text(
    contour: np.ndarray,
    text_regions: list[OCRText] | None,
) -> bool:
    """Return true when a colored contour is the background of a plan label.

    A bounding-box overlap alone is unsafe: room text can sit inside the empty
    center of an L-shaped kitchen counter. Testing the text center against the
    actual contour rejects filled label plaques while preserving surrounding
    furniture and counters.
    """
    if not text_regions:
        return False
    semantic_types = {
        "room_label",
        "dimension",
        "balcony_label",
        "lift_label",
        "stair_label",
    }
    for item in text_regions:
        if item.semantic_type not in semantic_types or item.confidence < 0.35:
            continue
        text_xs = [point[0] for point in item.polygon]
        text_ys = [point[1] for point in item.polygon]
        text_x0, text_x1 = min(text_xs), max(text_xs)
        text_y0, text_y1 = min(text_ys), max(text_ys)
        center_x = sum(point[0] for point in item.polygon) / len(item.polygon)
        center_y = sum(point[1] for point in item.polygon) / len(item.polygon)
        if cv2.pointPolygonTest(contour, (center_x, center_y), False) >= 0:
            return True
        # Label plaques often knock the glyph line out to white, leaving a
        # concave colored contour whose polygon technically excludes the text
        # center. Its overall box is still tightly wrapped around that text.
        contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(contour)
        overlap_w = max(0.0, min(contour_x + contour_w, text_x1) - max(contour_x, text_x0))
        overlap_h = max(0.0, min(contour_y + contour_h, text_y1) - max(contour_y, text_y0))
        text_area = max((text_x1 - text_x0) * (text_y1 - text_y0), 1.0)
        contour_box_area = float(contour_w * contour_h)
        if overlap_w * overlap_h >= text_area * 0.8 and contour_box_area <= text_area * 4.0:
            return True
    return False


def detect_colored_object_footprints(
    image_path: Path,
    max_side: int = 1600,
    text_regions: list[OCRText] | None = None,
) -> list[LocalObjectFootprint]:
    """Detect local object extents in pixels before reconstruction scale exists."""

    image = resize_preserving_aspect(load_image(image_path), max_side=max_side)
    mask = _mask_for_colored_objects(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height_px, width_px = image.shape[:2]
    min_area = max(90, int(width_px * height_px * 0.00012))
    counter_bars, counter_coverage = _counter_bar_footprints(image, min_area, text_regions)
    footprints: list[LocalObjectFootprint] = list(counter_bars)
    for contour in contours:
        area = cv2.contourArea(contour)
        if (
            area < min_area
            or _is_giant_connected_region(contour, area, width_px, height_px)
            or _contains_semantic_text(contour, text_regions)
            or _contour_mask_overlap(contour, counter_coverage) >= 0.42
        ):
            continue
        (center_x, center_y), (box_w, box_h), angle = cv2.minAreaRect(contour)
        if box_w <= 3 or box_h <= 3:
            continue
        category = _category_for_patch(image, contour, width_px, height_px)
        rotation = float(angle)
        if box_w < box_h:
            box_w, box_h = box_h, box_w
            rotation += 90.0
        footprints.append(
            LocalObjectFootprint(
                category=category,
                center=Point2D(x=float(center_x), y=float(center_y)),
                width_px=float(box_w),
                depth_px=float(box_h),
                rotation_deg=rotation,
            )
        )

    kept: list[LocalObjectFootprint] = []
    for item in sorted(footprints, key=lambda value: value.width_px * value.depth_px, reverse=True):
        duplicate = any(
            item.center.distance_to(existing.center)
            < max(8.0, min(item.width_px, item.depth_px, existing.width_px, existing.depth_px) * 0.5)
            for existing in kept
        )
        if not duplicate:
            kept.append(item)
    return kept[:80]


def detect_furniture_from_image(
    image_path: Path,
    pixels_per_metre: float,
    image_height_px: int,
    max_side: int = 1600,
    text_regions: list[OCRText] | None = None,
) -> list[FurniturePlacement]:
    placements: list[FurniturePlacement] = []
    footprints = detect_colored_object_footprints(
        image_path,
        max_side=max_side,
        text_regions=text_regions,
    )
    for footprint in footprints:
        width_m = max(footprint.width_px / pixels_per_metre, 0.25)
        depth_m = max(footprint.depth_px / pixels_per_metre, 0.25)
        if width_m > 5.0 or depth_m > 5.0:
            continue
        placements.append(
            FurniturePlacement(
                category=footprint.category,
                center=Point2D(
                    x=footprint.center.x / pixels_per_metre,
                    y=(image_height_px - footprint.center.y) / pixels_per_metre,
                ),
                width_m=width_m,
                depth_m=depth_m,
                rotation_deg=footprint.rotation_deg,
            )
        )

    return _deduplicate_furniture(placements)


def _placement_from_footprint(
    footprint: LocalObjectFootprint,
    category: str,
    pixels_per_metre: float,
    image_height_px: int,
) -> FurniturePlacement | None:
    width_m = max(footprint.width_px / pixels_per_metre, 0.25)
    depth_m = max(footprint.depth_px / pixels_per_metre, 0.25)
    if width_m > 5.0 or depth_m > 5.0:
        return None
    return FurniturePlacement(
        category=category,
        center=Point2D(
            x=footprint.center.x / pixels_per_metre,
            y=(image_height_px - footprint.center.y) / pixels_per_metre,
        ),
        width_m=width_m,
        depth_m=depth_m,
        rotation_deg=footprint.rotation_deg,
    )


def _normalized_furniture_category(value: object) -> str:
    return "_".join(str(value or "furniture").strip().lower().replace("-", " ").split()) or "furniture"


def _hint_is_furniture(category: str) -> bool:
    structural_tokens = (
        "wall",
        "door",
        "window",
        "balcony",
        "lift",
        "stair",
        "entrance",
        "railing",
    )
    return not any(token in category for token in structural_tokens)


def _point_in_expanded_footprint(
    point: Point2D,
    footprint: LocalObjectFootprint,
    expansion_px: float,
) -> bool:
    angle = math.radians(-footprint.rotation_deg)
    dx = point.x - footprint.center.x
    dy = point.y - footprint.center.y
    local_x = dx * math.cos(angle) - dy * math.sin(angle)
    local_y = dx * math.sin(angle) + dy * math.cos(angle)
    return (
        abs(local_x) <= footprint.width_px / 2 + expansion_px
        and abs(local_y) <= footprint.depth_px / 2 + expansion_px
    )


def ground_ai_furniture_semantics(
    local_footprints: list[LocalObjectFootprint],
    ai_hints: Any,
    image_width_px: int,
    image_height_px: int,
    pixels_per_metre: float,
    min_confidence: float = 0.55,
) -> GroundedFurnitureSemantics:
    """Apply AI categories only to compatible, nearby local color footprints.

    The returned geometry is derived exclusively from ``local_footprints``.
    Eligible AI hints participate in a one-to-one match; unmatched hints are
    counted as rejected and never materialize as scene objects.
    """

    if pixels_per_metre <= 0:
        raise ValueError("pixels_per_metre must be positive")

    valid_footprints = [
        footprint
        for footprint in local_footprints
        if footprint.width_px / pixels_per_metre <= 5.0
        and footprint.depth_px / pixels_per_metre <= 5.0
    ]
    raw_hints = list(getattr(ai_hints, "furniture", [])) if ai_hints is not None else []
    eligible_hints: list[tuple[Any, str, float]] = []
    for hint in raw_hints:
        confidence = float(getattr(hint, "confidence", 0.0))
        category = _normalized_furniture_category(getattr(hint, "category", "furniture"))
        if confidence >= min_confidence and _hint_is_furniture(category):
            eligible_hints.append((hint, category, confidence))

    basis = max(image_width_px, image_height_px, 1)
    candidate_pairs: list[tuple[float, int, int]] = []
    for hint_index, (hint, _category, confidence) in enumerate(eligible_hints):
        center = Point2D(
            x=float(hint.center.x) * image_width_px,
            y=float(hint.center.y) * image_height_px,
        )
        hint_width = max(2.0, float(hint.width) * image_width_px)
        hint_depth = max(2.0, float(hint.depth) * image_height_px)
        hint_diagonal = math.hypot(hint_width, hint_depth)
        hint_area = hint_width * hint_depth
        for footprint_index, footprint in enumerate(valid_footprints):
            local_diagonal = math.hypot(footprint.width_px, footprint.depth_px)
            max_distance = max(12.0, min(basis * 0.06, max(local_diagonal, hint_diagonal) * 0.70))
            distance = center.distance_to(footprint.center)
            if distance > max_distance:
                continue
            local_area = max(footprint.width_px * footprint.depth_px, 1.0)
            size_ratio = min(local_area, hint_area) / max(local_area, hint_area)
            locally_contained = _point_in_expanded_footprint(
                center,
                footprint,
                expansion_px=max(6.0, min(footprint.width_px, footprint.depth_px) * 0.30),
            )
            if not locally_contained and size_ratio < 0.12:
                continue
            score = distance / max_distance + (1.0 - size_ratio) * 0.30 - confidence * 0.05
            candidate_pairs.append((score, hint_index, footprint_index))

    matched_hints: set[int] = set()
    matched_footprints: set[int] = set()
    category_by_footprint: dict[int, str] = {}
    for _score, hint_index, footprint_index in sorted(candidate_pairs):
        if hint_index in matched_hints or footprint_index in matched_footprints:
            continue
        matched_hints.add(hint_index)
        matched_footprints.add(footprint_index)
        category_by_footprint[footprint_index] = eligible_hints[hint_index][1]

    furniture: list[FurniturePlacement] = []
    for index, footprint in enumerate(valid_footprints):
        category = category_by_footprint.get(index, footprint.category)
        placement = _placement_from_footprint(
            footprint,
            category,
            pixels_per_metre,
            image_height_px,
        )
        if placement is not None:
            furniture.append(placement)

    return GroundedFurnitureSemantics(
        furniture=_deduplicate_furniture(furniture),
        matched_hint_count=len(matched_hints),
        rejected_hint_count=len(eligible_hints) - len(matched_hints),
    )


def _deduplicate_furniture(items: list[FurniturePlacement]) -> list[FurniturePlacement]:
    kept: list[FurniturePlacement] = []
    for item in sorted(items, key=lambda value: value.width_m * value.depth_m, reverse=True):
        too_close = False
        for existing in kept:
            distance = item.center.distance_to(existing.center)
            footprint = min(item.width_m, item.depth_m, existing.width_m, existing.depth_m)
            if distance < max(0.25, footprint * 0.5):
                too_close = True
                break
        if not too_close:
            kept.append(item)
    return kept[:80]
