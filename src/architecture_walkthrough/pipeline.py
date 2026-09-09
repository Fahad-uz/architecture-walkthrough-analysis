from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, cast

import cv2

from architecture_walkthrough.ai.floorplan_vision import (
    FloorPlanVisionAnalysis,
    FloorPlanVisionHints,
    GeminiFloorPlanVisionAnalyzer,
    GeminiFloorPlanVisionError,
)
from architecture_walkthrough.ai.sanity_check import GeminiLayoutSanityChecker, SanityCheckResult
from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.models import (
    ArchitecturalElement,
    BalconyPolygon,
    CeilingSettings,
    CoordinateSystem,
    FloorPlanModel,
    FurniturePlacement,
    Point2D,
    ReconstructionMetadata,
    ValidationIssue,
    WallSegment,
)
from architecture_walkthrough.geometry.balcony_ownership import (
    reconcile_room_balcony_ownership,
)
from architecture_walkthrough.geometry.furniture_layout import fit_furniture_to_rooms
from architecture_walkthrough.geometry.reconstruction import reconstruct_walls
from architecture_walkthrough.geometry.room_extraction import extract_rooms_from_walls
from architecture_walkthrough.geometry.scale import ScaleConverter
from architecture_walkthrough.geometry.scale_solver import ScaleConstraint, solve_scale
from architecture_walkthrough.geometry.wall_graph import collinear_gaps
from architecture_walkthrough.geometry.validation import (
    SourceEvidence,
    evaluate_quality,
    load_source_evidence,
    validate_reconstruction,
)
from architecture_walkthrough.scene.export_glb import export_floorplan_glb
from architecture_walkthrough.scene.glb_optimizer import optimize_glb
from architecture_walkthrough.scene.glb_validator import validate_glb
from architecture_walkthrough.scene.blender_runner import run_blender_script
from architecture_walkthrough.scene.scene_builder import build_blender_script
from architecture_walkthrough.security.file_validation import validate_image_file
from architecture_walkthrough.vision.ocr import OCRText, parse_dimension_pair, run_ocr
from architecture_walkthrough.vision.furniture_detection import (
    colored_object_mask_from_image,
    detect_colored_object_footprints,
    ground_ai_furniture_semantics,
)
from architecture_walkthrough.vision.grounded_hints import ground_ai_wall_hints
from architecture_walkthrough.vision.semantic_filter import (
    filter_walls_crossing_repetitive_details,
    filter_walls_inside_semantic_objects,
    filter_walls_near_colored_objects,
    filter_walls_near_text_regions,
    semantic_exclusions_from_local_footprints,
)
from architecture_walkthrough.vision.local_openings import build_thin_line_mask, detect_local_openings
from architecture_walkthrough.vision.overlay import write_analysis_overlay
from architecture_walkthrough.vision.plan_roi import detect_plan_roi
from architecture_walkthrough.vision.preprocessing import load_image, preprocess_array
from architecture_walkthrough.vision.providers import build_geometry_provider
from architecture_walkthrough.vision.wall_detection import RepetitiveDetailRegion, WallBand
from architecture_walkthrough.walkthrough.path_planner import camera_waypoints_for_model
from architecture_walkthrough.walkthrough.render_video import encode_frames_to_mp4

LOGGER = logging.getLogger(__name__)
FLOORPLAN_VISION_CACHE_VERSION = "2026-07-grounded-hints-v2"
MIN_GEMINI_REQUEST_TIMEOUT_SECONDS = 5
LOCAL_ANALYSIS_RESERVE_SECONDS = 15


class StageLogger:
    def __init__(self) -> None:
        self.stages: list[dict[str, Any]] = []

    def record(self, name: str, started: float, **outputs: Any) -> None:
        self.stages.append({"name": name, "duration_seconds": round(time.perf_counter() - started, 4), **outputs})


def _write_roi_image(path: Path, image) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write ROI image: {path}")
    return path


def _read_grayscale(path: Path, label: str):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read {label}: {path}")
    return image


def _floorplan_vision_analysis(
    image_path: Path,
    output_dir: Path,
    config: AppConfig,
    require_success: bool,
    skip_request_error: str | None = None,
) -> tuple[FloorPlanVisionAnalysis, bool]:
    """Reuse validated semantic hints for the same ROI/model when available."""

    cache_path = output_dir / "floorplan_vision_hints.json"
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if config.ai.gemini_enabled and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("cache_version") == FLOORPLAN_VISION_CACHE_VERSION
                and cached.get("image_sha256") == image_sha256
                and cached.get("model") == config.ai.gemini_model
            ):
                hints = FloorPlanVisionHints.model_validate(cached.get("hints"))
                return FloorPlanVisionAnalysis(attempted=True, succeeded=True, hints=hints), True
        except (OSError, ValueError, TypeError):
            LOGGER.warning("ignoring invalid cached floor-plan vision hints at %s", cache_path)

    if skip_request_error is not None:
        if require_success:
            raise GeminiFloorPlanVisionError(skip_request_error)
        return FloorPlanVisionAnalysis(error=skip_request_error), False

    analysis = GeminiFloorPlanVisionAnalyzer(config.ai).analyze_with_diagnostics(
        image_path,
        require_success=require_success,
    )
    if analysis.hints is not None:
        _write_json(
            cache_path,
            {
                "cache_version": FLOORPLAN_VISION_CACHE_VERSION,
                "image_sha256": image_sha256,
                "model": config.ai.gemini_model,
                "hints": analysis.hints.model_dump(mode="json"),
            },
        )
    return analysis, False


def _runtime_ai_config(config: AppConfig) -> tuple[AppConfig, str | None]:
    """Fit optional Gemini calls inside the configured worker lifetime."""

    if not config.ai.gemini_enabled:
        return config, None
    attempts = config.ai.gemini_retry_attempts
    retry_delay = sum(
        config.ai.gemini_retry_base_delay_seconds * (2**attempt)
        for attempt in range(attempts - 1)
    )
    available_seconds = config.limits.processing_timeout_seconds - LOCAL_ANALYSIS_RESERVE_SECONDS
    # The semantic pass may consume every configured attempt; one additional
    # slot covers sanity when semantics succeeds. This keeps the local pipeline
    # and artifact publication outside the optional-provider budget.
    request_slots = attempts + 1
    max_request_seconds = int((available_seconds - retry_delay) / request_slots)
    if max_request_seconds < MIN_GEMINI_REQUEST_TIMEOUT_SECONDS:
        return (
            config,
            "Gemini skipped because the configured analysis timeout leaves "
            "insufficient time for local reconstruction",
        )
    effective_timeout = min(
        config.ai.gemini_request_timeout_seconds,
        max_request_seconds,
    )
    if effective_timeout == config.ai.gemini_request_timeout_seconds:
        return config, None
    runtime_ai = config.ai.model_copy(
        update={"gemini_request_timeout_seconds": effective_timeout}
    )
    return config.model_copy(update={"ai": runtime_ai}), None


def _convert_walls_to_metres(
    walls: list[WallSegment],
    pixels_per_metre: float,
    image_height_px: int,
) -> list[WallSegment]:
    converter = ScaleConverter(pixels_per_metre=pixels_per_metre)
    converted: list[WallSegment] = []
    for wall in walls:
        converted.append(
            wall.model_copy(
                update={
                    "start": Point2D(
                        x=converter.px_to_m(wall.start.x),
                        y=converter.px_to_m(image_height_px - wall.start.y),
                    ),
                    "end": Point2D(
                        x=converter.px_to_m(wall.end.x),
                        y=converter.px_to_m(image_height_px - wall.end.y),
                    ),
                }
            )
        )
    return converted


def _manual_pixels_per_metre(manual_scale: float | None) -> float | None:
    if manual_scale is None:
        return None
    if not math.isfinite(manual_scale) or manual_scale <= 0:
        raise ValueError("manual scale must be finite positive metres per pixel")
    return 1.0 / manual_scale


def _ocr_labels(ocr_results: list[OCRText]) -> list[OCRText]:
    return [item for item in ocr_results if item.semantic_type in {"room_label", "dimension", "balcony_label", "lift_label", "stair_label"}]


def _semantic_labels_from_gemini(ai_hints, image_width_px: int, image_height_px: int) -> tuple[list[OCRText], list[OCRText]]:
    if ai_hints is None:
        return [], []
    room_labels: list[OCRText] = []
    special_labels: list[OCRText] = []
    for room in ai_hints.rooms:
        if not room.name or room.confidence < 0.35 or not room.points:
            continue
        cx = sum(point.x for point in room.points) / len(room.points) * image_width_px
        cy = sum(point.y for point in room.points) / len(room.points) * image_height_px
        room_labels.append(
            OCRText(
                text=room.name,
                polygon=[(cx - 5, cy - 5), (cx + 5, cy - 5), (cx + 5, cy + 5), (cx - 5, cy + 5)],
                confidence=room.confidence,
                normalized_text=room.name,
                semantic_type="room_label",
            )
        )
    for furniture in ai_hints.furniture:
        category = furniture.category.lower()
        if furniture.confidence < 0.35:
            continue
        if not any(token in category for token in ("stair", "lift", "balcony", "shelf", "counter", "entrance")):
            continue
        cx = furniture.center.x * image_width_px
        cy = furniture.center.y * image_height_px
        semantic_type = "text"
        if "stair" in category:
            semantic_type = "stair_label"
        elif "lift" in category:
            semantic_type = "lift_label"
        elif "balcony" in category:
            semantic_type = "balcony_label"
        special_labels.append(
            OCRText(
                text=furniture.category,
                polygon=[(cx - 5, cy - 5), (cx + 5, cy - 5), (cx + 5, cy + 5), (cx - 5, cy + 5)],
                confidence=furniture.confidence,
                normalized_text=furniture.category,
                semantic_type=semantic_type,
            )
        )
    return room_labels, special_labels


def _dimension_labels_from_gemini(ai_hints, image_width_px: int, image_height_px: int) -> list[OCRText]:
    """Dimension annotations transcribed by Gemini become dimension labels.

    Gemini reads the text; the geometry the dimension applies to still comes
    from locally detected room polygons that the label falls inside.
    """
    if ai_hints is None:
        return []
    labels: list[OCRText] = []
    for item in getattr(ai_hints, "dimension_texts", []):
        if item.confidence < 0.35:
            continue
        try:
            parsed = parse_dimension_pair(item.text)
        except ValueError:
            continue
        if parsed is None:
            continue
        cx = item.center.x * image_width_px
        cy = item.center.y * image_height_px
        labels.append(
            OCRText(
                text=item.text,
                polygon=[(cx - 5, cy - 5), (cx + 5, cy - 5), (cx + 5, cy + 5), (cx - 5, cy + 5)],
                confidence=item.confidence,
                normalized_text=item.text,
                semantic_type="dimension",
            )
        )
    return labels


def _scale_constraints_from_rooms(preliminary_rooms) -> list[ScaleConstraint]:
    """Dimension labels matched to detected room polygons.

    Text-box-size guessing is intentionally not a scale source: a dimension
    only counts once it is associated with real geometry.
    """
    constraints: list[ScaleConstraint] = []
    for room in preliminary_rooms:
        if room.dimension_m is None:
            continue
        xs = [point.x for point in room.points]
        ys = [point.y for point in room.points]
        constraints.append(
            ScaleConstraint(
                id=f"room_dim_{len(constraints):03d}",
                source="dimension_label_in_room_polygon",
                label=room.name,
                measured_px=(max(xs) - min(xs), max(ys) - min(ys)),
                expected_m=room.dimension_m,
                weight=1.0,
                tier="room_dimension",
            )
        )
    return constraints


STANDARD_DOOR_WIDTH_M = 0.85


def _scale_constraints_from_door_gaps(walls_px: list[WallSegment]) -> list[ScaleConstraint]:
    """Collinear wall gaps of door-like proportion, assuming a ~0.85 m leaf.

    A deliberate low-confidence fallback for undimensioned plans; the tiered
    solver only consults it when no real dimension source exists.
    """
    identified = [wall for wall in walls_px if wall.id]
    if len(identified) < 2:
        return []
    gaps = collinear_gaps(identified, coord_tol=6.0)
    xs = [point for wall in identified for point in (wall.start.x, wall.end.x)]
    ys = [point for wall in identified for point in (wall.start.y, wall.end.y)]
    plan_span_px = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

    # A generic break between collinear fragments is not automatically a
    # door.  Restrict the fallback to repeated, door-sized gap populations.
    # This prevents a few tiny junction slivers and huge missing-wall spans
    # from being averaged together as if every one were a 0.85 m leaf.
    plausible = sorted(
        (
            gap
            for gap in gaps
            if plan_span_px * 0.025
            <= float(cast(float, gap["length_m"]))
            <= plan_span_px * 0.16
        ),
        key=lambda gap: float(cast(float, gap["length_m"])),
    )
    clusters: list[list[dict[str, object]]] = []
    for gap in plausible:
        length = float(cast(float, gap["length_m"]))
        if not clusters:
            clusters.append([gap])
            continue
        previous = float(cast(float, clusters[-1][-1]["length_m"]))
        if length <= previous * 1.22:
            clusters[-1].append(gap)
        else:
            clusters.append([gap])
    repeated = [cluster for cluster in clusters if len(cluster) >= 2]
    if not repeated:
        return []
    # Repeated standard single-leaf doors are normally the smallest dominant
    # population; wider entrance/double-door gaps form a separate cluster.
    selected = min(
        repeated,
        key=lambda cluster: (
            -len(cluster),
            float(cast(float, cluster[len(cluster) // 2]["length_m"])),
        ),
    )
    constraints: list[ScaleConstraint] = []
    for gap in selected:
        gap_px = float(cast(float, gap["length_m"]))  # pixel units here
        if gap_px <= 2:
            continue
        constraints.append(
            ScaleConstraint(
                id=f"door_gap_{len(constraints):03d}",
                source=f"gap_{gap['wall_a']}_{gap['wall_b']}",
                measured_px=(gap_px, gap_px),
                expected_m=(STANDARD_DOOR_WIDTH_M, STANDARD_DOOR_WIDTH_M),
                weight=1.0,
                tier="door_width",
            )
        )
    return constraints


def _scale_constraints_from_dimension_annotations(
    labels: list[OCRText],
    walls_px: list[WallSegment],
) -> list[ScaleConstraint]:
    """Match OCR dimension pairs to four locally measured room boundaries.

    A dimension text box is not itself a ruler. For each readable pair, find
    the nearest vertical walls to its left/right whose spans cross the label,
    and the equivalent horizontal walls above/below. Open-plan labels that do
    not have four supporting boundaries produce no constraint; inconsistent
    rectangles are rejected later by the scale solver's aspect residual.
    """

    horizontal: list[tuple[float, float, float]] = []
    vertical: list[tuple[float, float, float]] = []
    for wall in walls_px:
        dx = abs(wall.end.x - wall.start.x)
        dy = abs(wall.end.y - wall.start.y)
        if dx >= dy * 2.5:
            horizontal.append(
                (
                    (wall.start.y + wall.end.y) / 2.0,
                    min(wall.start.x, wall.end.x),
                    max(wall.start.x, wall.end.x),
                )
            )
        elif dy >= dx * 2.5:
            vertical.append(
                (
                    (wall.start.x + wall.end.x) / 2.0,
                    min(wall.start.y, wall.end.y),
                    max(wall.start.y, wall.end.y),
                )
            )

    constraints: list[ScaleConstraint] = []
    for label in labels:
        try:
            parsed = parse_dimension_pair(label.normalized_text)
        except ValueError:
            continue
        if parsed is None or len(label.polygon) < 3:
            continue
        center_x = sum(point[0] for point in label.polygon) / len(label.polygon)
        center_y = sum(point[1] for point in label.polygon) / len(label.polygon)
        crossing_vertical = [
            coordinate
            for coordinate, start, end in vertical
            if start - 2.0 <= center_y <= end + 2.0
        ]
        crossing_horizontal = [
            coordinate
            for coordinate, start, end in horizontal
            if start - 2.0 <= center_x <= end + 2.0
        ]
        left = [coordinate for coordinate in crossing_vertical if coordinate < center_x]
        right = [coordinate for coordinate in crossing_vertical if coordinate > center_x]
        above = [coordinate for coordinate in crossing_horizontal if coordinate < center_y]
        below = [coordinate for coordinate in crossing_horizontal if coordinate > center_y]
        if not left or not right or not above or not below:
            continue
        measured_width = min(right) - max(left)
        measured_height = min(below) - max(above)
        if measured_width <= 4.0 or measured_height <= 4.0:
            continue
        constraints.append(
            ScaleConstraint(
                id=f"dimension_annotation_{len(constraints):03d}",
                source="ocr_dimension_between_local_wall_boundaries",
                label=label.normalized_text,
                measured_px=(measured_width, measured_height),
                expected_m=(parsed.width_m, parsed.height_m),
                weight=max(0.35, min(1.0, label.confidence)),
                tier="dimension_annotation",
            )
        )
    return constraints


def _scale_constraint_from_plan_extent(
    walls_px: list[WallSegment],
    expected_long_side_m: float,
) -> list[ScaleConstraint]:
    """Low-confidence scale fallback for plans without readable dimensions."""

    if not walls_px or expected_long_side_m <= 0:
        return []
    xs = [point for wall in walls_px for point in (wall.start.x, wall.end.x)]
    ys = [point for wall in walls_px for point in (wall.start.y, wall.end.y)]
    span_px = max(max(xs) - min(xs), max(ys) - min(ys))
    if span_px <= 0:
        return []
    return [
        ScaleConstraint(
            id="plan_extent_000",
            source="assumed_residential_long_side",
            measured_px=(span_px, span_px),
            expected_m=(expected_long_side_m, expected_long_side_m),
            weight=0.15,
            tier="plan_extent",
        )
    ]


def _scale_constraints_from_wall_bands(
    bands: list[WallBand],
    internal_thickness_m: float,
    external_thickness_m: float,
) -> list[ScaleConstraint]:
    constraints: list[ScaleConstraint] = []
    for band in bands:
        expected = external_thickness_m if band.external else internal_thickness_m
        constraints.append(
            ScaleConstraint(
                id=f"wall_thickness_{len(constraints):03d}",
                source="assumed_band_thickness",
                label=band.id,
                measured_px=(band.thickness_px, band.thickness_px),
                expected_m=(expected, expected),
                weight=0.25 if band.external else 0.18,
                tier="wall_thickness",
            )
        )
    return constraints


def _classify_special_elements(
    repetitive_regions: list[RepetitiveDetailRegion],
    pixels_per_metre: float,
    image_height_px: int,
) -> list[ArchitecturalElement]:
    """Build spatial specials only from measured local geometry.

    Text and AI labels are useful names, but their tiny label boxes are not the
    footprint of a stair, lift, balcony or kitchen counter.  Repeated tread
    geometry provides an actual staircase rectangle and step count; other
    special types remain absent until equivalent local evidence exists.
    """

    elements: list[ArchitecturalElement] = []
    converter = ScaleConverter(pixels_per_metre=pixels_per_metre)
    for region in repetitive_regions:
        if not region.is_stair_like():
            continue
        x, y, width, height = region.rect
        x = max(0.0, x)
        y = max(0.0, y)
        width = max(1.0, width)
        height = max(1.0, min(height, image_height_px - y))
        left = converter.px_to_m(x)
        right = converter.px_to_m(x + width)
        top = converter.px_to_m(image_height_px - y)
        bottom = converter.px_to_m(image_height_px - (y + height))
        elements.append(
            ArchitecturalElement(
                id=f"staircase_{len(elements):03d}",
                kind="staircase",
                polygon=[
                    Point2D(x=left, y=bottom),
                    Point2D(x=right, y=bottom),
                    Point2D(x=right, y=top),
                    Point2D(x=left, y=top),
                ],
                center=Point2D(x=(left + right) / 2.0, y=(bottom + top) / 2.0),
                width_m=max(0.3, right - left),
                depth_m=max(0.3, top - bottom),
                rotation_deg=0.0,
                confidence=min(1.0, 0.72 + region.line_count * 0.02),
                evidence_source="repetitive_parallel_treads",
                metadata={
                    "step_count": region.line_count,
                    "tread_spacing_m": converter.px_to_m(region.spacing_px),
                    "tread_orientation": region.orientation,
                },
            )
        )
    return elements


def _balconies_from_patterned_regions(
    repetitive_regions: list[RepetitiveDetailRegion],
    labels: list[OCRText],
    pixels_per_metre: float,
    image_height_px: int,
) -> list[BalconyPolygon]:
    """Ground balcony semantics to locally measured patterned footprints.

    A balcony OCR label alone has no usable geometry. Wide, shallow hatch
    fields are independently measured local evidence, though, and are
    deliberately excluded from staircase classification. Only a balcony label
    whose centre falls inside such a field may name it; unmatched labels and
    unmatched hatching are ignored.
    """

    balcony_labels = [
        label
        for label in labels
        if label.semantic_type == "balcony_label" and len(label.polygon) >= 3
    ]
    converter = ScaleConverter(pixels_per_metre=pixels_per_metre)
    balconies: list[BalconyPolygon] = []
    consumed_labels: set[int] = set()
    for region in repetitive_regions:
        if region.is_stair_like():
            continue
        x, y, width, height = region.rect
        if width <= 1.0 or height <= 1.0:
            continue
        matched: tuple[int, OCRText] | None = None
        for label_index, label in enumerate(balcony_labels):
            if label_index in consumed_labels:
                continue
            center_x = sum(point[0] for point in label.polygon) / len(label.polygon)
            center_y = sum(point[1] for point in label.polygon) / len(label.polygon)
            if x <= center_x <= x + width and y <= center_y <= y + height:
                matched = (label_index, label)
                break
        if matched is None:
            continue
        label_index, label = matched
        consumed_labels.add(label_index)
        left = converter.px_to_m(max(0.0, x))
        right = converter.px_to_m(max(0.0, x + width))
        top = converter.px_to_m(max(0.0, image_height_px - y))
        bottom = converter.px_to_m(max(0.0, image_height_px - (y + height)))
        if right - left < 0.3 or top - bottom < 0.3:
            continue
        balconies.append(
            BalconyPolygon(
                id=f"balcony_{len(balconies):03d}",
                name=label.normalized_text,
                points=[
                    Point2D(x=left, y=bottom),
                    Point2D(x=right, y=bottom),
                    Point2D(x=right, y=top),
                    Point2D(x=left, y=top),
                ],
                confidence=min(label.confidence, 0.78 + min(region.line_count, 12) * 0.015),
                evidence_source="patterned_region+ocr_balcony_label",
            )
        )
    return balconies


def _furniture_from_gemini(
    ai_hints,
    image_width_px: int,
    image_height_px: int,
    pixels_per_metre: float,
    min_confidence: float,
) -> list[FurniturePlacement]:
    if ai_hints is None:
        return []
    placements: list[FurniturePlacement] = []
    structural_tokens = ("wall", "door", "window", "balcony", "lift", "stair", "entrance", "railing")
    for hint in ai_hints.furniture:
        category = hint.category.strip().lower().replace(" ", "_")
        if hint.confidence < min_confidence * 0.65 or any(token in category for token in structural_tokens):
            continue
        width_m = max(0.20, hint.width * image_width_px / pixels_per_metre)
        depth_m = max(0.20, hint.depth * image_height_px / pixels_per_metre)
        if width_m > 5.0 or depth_m > 5.0:
            continue
        placements.append(
            FurniturePlacement(
                category=category,
                center=Point2D(
                    x=hint.center.x * image_width_px / pixels_per_metre,
                    y=(1.0 - hint.center.y) * image_height_px / pixels_per_metre,
                ),
                width_m=width_m,
                depth_m=depth_m,
                rotation_deg=hint.rotation_deg,
            )
        )
    return placements


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _validation_report(model: FloorPlanModel) -> dict[str, Any]:
    return {
        "schema_version": model.schema_version,
        "quality_state": model.reconstruction.quality_state,
        "quality_score": model.reconstruction.quality_score,
        "pixels_per_metre": model.pixels_per_metre,
        "scale_constraints": [item.model_dump() for item in model.scale_constraints],
        "issues": [item.model_dump() for item in model.validation_issues],
        "metadata": model.metadata,
    }


def analyze_image(
    input_path: Path,
    output_dir: Path,
    config: AppConfig,
    manual_scale: float | None = None,
    require_ai_success: bool = False,
    crop_rect: tuple[int, int, int, int] | None = None,
) -> FloorPlanModel:
    source_pixels_per_metre = _manual_pixels_per_metre(manual_scale)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    stages = StageLogger()

    started = time.perf_counter()
    image = load_image(input_path)
    stages.record("validate_input", started, image_shape=list(image.shape))

    started = time.perf_counter()
    roi_result = detect_plan_roi(
        image,
        padding_ratio=config.roi_detection.padding_ratio,
        min_confidence=config.roi_detection.min_confidence,
        crop_rect=crop_rect,
        debug_dir=debug_dir,
    )
    _write_roi_image(debug_dir / "plan_roi.png", roi_result.image)
    stages.record("detect_plan_roi", started, roi=roi_result.roi.model_dump())

    started = time.perf_counter()
    preprocessing = preprocess_array(
        roi_result.image,
        debug_dir,
        max_side=config.preprocessing.max_side_px,
        options=config.preprocessing.model_dump(),
    )
    resized_height, resized_width = preprocessing.resized_shape[:2]
    resize_ratio = max(resized_height, resized_width) / max(preprocessing.original_shape[:2])
    stages.record("preprocess_layers", started, layers={key: str(value) for key, value in preprocessing.layers.items()})

    started = time.perf_counter()
    analysis_image_path = preprocessing.layers["original_roi"]
    ocr_results = run_ocr(analysis_image_path, config.ocr.backend, config.ocr.min_confidence) if config.ocr.enabled else []
    stages.record("run_ocr", started, text_count=len(ocr_results))

    started = time.perf_counter()
    # All downstream coordinates live in the resized plan ROI.  Asking Gemini
    # to analyse the uncropped upload and then multiplying its normalized
    # coordinates by the ROI dimensions shifted every semantic label,
    # dimension, furniture footprint and warning marker whenever a page had a
    # margin or title block.
    runtime_config, ai_budget_error = _runtime_ai_config(config)
    ai_analysis, ai_hints_cached = _floorplan_vision_analysis(
        analysis_image_path,
        output_dir,
        runtime_config,
        require_ai_success,
        skip_request_error=ai_budget_error,
    )
    ai_hints = ai_analysis.hints
    stages.record(
        "gemini_semantic_hints",
        started,
        attempted=ai_analysis.attempted,
        succeeded=ai_analysis.succeeded,
        cached=ai_hints_cached,
        request_timeout_seconds=runtime_config.ai.gemini_request_timeout_seconds,
    )

    started = time.perf_counter()
    geometry_provider = build_geometry_provider(config)
    wall_detection = geometry_provider.detect(preprocessing.layers, debug_dir, config)
    if not wall_detection.walls:
        raise ValueError("no structural walls were found in the plan ROI")
    stages.record(
        "detect_raw_walls",
        started,
        provider=geometry_provider.name,
        accuracy_tier=geometry_provider.accuracy_tier,
        raw_wall_count=len(wall_detection.walls),
        band_count=len(wall_detection.bands),
    )

    # Gemini wall coordinates are proposals, never authority.  For thin CAD
    # plans the local morphology pass can fragment otherwise obvious walls;
    # recover only proposals whose full span is supported by local structural
    # pixels, and snap their centerlines back onto that evidence.
    started = time.perf_counter()
    structural_mask = _read_grayscale(
        preprocessing.layers["dark_structural_stroke"],
        "structural mask",
    )
    grounded_hints = ground_ai_wall_hints(
        ai_hints,
        structural_mask,
        resized_width,
        resized_height,
        internal_thickness_m=config.defaults.internal_wall_thickness_m,
        external_thickness_m=config.defaults.external_wall_thickness_m,
        wall_height_m=config.defaults.wall_height_m,
        min_hint_confidence=config.ai.gemini_min_confidence,
        angle_tolerance_deg=config.snapping.angle_tolerance_deg + 3.0,
        debug_dir=debug_dir,
    )
    local_colored_object_mask = colored_object_mask_from_image(
        analysis_image_path,
        max_side=config.preprocessing.max_side_px,
    )
    local_color_wall_filter = filter_walls_near_colored_objects(
        wall_detection.walls,
        local_colored_object_mask,
    )
    local_text_wall_filter = filter_walls_near_text_regions(
        local_color_wall_filter.walls,
        ocr_results,
    )
    local_object_footprints = detect_colored_object_footprints(
        analysis_image_path,
        max_side=config.preprocessing.max_side_px,
        text_regions=ocr_results,
    )
    local_footprint_wall_filter = filter_walls_inside_semantic_objects(
        local_text_wall_filter.walls,
        semantic_exclusions_from_local_footprints(local_object_footprints),
        coverage_threshold=0.70,
        margin_px=4.0,
        preserve_external=True,
    )
    grounded_detail_filter = filter_walls_crossing_repetitive_details(
        grounded_hints.walls,
        wall_detection.repetitive_detail_regions,
    )
    grounded_color_filter = filter_walls_near_colored_objects(
        grounded_detail_filter.walls,
        local_colored_object_mask,
        coverage_threshold=0.50,
        preserve_long_external=False,
    )
    # AI wall coordinates are too unstable to become geometry.  Even after
    # pixel grounding, a changed response can trace stair strings, balcony
    # hatching or furniture into a plausible-looking closed rectangle.  Keep
    # the grounded pass as diagnostics, but reconstruct exclusively from local
    # measured bands and locally measured object exclusions.
    candidate_walls_px = list(local_footprint_wall_filter.walls)
    stages.record(
        "ground_ai_wall_proposals",
        started,
        accepted_diagnostic_only=len(grounded_hints.walls),
        accepted_for_geometry=0,
        rejected_low_support=grounded_hints.rejected_low_support,
        rejected_non_axis=grounded_hints.rejected_non_axis,
        rejected_off_envelope=grounded_hints.rejected_off_envelope,
        repetitive_detail_regions=len(wall_detection.repetitive_detail_regions),
        repetitive_detail_bands_rejected=len(wall_detection.rejected_detail_bands),
        grounded_walls_rejected_as_repetitive_detail=len(grounded_detail_filter.rejected),
        local_colored_object_walls_rejected=(
            len(local_color_wall_filter.rejected) + len(grounded_color_filter.rejected)
        ),
        local_text_walls_rejected=len(local_text_wall_filter.rejected),
        local_object_footprints=len(local_object_footprints),
        local_object_footprint_walls_rejected=len(local_footprint_wall_filter.rejected),
        ai_object_boxes_used_for_geometry=False,
    )

    started = time.perf_counter()
    semantic_room_labels, semantic_special_labels = _semantic_labels_from_gemini(ai_hints, resized_width, resized_height)
    gemini_dimension_labels = _dimension_labels_from_gemini(ai_hints, resized_width, resized_height)
    local_room_labels = _ocr_labels(ocr_results)
    local_has_room_names = any(label.semantic_type != "dimension" for label in local_room_labels)
    local_has_dimensions = any(label.semantic_type == "dimension" for label in local_room_labels)
    fallback_room_labels = [] if local_has_room_names else semantic_room_labels
    fallback_dimension_labels = [] if local_has_dimensions else gemini_dimension_labels
    all_room_labels = [
        *local_room_labels,
        *fallback_room_labels,
        *fallback_dimension_labels,
    ]

    # Raw wall bands are still in pixel units here, so disable metric junction
    # snapping; bands cross each other, which polygonize nodes on its own.
    preliminary_rooms = extract_rooms_from_walls(
        candidate_walls_px,
        all_room_labels,
        pixels_per_metre=1.0,
        junction_snap_m=0.0,
    ).rooms
    stages.record("estimate_preliminary_rooms", started, room_count=len(preliminary_rooms))

    started = time.perf_counter()
    constraints = [
        *_scale_constraints_from_dimension_annotations(ocr_results, candidate_walls_px),
        *_scale_constraints_from_rooms(preliminary_rooms),
        *_scale_constraint_from_plan_extent(
            candidate_walls_px,
            config.defaults.auto_plan_long_side_m,
        ),
        *_scale_constraints_from_door_gaps(candidate_walls_px),
        *_scale_constraints_from_wall_bands(
            wall_detection.bands,
            config.defaults.internal_wall_thickness_m,
            config.defaults.external_wall_thickness_m,
        ),
    ]
    scale_result = solve_scale(
        constraints,
        manual_pixels_per_metre=(
            source_pixels_per_metre * resize_ratio
            if source_pixels_per_metre is not None else None
        ),
        min_pixels_per_metre=config.scale_solver.min_pixels_per_metre,
        max_pixels_per_metre=config.scale_solver.max_pixels_per_metre,
        outlier_mad_factor=config.scale_solver.outlier_mad_factor,
    )
    pixels_per_metre = scale_result.pixels_per_metre
    stages.record(
        "solve_scale",
        started,
        pixels_per_metre=pixels_per_metre,
        source=scale_result.source,
        used=len(scale_result.constraints_used),
        rejected=len(scale_result.rejected_constraints),
    )

    started = time.perf_counter()
    raw_walls_m = _convert_walls_to_metres(candidate_walls_px, pixels_per_metre, resized_height)
    estimated_thickness = max(config.defaults.internal_wall_thickness_m, 1.0 / pixels_per_metre * 4.0)
    reconstruction = reconstruct_walls(
        raw_walls_m,
        estimated_thickness_m=estimated_thickness,
        angle_tolerance_deg=config.snapping.angle_tolerance_deg,
        gap_tolerance_factor=config.snapping.gap_tolerance_thickness_factor,
        merge_overlap_tolerance_factor=config.snapping.merge_overlap_tolerance_factor,
        min_wall_length_m=config.snapping.min_wall_length_m,
    )
    stages.record("optimize_wall_topology", started, optimized_wall_count=len(reconstruction.walls))

    started = time.perf_counter()
    dark_mask = _read_grayscale(
        preprocessing.layers["dark_structural_stroke"],
        "dark structural mask",
    )
    adaptive_mask = _read_grayscale(
        preprocessing.layers["adaptive_binary"],
        "adaptive binary mask",
    )
    colored_mask = _read_grayscale(
        preprocessing.layers["furniture_fixture_mask"],
        "furniture/fixture mask",
    )
    thin_mask = build_thin_line_mask(adaptive_mask, dark_mask, colored_mask)
    local_openings = detect_local_openings(
        reconstruction.walls,
        dark_mask,
        thin_mask,
        pixels_per_metre,
        resized_height,
        config.opening_detection,
        config.defaults,
        color_image=cv2.imread(str(analysis_image_path), cv2.IMREAD_COLOR),
    )
    final_walls = local_openings.walls
    stages.record(
        "detect_local_openings",
        started,
        doors=len(local_openings.doors),
        windows=len(local_openings.windows),
        ambiguous=len(local_openings.ambiguous),
        walls_merged_across_openings=len(reconstruction.walls) - len(final_walls),
    )

    started = time.perf_counter()
    room_result = extract_rooms_from_walls(
        final_walls,
        all_room_labels,
        pixels_per_metre=pixels_per_metre,
        image_height_px=resized_height,
        doors=local_openings.doors,
        windows=local_openings.windows,
        bridge_ambiguous_openings=True,
    )
    final_rooms = room_result.rooms
    unclosed_gap_reports = room_result.rejected
    # Failure to close the measured wall graph is evidence for review. A
    # morphological mask fallback used to close 90-pixel gaps indiscriminately
    # and invent rooms that disagreed with the walls rendered in the GLB.
    stages.record("extract_final_rooms", started, room_count=len(final_rooms), unclosed_gaps=len(unclosed_gap_reports))

    started = time.perf_counter()
    special_elements = _classify_special_elements(
        wall_detection.repetitive_detail_regions,
        pixels_per_metre,
        resized_height,
    )
    balconies = _balconies_from_patterned_regions(
        wall_detection.repetitive_detail_regions,
        all_room_labels,
        pixels_per_metre,
        resized_height,
    )
    balcony_ownership = reconcile_room_balcony_ownership(
        final_rooms,
        balconies,
    )
    final_rooms = balcony_ownership.rooms
    balconies = balcony_ownership.balconies
    stages.record(
        "classify_special_elements",
        started,
        element_count=len(special_elements),
        balcony_count=len(balconies),
        balcony_topology_faces_matched=(
            balcony_ownership.matched_topology_faces
        ),
        rooms_trimmed_for_balconies=(
            balcony_ownership.subtracted_room_count
        ),
    )

    started = time.perf_counter()
    grounded_furniture = ground_ai_furniture_semantics(
        local_object_footprints,
        ai_hints,
        resized_width,
        resized_height,
        pixels_per_metre,
        min_confidence=config.ai.gemini_min_confidence,
    )
    local_furniture = grounded_furniture.furniture
    furniture = fit_furniture_to_rooms(
        local_furniture,
        final_rooms,
        preserve_unassigned=True,
    )
    stages.record(
        "detect_optional_furniture",
        started,
        local_furniture=len(local_furniture),
        gemini_furniture_matched=grounded_furniture.matched_hint_count,
        gemini_furniture_rejected=grounded_furniture.rejected_hint_count,
        accepted_furniture=len(furniture),
    )

    raw_model = FloorPlanModel(
        coordinate_system=CoordinateSystem.PIXELS,
        pixels_per_metre=pixels_per_metre,
        plan_roi=roi_result.roi,
        walls=wall_detection.walls,
        metadata={"source_image": str(input_path), "coordinate_note": "raw wall coordinates are ROI pixels"},
    )
    raw_model.save_json(output_dir / "floorplan.raw.json")

    semantic_used = [f"room_label:{label.normalized_text}" for label in fallback_room_labels]
    ambiguous_openings = [
        f"{candidate.kind} on {candidate.wall_id} at {candidate.start_offset_m:.2f}-{candidate.end_offset_m:.2f} m (width heuristic only)"
        for candidate in local_openings.ambiguous
    ]
    model = FloorPlanModel(
        coordinate_system=CoordinateSystem.METRES,
        pixels_per_metre=pixels_per_metre,
        plan_roi=roi_result.roi,
        walls=final_walls,
        doors=local_openings.doors,
        windows=local_openings.windows,
        rooms=final_rooms,
        balconies=balconies,
        special_elements=special_elements,
        furniture=furniture,
        ceiling=CeilingSettings(
            enabled=config.quality.generate_ceilings,
            height_m=config.defaults.wall_height_m,
            thickness_m=config.defaults.ceiling_thickness_m,
        ),
        scale_constraints=[*scale_result.constraints_used, *scale_result.rejected_constraints],
        metadata={
            "source_image": str(input_path),
            "roi_image": str(analysis_image_path),
            "scale_source": scale_result.source,
            "scale_confidence": scale_result.confidence,
            "analysis_resize_ratio": resize_ratio,
            "manual_source_metres_per_pixel": manual_scale,
            "ai_provider": "gemini",
            "ai_assist_enabled": config.ai.gemini_enabled,
            "ai_assist_attempted": ai_analysis.attempted,
            "ai_assist_succeeded": ai_analysis.succeeded,
            "ai_assist_error": ai_analysis.error,
            "ai_hints_cached": ai_hints_cached,
            "ai_wall_hints_rejected_as_geometry": len(ai_hints.walls) if ai_hints else 0,
            "ai_wall_hints_grounded_to_local_evidence": 0,
            "ai_room_hints_rejected_as_geometry": len(ai_hints.rooms) if ai_hints else 0,
            "geometry_originated_only_from_ai": False,
            "ocr_text_count": len(ocr_results),
            "gemini_semantic_label_count": len(semantic_room_labels) + len(semantic_special_labels),
            "gemini_dimension_text_count": len(gemini_dimension_labels),
            "unclosed_wall_gaps": unclosed_gap_reports,
            "local_furniture_count": len(local_furniture),
            "gemini_furniture_count": grounded_furniture.matched_hint_count,
            "gemini_furniture_rejected_unmatched": grounded_furniture.rejected_hint_count,
            "accepted_furniture_count": len(furniture),
            "raw_wall_count": len(wall_detection.walls),
            "wall_band_count": len(wall_detection.bands),
            "grounded_ai_wall_count": len(grounded_hints.walls),
            "grounded_ai_walls_used_for_geometry": 0,
            "grounded_ai_wall_rejected_low_support": grounded_hints.rejected_low_support,
            "grounded_ai_wall_rejected_off_envelope": grounded_hints.rejected_off_envelope,
            "repetitive_detail_region_count": len(wall_detection.repetitive_detail_regions),
            "grounded_balcony_count": len(balconies),
            "balcony_topology_faces_matched": (
                balcony_ownership.matched_topology_faces
            ),
            "rooms_trimmed_for_balconies": (
                balcony_ownership.subtracted_room_count
            ),
            "repetitive_detail_band_rejection_count": len(wall_detection.rejected_detail_bands),
            "grounded_wall_repetitive_detail_rejection_count": len(grounded_detail_filter.rejected),
            "local_colored_object_wall_rejection_count": (
                len(local_color_wall_filter.rejected) + len(grounded_color_filter.rejected)
            ),
            "local_object_footprint_wall_rejection_count": len(local_footprint_wall_filter.rejected),
            "semantic_object_wall_rejection_count": 0,
            "opening_detection_source": "local_evidence",
            "ambiguous_openings": ambiguous_openings,
            "room_extraction_source": "wall_graph_faces",
        },
        reconstruction=ReconstructionMetadata(
            ai_geometry_originated=False,
            stages=[*stages.stages, *reconstruction.audit_trail],
            semantic_hints_used=semantic_used,
            semantic_hints_rejected=[],
        ),
    )
    started = time.perf_counter()
    try:
        route = camera_waypoints_for_model(model)
        model = model.model_copy(
            update={
                "camera_waypoints": route,
                "metadata": {**model.metadata, "walkthrough_route_error": None},
            }
        )
    except ValueError as exc:
        model = model.model_copy(
            update={"metadata": {**model.metadata, "walkthrough_route_error": str(exc)}}
        )
    stages.record(
        "plan_walkthrough_route",
        started,
        waypoint_count=len(model.camera_waypoints),
    )
    band_h = cv2.imread(str(preprocessing.layers["horizontal_wall_band"]), cv2.IMREAD_GRAYSCALE)
    band_v = cv2.imread(str(preprocessing.layers["vertical_wall_band"]), cv2.IMREAD_GRAYSCALE)
    band_mask = cv2.bitwise_or(band_h, band_v) if band_h is not None and band_v is not None else None
    evidence = SourceEvidence(
        dark_mask=dark_mask,
        pixels_per_metre=pixels_per_metre,
        image_height_px=resized_height,
        band_mask=band_mask,
    )
    issues = validate_reconstruction(model, evidence)
    for report in ambiguous_openings:
        issues.append(ValidationIssue(code="ambiguous_opening", severity="warning", message=report))
    for gap_report in unclosed_gap_reports:
        issues.append(ValidationIssue(code="unclosed_wall_gap", severity="warning", message=gap_report))

    started = time.perf_counter()
    if ai_budget_error is not None:
        sanity = SanityCheckResult(error=ai_budget_error)
    elif ai_analysis.attempted and not ai_analysis.succeeded:
        # A second call to the same unavailable provider only delays delivery of
        # the complete local reconstruction. The semantic error is already
        # recorded, so preserve that result and skip the advisory sanity pass.
        sanity = SanityCheckResult(error="skipped after Gemini semantic analysis failed")
    else:
        sanity = GeminiLayoutSanityChecker(runtime_config.ai).check(
            analysis_image_path,
            model,
            resized_width,
            resized_height,
        )
    for warning in sanity.warnings:
        issues.append(
            ValidationIssue(
                code=f"gemini_{warning.kind}",
                severity="warning",
                message=f"{warning.description} (at {warning.x:.2f}, {warning.y:.2f} normalized)",
            )
        )
    model = model.model_copy(
        update={
            "metadata": {
                **model.metadata,
                "sanity_check_attempted": sanity.attempted,
                "sanity_check_succeeded": sanity.succeeded,
                "sanity_check_error": sanity.error,
                "sanity_warnings": [warning.model_dump() for warning in sanity.warnings],
            }
        }
    )
    stages.record("gemini_sanity_check", started, attempted=sanity.attempted, warnings=len(sanity.warnings))
    quality = evaluate_quality(model.model_copy(update={"validation_issues": issues}), evidence)
    model = model.model_copy(
        update={
            "validation_issues": issues,
            "reconstruction": model.reconstruction.model_copy(
                update={"quality_score": quality.score, "quality_state": quality.state}
            ),
            "metadata": {**model.metadata, "quality_components": quality.components},
        }
    )
    model.save_json(output_dir / "floorplan.optimized.json")
    model.save_json(output_dir / "floorplan.json")
    _write_json(output_dir / "validation_report.json", _validation_report(model))
    write_analysis_overlay(
        analysis_image_path,
        model,
        output_dir / "analysis_overlay.svg",
        output_dir / "analysis_overlay.png",
        raw_walls=raw_walls_m,
    )
    LOGGER.info("saved optimized floorplan JSON to %s", output_dir / "floorplan.optimized.json")
    return model


def _select_floorplan_for_build(path: Path) -> Path:
    if path.is_dir():
        for name in ("floorplan.corrected.json", "floorplan.optimized.json", "floorplan.json"):
            candidate = path / name
            if candidate.exists():
                return candidate
    return path


def build_model(
    floorplan_path: Path,
    output_glb: Path,
    config: AppConfig,
    run_blender: bool = False,
    force: bool = False,
    bake_mode: str | None = None,
) -> Path:
    """Export a GLB, gated on reconstruction quality.

    Low-quality scenes are blocked so silently-bad exports never happen;
    `force=True` is the explicit user override.
    """
    selected = _select_floorplan_for_build(floorplan_path)
    model = FloorPlanModel.load_json(selected)
    if not force:
        evidence = load_source_evidence(selected.parent, model)
        deterministic = validate_reconstruction(model, evidence)
        review_issues = [
            issue
            for issue in model.validation_issues
            if issue.code.startswith("gemini_")
            or issue.code in {"ambiguous_opening", "unclosed_wall_gap"}
        ]
        issues: list[ValidationIssue] = []
        seen: set[tuple[str, str]] = set()
        for issue in [*deterministic, *review_issues]:
            key = (issue.code, issue.message)
            if key not in seen:
                seen.add(key)
                issues.append(issue)
        report = evaluate_quality(model.model_copy(update={"validation_issues": issues}), evidence)
        score, state = report.score, report.state
        blocking = [issue for issue in issues if issue.severity in {"error", "severe"}]
        if blocking and config.overlay.severe_error_blocks_glb:
            raise ValueError(
                "cannot generate GLB with unresolved validation errors: "
                f"{', '.join(issue.code for issue in blocking)}; fix them or pass force=True to override"
            )
        if state == "failed":
            raise ValueError("cannot generate GLB: reconstruction quality is 'failed'; fix the issues in the correction editor or pass force=True to override")
        if score < config.reconstruction_quality.min_glb_quality_score:
            raise ValueError(
                f"cannot generate GLB: quality score {score:.2f} is below the "
                f"export threshold {config.reconstruction_quality.min_glb_quality_score:.2f}; review the validation "
                "report in the correction editor or pass force=True to override"
            )
    exported = export_floorplan_glb(model, output_glb, config, run_blender=run_blender, bake_mode=bake_mode)
    if run_blender and config.optimize.enabled:
        optimized = optimize_glb(exported, exported.with_suffix(".opt.glb"), config, texture_size=config.optimize.texture_size)
        if optimized != exported and optimized.exists():
            optimized.replace(exported)
    if config.export.validation_enabled:
        validation = validate_glb(exported)
        if not validation.get("valid"):
            issues = validation.get("issues") or ["unknown GLB validation failure"]
            raise RuntimeError(f"generated GLB failed validation: {'; '.join(str(issue) for issue in issues)}")
    target_bytes = config.optimize.target_max_mb * 1024 * 1024
    if exported.stat().st_size > target_bytes:
        actual_mb = exported.stat().st_size / (1024 * 1024)
        raise RuntimeError(
            f"generated GLB is {actual_mb:.2f} MiB, above the configured "
            f"{config.optimize.target_max_mb} MiB browser-delivery limit"
        )
    return exported


def convert_image_to_glb(
    input_image: Path,
    output_glb: Path,
    config: AppConfig,
    manual_scale: float | None = None,
    work_dir: Path | None = None,
    run_blender: bool = False,
    require_ai_success: bool = False,
    crop_rect: tuple[int, int, int, int] | None = None,
) -> Path:
    validate_image_file(input_image, config.limits)
    if output_glb.suffix.lower() != ".glb":
        raise ValueError("output path must end with .glb")
    work_dir = work_dir or output_glb.parent / f"{output_glb.stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    analyze_image(
        input_image,
        work_dir,
        config,
        manual_scale=manual_scale,
        require_ai_success=require_ai_success,
        crop_rect=crop_rect,
    )
    return build_model(work_dir, output_glb, config, run_blender=run_blender)


def prepare_walkthrough_floorplan(floorplan_path: Path, output_path: Path) -> FloorPlanModel:
    model = FloorPlanModel.load_json(_select_floorplan_for_build(floorplan_path))
    waypoints = camera_waypoints_for_model(model)
    updated = model.model_copy(update={"camera_waypoints": waypoints})
    updated.save_json(output_path)
    return updated


def render_walkthrough(floorplan_path: Path, output_mp4: Path, config: AppConfig, mode: str = "preview") -> Path:
    frames_dir = output_mp4.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_mp4.with_suffix(".glb")
    model = prepare_walkthrough_floorplan(floorplan_path, output_mp4.with_suffix(".floorplan.json"))
    frame_count = min(config.limits.max_frames, 120 if mode == "preview" else 360)
    script_path = output_mp4.with_suffix(".walkthrough.blender.py")
    script_path.write_text(
        build_blender_script(
            model,
            model_path,
            render_frames_dir=frames_dir,
            frame_count=frame_count,
            config=config,
        ),
        encoding="utf-8",
    )
    run_blender_script(str(config.paths.blender_executable), script_path, config.limits.subprocess_timeout_seconds)
    return encode_frames_to_mp4(frames_dir / "frame_%04d.png", output_mp4, config)
