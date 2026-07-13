from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2

from architecture_walkthrough.ai.floorplan_vision import GeminiFloorPlanVisionAnalyzer
from architecture_walkthrough.ai.sanity_check import GeminiLayoutSanityChecker
from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.models import (
    ArchitecturalElement,
    CoordinateSystem,
    FloorPlanModel,
    FurniturePlacement,
    Point2D,
    ReconstructionMetadata,
    ValidationIssue,
    WallSegment,
)
from architecture_walkthrough.geometry.furniture_layout import deduplicate_furniture, fit_furniture_to_rooms
from architecture_walkthrough.geometry.reconstruction import reconstruct_walls
from architecture_walkthrough.geometry.room_extraction import (
    extract_rooms_from_geometry_mask,
    extract_rooms_from_walls,
)
from architecture_walkthrough.geometry.scale import ScaleConverter
from architecture_walkthrough.geometry.scale_solver import ScaleConstraint, solve_scale
from architecture_walkthrough.geometry.wall_graph import collinear_gaps
from architecture_walkthrough.geometry.validation import (
    SourceEvidence,
    evaluate_quality,
    validate_reconstruction,
)
from architecture_walkthrough.scene.export_glb import export_floorplan_glb
from architecture_walkthrough.scene.glb_optimizer import optimize_glb
from architecture_walkthrough.scene.blender_runner import run_blender_script
from architecture_walkthrough.scene.scene_builder import build_blender_script
from architecture_walkthrough.security.file_validation import validate_image_file
from architecture_walkthrough.vision.ocr import OCRText, parse_dimension_pair, run_ocr
from architecture_walkthrough.vision.furniture_detection import detect_furniture_from_image
from architecture_walkthrough.vision.local_openings import build_thin_line_mask, detect_local_openings
from architecture_walkthrough.vision.overlay import write_analysis_overlay
from architecture_walkthrough.vision.plan_roi import detect_plan_roi
from architecture_walkthrough.vision.preprocessing import load_image, preprocess_array
from architecture_walkthrough.vision.providers import build_geometry_provider
from architecture_walkthrough.vision.wall_detection import WallBand
from architecture_walkthrough.walkthrough.camera_animation import waypoints_from_points
from architecture_walkthrough.walkthrough.path_planner import manual_or_auto_waypoints
from architecture_walkthrough.walkthrough.render_video import encode_frames_to_mp4

LOGGER = logging.getLogger(__name__)


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
    if manual_scale <= 0:
        raise ValueError("manual scale must be positive metres per pixel")
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
    constraints: list[ScaleConstraint] = []
    for gap in gaps:
        gap_px = float(gap["length_m"])  # pixel units here: walls are in px
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


def _classify_special_elements(ocr_results: list[OCRText], pixels_per_metre: float, image_height_px: int) -> list[ArchitecturalElement]:
    elements: list[ArchitecturalElement] = []
    converter = ScaleConverter(pixels_per_metre=pixels_per_metre)
    for item in ocr_results:
        kind = None
        if item.semantic_type == "stair_label":
            kind = "staircase"
        elif item.semantic_type == "lift_label":
            kind = "lift"
        elif item.semantic_type == "balcony_label":
            kind = "balcony"
        elif "shelf" in item.normalized_text.lower() or "cabinet" in item.normalized_text.lower():
            kind = "shelf"
        elif "counter" in item.normalized_text.lower() or "kitchen" in item.normalized_text.lower():
            kind = "kitchen_counter"
        elif "entrance" in item.normalized_text.lower():
            kind = "entrance"
        if kind is None:
            continue
        cx = sum(point[0] for point in item.polygon) / len(item.polygon)
        cy = sum(point[1] for point in item.polygon) / len(item.polygon)
        elements.append(
            ArchitecturalElement(
                id=f"{kind}_{len(elements):03d}",
                kind=kind,
                center=Point2D(x=converter.px_to_m(cx), y=converter.px_to_m(image_height_px - cy)),
                width_m=max(0.3, converter.px_to_m(max(point[0] for point in item.polygon) - min(point[0] for point in item.polygon))),
                depth_m=max(0.3, converter.px_to_m(max(point[1] for point in item.polygon) - min(point[1] for point in item.polygon))),
                confidence=item.confidence,
                evidence_source="ocr_label",
            )
        )
    return elements


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
    stages.record("preprocess_layers", started, layers={key: str(value) for key, value in preprocessing.layers.items()})

    started = time.perf_counter()
    analysis_image_path = preprocessing.layers["original_roi"]
    ocr_results = run_ocr(analysis_image_path, config.ocr.backend, config.ocr.min_confidence) if config.ocr.enabled else []
    stages.record("run_ocr", started, text_count=len(ocr_results))

    started = time.perf_counter()
    ai_analysis = GeminiFloorPlanVisionAnalyzer(config.ai).analyze_with_diagnostics(
        input_path,
        require_success=require_ai_success,
    )
    ai_hints = ai_analysis.hints
    stages.record("gemini_semantic_hints", started, attempted=ai_analysis.attempted, succeeded=ai_analysis.succeeded)

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

    started = time.perf_counter()
    semantic_room_labels, semantic_special_labels = _semantic_labels_from_gemini(ai_hints, resized_width, resized_height)
    gemini_dimension_labels = _dimension_labels_from_gemini(ai_hints, resized_width, resized_height)
    all_room_labels = [*_ocr_labels(ocr_results), *semantic_room_labels, *gemini_dimension_labels]

    # Raw wall bands are still in pixel units here, so disable metric junction
    # snapping; bands cross each other, which polygonize nodes on its own.
    preliminary_rooms = extract_rooms_from_walls(
        wall_detection.walls,
        all_room_labels,
        pixels_per_metre=1.0,
        junction_snap_m=0.0,
    ).rooms
    stages.record("estimate_preliminary_rooms", started, room_count=len(preliminary_rooms))

    started = time.perf_counter()
    constraints = [
        *_scale_constraints_from_rooms(preliminary_rooms),
        *_scale_constraints_from_door_gaps(wall_detection.walls),
        *_scale_constraints_from_wall_bands(
            wall_detection.bands,
            config.defaults.internal_wall_thickness_m,
            config.defaults.external_wall_thickness_m,
        ),
    ]
    scale_result = solve_scale(
        constraints,
        manual_pixels_per_metre=_manual_pixels_per_metre(manual_scale),
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
    raw_walls_m = _convert_walls_to_metres(wall_detection.walls, pixels_per_metre, resized_height)
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
    dark_mask = cv2.imread(str(preprocessing.layers["dark_structural_stroke"]), cv2.IMREAD_GRAYSCALE)
    adaptive_mask = cv2.imread(str(preprocessing.layers["adaptive_binary"]), cv2.IMREAD_GRAYSCALE)
    colored_mask = cv2.imread(str(preprocessing.layers["furniture_fixture_mask"]), cv2.IMREAD_GRAYSCALE)
    thin_mask = build_thin_line_mask(adaptive_mask, dark_mask, colored_mask)
    local_openings = detect_local_openings(
        reconstruction.walls,
        dark_mask,
        thin_mask,
        pixels_per_metre,
        resized_height,
        config.opening_detection,
        config.defaults,
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
    )
    final_rooms = room_result.rooms
    unclosed_gap_reports = room_result.rejected
    if not final_rooms:
        final_rooms = extract_rooms_from_geometry_mask(
            preprocessing.layers["cleaned_geometry_only"],
            all_room_labels,
            pixels_per_metre=pixels_per_metre,
            image_height_px=resized_height,
        ).rooms
    stages.record("extract_final_rooms", started, room_count=len(final_rooms), unclosed_gaps=len(unclosed_gap_reports))

    started = time.perf_counter()
    special_elements = _classify_special_elements([*ocr_results, *semantic_special_labels], pixels_per_metre, resized_height)
    stages.record("classify_special_elements", started, element_count=len(special_elements))

    started = time.perf_counter()
    local_furniture = detect_furniture_from_image(analysis_image_path, pixels_per_metre, resized_height, max_side=config.preprocessing.max_side_px)
    gemini_furniture = _furniture_from_gemini(
        ai_hints,
        resized_width,
        resized_height,
        pixels_per_metre,
        config.ai.gemini_min_confidence,
    )
    furniture = fit_furniture_to_rooms(deduplicate_furniture([*local_furniture, *gemini_furniture]), final_rooms)
    stages.record(
        "detect_optional_furniture",
        started,
        local_furniture=len(local_furniture),
        gemini_furniture=len(gemini_furniture),
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

    semantic_used = [f"room_label:{label.normalized_text}" for label in semantic_room_labels]
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
        special_elements=special_elements,
        furniture=furniture,
        scale_constraints=[*scale_result.constraints_used, *scale_result.rejected_constraints],
        metadata={
            "source_image": str(input_path),
            "roi_image": str(analysis_image_path),
            "scale_source": scale_result.source,
            "scale_confidence": scale_result.confidence,
            "ai_provider": "gemini",
            "ai_assist_enabled": config.ai.gemini_enabled,
            "ai_assist_attempted": ai_analysis.attempted,
            "ai_assist_succeeded": ai_analysis.succeeded,
            "ai_assist_error": ai_analysis.error,
            "ai_wall_hints_rejected_as_geometry": len(ai_hints.walls) if ai_hints else 0,
            "ai_room_hints_rejected_as_geometry": len(ai_hints.rooms) if ai_hints else 0,
            "geometry_originated_only_from_ai": False,
            "ocr_text_count": len(ocr_results),
            "gemini_semantic_label_count": len(semantic_room_labels) + len(semantic_special_labels),
            "unclosed_wall_gaps": unclosed_gap_reports,
            "local_furniture_count": len(local_furniture),
            "gemini_furniture_count": len(gemini_furniture),
            "accepted_furniture_count": len(furniture),
            "raw_wall_count": len(wall_detection.walls),
            "wall_band_count": len(wall_detection.bands),
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
    issues = validate_reconstruction(model)
    for report in ambiguous_openings:
        issues.append(ValidationIssue(code="ambiguous_opening", severity="warning", message=report))
    for gap_report in unclosed_gap_reports:
        issues.append(ValidationIssue(code="unclosed_wall_gap", severity="warning", message=gap_report))

    started = time.perf_counter()
    sanity = GeminiLayoutSanityChecker(config.ai).check(analysis_image_path, model, resized_width, resized_height)
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
    band_h = cv2.imread(str(preprocessing.layers["horizontal_wall_band"]), cv2.IMREAD_GRAYSCALE)
    band_v = cv2.imread(str(preprocessing.layers["vertical_wall_band"]), cv2.IMREAD_GRAYSCALE)
    band_mask = cv2.bitwise_or(band_h, band_v) if band_h is not None and band_v is not None else None
    evidence = SourceEvidence(
        dark_mask=dark_mask,
        pixels_per_metre=pixels_per_metre,
        image_height_px=resized_height,
        band_mask=band_mask,
    )
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
        score = model.reconstruction.quality_score
        state = model.reconstruction.quality_state
        issues = model.validation_issues
        if not issues and score == 0.0:
            # Hand-authored or corrected JSON that never went through scoring:
            # assess it now instead of trusting (or zero-blocking) stale metadata.
            report = evaluate_quality(model)
            score, state, issues = report.score, report.state, report.issues
        severe = [issue for issue in issues if issue.severity == "severe"]
        if severe and config.overlay.severe_error_blocks_glb:
            raise ValueError(f"cannot generate GLB with severe validation errors: {', '.join(issue.code for issue in severe)}")
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
    path = manual_or_auto_waypoints(model)
    updated = model.model_copy(update={"camera_waypoints": waypoints_from_points(path)})
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
        build_blender_script(model, model_path, render_frames_dir=frames_dir, frame_count=frame_count),
        encoding="utf-8",
    )
    run_blender_script(str(config.paths.blender_executable), script_path, config.limits.subprocess_timeout_seconds)
    return encode_frames_to_mp4(frames_dir / "frame_%04d.png", output_mp4, config)
