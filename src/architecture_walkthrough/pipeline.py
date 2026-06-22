from __future__ import annotations

import logging
from pathlib import Path

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.ai.floorplan_vision import OpenAIFloorPlanVisionAnalyzer, hints_to_floorplan_geometry
from architecture_walkthrough.geometry.cleanup import cleanup_walls
from architecture_walkthrough.geometry.models import CoordinateSystem, FloorPlanModel
from architecture_walkthrough.geometry.models import Point2D, WallSegment
from architecture_walkthrough.geometry.scale import ScaleConverter
from architecture_walkthrough.scene.export_glb import export_floorplan_glb
from architecture_walkthrough.scene.blender_runner import run_blender_script
from architecture_walkthrough.scene.scene_builder import build_blender_script
from architecture_walkthrough.vision.preprocessing import preprocess_image
from architecture_walkthrough.vision.furniture_detection import detect_furniture_from_image, suggest_rendered_plan_furniture
from architecture_walkthrough.vision.wall_detection import detect_wall_lines
from architecture_walkthrough.walkthrough.camera_animation import waypoints_from_points
from architecture_walkthrough.walkthrough.path_planner import manual_or_auto_waypoints
from architecture_walkthrough.walkthrough.render_video import encode_frames_to_mp4
from architecture_walkthrough.security.file_validation import validate_image_file

LOGGER = logging.getLogger(__name__)


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


def _fallback_perimeter_walls(width_px: int, height_px: int, pixels_per_metre: float, config: AppConfig) -> list[WallSegment]:
    converter = ScaleConverter(pixels_per_metre=pixels_per_metre)
    width_m = converter.px_to_m(width_px)
    height_m = converter.px_to_m(height_px)
    thickness = config.defaults.external_wall_thickness_m
    wall_height = config.defaults.wall_height_m
    return [
        WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=width_m, y=0), thickness_m=thickness, height_m=wall_height, external=True),
        WallSegment(start=Point2D(x=width_m, y=0), end=Point2D(x=width_m, y=height_m), thickness_m=thickness, height_m=wall_height, external=True),
        WallSegment(start=Point2D(x=width_m, y=height_m), end=Point2D(x=0, y=height_m), thickness_m=thickness, height_m=wall_height, external=True),
        WallSegment(start=Point2D(x=0, y=height_m), end=Point2D(x=0, y=0), thickness_m=thickness, height_m=wall_height, external=True),
    ]


def analyze_image(input_path: Path, output_dir: Path, config: AppConfig, manual_scale: float | None = None) -> FloorPlanModel:
    debug_dir = output_dir / "debug"
    result = preprocess_image(input_path, debug_dir)
    cv_walls = detect_wall_lines(
        result.edges_path,
        thickness_m=config.defaults.internal_wall_thickness_m,
        height_m=config.defaults.wall_height_m,
    )
    resized_height, resized_width = result.resized_shape[:2]
    if manual_scale:
        pixels_per_metre = 1.0 / manual_scale
        scale_source = "manual"
    else:
        pixels_per_metre = max(resized_width, resized_height) / config.defaults.auto_plan_long_side_m
        scale_source = "auto_assumed_long_side"
    ai_hints = OpenAIFloorPlanVisionAnalyzer(config.ai).analyze(input_path)
    detected_furniture = detect_furniture_from_image(input_path, pixels_per_metre, resized_height)
    template_furniture = suggest_rendered_plan_furniture(resized_width, resized_height, pixels_per_metre)
    rooms = []
    doors = []
    windows = []
    furniture = []
    ai_wall_count = 0
    ai_furniture_count = 0
    if ai_hints:
        ai_walls, rooms, doors, windows, furniture = hints_to_floorplan_geometry(
            ai_hints,
            image_width_px=resized_width,
            image_height_px=resized_height,
            pixels_per_metre=pixels_per_metre,
            config=config,
        )
        ai_wall_count = len(ai_walls)
        ai_furniture_count = len(furniture)
        walls = ai_walls if ai_walls else _convert_walls_to_metres(cv_walls, pixels_per_metre, resized_height)
        if not furniture:
            furniture = detected_furniture
    else:
        walls = _convert_walls_to_metres(cv_walls, pixels_per_metre, resized_height)
        furniture = detected_furniture
    semantic_categories = {item.category for item in furniture}
    if len(furniture) < 15 or semantic_categories <= {"furniture"}:
        furniture = _merge_furniture(furniture, template_furniture)
    if not walls:
        walls = _fallback_perimeter_walls(resized_width, resized_height, pixels_per_metre, config)
    model = FloorPlanModel(
        coordinate_system=CoordinateSystem.METRES,
        pixels_per_metre=pixels_per_metre,
        walls=cleanup_walls(walls),
        doors=doors,
        windows=windows,
        rooms=rooms,
        furniture=furniture,
        metadata={
            "source_image": str(input_path),
            "preprocessing": {key: str(value) for key, value in result.__dict__.items()},
            "scale_source": scale_source,
            "approximate_reconstruction": True,
            "ai_assist_enabled": config.ai.openai_enabled,
            "ai_wall_hints_used": ai_wall_count,
            "ai_furniture_hints_used": ai_furniture_count,
            "local_furniture_hints_used": len(detected_furniture),
            "template_furniture_hints_used": len(template_furniture),
        },
    )
    model.save_json(output_dir / "floorplan.json")
    LOGGER.info("saved floorplan JSON to %s", output_dir / "floorplan.json")
    return model


def _merge_furniture(base: list, additions: list) -> list:
    merged = list(base)
    always_keep_prefixes = (
        "floor_patch",
        "railing",
        "stair",
        "door",
        "window",
        "lamp",
        "sink",
        "stove",
        "appliance",
        "tv",
        "wardrobe",
    )
    for item in additions:
        if item.category.startswith(always_keep_prefixes):
            merged.append(item)
            continue
        if any(
            item.category == existing.category
            and item.center.distance_to(existing.center) < max(0.35, min(item.width_m, item.depth_m) * 0.5)
            for existing in merged
        ):
            continue
        merged.append(item)
    return merged


def build_model(floorplan_path: Path, output_glb: Path, config: AppConfig, run_blender: bool = False) -> Path:
    model = FloorPlanModel.load_json(floorplan_path)
    return export_floorplan_glb(model, output_glb, config, run_blender=run_blender)


def convert_image_to_glb(
    input_image: Path,
    output_glb: Path,
    config: AppConfig,
    manual_scale: float | None = None,
    work_dir: Path | None = None,
    run_blender: bool = False,
) -> Path:
    validate_image_file(input_image, config.limits)
    if output_glb.suffix.lower() != ".glb":
        raise ValueError("output path must end with .glb")
    work_dir = work_dir or output_glb.parent / f"{output_glb.stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    model = analyze_image(input_image, work_dir, config, manual_scale=manual_scale)
    floorplan_path = work_dir / "floorplan.json"
    model.save_json(floorplan_path)
    return build_model(floorplan_path, output_glb, config, run_blender=run_blender)


def prepare_walkthrough_floorplan(floorplan_path: Path, output_path: Path) -> FloorPlanModel:
    model = FloorPlanModel.load_json(floorplan_path)
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
