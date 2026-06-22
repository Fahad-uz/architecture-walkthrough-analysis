from __future__ import annotations

import logging
from pathlib import Path

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.cleanup import cleanup_walls
from architecture_walkthrough.geometry.models import CoordinateSystem, FloorPlanModel
from architecture_walkthrough.geometry.models import Point2D, WallSegment
from architecture_walkthrough.geometry.scale import ScaleConverter
from architecture_walkthrough.scene.export_glb import export_floorplan_glb
from architecture_walkthrough.scene.blender_runner import run_blender_script
from architecture_walkthrough.scene.scene_builder import build_blender_script
from architecture_walkthrough.vision.preprocessing import preprocess_image
from architecture_walkthrough.vision.wall_detection import detect_wall_lines
from architecture_walkthrough.walkthrough.camera_animation import waypoints_from_points
from architecture_walkthrough.walkthrough.path_planner import manual_or_auto_waypoints
from architecture_walkthrough.walkthrough.render_video import encode_frames_to_mp4
from architecture_walkthrough.security.file_validation import validate_image_file

LOGGER = logging.getLogger(__name__)


def _convert_walls_to_metres(walls: list[WallSegment], pixels_per_metre: float) -> list[WallSegment]:
    converter = ScaleConverter(pixels_per_metre=pixels_per_metre)
    converted: list[WallSegment] = []
    for wall in walls:
        converted.append(
            wall.model_copy(
                update={
                    "start": Point2D(x=converter.px_to_m(wall.start.x), y=converter.px_to_m(wall.start.y)),
                    "end": Point2D(x=converter.px_to_m(wall.end.x), y=converter.px_to_m(wall.end.y)),
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
    walls = detect_wall_lines(
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
    walls = _convert_walls_to_metres(walls, pixels_per_metre)
    if not walls:
        walls = _fallback_perimeter_walls(resized_width, resized_height, pixels_per_metre, config)
    model = FloorPlanModel(
        coordinate_system=CoordinateSystem.METRES,
        pixels_per_metre=pixels_per_metre,
        walls=cleanup_walls(walls),
        metadata={
            "source_image": str(input_path),
            "preprocessing": {key: str(value) for key, value in result.__dict__.items()},
            "scale_source": scale_source,
            "approximate_reconstruction": True,
        },
    )
    model.save_json(output_dir / "floorplan.json")
    LOGGER.info("saved floorplan JSON to %s", output_dir / "floorplan.json")
    return model


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
