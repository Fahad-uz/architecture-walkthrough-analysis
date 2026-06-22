from __future__ import annotations

import logging
from pathlib import Path

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.cleanup import cleanup_walls
from architecture_walkthrough.geometry.models import CoordinateSystem, FloorPlanModel
from architecture_walkthrough.geometry.scale import ScaleConverter
from architecture_walkthrough.scene.export_glb import export_floorplan_glb
from architecture_walkthrough.scene.blender_runner import run_blender_script
from architecture_walkthrough.scene.scene_builder import build_blender_script
from architecture_walkthrough.vision.preprocessing import preprocess_image
from architecture_walkthrough.vision.wall_detection import detect_wall_lines
from architecture_walkthrough.walkthrough.camera_animation import waypoints_from_points
from architecture_walkthrough.walkthrough.path_planner import manual_or_auto_waypoints
from architecture_walkthrough.walkthrough.render_video import encode_frames_to_mp4

LOGGER = logging.getLogger(__name__)


def analyze_image(input_path: Path, output_dir: Path, config: AppConfig, manual_scale: float | None = None) -> FloorPlanModel:
    debug_dir = output_dir / "debug"
    result = preprocess_image(input_path, debug_dir)
    walls = detect_wall_lines(
        result.edges_path,
        thickness_m=config.defaults.internal_wall_thickness_m,
        height_m=config.defaults.wall_height_m,
    )
    if manual_scale:
        converter = ScaleConverter(pixels_per_metre=1.0 / manual_scale)
        walls = [
            wall.model_copy(
                update={
                    "start": wall.start.model_copy(update={"x": converter.px_to_m(wall.start.x), "y": converter.px_to_m(wall.start.y)}),
                    "end": wall.end.model_copy(update={"x": converter.px_to_m(wall.end.x), "y": converter.px_to_m(wall.end.y)}),
                }
            )
            for wall in walls
        ]
        coordinate_system = CoordinateSystem.METRES
        pixels_per_metre = converter.pixels_per_metre
    else:
        coordinate_system = CoordinateSystem.PIXELS
        pixels_per_metre = None
    model = FloorPlanModel(
        coordinate_system=coordinate_system,
        pixels_per_metre=pixels_per_metre,
        walls=cleanup_walls(walls),
        metadata={"source_image": str(input_path), "preprocessing": result.__dict__},
    )
    model.save_json(output_dir / "floorplan.json")
    LOGGER.info("saved floorplan JSON to %s", output_dir / "floorplan.json")
    return model


def build_model(floorplan_path: Path, output_glb: Path, config: AppConfig, run_blender: bool = True) -> Path:
    model = FloorPlanModel.load_json(floorplan_path)
    return export_floorplan_glb(model, output_glb, config, run_blender=run_blender)


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
