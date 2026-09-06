from __future__ import annotations

from pathlib import Path

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.models import FloorPlanModel
from architecture_walkthrough.pipeline import analyze_image, build_model
from architecture_walkthrough.pipeline import convert_image_to_glb as _convert_image_to_glb


def analyze_floorplan_image(
    input_path: Path,
    output_dir: Path,
    config: AppConfig,
    manual_scale: float | None = None,
    require_ai_success: bool = False,
    crop_rect: tuple[int, int, int, int] | None = None,
) -> FloorPlanModel:
    """Analyze a floor-plan image and write reconstruction artifacts."""
    return analyze_image(
        input_path,
        output_dir,
        config,
        manual_scale=manual_scale,
        require_ai_success=require_ai_success,
        crop_rect=crop_rect,
    )


def build_glb_model(
    floorplan_path: Path,
    output_glb: Path,
    config: AppConfig,
    run_blender: bool = False,
    force: bool = False,
    bake_mode: str | None = None,
) -> Path:
    """Build a GLB from a corrected, optimized, or compatibility floorplan JSON."""
    return build_model(
        floorplan_path, output_glb, config, run_blender=run_blender, force=force, bake_mode=bake_mode
    )


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
    """Run the full image analysis and GLB generation flow."""
    return _convert_image_to_glb(
        input_image,
        output_glb,
        config,
        manual_scale=manual_scale,
        work_dir=work_dir,
        run_blender=run_blender,
        require_ai_success=require_ai_success,
        crop_rect=crop_rect,
    )
