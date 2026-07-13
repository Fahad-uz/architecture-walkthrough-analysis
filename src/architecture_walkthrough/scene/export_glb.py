from __future__ import annotations

from pathlib import Path

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.models import FloorPlanModel
from architecture_walkthrough.scene.blender_generator import generate_glb_with_blender
from architecture_walkthrough.scene.simple_glb import export_simple_glb


def export_floorplan_glb(
    model: FloorPlanModel,
    output_glb: Path,
    config: AppConfig,
    run_blender: bool = False,
    bake_mode: str | None = None,
) -> Path:
    """Blender is the quality path (booleans, PBR, baked lighting); the pure-
    Python trimesh exporter remains the instant preview/testing fallback."""
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    if not run_blender:
        return export_simple_glb(model, output_glb)
    return generate_glb_with_blender(model, output_glb, config, mode=bake_mode)
