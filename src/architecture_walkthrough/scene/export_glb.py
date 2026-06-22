from __future__ import annotations

from pathlib import Path

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.models import FloorPlanModel
from architecture_walkthrough.scene.blender_runner import run_blender_script
from architecture_walkthrough.scene.scene_builder import build_blender_script
from architecture_walkthrough.scene.simple_glb import export_simple_glb


def export_floorplan_glb(model: FloorPlanModel, output_glb: Path, config: AppConfig, run_blender: bool = False) -> Path:
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    if not run_blender:
        return export_simple_glb(model, output_glb)
    script_path = output_glb.with_suffix(".blender.py")
    preview = output_glb.with_suffix(".preview.png")
    blend = output_glb.with_suffix(".blend")
    script_path.write_text(build_blender_script(model, output_glb, preview, blend), encoding="utf-8")
    if run_blender:
        run_blender_script(str(config.paths.blender_executable), script_path, config.limits.subprocess_timeout_seconds)
        if not output_glb.exists():
            raise RuntimeError(f"Blender completed without creating expected GLB: {output_glb}")
    return output_glb
