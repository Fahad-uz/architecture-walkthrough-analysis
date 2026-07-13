from __future__ import annotations

import logging
from pathlib import Path

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.geometry.models import FloorPlanModel
from architecture_walkthrough.security.sandbox import require_executable, run_subprocess

LOGGER = logging.getLogger(__name__)

TEMPLATE = Path(__file__).parent / "blender_templates" / "generate_building.py"

BAKE_MODES = ("final", "draft", "none")


def blender_generate_command(
    blender_executable: str,
    floorplan_json: Path,
    output_glb: Path,
    mode: str,
    samples: int,
    lightmap_px: int,
    denoise: bool,
) -> list[str]:
    command = [
        blender_executable,
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(TEMPLATE),
        "--",
        "--floorplan",
        str(floorplan_json),
        "--output",
        str(output_glb),
        "--mode",
        mode,
        "--samples",
        str(samples),
        "--lightmap-px",
        str(lightmap_px),
    ]
    if not denoise:
        command.append("--no-denoise")
    return command


def generate_glb_with_blender(
    model: FloorPlanModel,
    output_glb: Path,
    config: AppConfig,
    mode: str | None = None,
) -> Path:
    """Generate a lit GLB via headless Blender.

    Geometry is identical across bake modes; `mode` only changes lighting:
    final (default, full-quality Cycles bake), draft (fast bake), none
    (no bake, KHR_lights_punctual exported for real-time fallback).
    """
    bake_mode = mode or config.bake.mode
    if bake_mode not in BAKE_MODES:
        raise ValueError(f"unknown bake mode: {bake_mode}; expected one of {BAKE_MODES}")
    executable = require_executable(str(config.paths.blender_executable), "Blender")
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    floorplan_json = output_glb.with_suffix(".blender_input.json")
    model.save_json(floorplan_json)
    samples = config.bake.final_samples if bake_mode == "final" else config.bake.draft_samples
    lightmap = config.bake.final_lightmap_px if bake_mode == "final" else config.bake.draft_lightmap_px
    command = blender_generate_command(
        executable, floorplan_json, output_glb, bake_mode, samples, lightmap, config.bake.denoise
    )
    timeout = config.bake.timeout_seconds if bake_mode != "none" else config.limits.subprocess_timeout_seconds
    LOGGER.info("running Blender generator (mode=%s, samples=%s, lightmap=%spx)", bake_mode, samples, lightmap)
    try:
        result = run_subprocess(command, timeout_seconds=timeout)
    except Exception as exc:
        stderr = getattr(exc, "stderr", "") or ""
        stdout = getattr(exc, "stdout", "") or ""
        raise RuntimeError(
            f"Blender generation failed.\nstderr tail: {stderr[-2000:]}\nstdout tail: {stdout[-1500:]}"
        ) from exc
    if not output_glb.exists():
        raise RuntimeError(
            f"Blender completed without creating the GLB: {output_glb}\n"
            f"stderr tail: {result.stderr[-2000:]}\nstdout tail: {result.stdout[-1500:]}"
        )
    return output_glb
