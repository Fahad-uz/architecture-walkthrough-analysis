from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt


class Resolution(BaseModel):
    width: PositiveInt
    height: PositiveInt


class PathSettings(BaseModel):
    work_root: Path = Path("outputs")
    blender_executable: str = "blender"
    ffmpeg_executable: str = "ffmpeg"


class LimitSettings(BaseModel):
    max_upload_mb: PositiveInt = 20
    max_image_width: PositiveInt = 6000
    max_image_height: PositiveInt = 6000
    processing_timeout_seconds: PositiveInt = 120
    subprocess_timeout_seconds: PositiveInt = 600
    max_concurrent_analyses: PositiveInt = 1
    max_concurrent_generations: PositiveInt = 1
    max_frames: PositiveInt = 900
    preview_resolution: Resolution = Resolution(width=960, height=540)
    final_resolution: Resolution = Resolution(width=1920, height=1080)


class GeometryDefaults(BaseModel):
    wall_height_m: PositiveFloat = 3.0
    external_wall_thickness_m: PositiveFloat = 0.20
    internal_wall_thickness_m: PositiveFloat = 0.12
    auto_plan_long_side_m: PositiveFloat = 12.0
    door_width_m: PositiveFloat = 0.90
    door_height_m: PositiveFloat = 2.10
    window_width_m: PositiveFloat = 1.20
    window_height_m: PositiveFloat = 1.20
    sill_height_m: PositiveFloat = 0.90
    camera_height_m: PositiveFloat = 1.65
    floor_thickness_m: PositiveFloat = 0.10
    ceiling_thickness_m: PositiveFloat = 0.08
    bevel_width_m: PositiveFloat = 0.008


class AISettings(BaseModel):
    segmentation_checkpoint: Path | None = None
    device: str = "cpu"
    gemini_enabled: bool = True
    gemini_model: str = "gemini-2.5-flash"
    gemini_min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    gemini_sanity_check_enabled: bool = True
    # Gemini is optional enrichment. Keep its complete retry budget well below
    # the 120-second analysis-worker deadline so local geometry always wins.
    gemini_request_timeout_seconds: int = Field(default=20, ge=5, le=20)
    gemini_retry_attempts: int = Field(default=2, ge=1, le=2)
    gemini_retry_base_delay_seconds: float = Field(default=2.0, ge=0.0, le=2.0)


class ROIDetectionSettings(BaseModel):
    enabled: bool = True
    padding_ratio: float = Field(default=0.02, ge=0.0, le=0.20)
    min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)


class PreprocessingSettings(BaseModel):
    max_side_px: PositiveInt = 1800
    adaptive_block_ratio: float = Field(default=0.018, gt=0.0, le=0.10)
    dark_threshold: PositiveInt = 170
    # Structural ink is black/grey; colored dark fills (kitchen counters,
    # brick hatches) are furniture, not walls.
    max_structural_saturation: PositiveInt = 80
    # CAD plans draw walls as two thin parallel lines; closing fuses them
    # into solid bands the band detector can see. Ratio of the max side.
    hollow_wall_close_ratio: float = Field(default=0.008, gt=0.0, le=0.05)
    min_symbol_area_ratio: float = Field(default=0.000002, gt=0.0, le=0.01)
    max_text_component_area_ratio: float = Field(default=0.0007, gt=0.0, le=0.05)


class WallBandSettings(BaseModel):
    min_length_ratio: float = Field(default=0.02, gt=0.0, le=0.5)
    merge_gap_ratio: float = Field(default=0.012, gt=0.0, le=0.1)
    coordinate_tolerance_ratio: float = Field(default=0.006, gt=0.0, le=0.1)
    min_thickness_px: PositiveInt = 3
    max_thickness_ratio: float = Field(default=0.04, gt=0.0, le=0.2)


class OCRSettings(BaseModel):
    backend: str = "auto"
    enabled: bool = True
    min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)


class ScaleSolverSettings(BaseModel):
    min_pixels_per_metre: PositiveFloat = 10.0
    max_pixels_per_metre: PositiveFloat = 1000.0
    outlier_mad_factor: PositiveFloat = 2.8
    min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)


class SnappingSettings(BaseModel):
    angle_tolerance_deg: PositiveFloat = 7.0
    gap_tolerance_thickness_factor: PositiveFloat = 2.5
    merge_overlap_tolerance_factor: PositiveFloat = 1.5
    min_wall_length_m: PositiveFloat = 0.20


class OpeningDetectionSettings(BaseModel):
    enabled: bool = True
    projection_tolerance_m: PositiveFloat = 0.35
    default_door_width_m: PositiveFloat = 0.90
    default_window_width_m: PositiveFloat = 1.20
    min_door_width_m: PositiveFloat = 0.55
    max_door_width_m: PositiveFloat = 1.40
    min_window_width_m: PositiveFloat = 0.45
    max_window_width_m: PositiveFloat = 3.20
    arc_coverage_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    leaf_coverage_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    window_line_coverage_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    min_flank_m: PositiveFloat = 0.25


class OverlaySettings(BaseModel):
    enabled: bool = True
    severe_error_blocks_glb: bool = True


class ReconstructionQualitySettings(BaseModel):
    min_glb_quality_score: float = Field(default=0.45, ge=0.0, le=1.0)
    allow_debug_fallback_rectangle: bool = False


class RenderSettings(BaseModel):
    preview_samples: PositiveInt = 16
    final_samples: PositiveInt = 96
    preview_fps: PositiveInt = 24
    camera_speed_mps: PositiveFloat = 1.0


class BakeSettings(BaseModel):
    # "final" (no compromises), "draft" (fast iteration), "none" (skip baking,
    # export KHR_lights_punctual so the viewer lights in real time).
    mode: str = "final"
    final_samples: PositiveInt = 256
    draft_samples: PositiveInt = 16
    final_lightmap_px: PositiveInt = 2048
    draft_lightmap_px: PositiveInt = 512
    denoise: bool = True
    timeout_seconds: PositiveInt = 7200


class TextureSettings(BaseModel):
    registry_path: Path = Path("assets/textures/material_registry.yaml")
    default_resolution: PositiveInt = 2048
    allow_4k: bool = False
    embed_in_glb: bool = True
    max_texture_size: PositiveInt = 4096


class AssetSettings(BaseModel):
    registry_path: Path = Path("assets/models/asset_registry.yaml")
    allow_placeholder_fallback: bool = True


class OptimizeSettings(BaseModel):
    enabled: bool = True
    texture_size: PositiveInt = 2048
    target_max_mb: PositiveInt = 25


class ExportSettings(BaseModel):
    include_cameras: bool = False
    include_lights: bool = False
    apply_modifiers: bool = True
    export_normals: bool = True
    export_tangents: bool = True
    validation_enabled: bool = True


class QualitySettings(BaseModel):
    preset: str = "high"
    generate_ceilings: bool = True
    generate_trims: bool = True
    generate_doors: bool = True
    generate_windows: bool = True


class AppConfig(BaseModel):
    paths: PathSettings = Field(default_factory=PathSettings)
    limits: LimitSettings = Field(default_factory=LimitSettings)
    defaults: GeometryDefaults = Field(default_factory=GeometryDefaults)
    ai: AISettings = Field(default_factory=AISettings)
    roi_detection: ROIDetectionSettings = Field(default_factory=ROIDetectionSettings)
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    wall_bands: WallBandSettings = Field(default_factory=WallBandSettings)
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    scale_solver: ScaleSolverSettings = Field(default_factory=ScaleSolverSettings)
    snapping: SnappingSettings = Field(default_factory=SnappingSettings)
    opening_detection: OpeningDetectionSettings = Field(default_factory=OpeningDetectionSettings)
    overlay: OverlaySettings = Field(default_factory=OverlaySettings)
    reconstruction_quality: ReconstructionQualitySettings = Field(default_factory=ReconstructionQualitySettings)
    render: RenderSettings = Field(default_factory=RenderSettings)
    bake: BakeSettings = Field(default_factory=BakeSettings)
    textures: TextureSettings = Field(default_factory=TextureSettings)
    assets: AssetSettings = Field(default_factory=AssetSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    optimize: OptimizeSettings = Field(default_factory=OptimizeSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)


def _environment_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _apply_environment_overrides(raw: dict[str, Any]) -> None:
    paths = raw.setdefault("paths", {})
    ai = raw.setdefault("ai", {})
    if not isinstance(paths, dict) or not isinstance(ai, dict):
        raise ValueError("config sections 'paths' and 'ai' must be mappings")

    if blender_path := os.getenv("ARCH_WALK_BLENDER_PATH"):
        paths["blender_executable"] = blender_path
    if ffmpeg_path := os.getenv("ARCH_WALK_FFMPEG_PATH"):
        paths["ffmpeg_executable"] = ffmpeg_path
    if checkpoint := os.getenv("ARCH_WALK_AI_CHECKPOINT"):
        ai["segmentation_checkpoint"] = checkpoint
    if model := os.getenv("ARCH_WALK_GEMINI_MODEL"):
        ai["gemini_model"] = model
    gemini_enabled = _environment_bool("ARCH_WALK_GEMINI_ENABLED")
    if gemini_enabled is not None:
        ai["gemini_enabled"] = gemini_enabled


def load_config(path: Path | str | None = None) -> AppConfig:
    # Pick up GEMINI_API_KEY and friends from a local .env; OS environment wins.
    load_dotenv(override=False)
    selected_path: Path | str = path if path is not None else os.getenv("ARCH_WALK_CONFIG") or "configs/default.yaml"
    config_path = Path(selected_path)
    if not config_path.exists():
        raw: dict[str, Any] = {}
    else:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    _apply_environment_overrides(raw)
    return AppConfig.model_validate(raw)
