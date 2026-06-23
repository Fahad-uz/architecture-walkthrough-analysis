from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
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


class RenderSettings(BaseModel):
    preview_samples: PositiveInt = 16
    final_samples: PositiveInt = 96
    preview_fps: PositiveInt = 24
    camera_speed_mps: PositiveFloat = 1.0


class TextureSettings(BaseModel):
    default_resolution: PositiveInt = 2048
    allow_4k: bool = False
    embed_in_glb: bool = True
    max_texture_size: PositiveInt = 4096


class AssetSettings(BaseModel):
    registry_path: Path = Path("assets/models/asset_registry.yaml")
    allow_placeholder_fallback: bool = True


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
    render: RenderSettings = Field(default_factory=RenderSettings)
    textures: TextureSettings = Field(default_factory=TextureSettings)
    assets: AssetSettings = Field(default_factory=AssetSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)


def load_config(path: Path | str = Path("configs/default.yaml")) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    return AppConfig.model_validate(raw)
