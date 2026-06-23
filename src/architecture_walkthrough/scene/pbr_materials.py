from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PBRMaterialPreset(BaseModel):
    name: str
    base_color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    base_color_texture: Path | None = None
    normal_texture: Path | None = None
    roughness_texture: Path | None = None
    metallic_texture: Path | None = None
    ao_texture: Path | None = None
    texture_scale_m: float = 1.0
    roughness: float = 0.65
    metallic: float = 0.0
    alpha_mode: str = "OPAQUE"
    double_sided: bool = False


class MaterialRegistry(BaseModel):
    materials: dict[str, PBRMaterialPreset] = Field(default_factory=dict)

    def require(self, name: str) -> PBRMaterialPreset:
        if name not in self.materials:
            raise KeyError(f"material preset not found: {name}")
        return self.materials[name]


def load_material_registry(path: Path) -> MaterialRegistry:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    entries = {
        key: PBRMaterialPreset(name=key, **(value or {}))
        for key, value in (raw.get("materials", {}) if isinstance(raw, dict) else {}).items()
    }
    return MaterialRegistry(materials=entries)
