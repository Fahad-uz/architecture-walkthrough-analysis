from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AssetRegistryEntry(BaseModel):
    category: str
    path: Path | None = None
    source_format: str = "glb"
    native_dimensions_m: tuple[float, float, float] = (1.0, 1.0, 1.0)
    default_scale: float = 1.0
    forward_axis: str = "Y"
    up_axis: str = "Z"
    vertical_floor_offset_m: float = 0.0
    material_override: str | None = None
    license: str | None = None
    source: str | None = None


class AssetRegistry(BaseModel):
    assets: dict[str, AssetRegistryEntry] = Field(default_factory=dict)

    def get_entry(self, category: str) -> AssetRegistryEntry | None:
        return self.assets.get(category)


def load_asset_registry(path: Path) -> AssetRegistry:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    entries = {
        key: AssetRegistryEntry(category=key, **(value or {}))
        for key, value in (raw.get("assets", {}) if isinstance(raw, dict) else {}).items()
    }
    return AssetRegistry(assets=entries)
