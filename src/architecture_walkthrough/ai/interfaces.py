from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class SceneStyle(BaseModel):
    lighting_preset: str = "neutral_studio"
    color_temperature: int = Field(default=4500, ge=1000, le=12000)
    light_intensity: float = Field(default=450.0, ge=0.0)
    material_palette: dict[str, str] = Field(default_factory=dict)
    exposure: float = Field(default=0.0, ge=-5.0, le=5.0)
    decorative_asset_suggestions: list[str] = Field(default_factory=list)


class SceneStyler(ABC):
    @abstractmethod
    def propose_style(self, prompt: str | None = None) -> SceneStyle:
        raise NotImplementedError


class DeterministicSceneStyler(SceneStyler):
    def propose_style(self, prompt: str | None = None) -> SceneStyle:
        return SceneStyle()
