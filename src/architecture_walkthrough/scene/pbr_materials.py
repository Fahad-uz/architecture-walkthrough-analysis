from __future__ import annotations

from collections.abc import Mapping
import logging
from pathlib import Path
import re
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from shapely.geometry import LineString, Polygon  # type: ignore[import-untyped]
from shapely.ops import unary_union  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, PositiveFloat, field_validator
from architecture_walkthrough.scene.furniture import render_furniture

from architecture_walkthrough.geometry.models import (
    FloorPlanModel,
    MaterialAssignment,
    RoomPolygon,
)

LOGGER = logging.getLogger(__name__)
MATERIAL_PLAN_SCHEMA_VERSION = 2
MATERIAL_PLAN_METADATA_KEY = "blender_material_plan"
SAFE_FALLBACK_PRESET = "fallback"
_MATERIAL_SEPARATOR = re.compile(r"[^a-z0-9]+")
_ROOM_TOKEN = re.compile(r"[a-z0-9]+")


def normalize_material_name(value: object) -> str:
    """Return the stable lookup key used by registries and renderer plans."""

    return _MATERIAL_SEPARATOR.sub("_", str(value or "").strip().casefold()).strip("_")


def _normalized_room_name(value: str | None) -> str:
    return " ".join(str(value or "").split()).casefold()


class PBRMaterialPreset(BaseModel):
    name: str
    base_color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    base_color_texture: Path | None = None
    normal_texture: Path | None = None
    roughness_texture: Path | None = None
    metallic_texture: Path | None = None
    ao_texture: Path | None = None
    texture_scale_m: PositiveFloat = 1.0
    roughness: float = Field(default=0.65, ge=0.0, le=1.0)
    metallic: float = Field(default=0.0, ge=0.0, le=1.0)
    alpha_mode: Literal["OPAQUE", "MASK", "BLEND"] = "OPAQUE"
    double_sided: bool = False

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: object) -> str:
        normalized = normalize_material_name(value)
        if not normalized:
            raise ValueError("material preset name cannot be empty")
        return normalized

    @field_validator("base_color")
    @classmethod
    def validate_base_color(
        cls,
        value: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        if any(component < 0.0 or component > 1.0 for component in value):
            raise ValueError("base_color components must be between 0 and 1")
        return value

    @field_validator("alpha_mode", mode="before")
    @classmethod
    def normalize_alpha_mode(cls, value: object) -> str:
        return str(value or "OPAQUE").strip().upper()

    def scalar_payload(self) -> dict[str, object]:
        """Return only values directly supported by the scalar Blender pass."""

        return {
            "base_color": list(self.base_color),
            "roughness": float(self.roughness),
            "metallic": float(self.metallic),
            "alpha_mode": self.alpha_mode,
            "double_sided": self.double_sided,
        }

    def render_payload(self) -> dict[str, object]:
        """Include available local maps in the self-contained renderer contract."""
        payload = self.scalar_payload()
        payload["texture_scale_m"] = float(self.texture_scale_m)
        for field in (
            "base_color_texture", "normal_texture", "roughness_texture",
            "metallic_texture", "ao_texture",
        ):
            path = getattr(self, field)
            if path is not None and path.is_file():
                payload[field] = str(path.resolve())
        return payload


class MaterialRegistry(BaseModel):
    materials: dict[str, PBRMaterialPreset] = Field(default_factory=dict)

    @field_validator("materials")
    @classmethod
    def normalize_materials(
        cls,
        value: dict[str, PBRMaterialPreset],
    ) -> dict[str, PBRMaterialPreset]:
        normalized: dict[str, PBRMaterialPreset] = {}
        for key, preset in value.items():
            name = normalize_material_name(key)
            if not name:
                raise ValueError("material registry key cannot be empty")
            if name in normalized:
                raise ValueError(f"duplicate normalized material preset: {name}")
            normalized[name] = (
                preset
                if preset.name == name
                else preset.model_copy(update={"name": name})
            )
        return normalized

    def get(self, name: object) -> PBRMaterialPreset | None:
        return self.materials.get(normalize_material_name(name))

    def require(self, name: str) -> PBRMaterialPreset:
        preset = self.get(name)
        if preset is None:
            raise KeyError(f"material preset not found: {normalize_material_name(name)}")
        return preset

    def scalar_payload(self) -> dict[str, dict[str, object]]:
        return {
            name: preset.scalar_payload()
            for name, preset in sorted(self.materials.items())
        }

    def render_payload(self) -> dict[str, dict[str, object]]:
        return {name: preset.render_payload() for name, preset in sorted(self.materials.items())}


def load_material_registry(path: Path) -> MaterialRegistry:
    if not path.is_file():
        raise FileNotFoundError(f"material registry not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("material registry root must be a mapping")
    raw_materials = raw.get("materials", {})
    if not isinstance(raw_materials, Mapping):
        raise ValueError("material registry 'materials' must be a mapping")

    entries: dict[str, dict[str, object]] = {}
    for key, value in raw_materials.items():
        if not isinstance(key, str):
            raise ValueError("material registry keys must be strings")
        if value is None:
            entry: dict[str, object] = {}
        elif isinstance(value, Mapping):
            entry = dict(value)
        else:
            raise ValueError(f"material preset {key!r} must be a mapping")
        entry["name"] = key
        for field in (
            "base_color_texture", "normal_texture", "roughness_texture",
            "metallic_texture", "ao_texture",
        ):
            if entry.get(field):
                texture_path = Path(str(entry[field])).expanduser()
                if not texture_path.is_absolute():
                    texture_path = path.parent / texture_path
                entry[field] = texture_path.resolve()
                if not texture_path.is_file():
                    LOGGER.warning("Material %s: unavailable %s map %s; using scalar fallback", key, field, texture_path)
        entries[key] = entry
    if not entries:
        raise ValueError(f"material registry defines no materials: {path}")
    return MaterialRegistry.model_validate({"materials": entries})


class _PresetResolver:
    def __init__(self, registry: MaterialRegistry) -> None:
        self.registry = registry
        self.warnings: list[str] = []
        self._seen_warnings: set[str] = set()

    def _warn(self, message: str) -> None:
        if message not in self._seen_warnings:
            self._seen_warnings.add(message)
            self.warnings.append(message)

    def resolve(
        self,
        requested: str | None,
        fallback: str,
        *,
        context: str,
    ) -> str:
        requested_name = normalize_material_name(requested)
        if requested_name and self.registry.get(requested_name) is not None:
            return requested_name
        if requested_name:
            self._warn(
                f"{context} requested unknown material {requested_name!r}; "
                f"using {normalize_material_name(fallback) or SAFE_FALLBACK_PRESET!r}"
            )

        fallback_name = normalize_material_name(fallback)
        if fallback_name == SAFE_FALLBACK_PRESET:
            return SAFE_FALLBACK_PRESET
        if fallback_name and self.registry.get(fallback_name) is not None:
            return fallback_name
        if fallback_name:
            self._warn(
                f"{context} fallback material {fallback_name!r} is unavailable; "
                f"using {SAFE_FALLBACK_PRESET!r}"
            )
        return SAFE_FALLBACK_PRESET


def _matching_assignment(
    assignments: list[MaterialAssignment],
    target: str | None,
    *,
    case_insensitive: bool,
) -> MaterialAssignment | None:
    if target is None or not target.strip():
        return None
    expected = _normalized_room_name(target) if case_insensitive else target.strip()
    for assignment in reversed(assignments):
        candidate = (
            _normalized_room_name(assignment.target)
            if case_insensitive
            else assignment.target.strip()
        )
        if candidate == expected:
            return assignment
    return None


def explicit_room_material(
    room: RoomPolygon,
    assignments: list[MaterialAssignment],
) -> str | None:
    """Resolve explicit room targeting by id, then face id, then room name."""

    match = _matching_assignment(assignments, room.id, case_insensitive=False)
    if match is None:
        match = _matching_assignment(
            assignments,
            room.face_id,
            case_insensitive=False,
        )
    if match is None:
        match = _matching_assignment(
            assignments,
            room.name,
            case_insensitive=True,
        )
    return match.preset if match is not None else None


def semantic_room_material(room_name: str | None) -> str | None:
    """Choose only unambiguous room-type defaults from an existing label."""

    tokens = set(_ROOM_TOKEN.findall(_normalized_room_name(room_name)))
    if tokens & {"bedroom", "study"}:
        return "wood"
    if tokens & {"kitchen", "bath", "bathroom", "toilet"}:
        return "ceramic_tile"
    return None


def build_material_plan(
    model: FloorPlanModel,
    registry: MaterialRegistry,
) -> dict[str, Any]:
    """Build the indexed, JSON-safe material contract consumed by renderers."""

    resolver = _PresetResolver(registry)
    wall_default = resolver.resolve(
        model.style.wall_material,
        "painted_wall",
        context="scene wall style",
    )
    floor_default = resolver.resolve(
        model.style.floor_material,
        "marble",
        context="scene floor style",
    )
    door_default = resolver.resolve(
        model.style.door_material,
        "wood",
        context="scene door style",
    )
    window_default = resolver.resolve(
        model.style.window_frame_material,
        "metal",
        context="scene window-frame style",
    )
    ceiling_preset = resolver.resolve(
        model.ceiling.material_preset,
        "painted_wall",
        context="ceiling",
    )
    balcony_default = resolver.resolve(
        "ceramic_tile",
        floor_default,
        context="balcony",
    )

    wall_presets = [
        resolver.resolve(
            wall.material_preset,
            wall_default,
            context=f"wall[{index}]",
        )
        for index, wall in enumerate(model.walls)
    ]
    room_floor_presets: list[str] = []
    room_floor_base_presets: list[str] = []
    for index, room in enumerate(model.rooms):
        explicit = explicit_room_material(room, model.material_assignments)
        automatic = explicit or semantic_room_material(room.name)
        room_floor_base_presets.append(resolver.resolve(
            automatic, floor_default, context=f"room[{index}] automatic floor",
        ))
        requested = room.floor_material_preset or automatic
        room_floor_presets.append(
            resolver.resolve(
                requested,
                floor_default,
                context=f"room[{index}] floor",
            )
        )
    slab_presets = [
        resolver.resolve(
            slab.material_preset,
            floor_default,
            context=f"slab[{index}]",
        )
        for index, slab in enumerate(model.slabs)
    ]
    balcony_floor_presets = []
    for index, balcony in enumerate(model.balconies):
        requested = explicit_room_material(
            balcony,
            model.material_assignments,
        )
        balcony_floor_presets.append(
            resolver.resolve(
                requested,
                balcony_default,
                context=f"balcony[{index}] floor",
            )
        )

    floor_regions = [
        {
            "room_index": index,
            "room_id": room.id,
            "face_id": room.face_id,
            "points": [[point.x, point.y] for point in room.points],
            "preset": room_floor_presets[index],
            "apply": room.floor_material_preset is not None,
            "source": (
                "room" if room.floor_material_preset else
                "assignment" if explicit_room_material(room, model.material_assignments) else
                "semantic" if semantic_room_material(room.name) else "scene"
            ),
        }
        for index, room in enumerate(model.rooms)
    ]
    floor_regions_enabled = any(room.floor_material_preset is not None for room in model.rooms)
    if floor_regions_enabled:
        for index, region in enumerate(floor_regions):
            polygon = Polygon(region["points"])
            if not polygon.is_valid:
                resolver.warnings.append(f"room[{index}] floor region is invalid; ignored")
                continue
            for previous in floor_regions[:index]:
                other = Polygon(previous["points"])
                if (other.is_valid and region["preset"] != previous["preset"]
                        and polygon.intersection(other).area > 1e-8):
                    resolver.warnings.append(
                        f"room[{previous['room_index']}] and room[{index}] floor finishes overlap; "
                        "original surface material retained in the overlap"
                    )
    floor_regions = [region for region in floor_regions if Polygon(region["points"]).is_valid]
    wall_regions = _wall_finish_regions(model, resolver, wall_default)

    materials = registry.render_payload()
    materials.setdefault(
        SAFE_FALLBACK_PRESET,
        PBRMaterialPreset(name=SAFE_FALLBACK_PRESET).scalar_payload(),
    )
    return {
        "schema_version": MATERIAL_PLAN_SCHEMA_VERSION,
        "fallback_preset": SAFE_FALLBACK_PRESET,
        "materials": materials,
        "defaults": {
            "wall": wall_default,
            "floor": floor_default,
            "door": door_default,
            "window_frame": window_default,
            "ceiling": ceiling_preset,
            "balcony": balcony_default,
        },
        "wall_presets": wall_presets,
        "room_floor_presets": room_floor_presets,
        "room_floor_base_presets": room_floor_base_presets,
        "floor_regions": floor_regions,
        "floor_regions_enabled": floor_regions_enabled,
        "wall_regions": wall_regions,
        "door_presets": [door_default for _door in model.doors],
        "window_frame_presets": [
            window_default
            for _window in model.windows
        ],
        "slab_presets": slab_presets,
        "balcony_floor_presets": balcony_floor_presets,
        "ceiling_preset": ceiling_preset,
        "warnings": resolver.warnings,
    }


def _wall_finish_regions(
    model: FloorPlanModel, resolver: _PresetResolver, wall_default: str,
) -> list[dict[str, Any]]:
    """Resolve only interior sides touching an explicitly finished room.

    Intersect the actual wall-side line with the room's boundary band. This
    supports concave rooms and multiple rooms along one long wall, while
    excluding walls embedded in the middle of a room polygon.
    """
    import math

    candidates: list[dict[str, Any]] = []
    for room_index, room in enumerate(model.rooms):
        if room.wall_material_preset is None:
            continue
        polygon = Polygon([(point.x, point.y) for point in room.points])
        if not polygon.is_valid:
            resolver.warnings.append(f"room[{room_index}] wall region is invalid; ignored")
            continue
        preset = resolver.resolve(
            room.wall_material_preset, wall_default, context=f"room[{room_index}] wall",
        )
        for wall_index, wall in enumerate(model.walls):
            dx, dy = wall.end.x - wall.start.x, wall.end.y - wall.start.y
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                continue
            ux, uy = dx / length, dy / length
            offset = wall.thickness_m / 2 + 1e-7
            parallel_edges = []
            coordinates = list(polygon.exterior.coords)
            for a, b in zip(coordinates, coordinates[1:]):
                ex, ey = b[0] - a[0], b[1] - a[1]
                edge_length = math.hypot(ex, ey)
                if edge_length > 1e-9 and abs((ex * ux + ey * uy) / edge_length) > 0.999:
                    parallel_edges.append(LineString([a, b]).buffer(offset + 1e-6, cap_style=2))
            if not parallel_edges:
                continue
            boundary_band = polygon.intersection(unary_union(parallel_edges))
            for side in (-1, 1):
                sx, sy = wall.start.x - uy * offset * side, wall.start.y + ux * offset * side
                line = LineString([(sx, sy), (sx + ux * length, sy + uy * length)])
                intersection = line.intersection(boundary_band)
                parts = list(intersection.geoms) if hasattr(intersection, "geoms") else [intersection]
                for part in parts:
                    if part.geom_type != "LineString" or part.length <= 1e-7:
                        continue
                    positions = [(x - sx) * ux + (y - sy) * uy for x, y in part.coords]
                    candidates.append({
                        "wall_index": wall_index, "side": side,
                        "start_m": max(0.0, min(positions)), "end_m": min(length, max(positions)),
                        "preset": preset, "room_index": room_index,
                    })
    regions = []
    for wall_index in range(len(model.walls)):
        for side in (-1, 1):
            selected = [region for region in candidates
                        if region["wall_index"] == wall_index and region["side"] == side]
            boundaries = sorted({value for region in selected
                                 for value in (region["start_m"], region["end_m"])})
            for start, end in zip(boundaries, boundaries[1:]):
                midpoint = (start + end) / 2
                matches = [region for region in selected
                           if region["start_m"] < midpoint < region["end_m"]]
                if not matches:
                    continue
                if len({region["preset"] for region in matches}) > 1:
                    resolver.warnings.append(
                        f"wall[{wall_index}] side {side} room finishes overlap at {start:g}-{end:g}m; "
                        "original surface material retained in the overlap"
                    )
                    continue
                regions.append({**matches[0], "start_m": start, "end_m": end})
    return regions


def model_with_material_plan(
    model: FloorPlanModel,
    registry: MaterialRegistry,
) -> FloorPlanModel:
    """Return a renderer copy with the material contract embedded in metadata."""

    material_plan = build_material_plan(model, registry)
    for warning in material_plan["warnings"]:
        LOGGER.warning("Material plan: %s", warning)
    return model.model_copy(
        update={
            "furniture": render_furniture(model),
            "metadata": {
                **model.metadata,
                MATERIAL_PLAN_METADATA_KEY: material_plan,
            }
        }
    )
