from __future__ import annotations

from typing import Any

from architecture_walkthrough.geometry.models import SCHEMA_VERSION


def _major(version: str) -> int:
    try:
        return int(str(version).split(".")[0])
    except (ValueError, IndexError):
        return 0


def _fill_opening_interval(opening: dict[str, Any], default_width_m: float) -> None:
    if opening.get("start_offset_m") is not None and opening.get("end_offset_m") is not None:
        return
    offset = opening.get("offset_m")
    if offset is None:
        return
    width = float(opening.get("width_m") or default_width_m)
    opening["start_offset_m"] = float(offset) - width / 2
    opening["end_offset_m"] = float(offset) + width / 2


def migrate_floorplan_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a floorplan JSON payload in place to the current schema version.

    Schema 2.x openings carry a midpoint `offset_m` plus `width_m`; schema 3
    represents openings as explicit intervals along their wall. Anything that
    cannot be derived is left for downstream re-attachment from `center`.
    """
    if not isinstance(data, dict):
        return data
    if _major(data.get("schema_version", "1.0")) >= _major(SCHEMA_VERSION):
        return data
    for door in data.get("doors", []) or []:
        if isinstance(door, dict):
            _fill_opening_interval(door, default_width_m=0.90)
    for window in data.get("windows", []) or []:
        if isinstance(window, dict):
            _fill_opening_interval(window, default_width_m=1.20)
    for index, room in enumerate(data.get("rooms", []) or []):
        if isinstance(room, dict) and not room.get("face_id"):
            room["face_id"] = room.get("id") or f"legacy_face_{index:03d}"
    data["schema_version"] = SCHEMA_VERSION
    return data
