from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import Polygon

from architecture_walkthrough.geometry.models import ArchitecturalElement, RoomPolygon, ValidationIssue


@dataclass
class StairCeilingVoids:
    room_ids: list[str] = field(default_factory=list)
    inferences: list[dict[str, Any]] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)


def infer_stair_ceiling_voids(
    rooms: list[RoomPolygon], elements: list[ArchitecturalElement]
) -> StairCeilingVoids:
    """Keep a measured stairwell open without cutting ceilings over general rooms.

    Repeated local treads must occupy a majority of one enclosing room. This
    infers an open stairwell ceiling, not an upper-storey slab or a precise stair
    aperture: neither of those is observable from a single floor plan.
    """

    result = StairCeilingVoids()
    eligible_rooms: list[tuple[RoomPolygon, Polygon]] = []
    for room in rooms:
        # A stair in a named habitable room needs an explicit, smaller opening;
        # removing that room's entire ceiling would be an unsupported guess.
        name = (room.name or "").strip().lower()
        if not room.id or (name and "stair" not in name):
            continue
        polygon = Polygon([(point.x, point.y) for point in room.points])
        if polygon.is_valid and not polygon.is_empty and polygon.area > 0:
            eligible_rooms.append((room, polygon))

    for element in elements:
        if (
            element.kind != "staircase"
            or element.evidence_source != "repetitive_parallel_treads"
            or element.confidence < 0.8
            or len(element.polygon) < 3
        ):
            continue
        stair = Polygon([(point.x, point.y) for point in element.polygon])
        if not stair.is_valid or stair.is_empty or stair.area <= 0:
            continue
        matches: list[tuple[RoomPolygon, float, float]] = []
        for room, polygon in eligible_rooms:
            intersection_area = stair.intersection(polygon).area
            containment = intersection_area / stair.area
            coverage = intersection_area / polygon.area
            if containment >= 0.98 and coverage > 0.5:
                matches.append((room, containment, coverage))
        # Overlapping room faces make ownership ambiguous even when each face
        # individually passes the measurements.
        if len(matches) != 1:
            continue
        room, containment, coverage = matches[0]
        if room.id in result.room_ids:
            continue
        assert room.id is not None
        result.room_ids.append(room.id)
        assumption = (
            "Ceiling omitted over the enclosing stair room. The upper-storey "
            "slab and exact stair opening dimensions are unknown; review this opening."
        )
        result.inferences.append({
            "room_id": room.id,
            "staircase_id": element.id,
            "stair_containment": containment,
            "room_coverage": coverage,
            "assumption": assumption,
        })
        result.issues.append(ValidationIssue(
            code="inferred_stair_ceiling_void",
            severity="warning",
            element_id=room.id,
            message=assumption,
        ))
    return result
