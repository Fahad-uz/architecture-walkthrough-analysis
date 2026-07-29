from __future__ import annotations

from dataclasses import dataclass
import re

from shapely.geometry import Polygon

from architecture_walkthrough.geometry.models import (
    BalconyPolygon,
    Point2D,
    RoomPolygon,
)

_BALCONY_TOKENS = {"balcony", "terrace"}
_SEMANTIC_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class BalconyOwnershipResult:
    rooms: list[RoomPolygon]
    balconies: list[BalconyPolygon]
    matched_topology_faces: int = 0
    subtracted_room_count: int = 0


@dataclass(frozen=True)
class _BalconyCandidate:
    balcony: BalconyPolygon
    topology_backed: bool
    subtract_from_room: bool


def _polygon(region: RoomPolygon) -> Polygon | None:
    polygon = Polygon([(point.x, point.y) for point in region.points])
    if not polygon.is_valid or polygon.area <= 0:
        return None
    return polygon


def _polygon_points(polygon: Polygon) -> list[Point2D]:
    return [
        Point2D(x=float(x), y=float(y))
        for x, y in list(polygon.exterior.coords)[:-1]
    ]


def _is_balcony_room(room: RoomPolygon) -> bool:
    tokens = set(_SEMANTIC_TOKEN.findall(str(room.name or "").casefold()))
    return bool(tokens & _BALCONY_TOKENS)


def _combined_evidence(*values: str) -> str:
    parts: list[str] = []
    for value in values:
        for part in str(value or "").split("+"):
            normalized = part.strip()
            if normalized and normalized != "unknown" and normalized not in parts:
                parts.append(normalized)
    return "+".join(parts) or "unknown"


def _canonical_balcony(
    room: RoomPolygon,
    detected: BalconyPolygon | None,
    balcony_id: str,
) -> BalconyPolygon:
    source = detected or BalconyPolygon(
        id=balcony_id,
        name=room.name or "BALCONY",
        points=room.points,
        confidence=room.confidence,
        evidence_source=room.evidence_source,
    )
    return source.model_copy(
        update={
            "id": balcony_id,
            "face_id": room.face_id,
            "name": room.name or source.name or "BALCONY",
            "points": room.points,
            "confidence": max(room.confidence, source.confidence),
            "evidence_source": _combined_evidence(
                room.evidence_source,
                source.evidence_source,
            ),
            "dimension_m": room.dimension_m or source.dimension_m,
        }
    )


def reconcile_room_balcony_ownership(
    rooms: list[RoomPolygon],
    balconies: list[BalconyPolygon],
    *,
    min_topology_overlap_ratio: float = 0.5,
    min_balcony_duplicate_overlap_ratio: float = 0.75,
    min_room_overlap_ratio: float = 0.5,
    boundary_tolerance_m: float = 0.08,
) -> BalconyOwnershipResult:
    """Give each balcony one floor/ceiling owner without inventing wall lines.

    A named wall-graph face is stronger geometry than a patterned balcony
    rectangle, so matching faces become canonical balconies and leave
    ``rooms``. An unmatched detected balcony may be cut from one dominant room
    only when it reaches that room's boundary and the subtraction stays a
    single, valid, hole-free polygon.
    """

    semantic_rooms = [
        (index, room, polygon)
        for index, room in enumerate(rooms)
        if _is_balcony_room(room)
        and (polygon := _polygon(room)) is not None
    ]
    semantic_room_indexes = {index for index, _room, _polygon in semantic_rooms}
    interior_rooms = [
        room
        for index, room in enumerate(rooms)
        if index not in semantic_room_indexes
    ]

    reserved_ids = {balcony.id for balcony in balconies if balcony.id}
    assigned_ids: set[str] = set()
    next_balcony_index = 0

    def unique_balcony_id(requested: str | None) -> str:
        nonlocal next_balcony_index
        if requested and requested not in assigned_ids:
            assigned_ids.add(requested)
            return requested
        while True:
            candidate = f"balcony_{next_balcony_index:03d}"
            next_balcony_index += 1
            if candidate not in reserved_ids and candidate not in assigned_ids:
                assigned_ids.add(candidate)
                return candidate

    used_semantic_rooms: set[int] = set()
    balcony_candidates: list[_BalconyCandidate] = []
    matched_topology_faces = 0

    for balcony in balconies:
        balcony = balcony.model_copy(
            update={"id": unique_balcony_id(balcony.id)}
        )
        balcony_polygon = _polygon(balcony)
        best_match: tuple[float, int, RoomPolygon] | None = None
        if balcony_polygon is not None:
            for semantic_index, (_room_index, room, room_polygon) in enumerate(
                semantic_rooms
            ):
                if semantic_index in used_semantic_rooms:
                    continue
                smaller_area = min(balcony_polygon.area, room_polygon.area)
                if smaller_area <= 0:
                    continue
                ratio = balcony_polygon.intersection(room_polygon).area / smaller_area
                if ratio >= min_topology_overlap_ratio and (
                    best_match is None or ratio > best_match[0]
                ):
                    best_match = (ratio, semantic_index, room)
        if best_match is not None:
            _ratio, semantic_index, room = best_match
            used_semantic_rooms.add(semantic_index)
            matched_topology_faces += 1
            balcony_candidates.append(
                _BalconyCandidate(
                    balcony=_canonical_balcony(
                        room,
                        balcony,
                        balcony_id=balcony.id or unique_balcony_id(None),
                    ),
                    topology_backed=True,
                    subtract_from_room=False,
                )
            )
        else:
            balcony_candidates.append(
                _BalconyCandidate(
                    balcony=balcony,
                    topology_backed=False,
                    subtract_from_room=True,
                )
            )

    for semantic_index, (_room_index, room, _polygon_value) in enumerate(
        semantic_rooms
    ):
        if semantic_index in used_semantic_rooms:
            continue
        balcony_candidates.append(
            _BalconyCandidate(
                balcony=_canonical_balcony(
                    room,
                    None,
                    balcony_id=unique_balcony_id(None),
                ),
                topology_backed=True,
                subtract_from_room=False,
            )
        )

    deduplicated_candidates: list[_BalconyCandidate] = []
    for candidate in balcony_candidates:
        candidate_polygon = _polygon(candidate.balcony)
        duplicate_index: int | None = None
        if candidate_polygon is not None:
            for index, existing in enumerate(deduplicated_candidates):
                existing_polygon = _polygon(existing.balcony)
                if existing_polygon is None:
                    continue
                smaller_area = min(
                    candidate_polygon.area,
                    existing_polygon.area,
                )
                overlap_ratio = (
                    candidate_polygon.intersection(existing_polygon).area
                    / smaller_area
                )
                if overlap_ratio >= min_balcony_duplicate_overlap_ratio:
                    duplicate_index = index
                    break
        if duplicate_index is None:
            deduplicated_candidates.append(candidate)
            continue
        existing = deduplicated_candidates[duplicate_index]
        if candidate.topology_backed and not existing.topology_backed:
            deduplicated_candidates[duplicate_index] = candidate

    subtracted_room_indexes: set[int] = set()
    for candidate in deduplicated_candidates:
        if not candidate.subtract_from_room:
            continue
        balcony = candidate.balcony
        balcony_polygon = _polygon(balcony)
        if balcony_polygon is None:
            continue
        best_room: tuple[float, int, Polygon] | None = None
        for room_index, room in enumerate(interior_rooms):
            room_polygon = _polygon(room)
            if room_polygon is None:
                continue
            overlap = room_polygon.intersection(balcony_polygon).area
            ratio = overlap / balcony_polygon.area
            if ratio < min_room_overlap_ratio:
                continue
            if room_polygon.boundary.distance(balcony_polygon.boundary) > (
                boundary_tolerance_m
            ):
                continue
            if best_room is None or overlap > best_room[0]:
                best_room = (overlap, room_index, room_polygon)
        if best_room is None:
            continue

        _overlap, room_index, room_polygon = best_room
        difference = room_polygon.difference(balcony_polygon)
        if (
            not isinstance(difference, Polygon)
            or not difference.is_valid
            or difference.area <= 0
            or len(difference.interiors) > 0
        ):
            continue
        interior_rooms[room_index] = interior_rooms[room_index].model_copy(
            update={
                "points": _polygon_points(difference),
                "evidence_source": _combined_evidence(
                    interior_rooms[room_index].evidence_source,
                    "balcony_subtracted",
                ),
            }
        )
        subtracted_room_indexes.add(room_index)

    return BalconyOwnershipResult(
        rooms=interior_rooms,
        balconies=[
            candidate.balcony
            for candidate in deduplicated_candidates
        ],
        matched_topology_faces=matched_topology_faces,
        subtracted_room_count=len(subtracted_room_indexes),
    )
