from __future__ import annotations

from dataclasses import dataclass, field

from architecture_walkthrough.geometry.constraints import is_horizontal, snap_wall_axis, sorted_wall, wall_length
from architecture_walkthrough.geometry.models import Point2D, WallSegment


@dataclass(frozen=True)
class ReconstructionResult:
    walls: list[WallSegment]
    audit_trail: list[dict[str, object]] = field(default_factory=list)


def _merge_collinear(walls: list[WallSegment], coord_tol: float, gap_tol: float, min_length: float) -> list[WallSegment]:
    normalized = [sorted_wall(wall) for wall in walls]
    merged: list[WallSegment] = []
    for horizontal in (True, False):
        oriented = [wall for wall in normalized if is_horizontal(wall) == horizontal]
        if horizontal:
            oriented.sort(key=lambda wall: (round(wall.start.y / coord_tol), wall.start.x))
        else:
            oriented.sort(key=lambda wall: (round(wall.start.x / coord_tol), wall.start.y))
        groups: list[list[WallSegment]] = []
        for wall in oriented:
            if not groups:
                groups.append([wall])
                continue
            prev = groups[-1][-1]
            if horizontal:
                same = abs(wall.start.y - prev.start.y) <= coord_tol
                touches = wall.start.x <= prev.end.x + gap_tol
            else:
                same = abs(wall.start.x - prev.start.x) <= coord_tol
                touches = wall.start.y <= prev.end.y + gap_tol
            if same and touches:
                groups[-1].append(wall)
            else:
                groups.append([wall])
        for group in groups:
            base = group[0]
            confidence = sum(wall.confidence for wall in group) / len(group)
            thickness = max(wall.thickness_m for wall in group)
            external = any(wall.external for wall in group)
            if horizontal:
                y = sum(wall.start.y for wall in group) / len(group)
                start = Point2D(x=min(wall.start.x for wall in group), y=y)
                end = Point2D(x=max(wall.end.x for wall in group), y=y)
            else:
                x = sum(wall.start.x for wall in group) / len(group)
                start = Point2D(x=x, y=min(wall.start.y for wall in group))
                end = Point2D(x=x, y=max(wall.end.y for wall in group))
            candidate = base.model_copy(
                update={
                    "start": start,
                    "end": end,
                    "thickness_m": thickness,
                    "external": external,
                    "wall_type": "external" if external else "internal",
                    "confidence": confidence,
                    "evidence_source": "optimized_wall_band",
                }
            )
            if wall_length(candidate) >= min_length:
                merged.append(candidate)
    return merged


def _snap_intersections(walls: list[WallSegment], tolerance: float) -> list[WallSegment]:
    updates = [wall.model_copy(deep=True) for wall in walls]
    for h_index, h_wall in enumerate(updates):
        if not is_horizontal(h_wall):
            continue
        hx0, hx1 = sorted((h_wall.start.x, h_wall.end.x))
        hy = h_wall.start.y
        for v_index, v_wall in enumerate(updates):
            if is_horizontal(v_wall):
                continue
            vx = v_wall.start.x
            vy0, vy1 = sorted((v_wall.start.y, v_wall.end.y))
            if hx0 - tolerance <= vx <= hx1 + tolerance and vy0 - tolerance <= hy <= vy1 + tolerance:
                h_start = h_wall.start
                h_end = h_wall.end
                v_start = v_wall.start
                v_end = v_wall.end
                if abs(vx - hx0) <= tolerance:
                    h_start = Point2D(x=vx, y=hy)
                if abs(vx - hx1) <= tolerance:
                    h_end = Point2D(x=vx, y=hy)
                if abs(hy - vy0) <= tolerance:
                    v_start = Point2D(x=vx, y=hy)
                if abs(hy - vy1) <= tolerance:
                    v_end = Point2D(x=vx, y=hy)
                updates[h_index] = updates[h_index].model_copy(update={"start": h_start, "end": h_end})
                updates[v_index] = updates[v_index].model_copy(update={"start": v_start, "end": v_end})
    return updates


def _assign_ids(walls: list[WallSegment]) -> list[WallSegment]:
    ordered = sorted(walls, key=lambda wall: (round(wall.start.y, 4), round(wall.start.x, 4), round(wall.end.y, 4), round(wall.end.x, 4)))
    return [wall.model_copy(update={"id": wall.id or f"w{index:03d}"}) for index, wall in enumerate(ordered)]


def reconstruct_walls(
    raw_walls: list[WallSegment],
    estimated_thickness_m: float = 0.12,
    angle_tolerance_deg: float = 7.0,
    gap_tolerance_factor: float = 2.5,
    merge_overlap_tolerance_factor: float = 1.5,
    min_wall_length_m: float = 0.20,
) -> ReconstructionResult:
    audit: list[dict[str, object]] = [{"stage": "input", "wall_count": len(raw_walls)}]
    axis = [snap_wall_axis(wall, angle_tolerance_deg) for wall in raw_walls]
    audit.append({"stage": "axis_snap", "wall_count": len(axis)})
    tolerance = max(estimated_thickness_m * merge_overlap_tolerance_factor, 0.04)
    gap = max(estimated_thickness_m * gap_tolerance_factor, 0.08)
    merged = _merge_collinear(axis, tolerance, gap, min_wall_length_m)
    audit.append({"stage": "merge_collinear", "wall_count": len(merged)})
    snapped = _snap_intersections(merged, gap)
    audit.append({"stage": "snap_intersections", "wall_count": len(snapped)})
    deduped = _merge_collinear(snapped, tolerance, 0.01, min_wall_length_m)
    audit.append({"stage": "dedupe", "wall_count": len(deduped)})
    return ReconstructionResult(walls=_assign_ids(deduped), audit_trail=audit)
