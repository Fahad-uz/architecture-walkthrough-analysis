"""Footprint-based stair parts shared by the preview and standalone Blender.

The local +Y axis is the ascending direction of a straight/first flight.
Dogleg stairs turn on a landing at +Y, then ascend back along the +X flight.
No geometry is inferred outside the explicitly supplied stair footprint.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


def stair_placement(element: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve polygon-only stair elements identically in both exporters."""
    angle = math.radians(float(element.get("rotation_deg") or 0.0))
    polygon = element.get("polygon") or []
    center = element.get("center")
    if center is None and polygon:
        center = {
            "x": (min(point["x"] for point in polygon) + max(point["x"] for point in polygon)) / 2,
            "y": (min(point["y"] for point in polygon) + max(point["y"] for point in polygon)) / 2,
        }
    center = center or {"x": 0.0, "y": 0.0}
    local = [
        (
            (point["x"] - center["x"]) * math.cos(angle) + (point["y"] - center["y"]) * math.sin(angle),
            -(point["x"] - center["x"]) * math.sin(angle) + (point["y"] - center["y"]) * math.cos(angle),
        )
        for point in polygon
    ]
    width = element.get("width_m")
    depth = element.get("depth_m")
    if width is None:
        width = max(point[0] for point in local) - min(point[0] for point in local) if local else 1.0
    if depth is None:
        depth = max(point[1] for point in local) - min(point[1] for point in local) if local else 2.0
    return {
        "center": dict(center),
        "width_m": float(width),
        "depth_m": float(depth),
        "rotation_deg": float(element.get("rotation_deg") or 0.0),
    }


def stair_parts(
    width_m: float, depth_m: float, metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return local boxes with contiguous tread surfaces and explicit rises.

    ``layout`` is ``straight`` (default) or ``dogleg``. Dogleg metadata accepts
    ``steps_per_flight``, ``landing_depth_m``, ``flight_gap_m`` and
    ``total_height_m``. Heights are construction assumptions unless supplied
    by the plan/user; defaults preserve the historic 0.16m rise per step.
    """
    metadata = metadata or {}
    if not all(math.isfinite(value) and value > 0 for value in (width_m, depth_m)):
        raise ValueError("stair width and depth must be finite and positive")
    layout = str(metadata.get("layout") or "straight").lower()
    if layout not in {"straight", "dogleg"}:
        raise ValueError(f"unsupported stair layout: {layout}")
    step_count = metadata.get("steps_per_flight" if layout == "dogleg" else "step_count", 10)
    if isinstance(step_count, bool):
        raise ValueError("stair step count must be an integer")
    steps = int(step_count)
    if float(step_count) != steps:
        raise ValueError("stair step count must be an integer")
    if not 1 <= steps <= 64:
        raise ValueError("stairs require between 1 and 64 steps per flight")
    total_steps = steps * (2 if layout == "dogleg" else 1)
    total_height = float(metadata.get("total_height_m", 0.16 * total_steps))
    if not math.isfinite(total_height) or total_height <= 0:
        raise ValueError("stair total height must be finite and positive")
    rise = total_height / total_steps
    parts: list[dict[str, Any]] = []

    def add(name: str, x: float, y: float, width: float, depth: float, top: float) -> None:
        parts.append({
            "name": name, "x": x, "y": y,
            "width": width, "depth": depth, "height": top, "z": top / 2,
        })

    if layout == "straight":
        tread = depth_m / steps
        for index in range(steps):
            add(f"Step_{index:02d}", 0.0, -depth_m / 2 + tread * (index + 0.5),
                width_m, tread, rise * (index + 1))
        return parts

    landing_depth = float(metadata.get("landing_depth_m", min(width_m / 2, depth_m / 3)))
    flight_gap = float(metadata.get("flight_gap_m", 0.0))
    if not math.isfinite(landing_depth) or not 0 < landing_depth < depth_m:
        raise ValueError("dogleg landing depth must fit inside the stair footprint")
    if not math.isfinite(flight_gap) or not 0 <= flight_gap < width_m:
        raise ValueError("dogleg flight gap must fit inside the stair width")
    flight_width = (width_m - flight_gap) / 2
    flight_x = (flight_width + flight_gap) / 2
    run_depth = depth_m - landing_depth
    tread = run_depth / steps
    landing_y = depth_m / 2 - landing_depth / 2
    add("Landing", 0.0, landing_y, width_m, landing_depth, total_height / 2)
    for index in range(steps):
        add(f"Flight_1_Step_{index:02d}", -flight_x,
            -depth_m / 2 + tread * (index + 0.5), flight_width, tread, rise * (index + 1))
        add(f"Flight_2_Step_{index:02d}", flight_x,
            depth_m / 2 - landing_depth - tread * (index + 0.5),
            flight_width, tread, total_height / 2 + rise * (index + 1))
    return parts
