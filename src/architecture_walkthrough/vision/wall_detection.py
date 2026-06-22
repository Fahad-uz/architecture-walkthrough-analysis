from __future__ import annotations

from pathlib import Path

import cv2

from architecture_walkthrough.geometry.models import Point2D, WallSegment


def _axis_aligned_segment(x1: int, y1: int, x2: int, y2: int) -> tuple[str, float, float, float] | None:
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dx < 1 and dy < 1:
        return None
    if dx >= dy * 2.5:
        return "h", (y1 + y2) / 2, min(x1, x2), max(x1, x2)
    if dy >= dx * 2.5:
        return "v", (x1 + x2) / 2, min(y1, y2), max(y1, y2)
    return None


def _merge_segments(
    segments: list[tuple[str, float, float, float]],
    coordinate_tolerance_px: float = 18.0,
    gap_tolerance_px: float = 24.0,
) -> list[tuple[str, float, float, float]]:
    merged: list[tuple[str, float, float, float]] = []
    for orientation in ("h", "v"):
        oriented = sorted(
            [segment for segment in segments if segment[0] == orientation],
            key=lambda item: (round(item[1] / coordinate_tolerance_px), item[2]),
        )
        clusters: list[list[tuple[str, float, float, float]]] = []
        for segment in oriented:
            if not clusters:
                clusters.append([segment])
                continue
            previous = clusters[-1][-1]
            same_line = abs(segment[1] - previous[1]) <= coordinate_tolerance_px
            touches = segment[2] <= previous[3] + gap_tolerance_px
            if same_line and touches:
                clusters[-1].append(segment)
            else:
                clusters.append([segment])
        for cluster in clusters:
            coord = sum(segment[1] for segment in cluster) / len(cluster)
            start = min(segment[2] for segment in cluster)
            end = max(segment[3] for segment in cluster)
            if end - start >= 60:
                merged.append((orientation, coord, start, end))
    return merged


def detect_wall_lines(edge_image_path: Path, thickness_m: float = 0.12, height_m: float = 3.0) -> list[WallSegment]:
    image = cv2.imread(str(edge_image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read edge image: {edge_image_path}")
    lines = cv2.HoughLinesP(image, 1, 3.14159 / 180, threshold=80, minLineLength=70, maxLineGap=12)
    if lines is None:
        return []
    axis_segments: list[tuple[str, float, float, float]] = []
    for [[x1, y1, x2, y2]] in lines:
        segment = _axis_aligned_segment(int(x1), int(y1), int(x2), int(y2))
        if segment is not None:
            axis_segments.append(segment)

    walls: list[WallSegment] = []
    for orientation, coord, start, end in _merge_segments(axis_segments):
        if orientation == "h":
            wall_start = Point2D(x=float(start), y=float(coord))
            wall_end = Point2D(x=float(end), y=float(coord))
        else:
            wall_start = Point2D(x=float(coord), y=float(start))
            wall_end = Point2D(x=float(coord), y=float(end))
        walls.append(
            WallSegment(
                start=wall_start,
                end=wall_end,
                thickness_m=thickness_m,
                height_m=height_m,
            )
        )
    return walls
