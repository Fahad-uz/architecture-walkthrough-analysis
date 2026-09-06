from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np

from architecture_walkthrough.geometry.models import FloorPlanModel, Point2D, WallSegment


def _pt(point: Point2D, ppm: float, height_px: int) -> tuple[float, float]:
    return point.x * ppm, height_px - point.y * ppm


def _wall_line(wall: WallSegment, ppm: float, height_px: int, color: str, width: int) -> str:
    x1, y1 = _pt(wall.start, ppm, height_px)
    x2, y2 = _pt(wall.end, ppm, height_px)
    label = escape(wall.id or "")
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" opacity="0.85"/>'
        f'<text x="{mx:.2f}" y="{my:.2f}" font-size="11" fill="{color}">{label}</text>'
    )


def write_analysis_overlay(
    source_image_path: Path,
    model: FloorPlanModel,
    output_svg: Path,
    output_png: Path | None = None,
    raw_walls: list[WallSegment] | None = None,
) -> tuple[Path, Path | None]:
    image = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read source image for overlay: {source_image_path}")
    height, width = image.shape[:2]
    ppm = model.pixels_per_metre or 1.0
    layers: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    if raw_walls:
        layers.append('<g id="raw-walls">')
        layers.extend(_wall_line(wall, ppm, height, "#d98000", 2) for wall in raw_walls)
        layers.append("</g>")
    layers.append('<g id="optimized-walls">')
    layers.extend(_wall_line(wall, ppm, height, "#0868d8", 4 if wall.external else 3) for wall in model.walls)
    layers.append("</g>")
    layers.append('<g id="rooms">')
    for room in model.rooms:
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in [_pt(point, ppm, height) for point in room.points])
        layers.append(f'<polygon points="{points}" fill="#2ca25f22" stroke="#2ca25f" stroke-width="2"/>')
        if room.name:
            cx = sum(point.x for point in room.points) / len(room.points) * ppm
            cy = height - (sum(point.y for point in room.points) / len(room.points) * ppm)
            layers.append(f'<text x="{cx:.2f}" y="{cy:.2f}" font-size="13" fill="#006d2c">{escape(room.name)}</text>')
    layers.append("</g>")
    layers.append('<g id="balconies">')
    for balcony in model.balconies:
        points = " ".join(
            f"{x:.2f},{y:.2f}"
            for x, y in [_pt(point, ppm, height) for point in balcony.points]
        )
        layers.append(
            f'<polygon points="{points}" fill="#3182bd22" '
            'stroke="#3182bd" stroke-width="2"/>'
        )
        if balcony.name:
            cx = sum(point.x for point in balcony.points) / len(balcony.points) * ppm
            cy = height - (
                sum(point.y for point in balcony.points)
                / len(balcony.points)
                * ppm
            )
            layers.append(
                f'<text x="{cx:.2f}" y="{cy:.2f}" font-size="13" '
                f'fill="#08519c">{escape(balcony.name)}</text>'
            )
    layers.append("</g>")
    layers.append('<g id="openings">')
    for door in model.doors:
        x, y = _pt(door.center, ppm, height)
        layers.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="#7b3294"><title>{escape(door.id or "door")}</title></circle>')
    for window in model.windows:
        x, y = _pt(window.center, ppm, height)
        layers.append(f'<rect x="{x - 5:.2f}" y="{y - 5:.2f}" width="10" height="10" fill="#00a6ca"><title>{escape(window.id or "window")}</title></rect>')
    layers.append("</g>")
    layers.append('<g id="special-elements">')
    for element in model.special_elements:
        if element.polygon:
            points = " ".join(f"{x:.2f},{y:.2f}" for x, y in [_pt(point, ppm, height) for point in element.polygon])
            layers.append(f'<polygon points="{points}" fill="#f03b2022" stroke="#f03b20" stroke-width="2"/>')
        elif element.center:
            x, y = _pt(element.center, ppm, height)
            layers.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="13" fill="#f03b20">{escape(element.kind)}</text>')
    layers.append("</g>")
    layers.append('<g id="quality">')
    for idx, issue in enumerate(model.validation_issues):
        color = "#cb181d" if issue.severity in {"error", "severe"} else "#a6761d"
        layers.append(f'<text x="12" y="{22 + idx * 16}" font-size="12" fill="{color}">{escape(issue.code)}: {escape(issue.message)}</text>')
    layers.append("</g>")
    layers.append("</svg>")
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text("\n".join(layers), encoding="utf-8")

    png_path: Path | None = None
    if output_png is not None:
        canvas = image.copy()
        if raw_walls:
            for wall in raw_walls:
                x1, y1 = _pt(wall.start, ppm, height)
                x2, y2 = _pt(wall.end, ppm, height)
                cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 128, 255), 2)
        for wall in model.walls:
            x1, y1 = _pt(wall.start, ppm, height)
            x2, y2 = _pt(wall.end, ppm, height)
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (216, 104, 8), 3)
        for room in model.rooms:
            pts = np.array([_pt(point, ppm, height) for point in room.points], dtype=np.int32)
            if len(pts) >= 3:
                cv2.polylines(canvas, [pts], True, (47, 163, 84), 2)
        for balcony in model.balconies:
            pts = np.array(
                [_pt(point, ppm, height) for point in balcony.points],
                dtype=np.int32,
            )
            if len(pts) >= 3:
                cv2.polylines(canvas, [pts], True, (189, 130, 49), 2)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_png), canvas)
        png_path = output_png
    return output_svg, png_path
