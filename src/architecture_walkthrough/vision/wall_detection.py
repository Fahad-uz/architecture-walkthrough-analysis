from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from architecture_walkthrough.geometry.models import Point2D, WallSegment


@dataclass(frozen=True)
class WallBand:
    id: str
    orientation: str
    rect: tuple[float, float, float, float]
    centerline: tuple[float, float, float, float]
    thickness_px: float
    confidence: float
    external: bool = False


@dataclass(frozen=True)
class WallDetectionResult:
    bands: list[WallBand]
    walls: list[WallSegment]
    debug_overlay_path: Path | None = None


def _write_overlay(image_shape: tuple[int, int], bands: list[WallBand], debug_dir: Path | None) -> Path | None:
    if debug_dir is None:
        return None
    debug_dir.mkdir(parents=True, exist_ok=True)
    overlay = np.full((image_shape[0], image_shape[1], 3), 255, dtype=np.uint8)
    for band in bands:
        x, y, w, h = band.rect
        color = (20, 120, 240) if band.orientation == "h" else (60, 170, 70)
        cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        x1, y1, x2, y2 = band.centerline
        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 1)
        cv2.putText(overlay, band.id, (int(x), int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    path = debug_dir / "wall_band_overlay.png"
    cv2.imwrite(str(path), overlay)
    return path


def _components_to_bands(
    mask: np.ndarray,
    orientation: str,
    min_length_px: float,
    min_thickness_px: float,
    max_thickness_px: float,
    prefix: str,
) -> list[WallBand]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bands: list[WallBand] = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        long_side = w if orientation == "h" else h
        thickness = h if orientation == "h" else w
        if long_side < min_length_px:
            continue
        if thickness < min_thickness_px or thickness > max_thickness_px:
            continue
        fill_ratio = area / max(w * h, 1)
        if fill_ratio < 0.22:
            continue
        if orientation == "h":
            center = y + h / 2
            centerline = (float(x), float(center), float(x + w), float(center))
        else:
            center = x + w / 2
            centerline = (float(center), float(y), float(center), float(y + h))
        confidence = max(0.1, min(1.0, fill_ratio * min(1.0, long_side / (min_length_px * 3))))
        bands.append(
            WallBand(
                id=f"{prefix}{len(bands):03d}",
                orientation=orientation,
                rect=(float(x), float(y), float(w), float(h)),
                centerline=centerline,
                thickness_px=float(thickness),
                confidence=confidence,
            )
        )
    return bands


def _merge_band_group(group: list[WallBand], band_id: str) -> WallBand:
    orientation = group[0].orientation
    xs = [band.rect[0] for band in group]
    ys = [band.rect[1] for band in group]
    x2s = [band.rect[0] + band.rect[2] for band in group]
    y2s = [band.rect[1] + band.rect[3] for band in group]
    x, y, x2, y2 = min(xs), min(ys), max(x2s), max(y2s)
    if orientation == "h":
        center = sum((band.centerline[1] * band.confidence) for band in group) / sum(band.confidence for band in group)
        centerline = (x, center, x2, center)
    else:
        center = sum((band.centerline[0] * band.confidence) for band in group) / sum(band.confidence for band in group)
        centerline = (center, y, center, y2)
    return WallBand(
        id=band_id,
        orientation=orientation,
        rect=(x, y, x2 - x, y2 - y),
        centerline=centerline,
        thickness_px=float(np.median([band.thickness_px for band in group])),
        confidence=float(np.mean([band.confidence for band in group])),
        external=any(band.external for band in group),
    )


def _merge_collinear_bands(
    bands: list[WallBand],
    coordinate_tolerance_px: float,
    gap_tolerance_px: float,
) -> list[WallBand]:
    merged: list[WallBand] = []
    for orientation in ("h", "v"):
        oriented = [band for band in bands if band.orientation == orientation]
        if orientation == "h":
            oriented.sort(key=lambda band: (round(band.centerline[1] / coordinate_tolerance_px), band.centerline[0]))
        else:
            oriented.sort(key=lambda band: (round(band.centerline[0] / coordinate_tolerance_px), band.centerline[1]))
        groups: list[list[WallBand]] = []
        for band in oriented:
            if not groups:
                groups.append([band])
                continue
            previous = groups[-1][-1]
            if orientation == "h":
                same_line = abs(band.centerline[1] - previous.centerline[1]) <= coordinate_tolerance_px
                touches = band.centerline[0] <= previous.centerline[2] + gap_tolerance_px
            else:
                same_line = abs(band.centerline[0] - previous.centerline[0]) <= coordinate_tolerance_px
                touches = band.centerline[1] <= previous.centerline[3] + gap_tolerance_px
            if same_line and touches:
                groups[-1].append(band)
            else:
                groups.append([band])
        for group in groups:
            merged.append(_merge_band_group(group, f"{orientation}wall_{len(merged):03d}"))
    return merged


def _classify_external(bands: list[WallBand], image_shape: tuple[int, int]) -> list[WallBand]:
    if not bands:
        return []
    centers_x = [(band.centerline[0] + band.centerline[2]) / 2 for band in bands]
    centers_y = [(band.centerline[1] + band.centerline[3]) / 2 for band in bands]
    min_x, max_x = min(centers_x), max(centers_x)
    min_y, max_y = min(centers_y), max(centers_y)
    tolerance = max(image_shape) * 0.035
    classified: list[WallBand] = []
    for band in bands:
        cx = (band.centerline[0] + band.centerline[2]) / 2
        cy = (band.centerline[1] + band.centerline[3]) / 2
        external = (
            abs(cx - min_x) <= tolerance
            or abs(cx - max_x) <= tolerance
            or abs(cy - min_y) <= tolerance
            or abs(cy - max_y) <= tolerance
        )
        classified.append(
            WallBand(
                id=band.id,
                orientation=band.orientation,
                rect=band.rect,
                centerline=band.centerline,
                thickness_px=band.thickness_px,
                confidence=band.confidence,
                external=external,
            )
        )
    return classified


def detect_wall_bands(
    horizontal_mask_path: Path,
    vertical_mask_path: Path,
    debug_dir: Path | None = None,
    min_length_ratio: float = 0.035,
    merge_gap_ratio: float = 0.012,
    coordinate_tolerance_ratio: float = 0.006,
    min_thickness_px: int = 3,
    max_thickness_ratio: float = 0.04,
    internal_thickness_m: float = 0.12,
    external_thickness_m: float = 0.20,
    wall_height_m: float = 3.0,
) -> WallDetectionResult:
    horizontal = cv2.imread(str(horizontal_mask_path), cv2.IMREAD_GRAYSCALE)
    vertical = cv2.imread(str(vertical_mask_path), cv2.IMREAD_GRAYSCALE)
    if horizontal is None or vertical is None:
        raise ValueError("failed to read wall-band masks")
    height, width = horizontal.shape[:2]
    basis = max(width, height)
    min_length_px = basis * min_length_ratio
    max_thickness_px = max(min_thickness_px + 1, basis * max_thickness_ratio, 12.0)
    gap_tolerance_px = basis * merge_gap_ratio
    coordinate_tolerance_px = basis * coordinate_tolerance_ratio
    h_bands = _components_to_bands(horizontal, "h", min_length_px, min_thickness_px, max_thickness_px, "hraw_")
    v_bands = _components_to_bands(vertical, "v", min_length_px, min_thickness_px, max_thickness_px, "vraw_")
    bands = _merge_collinear_bands(h_bands + v_bands, coordinate_tolerance_px, gap_tolerance_px)
    bands = _classify_external(bands, (height, width))
    walls: list[WallSegment] = []
    for index, band in enumerate(bands):
        x1, y1, x2, y2 = band.centerline
        walls.append(
            WallSegment(
                id=f"w{index:03d}",
                start=Point2D(x=x1, y=y1),
                end=Point2D(x=x2, y=y2),
                thickness_m=external_thickness_m if band.external else internal_thickness_m,
                height_m=wall_height_m,
                external=band.external,
                wall_type="external" if band.external else "internal",
                confidence=band.confidence,
                evidence_source="wall_band",
                source_band_id=band.id,
            )
        )
    overlay_path = _write_overlay((height, width), bands, debug_dir)
    return WallDetectionResult(bands=bands, walls=walls, debug_overlay_path=overlay_path)


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


def detect_wall_lines(edge_image_path: Path, thickness_m: float = 0.12, height_m: float = 3.0) -> list[WallSegment]:
    image = cv2.imread(str(edge_image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read edge image: {edge_image_path}")
    lines = cv2.HoughLinesP(image, 1, 3.14159 / 180, threshold=80, minLineLength=70, maxLineGap=12)
    if lines is None:
        return []
    walls: list[WallSegment] = []
    for [[x1, y1, x2, y2]] in lines:
        segment = _axis_aligned_segment(int(x1), int(y1), int(x2), int(y2))
        if segment is None:
            continue
        orientation, coord, start, end = segment
        if end - start < 60:
            continue
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
                confidence=0.25,
                evidence_source="legacy_hough_supplement",
            )
        )
    return walls
