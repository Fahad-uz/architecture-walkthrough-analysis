from __future__ import annotations

from dataclasses import dataclass, field
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
class RepetitiveDetailRegion:
    """Dense, regularly spaced linework such as stair treads or hatching."""

    orientation: str
    rect: tuple[float, float, float, float]
    spacing_px: float
    line_count: int

    def is_stair_like(self) -> bool:
        """Separate tread fields from wide, shallow material hatching."""

        _x, _y, width, height = self.rect
        run_span = width if self.orientation == "h" else height
        progression_span = height if self.orientation == "h" else width
        if run_span <= 0 or progression_span <= 0:
            return False
        aspect = progression_span / run_span
        relative_pitch = self.spacing_px / run_span
        return 4 <= self.line_count <= 30 and 0.35 <= aspect <= 2.8 and 0.045 <= relative_pitch <= 0.35


@dataclass(frozen=True)
class WallDetectionResult:
    bands: list[WallBand]
    walls: list[WallSegment]
    debug_overlay_path: Path | None = None
    repetitive_detail_regions: list[RepetitiveDetailRegion] = field(default_factory=list)
    rejected_detail_bands: list[WallBand] = field(default_factory=list)


def _write_overlay(
    image_shape: tuple[int, int],
    bands: list[WallBand],
    debug_dir: Path | None,
    rejected_detail_bands: list[WallBand] | None = None,
    repetitive_detail_regions: list[RepetitiveDetailRegion] | None = None,
) -> Path | None:
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
    for region in repetitive_detail_regions or []:
        x, y, w, h = region.rect
        cv2.rectangle(
            overlay,
            (int(round(x)), int(round(y))),
            (int(round(x + w)), int(round(y + h))),
            (180, 70, 180),
            2,
        )
        cv2.putText(
            overlay,
            f"detail x{region.line_count}",
            (int(round(x)), max(12, int(round(y)) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 70, 180),
            1,
        )
    for band in rejected_detail_bands or []:
        x1, y1, x2, y2 = band.centerline
        cv2.line(
            overlay,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (80, 80, 190),
            1,
        )
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


def _true_runs(values: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    """Return half-open true runs, bridging only tiny rasterization gaps."""

    active = np.asarray(values, dtype=bool).copy()
    if max_gap > 0:
        index = 0
        while index < len(active):
            if active[index]:
                index += 1
                continue
            end = index
            while end < len(active) and not active[end]:
                end += 1
            if index > 0 and end < len(active) and end - index <= max_gap:
                active[index:end] = True
            index = end
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(active)))
    return runs


def _dense_outer_detail_components(
    mask: np.ndarray,
    orientation: str,
    min_length_px: float,
    max_thickness_px: float,
) -> tuple[list[WallBand], list[RepetitiveDetailRegion]]:
    """Recover the envelope edge hidden inside an outer patterned component.

    Balcony brickwork, facade hatching, and deck boards can connect to a real
    exterior line, producing one component too thick for normal wall-band
    extraction. The component's outward-most long run is still direct wall
    evidence. Its dense interior is also recorded so perpendicular hatch
    strokes cannot later become walls.
    """

    image_height, image_width = mask.shape[:2]
    image_progression = image_height if orientation == "h" else image_width
    image_run = image_width if orientation == "h" else image_height
    basis = max(image_width, image_height)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    recovered: list[WallBand] = []
    regions: list[RepetitiveDetailRegion] = []

    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        long_side = width if orientation == "h" else height
        thickness = height if orientation == "h" else width
        fill_ratio = area / max(width * height, 1)
        progression_start = y if orientation == "h" else x
        progression_end = progression_start + thickness
        near_start = progression_start <= image_progression * 0.16
        near_end = progression_end >= image_progression * 0.84
        if (
            long_side < max(min_length_px * 4.0, image_run * 0.25)
            or thickness <= max_thickness_px
            or thickness > basis * 0.22
            or long_side / max(thickness, 1) < 3.0
            or not 0.10 <= fill_ratio <= 0.82
            or not (near_start or near_end)
        ):
            continue

        component = labels[y : y + height, x : x + width] == label
        progression_view = component if orientation == "h" else component.T
        use_start_edge = near_start and (not near_end or progression_start < image_progression / 2)
        edge_depth = max(3, min(thickness, int(round(thickness * 0.18))))
        edge_indices = range(0, edge_depth) if use_start_edge else range(thickness - edge_depth, thickness)
        boundary_candidates: list[tuple[int, int, int]] = []
        for progression_index in edge_indices:
            runs = _true_runs(progression_view[progression_index], max_gap=2)
            if not runs:
                continue
            run_start, run_end = max(runs, key=lambda run: run[1] - run[0])
            boundary_candidates.append((run_end - run_start, progression_index, run_start))
        if not boundary_candidates:
            continue
        boundary_length, progression_index, run_start = max(boundary_candidates)
        if boundary_length < max(min_length_px * 2.0, image_run * 0.20):
            continue
        run_end = run_start + boundary_length

        run_density = progression_view.sum(axis=0)
        core_threshold = max(4, int(round(thickness * 0.10)))
        core_runs = _true_runs(run_density >= core_threshold, max_gap=max(2, int(round(basis * 0.004))))
        if not core_runs:
            continue
        core_start, core_end = max(core_runs, key=lambda run: run[1] - run[0])
        if core_end - core_start < min_length_px * 2.0:
            continue

        projection = progression_view[:, core_start:core_end].sum(axis=1)
        course_runs = _true_runs(projection >= (core_end - core_start) * 0.42, max_gap=1)
        course_centers = [(start + end - 1) / 2.0 for start, end in course_runs]
        spacings = [
            second - first
            for first, second in zip(course_centers, course_centers[1:])
            if second - first >= 3.0
        ]
        spacing = float(np.median(spacings)) if spacings else max(3.0, thickness / 6.0)
        line_count = max(4, len(course_centers))

        coordinate = progression_start + progression_index + 0.5
        global_run_start = (x if orientation == "h" else y) + run_start
        global_run_end = (x if orientation == "h" else y) + run_end
        band_thickness = 3.0
        if orientation == "h":
            centerline = (float(global_run_start), coordinate, float(global_run_end), coordinate)
            rect = (
                float(global_run_start),
                coordinate - band_thickness / 2.0,
                float(boundary_length),
                band_thickness,
            )
            region_rect = (
                float(x + core_start),
                float(y),
                float(core_end - core_start),
                float(height),
            )
        else:
            centerline = (coordinate, float(global_run_start), coordinate, float(global_run_end))
            rect = (
                coordinate - band_thickness / 2.0,
                float(global_run_start),
                band_thickness,
                float(boundary_length),
            )
            region_rect = (
                float(x),
                float(y + core_start),
                float(width),
                float(core_end - core_start),
            )
        recovered.append(
            WallBand(
                id=f"recovered_outer_{orientation}_{len(recovered):03d}",
                orientation=orientation,
                rect=rect,
                centerline=centerline,
                thickness_px=band_thickness,
                confidence=0.82,
                external=True,
            )
        )
        regions.append(
            RepetitiveDetailRegion(
                orientation=orientation,
                rect=region_rect,
                spacing_px=spacing,
                line_count=line_count,
            )
        )
    return recovered, regions


def _suppress_bands_inside_dense_details(
    bands: list[WallBand],
    regions: list[RepetitiveDetailRegion],
) -> tuple[list[WallBand], list[WallBand]]:
    """Remove perpendicular mortar/hatch strokes while keeping strip edges."""

    kept: list[WallBand] = []
    rejected: list[WallBand] = []
    for band in bands:
        reject = False
        span_start, span_end = _band_span(band)
        span_length = max(span_end - span_start, 1.0)
        for region in regions:
            if band.orientation == region.orientation:
                continue
            x, y, width, height = region.rect
            edge_margin = max(6.0, region.spacing_px * 2.0)
            if region.orientation == "h":
                coordinate = _band_coordinate(band)
                overlap = max(0.0, min(span_end, y + height) - max(span_start, y))
                interior = x + edge_margin < coordinate < x + width - edge_margin
            else:
                coordinate = _band_coordinate(band)
                overlap = max(0.0, min(span_end, x + width) - max(span_start, x))
                interior = y + edge_margin < coordinate < y + height - edge_margin
            if interior and overlap / span_length >= 0.68:
                reject = True
                break
        (rejected if reject else kept).append(band)
    return kept, rejected


def _recover_dense_detail_side_extensions(
    regions: list[RepetitiveDetailRegion],
    structural_bands: list[WallBand],
    image_shape: tuple[int, int],
) -> list[WallBand]:
    """Extend a measured side wall through a connected outer hatch strip.

    A side is recovered only when a perpendicular structural band is already
    aligned with that texture edge and touches or overlaps the strip. The
    hatch therefore cannot create a new partition by itself; it can only fill
    the portion of an existing wall hidden by connected pattern linework.
    """

    basis = float(max(image_shape))
    coordinate_tolerance = max(8.0, basis * 0.028)
    span_tolerance = max(6.0, basis * 0.018)
    recovered: list[WallBand] = []
    for region in regions:
        x, y, width, height = region.rect
        if region.orientation == "h":
            candidates = [band for band in structural_bands if band.orientation == "v"]
            side_coordinates = (x, x + width)
            region_start, region_end = y, y + height
        else:
            candidates = [band for band in structural_bands if band.orientation == "h"]
            side_coordinates = (y, y + height)
            region_start, region_end = x, x + width
        for side_coordinate in side_coordinates:
            aligned: list[tuple[float, float, WallBand]] = []
            for band in candidates:
                coordinate_delta = abs(_band_coordinate(band) - side_coordinate)
                if coordinate_delta > coordinate_tolerance:
                    continue
                band_start, band_end = _band_span(band)
                span_gap = max(0.0, max(band_start, region_start) - min(band_end, region_end))
                if span_gap <= span_tolerance:
                    aligned.append((coordinate_delta, span_gap, band))
            if not aligned:
                continue
            source = min(aligned, key=lambda value: (value[0], value[1]))[2]
            coordinate = _band_coordinate(source)
            thickness = max(3.0, min(source.thickness_px, basis * 0.025))
            if region.orientation == "h":
                centerline = (coordinate, region_start, coordinate, region_end)
                rect = (
                    coordinate - thickness / 2.0,
                    region_start,
                    thickness,
                    region_end - region_start,
                )
                orientation = "v"
            else:
                centerline = (region_start, coordinate, region_end, coordinate)
                rect = (
                    region_start,
                    coordinate - thickness / 2.0,
                    region_end - region_start,
                    thickness,
                )
                orientation = "h"
            recovered.append(
                WallBand(
                    id=f"recovered_detail_side_{orientation}_{len(recovered):03d}",
                    orientation=orientation,
                    rect=rect,
                    centerline=centerline,
                    thickness_px=thickness,
                    confidence=min(0.90, source.confidence + 0.06),
                    external=source.external,
                )
            )
    return recovered


def _band_coordinate(band: WallBand) -> float:
    return band.centerline[1] if band.orientation == "h" else band.centerline[0]


def _band_span(band: WallBand) -> tuple[float, float]:
    values = (
        (band.centerline[0], band.centerline[2])
        if band.orientation == "h"
        else (band.centerline[1], band.centerline[3])
    )
    return min(values), max(values)


def _longest_regular_sequence(
    bands: list[WallBand],
    *,
    min_count: int,
    min_spacing_px: float,
    max_spacing_px: float,
) -> tuple[list[WallBand], float]:
    """Find the strongest arithmetic run of parallel line coordinates."""

    ordered = sorted(bands, key=_band_coordinate)
    best: list[WallBand] = []
    best_spacing = 0.0
    for first_index, first in enumerate(ordered):
        for second_index in range(first_index + 1, len(ordered)):
            spacing = _band_coordinate(ordered[second_index]) - _band_coordinate(first)
            if spacing < min_spacing_px:
                continue
            if spacing > max_spacing_px:
                break
            tolerance = max(2.0, spacing * 0.22)
            sequence = [first, ordered[second_index]]
            expected = _band_coordinate(ordered[second_index]) + spacing
            for candidate in ordered[second_index + 1 :]:
                coordinate = _band_coordinate(candidate)
                if coordinate < expected - tolerance:
                    continue
                if coordinate > expected + tolerance:
                    break
                sequence.append(candidate)
                expected += spacing
            if len(sequence) > len(best):
                best = sequence
                best_spacing = spacing
    if len(best) < min_count:
        return [], 0.0
    return best, best_spacing


def _suppress_repetitive_detail_bands(
    bands: list[WallBand],
    image_shape: tuple[int, int],
    *,
    min_count: int = 4,
) -> tuple[list[WallBand], list[WallBand], list[RepetitiveDetailRegion]]:
    """Reject repeated thin runs before they can become walls.

    Stair treads, balcony brick courses and material hatches share a useful
    geometric signature: at least four thin, similarly spanning lines at a
    regular pitch.  Real wall faces can be parallel too, but they do not form
    a dense arithmetic family with almost identical endpoints.  Suppression
    happens before collinear merging so a tread touching a true room boundary
    cannot drag that boundary across the stairwell.
    """

    if len(bands) < min_count:
        return list(bands), [], []
    basis = float(max(image_shape))
    thin_limit = max(5.0, basis * 0.006)
    endpoint_tolerance = max(5.0, basis * 0.012)
    min_spacing = max(5.0, basis * 0.005)
    max_spacing = max(min_spacing + 1.0, basis * 0.055)
    rejected_ids: set[int] = set()
    regions: list[RepetitiveDetailRegion] = []

    for orientation in ("h", "v"):
        eligible = [
            band
            for band in bands
            if band.orientation == orientation and band.thickness_px <= thin_limit
        ]
        parents = list(range(len(eligible)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(first: int, second: int) -> None:
            first_root = find(first)
            second_root = find(second)
            if first_root != second_root:
                parents[second_root] = first_root

        for first_index, first in enumerate(eligible):
            first_start, first_end = _band_span(first)
            first_length = first_end - first_start
            for second_index in range(first_index + 1, len(eligible)):
                second = eligible[second_index]
                second_start, second_end = _band_span(second)
                second_length = second_end - second_start
                length_ratio = min(first_length, second_length) / max(first_length, second_length, 1.0)
                if (
                    length_ratio >= 0.78
                    and abs(first_start - second_start) <= endpoint_tolerance
                    and abs(first_end - second_end) <= endpoint_tolerance
                ):
                    union(first_index, second_index)

        grouped: dict[int, list[WallBand]] = {}
        for index, band in enumerate(eligible):
            grouped.setdefault(find(index), []).append(band)
        for group in grouped.values():
            sequence, spacing = _longest_regular_sequence(
                group,
                min_count=min_count,
                min_spacing_px=min_spacing,
                max_spacing_px=max_spacing,
            )
            if not sequence:
                continue
            for band in sequence:
                rejected_ids.add(id(band))
            span_start = min(_band_span(band)[0] for band in sequence)
            span_end = max(_band_span(band)[1] for band in sequence)
            coordinate_start = min(_band_coordinate(band) for band in sequence) - spacing * 0.5
            coordinate_end = max(_band_coordinate(band) for band in sequence) + spacing * 0.5
            rect = (
                (span_start, coordinate_start, span_end - span_start, coordinate_end - coordinate_start)
                if orientation == "h"
                else (coordinate_start, span_start, coordinate_end - coordinate_start, span_end - span_start)
            )
            regions.append(
                RepetitiveDetailRegion(
                    orientation=orientation,
                    rect=rect,
                    spacing_px=spacing,
                    line_count=len(sequence),
                )
            )

    # A thin line running through the middle of repeated treads is usually a
    # stair centre/stringer or a hatch stroke, not a full-height wall.  Keep
    # perimeter faces by protecting a margin along the region's two sides.
    for band in bands:
        if id(band) in rejected_ids or band.thickness_px > thin_limit:
            continue
        for region in regions:
            if band.orientation == region.orientation:
                continue
            x, y, width, height = region.rect
            edge_margin = max(3.0, region.spacing_px * 0.35)
            if region.orientation == "h":
                coordinate = band.centerline[0]
                span_start, span_end = sorted((band.centerline[1], band.centerline[3]))
                overlap = max(0.0, min(span_end, y + height) - max(span_start, y))
                interior = x + edge_margin < coordinate < x + width - edge_margin
            else:
                coordinate = band.centerline[1]
                span_start, span_end = sorted((band.centerline[0], band.centerline[2]))
                overlap = max(0.0, min(span_end, x + width) - max(span_start, x))
                interior = y + edge_margin < coordinate < y + height - edge_margin
            covered = overlap / max(span_end - span_start, 1.0)
            if interior and covered >= 0.65:
                rejected_ids.add(id(band))
                break

    rejected = [band for band in bands if id(band) in rejected_ids]
    kept = [band for band in bands if id(band) not in rejected_ids]
    return kept, rejected, regions


def _pair_parallel_wall_faces(
    bands: list[WallBand],
    image_shape: tuple[int, int],
) -> list[WallBand]:
    """Collapse two thin CAD wall faces onto their physical centerline.

    Globally increasing the collinear tolerance also merges nearby furniture
    edges and can erase short real walls.  Pair only highly overlapping thin
    faces at a plausible drawn wall thickness, then let the normal collinear
    pass handle fragmentation along that new centerline.
    """

    basis = float(max(image_shape))
    thin_limit = max(5.0, basis * 0.006)
    min_separation = max(5.0, basis * 0.007)
    max_separation = max(18.0, basis * 0.024)
    target_separation = max(10.0, basis * 0.015)
    available = [band for band in bands]
    paired_ids: set[int] = set()
    pair_candidates: list[tuple[float, int, int]] = []

    for first_index, first in enumerate(available):
        if first.thickness_px > thin_limit:
            continue
        first_start, first_end = _band_span(first)
        first_length = first_end - first_start
        for second_index in range(first_index + 1, len(available)):
            second = available[second_index]
            if second.orientation != first.orientation or second.thickness_px > thin_limit:
                continue
            separation = abs(_band_coordinate(second) - _band_coordinate(first))
            if not min_separation <= separation <= max_separation:
                continue
            second_start, second_end = _band_span(second)
            second_length = second_end - second_start
            overlap = max(0.0, min(first_end, second_end) - max(first_start, second_start))
            overlap_ratio = overlap / max(first_length, second_length, 1.0)
            if overlap_ratio < 0.72:
                continue
            separation_score = abs(separation - target_separation) / max(target_separation, 1.0)
            pair_candidates.append((overlap_ratio - separation_score * 0.12, first_index, second_index))

    pairs: list[WallBand] = []
    for _score, first_index, second_index in sorted(pair_candidates, reverse=True):
        first = available[first_index]
        second = available[second_index]
        if id(first) in paired_ids or id(second) in paired_ids:
            continue
        paired_ids.update((id(first), id(second)))
        orientation = first.orientation
        coordinate = (_band_coordinate(first) + _band_coordinate(second)) / 2.0
        span_start = min(_band_span(first)[0], _band_span(second)[0])
        span_end = max(_band_span(first)[1], _band_span(second)[1])
        separation = abs(_band_coordinate(second) - _band_coordinate(first))
        thickness = separation + (first.thickness_px + second.thickness_px) / 2.0
        rect = (
            (span_start, coordinate - thickness / 2.0, span_end - span_start, thickness)
            if orientation == "h"
            else (coordinate - thickness / 2.0, span_start, thickness, span_end - span_start)
        )
        centerline = (
            (span_start, coordinate, span_end, coordinate)
            if orientation == "h"
            else (coordinate, span_start, coordinate, span_end)
        )
        pairs.append(
            WallBand(
                id=f"paired_{orientation}_{len(pairs):03d}",
                orientation=orientation,
                rect=rect,
                centerline=centerline,
                thickness_px=thickness,
                confidence=min(1.0, (first.confidence + second.confidence) / 2.0 + 0.12),
                external=first.external or second.external,
            )
        )

    unpaired = [band for band in available if id(band) not in paired_ids]
    return [*unpaired, *pairs]


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
    """Merge nearby, overlapping runs without inventing the span between them.

    The old implementation sorted primarily by the rounded line coordinate and
    then used a one-sided ``next.start <= previous.end + tolerance`` check.  A
    segment in the next coordinate bucket could therefore appear *after* a
    far-right segment while starting hundreds of pixels to its left.  The
    one-sided check treated those disjoint runs as touching and emitted a
    phantom wall covering their entire bounding box.

    Build connected components with a symmetric interval-distance test instead.
    This remains transitive for genuinely fragmented wall runs, while two spans
    only join when their projections actually overlap or are separated by at
    most ``gap_tolerance_px``.
    """

    def line_coordinate(band: WallBand) -> float:
        return band.centerline[1] if band.orientation == "h" else band.centerline[0]

    def span(band: WallBand) -> tuple[float, float]:
        values = (
            (band.centerline[0], band.centerline[2])
            if band.orientation == "h"
            else (band.centerline[1], band.centerline[3])
        )
        return min(values), max(values)

    def interval_gap(first: WallBand, second: WallBand) -> float:
        first_start, first_end = span(first)
        second_start, second_end = span(second)
        return max(0.0, max(first_start, second_start) - min(first_end, second_end))

    merged: list[WallBand] = []
    for orientation in ("h", "v"):
        oriented = [band for band in bands if band.orientation == orientation]
        oriented.sort(key=lambda band: (line_coordinate(band), span(band)[0], span(band)[1]))

        parents = list(range(len(oriented)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(first: int, second: int) -> None:
            root_first = find(first)
            root_second = find(second)
            if root_first != root_second:
                parents[root_second] = root_first

        for first_index, first in enumerate(oriented):
            for second_index in range(first_index + 1, len(oriented)):
                second = oriented[second_index]
                coordinate_delta = line_coordinate(second) - line_coordinate(first)
                if coordinate_delta > coordinate_tolerance_px:
                    break
                if interval_gap(first, second) <= gap_tolerance_px:
                    union(first_index, second_index)

        grouped: dict[int, list[WallBand]] = {}
        for index, band in enumerate(oriented):
            grouped.setdefault(find(index), []).append(band)
        groups = sorted(
            grouped.values(),
            key=lambda group: (min(line_coordinate(band) for band in group), min(span(band)[0] for band in group)),
        )
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
    recovered_h, dense_h_regions = _dense_outer_detail_components(
        horizontal,
        "h",
        min_length_px,
        max_thickness_px,
    )
    recovered_v, dense_v_regions = _dense_outer_detail_components(
        vertical,
        "v",
        min_length_px,
        max_thickness_px,
    )
    structural_bands, rejected_detail_bands, repetitive_detail_regions = _suppress_repetitive_detail_bands(
        h_bands + v_bands,
        (height, width),
    )
    dense_regions = [*dense_h_regions, *dense_v_regions]
    structural_bands, dense_rejected_bands = _suppress_bands_inside_dense_details(
        structural_bands,
        dense_regions,
    )
    recovered_sides = _recover_dense_detail_side_extensions(
        dense_regions,
        structural_bands,
        (height, width),
    )
    rejected_detail_bands.extend(dense_rejected_bands)
    repetitive_detail_regions.extend(dense_regions)
    structural_bands = _pair_parallel_wall_faces(structural_bands, (height, width))
    structural_bands.extend([*recovered_h, *recovered_v, *recovered_sides])
    bands = _merge_collinear_bands(structural_bands, coordinate_tolerance_px, gap_tolerance_px)
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
    overlay_path = _write_overlay(
        (height, width),
        bands,
        debug_dir,
        rejected_detail_bands,
        repetitive_detail_regions,
    )
    return WallDetectionResult(
        bands=bands,
        walls=walls,
        debug_overlay_path=overlay_path,
        repetitive_detail_regions=repetitive_detail_regions,
        rejected_detail_bands=rejected_detail_bands,
    )


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
