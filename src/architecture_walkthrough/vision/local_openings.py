from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from architecture_walkthrough.config import GeometryDefaults, OpeningDetectionSettings
from architecture_walkthrough.geometry.models import DoorOpening, Point2D, WallSegment, WindowOpening
from architecture_walkthrough.geometry.wall_graph import collinear_gaps, perpendicular_endpoint_gaps

# Openings must come from local image evidence. Gemini may only classify
# ambiguous candidates (door vs cabinet arc); it never supplies coordinates.


@dataclass(frozen=True)
class OpeningCandidate:
    wall_id: str
    start_offset_m: float
    end_offset_m: float
    kind: str  # "door" | "window" | "ambiguous"
    confidence: float
    evidence: tuple[str, ...]
    hinge_side: str | None = None
    swing_side: str | None = None

    @property
    def width_m(self) -> float:
        return self.end_offset_m - self.start_offset_m


@dataclass
class LocalOpeningResult:
    walls: list[WallSegment]
    doors: list[DoorOpening] = field(default_factory=list)
    windows: list[WindowOpening] = field(default_factory=list)
    ambiguous: list[OpeningCandidate] = field(default_factory=list)
    merged_wall_ids: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class SwingSymbol:
    center_px: tuple[float, float]
    diagonal_px: float
    confidence: float


def detect_colored_swing_symbols(image_bgr: np.ndarray | None) -> list[SwingSymbol]:
    """Detect sparse, diagonally oriented red door-swing glyphs.

    The rendered-plan style used by many design tools draws swings as a red
    45-degree hatched sector instead of a clean black circular arc. Beds and
    counters can also be red, so candidates must be sparse contours with a
    diagonal minimum-area box. They are not openings by themselves; the caller
    still requires a nearby measured wall gap of architectural width.
    """

    if image_bgr is None or image_bgr.ndim != 3:
        return []
    height, width = image_bgr.shape[:2]
    basis = max(width, height)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red = ((hue <= 8) | (hue >= 172)) & (saturation >= 100) & (value >= 180)
    mask = np.where(red, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    symbols: list[SwingSymbol] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if (
            min(box_width, box_height) < basis * 0.035
            or max(box_width, box_height) > basis * 0.18
        ):
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        contour_area = float(cv2.contourArea(contour))
        if hull_area < basis * basis * 0.0015 or contour_area / max(hull_area, 1.0) > 0.24:
            continue
        rect = cv2.minAreaRect(contour)
        points = cv2.boxPoints(rect)
        edges = [points[(index + 1) % 4] - points[index] for index in range(4)]
        longest = max(edges, key=lambda edge: float(np.hypot(*edge)))
        angle = abs(math.degrees(math.atan2(float(longest[1]), float(longest[0])))) % 90.0
        diagonal_deviation = abs(angle - 45.0)
        if diagonal_deviation > 12.0:
            continue
        diagonal = math.hypot(box_width, box_height)
        confidence = max(0.55, min(0.92, 0.82 - diagonal_deviation / 60.0))
        symbols.append(
            SwingSymbol(
                center_px=(x + box_width / 2.0, y + box_height / 2.0),
                diagonal_px=diagonal,
                confidence=confidence,
            )
        )
    return symbols


def _swing_symbol_support(
    gap_start_px: np.ndarray,
    gap_end_px: np.ndarray,
    symbols: list[SwingSymbol],
    consumed: set[int],
) -> tuple[int, float] | None:
    gap_px = float(np.hypot(*(gap_end_px - gap_start_px)))
    if gap_px <= 1.0:
        return None
    midpoint = (gap_start_px + gap_end_px) / 2.0
    best: tuple[float, int, float] | None = None
    for index, symbol in enumerate(symbols):
        if index in consumed:
            continue
        size_ratio = symbol.diagonal_px / gap_px
        if not 0.65 <= size_ratio <= 2.10:
            continue
        distance = float(
            np.hypot(
                symbol.center_px[0] - midpoint[0],
                symbol.center_px[1] - midpoint[1],
            )
        )
        max_distance = max(gap_px * 1.85, symbol.diagonal_px * 1.35)
        if distance > max_distance:
            continue
        support = (1.0 - distance / max_distance) * 0.55 + symbol.confidence * 0.45
        if best is None or support > best[0]:
            best = (support, index, symbol.confidence)
    if best is None or best[0] < 0.48:
        return None
    return best[1], min(0.9, 0.50 + best[0] * 0.35)


def build_thin_line_mask(
    adaptive_binary: np.ndarray,
    dark_structural: np.ndarray,
    colored_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Strokes that are drawn but not bold structure: arcs, leaves, glazing lines."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bold = cv2.dilate(dark_structural, kernel, iterations=2)
    thin = cv2.bitwise_and(adaptive_binary, cv2.bitwise_not(bold))
    if colored_mask is not None:
        thin = cv2.bitwise_or(thin, colored_mask)
    return cv2.dilate(thin, kernel, iterations=1)


def _to_px(point: Point2D, ppm: float, image_height_px: int) -> np.ndarray:
    return np.array([point.x * ppm, image_height_px - point.y * ppm], dtype=np.float64)


def _coverage(mask: np.ndarray, points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    height, width = mask.shape[:2]
    xs = np.clip(np.round(points[:, 0]).astype(int), 0, width - 1)
    ys = np.clip(np.round(points[:, 1]).astype(int), 0, height - 1)
    return float((mask[ys, xs] > 0).mean())


def _segment_points(p0: np.ndarray, p1: np.ndarray, samples: int) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, max(samples, 2))
    return p0[None, :] + ts[:, None] * (p1 - p0)[None, :]


def _presence_profile(mask: np.ndarray, p0: np.ndarray, p1: np.ndarray, half_thickness_px: float) -> np.ndarray:
    """Per-step booleans: is bold wall material present across the band here?"""
    length = float(np.hypot(*(p1 - p0)))
    steps = max(int(length), 2)
    base = _segment_points(p0, p1, steps)
    direction = (p1 - p0) / max(length, 1e-9)
    normal = np.array([-direction[1], direction[0]])
    presence = np.zeros(steps, dtype=bool)
    height, width = mask.shape[:2]
    for offset in np.linspace(-half_thickness_px, half_thickness_px, 5):
        pts = base + normal[None, :] * offset
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, width - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, height - 1)
        presence |= mask[ys, xs] > 0
    return presence


def _absent_runs(presence: np.ndarray, min_len: int, max_len: int, min_flank: int) -> list[tuple[int, int]]:
    """Interior gaps in the presence profile, flanked by real wall on both sides."""
    # Bridge single-pixel noise so hairline speckles don't split a gap.
    smoothed = presence.copy()
    for index in range(1, len(smoothed) - 1):
        if not smoothed[index] and smoothed[index - 1] and smoothed[index + 1]:
            smoothed[index] = True
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, on in enumerate(smoothed):
        if not on and start is None:
            start = index
        elif on and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(smoothed)))
    accepted: list[tuple[int, int]] = []
    for run_start, run_end in runs:
        length = run_end - run_start
        if not min_len <= length <= max_len:
            continue
        if run_start < min_flank or len(smoothed) - run_end < min_flank:
            continue
        accepted.append((run_start, run_end))
    return accepted


def _arc_coverage(
    mask: np.ndarray,
    center: np.ndarray,
    radius: float,
    dir_into_gap: np.ndarray,
    side_sign: float,
    samples: int = 28,
) -> float:
    """Coverage of a quarter arc swept from the gap direction toward the wall normal."""
    normal = np.array([-dir_into_gap[1], dir_into_gap[0]]) * side_sign
    start_angle = math.atan2(dir_into_gap[1], dir_into_gap[0])
    end_angle = math.atan2(normal[1], normal[0])
    delta = end_angle - start_angle
    while delta > math.pi:
        delta -= 2 * math.pi
    while delta < -math.pi:
        delta += 2 * math.pi
    ts = np.linspace(0.10, 0.90, samples)
    angles = start_angle + ts * delta
    points = center[None, :] + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return _coverage(mask, points)


def _score_candidate(
    thin_mask: np.ndarray,
    gap_start_px: np.ndarray,
    gap_end_px: np.ndarray,
    half_thickness_px: float,
    settings: OpeningDetectionSettings,
    external_wall: bool = False,
) -> tuple[str, float, tuple[str, ...], str | None, str | None]:
    """Score door/window evidence for a gap given in pixel coordinates."""
    gap_vector = gap_end_px - gap_start_px
    gap_px = float(np.hypot(*gap_vector))
    if gap_px <= 1:
        return "ambiguous", 0.0, ("wall_gap",), None, None
    direction = gap_vector / gap_px

    best_arc = 0.0
    best_hinge: str | None = None
    best_side: str | None = None
    for hinge_name, hinge_point, into_gap in (("start", gap_start_px, direction), ("end", gap_end_px, -direction)):
        for side_name, side_sign in (("left", 1.0), ("right", -1.0)):
            coverage = _arc_coverage(thin_mask, hinge_point, gap_px, into_gap, side_sign)
            if coverage > best_arc:
                best_arc, best_hinge, best_side = coverage, hinge_name, side_name

    leaf = 0.0
    if best_hinge is not None and best_side is not None:
        hinge_point = gap_start_px if best_hinge == "start" else gap_end_px
        into_gap = direction if best_hinge == "start" else -direction
        normal = np.array([-into_gap[1], into_gap[0]]) * (1.0 if best_side == "left" else -1.0)
        leaf = _coverage(thin_mask, _segment_points(hinge_point, hinge_point + normal * gap_px, 20))

    window_coverages: list[float] = []
    normal = np.array([-direction[1], direction[0]])
    for offset in (-half_thickness_px * 0.75, 0.0, half_thickness_px * 0.75):
        pts = _segment_points(gap_start_px + normal * offset, gap_end_px + normal * offset, 24)
        window_coverages.append(_coverage(thin_mask, pts))

    evidence: list[str] = ["wall_gap"]
    strong_arc_threshold = max(0.42, settings.arc_coverage_threshold + 0.12)
    if best_arc >= settings.arc_coverage_threshold and (
        best_arc >= strong_arc_threshold or leaf >= settings.leaf_coverage_threshold
    ):
        evidence.append("swing_arc")
        if leaf >= settings.leaf_coverage_threshold:
            evidence.append("leaf_line")
        confidence = min(0.95, 0.5 + best_arc * 0.4 + (0.1 if leaf >= settings.leaf_coverage_threshold else 0.0))
        return "door", confidence, tuple(evidence), best_hinge, best_side
    supported_tracks = sorted(
        (coverage for coverage in window_coverages if coverage >= settings.window_line_coverage_threshold),
        reverse=True,
    )
    # A single annotation, counter edge, or wall outline is not glazing.
    # Require two separately offset tracks before calling the gap a window.
    if len(supported_tracks) >= 2:
        window_cov = sum(supported_tracks[:2]) / 2
        evidence.append("parallel_lines")
        return "window", min(0.9, 0.45 + window_cov * 0.4), tuple(evidence), None, None
    if external_wall and supported_tracks and supported_tracks[0] >= 0.80:
        # Hollow exterior walls often retain one measured glazing track while
        # the second coincides with (and is removed as) the bold wall face.
        evidence.append("external_glazing_line")
        return "window", min(0.82, 0.48 + supported_tracks[0] * 0.28), tuple(evidence), None, None
    return "ambiguous", 0.35, tuple(evidence), None, None


def _merge_collinear_pair(wall_a: WallSegment, wall_b: WallSegment) -> WallSegment:
    horizontal = abs(wall_a.end.x - wall_a.start.x) >= abs(wall_a.end.y - wall_a.start.y)
    if horizontal:
        y = (wall_a.start.y + wall_a.end.y + wall_b.start.y + wall_b.end.y) / 4
        xs = [wall_a.start.x, wall_a.end.x, wall_b.start.x, wall_b.end.x]
        start, end = Point2D(x=min(xs), y=y), Point2D(x=max(xs), y=y)
    else:
        x = (wall_a.start.x + wall_a.end.x + wall_b.start.x + wall_b.end.x) / 4
        ys = [wall_a.start.y, wall_a.end.y, wall_b.start.y, wall_b.end.y]
        start, end = Point2D(x=x, y=min(ys)), Point2D(x=x, y=max(ys))
    return wall_a.model_copy(
        update={
            "start": start,
            "end": end,
            "thickness_m": max(wall_a.thickness_m, wall_b.thickness_m),
            "external": wall_a.external or wall_b.external,
            "wall_type": "external" if (wall_a.external or wall_b.external) else wall_a.wall_type,
            "confidence": (wall_a.confidence + wall_b.confidence) / 2,
            "evidence_source": "merged_across_opening",
        }
    )


def _extend_wall_across_gap(
    wall: WallSegment,
    gap_start: Point2D,
    gap_end: Point2D,
) -> WallSegment:
    horizontal = abs(wall.end.x - wall.start.x) >= abs(wall.end.y - wall.start.y)
    if horizontal:
        xs = [wall.start.x, wall.end.x, gap_start.x, gap_end.x]
        y = (wall.start.y + wall.end.y + gap_start.y + gap_end.y) / 4.0
        start = Point2D(x=min(xs), y=y)
        end = Point2D(x=max(xs), y=y)
    else:
        ys = [wall.start.y, wall.end.y, gap_start.y, gap_end.y]
        x = (wall.start.x + wall.end.x + gap_start.x + gap_end.x) / 4.0
        start = Point2D(x=x, y=min(ys))
        end = Point2D(x=x, y=max(ys))
    return wall.model_copy(
        update={
            "start": start,
            "end": end,
            "evidence_source": f"{wall.evidence_source}+extended_across_confirmed_opening",
        }
    )


def _offset_on_wall(wall: WallSegment, point: Point2D) -> float:
    length = wall.start.distance_to(wall.end)
    if length <= 0:
        return 0.0
    ux = (wall.end.x - wall.start.x) / length
    uy = (wall.end.y - wall.start.y) / length
    return (point.x - wall.start.x) * ux + (point.y - wall.start.y) * uy


def _point_at_offset(wall: WallSegment, offset: float) -> Point2D:
    length = wall.start.distance_to(wall.end)
    if length <= 0:
        return wall.start
    ux = (wall.end.x - wall.start.x) / length
    uy = (wall.end.y - wall.start.y) / length
    return Point2D(x=wall.start.x + ux * offset, y=wall.start.y + uy * offset)


def _candidate_to_opening(
    candidate: OpeningCandidate,
    wall: WallSegment,
    defaults: GeometryDefaults,
    index: int,
) -> DoorOpening | WindowOpening:
    center = _point_at_offset(wall, (candidate.start_offset_m + candidate.end_offset_m) / 2)
    if candidate.kind == "door":
        return DoorOpening(
            id=f"door_{index:03d}",
            center=center,
            wall_id=candidate.wall_id,
            start_offset_m=candidate.start_offset_m,
            end_offset_m=candidate.end_offset_m,
            width_m=candidate.width_m,
            height_m=defaults.door_height_m,
            hinge_side=candidate.hinge_side,  # type: ignore[arg-type]
            swing_side=candidate.swing_side,  # type: ignore[arg-type]
            confidence=candidate.confidence,
            evidence_source="+".join(candidate.evidence),
        )
    return WindowOpening(
        id=f"window_{index:03d}",
        center=center,
        wall_id=candidate.wall_id,
        start_offset_m=candidate.start_offset_m,
        end_offset_m=candidate.end_offset_m,
        width_m=candidate.width_m,
        height_m=defaults.window_height_m,
        sill_height_m=defaults.sill_height_m,
        confidence=candidate.confidence,
        evidence_source="+".join(candidate.evidence),
    )


def detect_local_openings(
    walls: list[WallSegment],
    dark_structural: np.ndarray,
    thin_mask: np.ndarray,
    pixels_per_metre: float,
    image_height_px: int,
    settings: OpeningDetectionSettings,
    defaults: GeometryDefaults,
    color_image: np.ndarray | None = None,
) -> LocalOpeningResult:
    """Detect door/window openings from local image evidence.

    Two candidate sources: interior breaks in the bold wall mask along a wall,
    and gaps between collinear wall segments. Each candidate is scored with
    swing-arc, door-leaf, and parallel-glazing-line evidence; a confirmed gap
    between two segments merges them into one wall carrying the opening.
    """
    ppm = pixels_per_metre
    min_open = min(settings.min_door_width_m, settings.min_window_width_m)
    max_open = max(settings.max_door_width_m, settings.max_window_width_m)
    min_flank_px = max(4, int(settings.min_flank_m * ppm))

    working = {wall.id: wall for wall in walls if wall.id}
    candidates: list[OpeningCandidate] = []
    swing_symbols = detect_colored_swing_symbols(color_image)
    consumed_symbols: set[int] = set()

    # Pass 1: gaps between collinear wall pairs (the common case for doors).
    thickness = min((wall.thickness_m for wall in walls), default=0.12)
    # Separate detector passes can place the two sides of one hollow wall on
    # slightly different centerlines. Honour the configured projection
    # tolerance, but cap it relative to wall thickness so genuinely parallel
    # nearby walls are not paired as one opening.
    coord_tol = max(0.04, min(settings.projection_tolerance_m, thickness * 1.5))
    for gap in collinear_gaps(list(working.values()), coord_tol):
        length_m = float(gap["length_m"])  # type: ignore[arg-type]
        if not min_open <= length_m <= max_open:
            continue
        wall_a = working.get(str(gap["wall_a"]))
        wall_b = working.get(str(gap["wall_b"]))
        if wall_a is None or wall_b is None:
            continue
        gap_line = gap["line"]
        (ax, ay), (bx, by) = list(gap_line.coords)  # type: ignore[attr-defined]
        gap_start_px = _to_px(Point2D(x=ax, y=ay), ppm, image_height_px)
        gap_end_px = _to_px(Point2D(x=bx, y=by), ppm, image_height_px)
        half_thickness_px = max(wall_a.thickness_m, wall_b.thickness_m) * ppm / 2
        kind, confidence, evidence, hinge, swing = _score_candidate(
            thin_mask,
            gap_start_px,
            gap_end_px,
            half_thickness_px,
            settings,
            external_wall=wall_a.external or wall_b.external,
        )
        if kind == "ambiguous":
            symbol_support = _swing_symbol_support(
                gap_start_px,
                gap_end_px,
                swing_symbols,
                consumed_symbols,
            )
            if symbol_support is not None:
                symbol_index, confidence = symbol_support
                consumed_symbols.add(symbol_index)
                kind = "door"
                evidence = (*evidence, "colored_swing_symbol")
                hinge = "end"
                swing = "left"
        gap_start = Point2D(x=ax, y=ay)
        gap_end = Point2D(x=bx, y=by)
        if kind == "ambiguous":
            # A wall-sized gap is only a hypothesis.  Keep it visible to the
            # correction UI, but do not invent a door/window or bridge the two
            # structural segments without local swing/glazing evidence.
            start_offset = _offset_on_wall(wall_a, gap_start)
            end_offset = _offset_on_wall(wall_a, gap_end)
            if end_offset < start_offset:
                start_offset, end_offset = end_offset, start_offset
            candidates.append(
                OpeningCandidate(
                    wall_id=str(wall_a.id),
                    start_offset_m=start_offset,
                    end_offset_m=end_offset,
                    kind="ambiguous",
                    confidence=confidence,
                    evidence=evidence,
                )
            )
            continue

        merged = _merge_collinear_pair(wall_a, wall_b)
        del working[wall_b.id]  # type: ignore[arg-type]
        working[merged.id] = merged  # type: ignore[index]
        start_offset = _offset_on_wall(merged, gap_start)
        end_offset = _offset_on_wall(merged, gap_end)
        if end_offset < start_offset:
            start_offset, end_offset = end_offset, start_offset
        candidates.append(
            OpeningCandidate(
                wall_id=str(merged.id),
                start_offset_m=start_offset,
                end_offset_m=end_offset,
                kind=kind,
                confidence=confidence,
                evidence=evidence,
                hinge_side=hinge,
                swing_side=swing,
            )
        )

    # Pass 1b: door-sized gaps from a wall endpoint to a perpendicular room
    # boundary. A confirmed colored swing extends the source wall for rendering
    # and stores the opening interval on that measured centerline.
    perpendicular_gaps = perpendicular_endpoint_gaps(
        list(working.values()),
        min_open,
        min(max_open, settings.max_door_width_m),
    )
    for gap in perpendicular_gaps:
        source = working.get(str(gap["wall_a"]))
        if source is None:
            continue
        gap_line = gap["line"]
        (ax, ay), (bx, by) = list(gap_line.coords)  # type: ignore[attr-defined]
        gap_start = Point2D(x=ax, y=ay)
        gap_end = Point2D(x=bx, y=by)
        gap_start_px = _to_px(gap_start, ppm, image_height_px)
        gap_end_px = _to_px(gap_end, ppm, image_height_px)
        half_thickness_px = source.thickness_m * ppm / 2.0
        kind, confidence, evidence, hinge, swing = _score_candidate(
            thin_mask,
            gap_start_px,
            gap_end_px,
            half_thickness_px,
            settings,
            external_wall=source.external,
        )
        if kind == "ambiguous":
            symbol_support = _swing_symbol_support(
                gap_start_px,
                gap_end_px,
                swing_symbols,
                consumed_symbols,
            )
            if symbol_support is not None:
                symbol_index, confidence = symbol_support
                consumed_symbols.add(symbol_index)
                kind = "door"
                evidence = (*evidence, "colored_swing_symbol", "perpendicular_endpoint_gap")
                hinge = "end"
                swing = "left"

        if kind == "ambiguous":
            continue
        target_wall = _extend_wall_across_gap(source, gap_start, gap_end)
        working[str(target_wall.id)] = target_wall
        start_offset = _offset_on_wall(target_wall, gap_start)
        end_offset = _offset_on_wall(target_wall, gap_end)
        if end_offset < start_offset:
            start_offset, end_offset = end_offset, start_offset
        candidates.append(
            OpeningCandidate(
                wall_id=str(target_wall.id),
                start_offset_m=start_offset,
                end_offset_m=end_offset,
                kind=kind,
                confidence=confidence,
                evidence=evidence,
                hinge_side=hinge,
                swing_side=swing,
            )
        )

    # Pass 2: interior breaks in the bold mask along each (possibly merged) wall.
    for wall in working.values():
        length_m = wall.start.distance_to(wall.end)
        if length_m < min_open + 2 * settings.min_flank_m:
            continue
        p0 = _to_px(wall.start, ppm, image_height_px)
        p1 = _to_px(wall.end, ppm, image_height_px)
        half_thickness_px = wall.thickness_m * ppm / 2
        presence = _presence_profile(dark_structural, p0, p1, half_thickness_px)
        px_per_step = length_m * ppm / max(len(presence) - 1, 1)
        runs = _absent_runs(
            presence,
            min_len=int(min_open * ppm / px_per_step),
            max_len=int(max_open * ppm / px_per_step),
            min_flank=min_flank_px,
        )
        existing = [c for c in candidates if c.wall_id == wall.id]
        for run_start, run_end in runs:
            start_offset = run_start * px_per_step / ppm
            end_offset = run_end * px_per_step / ppm
            if any(
                min(end_offset, c.end_offset_m) - max(start_offset, c.start_offset_m) > -0.15
                for c in existing
            ):
                continue
            gap_start_px = p0 + (p1 - p0) * (run_start / max(len(presence) - 1, 1))
            gap_end_px = p0 + (p1 - p0) * (run_end / max(len(presence) - 1, 1))
            kind, confidence, evidence, hinge, swing = _score_candidate(
                thin_mask,
                gap_start_px,
                gap_end_px,
                half_thickness_px,
                settings,
                external_wall=wall.external,
            )
            candidates.append(
                OpeningCandidate(
                    wall_id=str(wall.id),
                    start_offset_m=start_offset,
                    end_offset_m=end_offset,
                    kind=kind,
                    confidence=confidence,
                    evidence=evidence,
                    hinge_side=hinge,
                    swing_side=swing,
                )
            )

    doors: list[DoorOpening] = []
    windows: list[WindowOpening] = []
    ambiguous = [candidate for candidate in candidates if candidate.kind == "ambiguous"]
    for candidate in candidates:
        if candidate.kind == "ambiguous":
            continue
        candidate_wall = working.get(candidate.wall_id)
        if candidate_wall is None:
            continue
        opening = _candidate_to_opening(
            candidate,
            candidate_wall,
            defaults,
            len(doors) if candidate.kind == "door" else len(windows),
        )
        if isinstance(opening, DoorOpening):
            doors.append(opening)
        else:
            windows.append(opening)

    ordered_walls = sorted(working.values(), key=lambda wall: str(wall.id))
    return LocalOpeningResult(walls=ordered_walls, doors=doors, windows=windows, ambiguous=ambiguous)
