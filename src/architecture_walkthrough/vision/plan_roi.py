from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from architecture_walkthrough.geometry.models import BoundingBox, PlanROI


@dataclass(frozen=True)
class ROIExtractionResult:
    image: np.ndarray
    roi: PlanROI
    transform: tuple[int, int]
    debug_path: Path | None = None


def _write_debug_overlay(image: np.ndarray, roi: PlanROI, debug_dir: Path | None) -> Path | None:
    if debug_dir is None:
        return None
    debug_dir.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    rect = roi.rect
    p1 = (int(rect.x), int(rect.y))
    p2 = (int(rect.x + rect.width), int(rect.y + rect.height))
    cv2.rectangle(overlay, p1, p2, (0, 128, 255), 3)
    cv2.putText(
        overlay,
        f"ROI {roi.confidence:.2f} {roi.source}",
        (p1[0], max(20, p1[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 128, 255),
        2,
        cv2.LINE_AA,
    )
    path = debug_dir / "00_plan_roi.png"
    cv2.imwrite(str(path), overlay)
    return path


def _manual_roi(
    image: np.ndarray,
    crop_rect: tuple[int, int, int, int],
    padding_px: int,
) -> PlanROI:
    height, width = image.shape[:2]
    x, y, w, h = crop_rect
    x0 = max(0, x - padding_px)
    y0 = max(0, y - padding_px)
    x1 = min(width, x + w + padding_px)
    y1 = min(height, y + h + padding_px)
    return PlanROI(
        rect=BoundingBox(x=x0, y=y0, width=max(1, x1 - x0), height=max(1, y1 - y0)),
        confidence=1.0,
        source="manual",
        padding_px=padding_px,
        full_image_width_px=width,
        full_image_height_px=height,
    )


def detect_plan_roi(
    image: np.ndarray,
    padding_ratio: float = 0.02,
    min_confidence: float = 0.35,
    crop_rect: tuple[int, int, int, int] | None = None,
    debug_dir: Path | None = None,
) -> ROIExtractionResult:
    height, width = image.shape[:2]
    padding_px = int(round(max(width, height) * padding_ratio))
    if crop_rect is not None:
        roi = _manual_roi(image, crop_rect, padding_px)
        rect = roi.rect
        debug_path = _write_debug_overlay(image, roi, debug_dir)
        return ROIExtractionResult(
            image=image[int(rect.y) : int(rect.y + rect.height), int(rect.x) : int(rect.x + rect.width)].copy(),
            roi=roi,
            transform=(int(rect.x), int(rect.y)),
            debug_path=debug_path,
        )

    from architecture_walkthrough.vision.preprocessing import strip_border_bars, structural_ink_mask

    # Black/grey ink only (colored furniture fills are not structure), page
    # frame bars stripped, hollow double-line walls fused into bands.
    binary = structural_ink_mask(image, dark_threshold=210, max_saturation=90)
    min_area = max(25, int(width * height * 0.00002))
    components, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros_like(binary)
    for label in range(1, components):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        if w > width * 0.95 and h > height * 0.95:
            continue
        mask[labels == label] = 255
    close_px = max(3, int(max(width, height) * 0.008) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px)))

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, width // 20), 3))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(25, height // 25)))
    structural = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
    structural = cv2.bitwise_or(structural, cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel))
    structural = strip_border_bars(structural)
    structural = cv2.dilate(structural, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
    contours, _ = cv2.findContours(structural, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        roi = PlanROI(
            rect=BoundingBox(x=0, y=0, width=width, height=height),
            confidence=0.0,
            source="full_image_fallback",
            full_image_width_px=width,
            full_image_height_px=height,
        )
        debug_path = _write_debug_overlay(image, roi, debug_dir)
        return ROIExtractionResult(image=image.copy(), roi=roi, transform=(0, 0), debug_path=debug_path)

    # The plan is the union of all significant structural regions, not just the
    # single largest blob — thin-walled plans fragment into many components.
    boxes = [cv2.boundingRect(contour) for contour in contours]
    significant = [box for box in boxes if box[2] * box[3] >= width * height * 0.005]
    if not significant:
        significant = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)[:1]
    x0u = min(box[0] for box in significant)
    y0u = min(box[1] for box in significant)
    x1u = max(box[0] + box[2] for box in significant)
    y1u = max(box[1] + box[3] for box in significant)
    x, y, w, h = x0u, y0u, x1u - x0u, y1u - y0u

    x0 = max(0, x - padding_px)
    y0 = max(0, y - padding_px)
    x1 = min(width, x + w + padding_px)
    y1 = min(height, y + h + padding_px)
    roi_area = (x1 - x0) * (y1 - y0)
    density = float((structural[y0:y1, x0:x1] > 0).mean()) if roi_area else 0.0
    coverage = roi_area / float(width * height)
    confidence = max(0.0, min(1.0, density * 8.0 + min(coverage, 0.7)))
    source = "auto"
    if confidence < min_confidence or roi_area < width * height * 0.08:
        x0, y0, x1, y1 = 0, 0, width, height
        confidence = min(confidence, 0.25)
        source = "full_image_fallback"

    roi = PlanROI(
        rect=BoundingBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0),
        confidence=confidence,
        source=source,
        padding_px=padding_px,
        full_image_width_px=width,
        full_image_height_px=height,
    )
    debug_path = _write_debug_overlay(image, roi, debug_dir)
    return ROIExtractionResult(image=image[y0:y1, x0:x1].copy(), roi=roi, transform=(x0, y0), debug_path=debug_path)
