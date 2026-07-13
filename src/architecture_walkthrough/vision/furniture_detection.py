from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from architecture_walkthrough.geometry.models import FurniturePlacement, Point2D
from architecture_walkthrough.vision.preprocessing import load_image, resize_preserving_aspect


def _mask_for_colored_objects(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Rendered plans usually draw floors and walls in low-saturation whites,
    # creams, and greys. Furniture, rugs, plants, wood, and counters carry more
    # chroma, so this isolates semantic objects without relying on text labels.
    colored = (saturation > 28) & (value > 35) & (value < 250)
    plants = (hue >= 35) & (hue <= 95) & (saturation > 40) & (value > 35)
    mask = np.where(colored | plants, 255, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _category_for_patch(image: np.ndarray, contour: np.ndarray, width: int, height: int) -> str:
    x, y, w, h = cv2.boundingRect(contour)
    patch = image[y : y + h, x : x + w]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = float(np.median(hsv[:, :, 0]))
    saturation = float(np.median(hsv[:, :, 1]))
    aspect = max(w, h) / max(min(w, h), 1)
    area_ratio = (w * h) / max(width * height, 1)

    if 35 <= hue <= 95 and saturation > 55 and area_ratio < 0.025:
        return "plant"
    if 16 <= hue <= 95 and saturation > 45:
        if area_ratio > 0.01 and aspect > 1.35:
            return "bed"
        if area_ratio > 0.004:
            return "sofa"
        return "chair"
    if 8 <= hue <= 28 and saturation > 45 and aspect > 2.0:
        return "kitchen_counter"
    if 8 <= hue <= 35 and saturation > 35:
        return "dining_table" if area_ratio > 0.004 else "side_table"
    if area_ratio > 0.008 and aspect > 1.5:
        return "rug"
    return "furniture"


def detect_furniture_from_image(
    image_path: Path,
    pixels_per_metre: float,
    image_height_px: int,
    max_side: int = 1600,
) -> list[FurniturePlacement]:
    image = resize_preserving_aspect(load_image(image_path), max_side=max_side)
    mask = _mask_for_colored_objects(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height_px, width_px = image.shape[:2]
    placements: list[FurniturePlacement] = []
    min_area = max(90, int(width_px * height_px * 0.00012))
    max_area = int(width_px * height_px * 0.08)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (box_w, box_h), angle = rect
        if box_w <= 3 or box_h <= 3:
            continue
        width_m = max(box_w / pixels_per_metre, 0.25)
        depth_m = max(box_h / pixels_per_metre, 0.25)
        if width_m > 5.0 or depth_m > 5.0:
            continue
        category = _category_for_patch(image, contour, width_px, height_px)
        rotation = float(angle)
        if width_m < depth_m:
            width_m, depth_m = depth_m, width_m
            rotation += 90.0
        placements.append(
            FurniturePlacement(
                category=category,
                center=Point2D(x=center_x / pixels_per_metre, y=(image_height_px - center_y) / pixels_per_metre),
                width_m=width_m,
                depth_m=depth_m,
                rotation_deg=rotation,
            )
        )

    return _deduplicate_furniture(placements)


def _deduplicate_furniture(items: list[FurniturePlacement]) -> list[FurniturePlacement]:
    kept: list[FurniturePlacement] = []
    for item in sorted(items, key=lambda value: value.width_m * value.depth_m, reverse=True):
        too_close = False
        for existing in kept:
            distance = item.center.distance_to(existing.center)
            footprint = min(item.width_m, item.depth_m, existing.width_m, existing.depth_m)
            if distance < max(0.25, footprint * 0.5):
                too_close = True
                break
        if not too_close:
            kept.append(item)
    return kept[:80]
