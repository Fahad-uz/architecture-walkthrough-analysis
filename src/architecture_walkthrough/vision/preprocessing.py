from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessingResult:
    original_shape: tuple[int, int, int]
    resized_shape: tuple[int, int, int]
    edges_path: Path
    grayscale_path: Path
    cleaned_path: Path
    layers: dict[str, Path]


def _write_debug(debug_dir: Path, name: str, image: np.ndarray) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / name
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write debug image: {path}")
    return path


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode image: {path}")
    return image


def resize_preserving_aspect(image: np.ndarray, max_side: int = 1600) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_side / max(width, height), 1.0)
    if scale == 1.0:
        return image.copy()
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def preprocess_array(image: np.ndarray, debug_dir: Path, max_side: int = 1600, options: dict[str, Any] | None = None) -> PreprocessingResult:
    options = options or {}
    resized = resize_preserving_aspect(image, max_side=max_side)
    layers: dict[str, Path] = {}
    layers["original_roi"] = _write_debug(debug_dir, "01_original_roi.png", resized)
    denoised = cv2.fastNlMeansDenoisingColored(resized, None, 7, 7, 7, 21)
    layers["denoised"] = _write_debug(debug_dir, "02_denoised.png", denoised)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    normalized = cv2.equalizeHist(gray)
    grayscale_path = _write_debug(debug_dir, "03_grayscale_normalized.png", normalized)
    layers["grayscale"] = grayscale_path
    block_size = int(max(resized.shape[:2]) * float(options.get("adaptive_block_ratio", 0.018)))
    block_size = max(11, block_size | 1)
    adaptive = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        9,
    )
    layers["adaptive_binary"] = _write_debug(debug_dir, "04_adaptive_binary.png", adaptive)
    dark_threshold = int(options.get("dark_threshold", 95))
    dark_mask = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)[1]
    layers["dark_structural_stroke"] = _write_debug(debug_dir, "05_dark_structural_stroke.png", dark_mask)

    area = resized.shape[0] * resized.shape[1]
    max_text_area = max(12, int(area * float(options.get("max_text_component_area_ratio", 0.0007))))
    text_mask = np.zeros_like(dark_mask)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    for label in range(1, components):
        x, y, w, h, component_area = stats[label]
        if component_area <= max_text_area and 2 <= h <= max(12, resized.shape[0] * 0.04):
            aspect = w / max(h, 1)
            if 0.12 <= aspect <= 12.0:
                text_mask[labels == label] = 255
    text_mask = cv2.dilate(text_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    layers["text_mask"] = _write_debug(debug_dir, "06_text_mask.png", text_mask)

    furniture_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    furniture_mask = cv2.morphologyEx(cv2.bitwise_and(adaptive, cv2.bitwise_not(text_mask)), cv2.MORPH_OPEN, furniture_kernel)
    layers["furniture_fixture_mask"] = _write_debug(debug_dir, "07_furniture_fixture_mask.png", furniture_mask)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, resized.shape[1] // 18), 3))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(15, resized.shape[0] // 18)))
    structural_no_text = cv2.bitwise_and(dark_mask, cv2.bitwise_not(text_mask))
    horizontal = cv2.morphologyEx(structural_no_text, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(structural_no_text, cv2.MORPH_OPEN, v_kernel)
    layers["horizontal_wall_band"] = _write_debug(debug_dir, "08_horizontal_wall_band.png", horizontal)
    layers["vertical_wall_band"] = _write_debug(debug_dir, "09_vertical_wall_band.png", vertical)

    door_candidates = cv2.Canny(normalized, 40, 120)
    layers["door_symbol_candidate"] = _write_debug(debug_dir, "10_door_symbol_candidate.png", door_candidates)
    window_candidates = cv2.bitwise_and(horizontal, cv2.dilate(vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))))
    layers["window_symbol_candidate"] = _write_debug(debug_dir, "11_window_symbol_candidate.png", window_candidates)

    geometry_mask = cv2.bitwise_or(horizontal, vertical)
    geometry_mask = cv2.morphologyEx(geometry_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    layers["cleaned_geometry_only"] = _write_debug(debug_dir, "12_cleaned_geometry_only.png", geometry_mask)

    edges = cv2.Canny(normalized, 50, 150)
    edges_path = _write_debug(debug_dir, "13_edges_legacy.png", edges)
    layers["edges"] = edges_path
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned_path = _write_debug(debug_dir, "14_morphology_cleaned_legacy.png", cleaned)
    layers["cleaned_edges"] = cleaned_path
    return PreprocessingResult(
        original_shape=image.shape,
        resized_shape=resized.shape,
        edges_path=edges_path,
        grayscale_path=grayscale_path,
        cleaned_path=cleaned_path,
        layers=layers,
    )


def preprocess_image(path: Path, debug_dir: Path, max_side: int = 1600) -> PreprocessingResult:
    image = load_image(path)
    return preprocess_array(image, debug_dir, max_side=max_side)
