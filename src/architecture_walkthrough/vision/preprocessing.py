from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessingResult:
    original_shape: tuple[int, int, int]
    resized_shape: tuple[int, int, int]
    edges_path: Path
    grayscale_path: Path
    cleaned_path: Path


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


def preprocess_image(path: Path, debug_dir: Path, max_side: int = 1600) -> PreprocessingResult:
    image = load_image(path)
    resized = resize_preserving_aspect(image, max_side=max_side)
    _write_debug(debug_dir, "01_resized.png", resized)
    denoised = cv2.fastNlMeansDenoisingColored(resized, None, 7, 7, 7, 21)
    _write_debug(debug_dir, "02_denoised.png", denoised)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    normalized = cv2.equalizeHist(gray)
    grayscale_path = _write_debug(debug_dir, "03_grayscale_normalized.png", normalized)
    edges = cv2.Canny(normalized, 50, 150)
    edges_path = _write_debug(debug_dir, "04_edges.png", edges)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned_path = _write_debug(debug_dir, "05_morphology_cleaned.png", cleaned)
    return PreprocessingResult(
        original_shape=image.shape,
        resized_shape=resized.shape,
        edges_path=edges_path,
        grayscale_path=grayscale_path,
        cleaned_path=cleaned_path,
    )
