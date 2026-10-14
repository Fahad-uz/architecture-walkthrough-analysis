from __future__ import annotations

from numbers import Integral

import cv2
import numpy as np

from architecture_walkthrough.geometry.models import PlanROI


def validate_quarter_turns(turns: int) -> None:
    if isinstance(turns, bool) or not isinstance(turns, Integral) or not 0 <= turns <= 3:
        raise ValueError("rotation_quarter_turns must be an integer from 0 to 3")


def prepared_dimensions(width: int, height: int, turns: int) -> tuple[int, int]:
    validate_quarter_turns(turns)
    return (height, width) if turns % 2 else (width, height)


def rotate_image(image: np.ndarray, turns: int) -> np.ndarray:
    """Rotate displayed source pixels clockwise without interpolation."""
    validate_quarter_turns(turns)
    if not turns:
        return image
    return cv2.rotate(image, {
        1: cv2.ROTATE_90_CLOCKWISE,
        2: cv2.ROTATE_180,
        3: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }[turns])


def preparation_metadata(
    source_width: int,
    source_height: int,
    turns: int,
    roi: PlanROI,
    analysis_width: int,
    analysis_height: int,
) -> dict[str, object]:
    """Map EXIF-oriented source edges to actual resized ROI edges, and back.

    Coordinates cover [0, width] × [0, height]; a pixel with array index (x, y)
    has centre (x + .5, y + .5). Cropping is lossy in content but this coordinate
    mapping remains invertible for all retained points.
    """
    width, height = prepared_dimensions(source_width, source_height, turns)
    if min(source_width, source_height, analysis_width, analysis_height) <= 0:
        raise ValueError("image preparation dimensions must be positive")
    rect = roi.rect
    if rect.width <= 0 or rect.height <= 0:
        raise ValueError("image preparation ROI must be nonempty")
    rotation = np.asarray((
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((0, -1, source_height), (1, 0, 0), (0, 0, 1)),
        ((-1, 0, source_width), (0, -1, source_height), (0, 0, 1)),
        ((0, 1, 0), (-1, 0, source_width), (0, 0, 1)),
    )[turns], dtype=float)
    scale_x, scale_y = analysis_width / rect.width, analysis_height / rect.height
    crop_resize = np.asarray((
        (scale_x, 0, -rect.x * scale_x),
        (0, scale_y, -rect.y * scale_y),
        (0, 0, 1),
    ), dtype=float)
    forward = crop_resize @ rotation
    return {
        "version": 1,
        "rotation_quarter_turns": int(turns),
        "source_width_px": source_width,
        "source_height_px": source_height,
        "prepared_width_px": width,
        "prepared_height_px": height,
        "analysis_width_px": analysis_width,
        "analysis_height_px": analysis_height,
        "source_to_analysis": forward.tolist(),
        "analysis_to_source": np.linalg.inv(forward).tolist(),
    }
