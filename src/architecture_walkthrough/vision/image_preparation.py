from __future__ import annotations

import math
from numbers import Integral, Real

import cv2
import numpy as np

from architecture_walkthrough.geometry.models import PlanROI


def validate_quarter_turns(turns: int) -> None:
    if isinstance(turns, bool) or not isinstance(turns, Integral) or not 0 <= turns <= 3:
        raise ValueError("rotation_quarter_turns must be an integer from 0 to 3")


def validate_straighten_degrees(degrees: float) -> None:
    if (isinstance(degrees, bool) or not isinstance(degrees, Real)
            or not math.isfinite(degrees) or not -15 <= degrees <= 15):
        raise ValueError("straighten_degrees must be a finite number from -15 to 15")


def _straightening_frame(
    width: int, height: int, degrees: float,
) -> tuple[int, int, np.ndarray]:
    validate_straighten_degrees(degrees)
    if degrees == 0:
        return width, height, np.eye(3)
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    # Match browser rounding at integer boundaries despite tiny trig differences.
    output_width = math.ceil(abs(width * cosine) + abs(height * sine) - 1e-9)
    output_height = math.ceil(abs(width * sine) + abs(height * cosine) - 1e-9)
    matrix = np.asarray((
        (cosine, -sine, output_width / 2 - cosine * width / 2 + sine * height / 2),
        (sine, cosine, output_height / 2 - sine * width / 2 - cosine * height / 2),
        (0, 0, 1),
    ), dtype=float)
    return output_width, output_height, matrix


def preparation_frame(
    width: int, height: int, turns: int, straighten_degrees: float = 0,
) -> tuple[int, int, np.ndarray]:
    """Expanded frame and source-edge mapping; positive angles turn clockwise."""
    validate_quarter_turns(turns)
    rotation = np.asarray((
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((0, -1, height), (1, 0, 0), (0, 0, 1)),
        ((-1, 0, width), (0, -1, height), (0, 0, 1)),
        ((0, 1, 0), (-1, 0, width), (0, 0, 1)),
    )[turns], dtype=float)
    turned_width, turned_height = (height, width) if turns % 2 else (width, height)
    output_width, output_height, fine_rotation = _straightening_frame(
        turned_width, turned_height, straighten_degrees,
    )
    return output_width, output_height, fine_rotation @ rotation


def prepared_dimensions(
    width: int, height: int, turns: int, straighten_degrees: float = 0,
) -> tuple[int, int]:
    output_width, output_height, _matrix = preparation_frame(
        width, height, turns, straighten_degrees,
    )
    return output_width, output_height


def rotate_image(image: np.ndarray, turns: int, straighten_degrees: float = 0) -> np.ndarray:
    """Apply lossless quarter-turns, then at most one expanded fine rotation."""
    validate_quarter_turns(turns)
    validate_straighten_degrees(straighten_degrees)
    if turns:
        image = cv2.rotate(image, {
            1: cv2.ROTATE_90_CLOCKWISE,
            2: cv2.ROTATE_180,
            3: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }[turns])
    if straighten_degrees == 0:
        return image
    height, width = image.shape[:2]
    output_width, output_height, edge_matrix = _straightening_frame(
        width, height, straighten_degrees,
    )
    # OpenCV locates pixel centres at integer indices; provenance and SVG use
    # edges, with centre (x + .5, y + .5). Convert the coordinate convention.
    to_edges = np.asarray(((1, 0, .5), (0, 1, .5), (0, 0, 1)))
    to_indices = np.asarray(((1, 0, -.5), (0, 1, -.5), (0, 0, 1)))
    index_matrix = to_indices @ edge_matrix @ to_edges
    return cv2.warpAffine(
        image, index_matrix[:2], (output_width, output_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )


def preparation_metadata(
    source_width: int,
    source_height: int,
    turns: int,
    roi: PlanROI,
    analysis_width: int,
    analysis_height: int,
    straighten_degrees: float = 0,
) -> dict[str, object]:
    """Map EXIF-oriented source edges to actual resized ROI edges, and back.

    Coordinates cover [0, width] × [0, height]; a pixel with array index (x, y)
    has centre (x + .5, y + .5). Cropping is lossy in content but this coordinate
    mapping remains invertible for all retained points.
    """
    width, height, rotation = preparation_frame(
        source_width, source_height, turns, straighten_degrees,
    )
    if min(source_width, source_height, analysis_width, analysis_height) <= 0:
        raise ValueError("image preparation dimensions must be positive")
    rect = roi.rect
    if rect.width <= 0 or rect.height <= 0:
        raise ValueError("image preparation ROI must be nonempty")
    scale_x, scale_y = analysis_width / rect.width, analysis_height / rect.height
    crop_resize = np.asarray((
        (scale_x, 0, -rect.x * scale_x),
        (0, scale_y, -rect.y * scale_y),
        (0, 0, 1),
    ), dtype=float)
    forward = crop_resize @ rotation
    record: dict[str, object] = {
        "version": 2 if straighten_degrees != 0 else 1,
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
    if straighten_degrees != 0:
        record["straighten_degrees"] = float(straighten_degrees)
    return record
