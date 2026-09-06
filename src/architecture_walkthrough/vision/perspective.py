from __future__ import annotations

import cv2
import numpy as np

from architecture_walkthrough.geometry.models import Point2D


def estimate_quadrilateral(edge_image: np.ndarray) -> list[Point2D] | None:
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) != 4:
        return None
    return [Point2D(x=float(point[0][0]), y=float(point[0][1])) for point in approx]


def apply_manual_perspective(image: np.ndarray, corners: list[Point2D], output_size: tuple[int, int]) -> np.ndarray:
    if len(corners) != 4:
        raise ValueError("manual perspective correction requires exactly four corners")
    src: np.ndarray = np.asarray(
        [[corner.x, corner.y] for corner in corners],
        dtype=np.float32,
    )
    width, height = output_size
    dst: np.ndarray = np.asarray(
        [[0, 0], [width, 0], [width, height], [0, height]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (width, height))
