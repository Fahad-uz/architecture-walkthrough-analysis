from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from architecture_walkthrough.config import AISettings
from architecture_walkthrough.geometry.models import Point2D


@dataclass(frozen=True)
class SegmentationResult:
    labels: dict[str, list[list[Point2D]]] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)


class Segmenter(ABC):
    @abstractmethod
    def segment(self, image_path: Path) -> SegmentationResult:
        raise NotImplementedError


class RuleBasedSegmenter(Segmenter):
    def segment(self, image_path: Path) -> SegmentationResult:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"failed to load image for segmentation: {image_path}")
        edges = cv2.Canny(image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wall_polygons: list[list[Point2D]] = []
        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue
            approx = cv2.approxPolyDP(contour, 2.0, True)
            wall_polygons.append([Point2D(x=float(p[0][0]), y=float(p[0][1])) for p in approx])
        return SegmentationResult(labels={"walls": wall_polygons, "unknown": []}, confidence={"walls": 0.35})


class AISegmenter(Segmenter):
    def __init__(self, settings: AISettings) -> None:
        self.settings = settings
        if not settings.segmentation_checkpoint:
            raise RuntimeError("AI segmentation checkpoint is not configured")
        if not settings.segmentation_checkpoint.exists():
            raise RuntimeError(f"AI segmentation checkpoint not found: {settings.segmentation_checkpoint}")

    def segment(self, image_path: Path) -> SegmentationResult:
        raise NotImplementedError("AI model loading hook exists, but no trained model is bundled")


def build_segmenter(settings: AISettings) -> Segmenter:
    if settings.segmentation_checkpoint:
        return AISegmenter(settings)
    return RuleBasedSegmenter()
