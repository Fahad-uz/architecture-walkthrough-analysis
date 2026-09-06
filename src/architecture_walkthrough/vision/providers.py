from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.vision.wall_detection import WallDetectionResult, detect_wall_bands

# Geometry must come from local image evidence (CV or a segmentation model).
# Vision-LLM coordinates are never accurate enough to be a geometry provider.


class GeometryProvider(ABC):
    """Source of wall geometry for the pipeline. Swappable so the CV baseline
    can be replaced by CubiCasa5k (or RoomFormer later) without touching
    downstream stages."""

    name: str = "unknown"
    accuracy_tier: str = "medium"  # "high" (trained model) | "medium" (CV) | "low"

    @abstractmethod
    def detect(self, layers: dict[str, Path], debug_dir: Path | None, config: AppConfig) -> WallDetectionResult:
        raise NotImplementedError


class WallBandGeometryProvider(GeometryProvider):
    """Morphological wall-band detector — the local-CV baseline."""

    name = "wallband_cv"
    accuracy_tier = "medium"

    def detect(self, layers: dict[str, Path], debug_dir: Path | None, config: AppConfig) -> WallDetectionResult:
        return detect_wall_bands(
            layers["horizontal_wall_band"],
            layers["vertical_wall_band"],
            debug_dir=debug_dir,
            min_length_ratio=config.wall_bands.min_length_ratio,
            merge_gap_ratio=config.wall_bands.merge_gap_ratio,
            coordinate_tolerance_ratio=config.wall_bands.coordinate_tolerance_ratio,
            min_thickness_px=config.wall_bands.min_thickness_px,
            max_thickness_ratio=config.wall_bands.max_thickness_ratio,
            internal_thickness_m=config.defaults.internal_wall_thickness_m,
            external_thickness_m=config.defaults.external_wall_thickness_m,
            wall_height_m=config.defaults.wall_height_m,
        )


class CubiCasaGeometryProvider(GeometryProvider):
    """CubiCasa5k segmentation model (preferred when its weights are available).

    Activates when `ai.segmentation_checkpoint` points at downloaded weights
    and torch is importable; see scripts/setup_cubicasa.py for setup steps.
    """

    name = "cubicasa5k"
    accuracy_tier = "high"
    # Flip to True once model loading + mask-to-band conversion lands.
    inference_implemented = False

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint

    @staticmethod
    def is_available(config: AppConfig) -> bool:
        if not CubiCasaGeometryProvider.inference_implemented:
            return False
        checkpoint = config.ai.segmentation_checkpoint
        if not checkpoint or not Path(checkpoint).exists():
            return False
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        return True

    def detect(self, layers: dict[str, Path], debug_dir: Path | None, config: AppConfig) -> WallDetectionResult:
        raise NotImplementedError(
            "CubiCasa5k inference is not wired up yet. Install the 'ai' extra "
            "(pip install -e .[ai]) and run scripts/setup_cubicasa.py to fetch "
            "weights; until then the wallband_cv provider is used."
        )


def build_geometry_provider(config: AppConfig) -> GeometryProvider:
    if CubiCasaGeometryProvider.is_available(config):
        checkpoint = config.ai.segmentation_checkpoint
        assert checkpoint is not None
        return CubiCasaGeometryProvider(Path(checkpoint))
    return WallBandGeometryProvider()
