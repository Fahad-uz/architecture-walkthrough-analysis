from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from .cleanup import cleanup_walls
from .models import FloorPlanModel


def load_corrected_floorplan(path: Path) -> FloorPlanModel:
    try:
        model = FloorPlanModel.load_json(path)
    except (OSError, ValueError, ValidationError) as exc:
        raise ValueError(f"invalid correction/floorplan JSON: {path}") from exc
    return model.model_copy(update={"walls": cleanup_walls(model.walls)})
