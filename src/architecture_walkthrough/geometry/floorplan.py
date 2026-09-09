from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from .models import FloorPlanModel


def load_corrected_floorplan(path: Path) -> FloorPlanModel:
    try:
        model = FloorPlanModel.load_json(path)
    except (OSError, ValueError, ValidationError) as exc:
        raise ValueError(f"invalid correction/floorplan JSON: {path}") from exc
    # Loading the authoritative model must not snap walls or drop duplicates:
    # either operation can invalidate opening offsets and room face geometry.
    # Geometry changes belong to the explicit correction/reconstruction flow.
    return model
