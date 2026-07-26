from __future__ import annotations

import json
import logging
import mimetypes
import os
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon

from architecture_walkthrough.config import AISettings
from architecture_walkthrough.geometry.models import FloorPlanModel, RoomPolygon

LOGGER = logging.getLogger(__name__)

# Sanity checking is Gemini's third semantic role: compare the vectorized
# layout against the original image and report discrepancies as warnings for
# the correction editor. Warnings never modify geometry automatically.

WARNING_KINDS = ("missed_door", "missed_window", "phantom_wall", "missed_wall", "wrong_room_label")


class SanityWarning(BaseModel):
    kind: str
    description: str
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SanityCheckResult(BaseModel):
    attempted: bool = False
    succeeded: bool = False
    warnings: list[SanityWarning] = Field(default_factory=list)
    error: str | None = None


def _warning_schema() -> dict:
    return {
        "type": "object",
        "required": ["warnings"],
        "properties": {
            "warnings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["kind", "description", "x", "y", "confidence"],
                    "properties": {
                        "kind": {"type": "string", "enum": list(WARNING_KINDS)},
                        "description": {"type": "string"},
                        "x": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "y": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                },
            },
        },
    }


def _layout_summary(model: FloorPlanModel, image_width_px: int, image_height_px: int) -> str:
    ppm = model.pixels_per_metre or 1.0

    def norm(x_m: float, y_m: float) -> dict[str, float]:
        return {
            "x": round(x_m * ppm / max(image_width_px, 1), 3),
            "y": round(1.0 - y_m * ppm / max(image_height_px, 1), 3),
        }

    def area_summary(area: RoomPolygon) -> dict[str, object]:
        return {
            "id": area.id,
            "name": area.name,
            "center": norm(
                sum(point.x for point in area.points) / len(area.points),
                sum(point.y for point in area.points) / len(area.points),
            ),
            "polygon": [norm(point.x, point.y) for point in area.points],
        }

    return json.dumps(
        {
            "walls": [
                {"start": norm(w.start.x, w.start.y), "end": norm(w.end.x, w.end.y), "external": w.external}
                for w in model.walls
            ],
            "doors": [norm(d.center.x, d.center.y) for d in model.doors],
            "windows": [norm(w.center.x, w.center.y) for w in model.windows],
            "rooms": [area_summary(room) for room in model.rooms],
            "balconies": [area_summary(balcony) for balcony in model.balconies],
        }
    )


def _sanity_prompt(model: FloorPlanModel, image_width_px: int, image_height_px: int) -> str:
    return (
        "You are reviewing an automatic floor-plan reconstruction. "
        "The JSON below describes the detected layout in normalized image "
        "coordinates (0-1, origin top-left). Compare it with the attached "
        "original plan image and report discrepancies only: doors or windows "
        "visible in the image but missing from the JSON (missed_door / "
        "missed_window), detected walls with no black wall stroke in the image "
        "(phantom_wall), and clearly drawn walls absent from the JSON (missed_wall). "
        "The JSON intentionally separates enclosed rooms in rooms[] from outdoor "
        "or semi-outdoor areas in balconies[]; both entries include their full polygon. "
        "Use wrong_room_label only when printed room text located inside a rooms[].polygon "
        "contradicts that room's non-empty name. A BALCONY or TERRACE label inside a "
        "balconies[].polygon is correct balcony semantics, not a wrong_room_label. "
        "Do not report a missing label or a balcony merely because it is absent from rooms[]. "
        "Place each warning at the discrepancy's location. Do not restate "
        "correct geometry. Return an empty list if the layout matches.\n\n"
        f"Detected layout JSON:\n{_layout_summary(model, image_width_px, image_height_px)}"
    )


def _discard_balcony_room_label_warnings(
    warnings: list[SanityWarning],
    model: FloorPlanModel,
    image_width_px: int,
    image_height_px: int,
) -> list[SanityWarning]:
    """Remove room-label warnings whose reported location is a known balcony."""

    ppm = model.pixels_per_metre or 1.0
    width = max(image_width_px, 1)
    height = max(image_height_px, 1)
    balcony_polygons = [
        Polygon(
            [
                (point.x * ppm / width, 1.0 - point.y * ppm / height)
                for point in balcony.points
            ]
        )
        for balcony in model.balconies
    ]
    return [
        warning
        for warning in warnings
        if not (
            warning.kind == "wrong_room_label"
            and any(
                polygon.is_valid
                and not polygon.is_empty
                and polygon.covers(ShapelyPoint(warning.x, warning.y))
                for polygon in balcony_polygons
            )
        )
    ]


class GeminiLayoutSanityChecker:
    def __init__(self, settings: AISettings) -> None:
        self.settings = settings

    def check(
        self,
        image_path: Path,
        model: FloorPlanModel,
        image_width_px: int,
        image_height_px: int,
    ) -> SanityCheckResult:
        if not self.settings.gemini_enabled or not self.settings.gemini_sanity_check_enabled:
            return SanityCheckResult(error="sanity check disabled in configuration")
        if not os.getenv("GEMINI_API_KEY"):
            return SanityCheckResult(attempted=True, error="GEMINI_API_KEY is not set")
        try:
            warnings = self._request(image_path, model, image_width_px, image_height_px)
            return SanityCheckResult(
                attempted=True,
                succeeded=True,
                warnings=_discard_balcony_room_label_warnings(
                    warnings,
                    model,
                    image_width_px,
                    image_height_px,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - warnings are best-effort
            LOGGER.warning("Gemini sanity check failed: %s", exc)
            return SanityCheckResult(attempted=True, error=str(exc))

    def _request(
        self,
        image_path: Path,
        model: FloorPlanModel,
        image_width_px: int,
        image_height_px: int,
    ) -> list[SanityWarning]:
        from google import genai
        from google.genai import types

        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        prompt = _sanity_prompt(model, image_width_px, image_height_px)
        from architecture_walkthrough.ai.retry import call_with_backoff

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        contents = types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type),
            ],
        )
        response = call_with_backoff(
            lambda: client.models.generate_content(
                model=self.settings.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_warning_schema(),
                ),
            )
        )
        content = getattr(response, "text", None)
        if not content:
            raise ValueError("Gemini sanity check returned no text")
        try:
            payload = json.loads(content)
            return [SanityWarning.model_validate(item) for item in payload.get("warnings", [])]
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"Gemini sanity check returned invalid payload: {exc}") from exc
