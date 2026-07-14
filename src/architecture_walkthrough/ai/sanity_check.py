from __future__ import annotations

import json
import logging
import mimetypes
import os
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from architecture_walkthrough.config import AISettings
from architecture_walkthrough.geometry.models import FloorPlanModel

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

    return json.dumps(
        {
            "walls": [
                {"start": norm(w.start.x, w.start.y), "end": norm(w.end.x, w.end.y), "external": w.external}
                for w in model.walls
            ],
            "doors": [norm(d.center.x, d.center.y) for d in model.doors],
            "windows": [norm(w.center.x, w.center.y) for w in model.windows],
            "rooms": [
                {"name": r.name, "center": norm(
                    sum(p.x for p in r.points) / len(r.points),
                    sum(p.y for p in r.points) / len(r.points),
                )}
                for r in model.rooms
            ],
        }
    )


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
            return SanityCheckResult(
                attempted=True,
                succeeded=True,
                warnings=self._request(image_path, model, image_width_px, image_height_px),
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
        prompt = (
            "You are reviewing an automatic floor-plan reconstruction. "
            "The JSON below describes the detected layout in normalized image "
            "coordinates (0-1, origin top-left). Compare it with the attached "
            "original plan image and report discrepancies only: doors or windows "
            "visible in the image but missing from the JSON (missed_door / "
            "missed_window), detected walls with no black wall stroke in the image "
            "(phantom_wall), clearly drawn walls absent from the JSON (missed_wall), "
            "and room labels that contradict the printed text (wrong_room_label). "
            "Place each warning at the discrepancy's location. Do not restate "
            "correct geometry. Return an empty list if the layout matches.\n\n"
            f"Detected layout JSON:\n{_layout_summary(model, image_width_px, image_height_px)}"
        )
        from architecture_walkthrough.ai.retry import call_with_backoff

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = call_with_backoff(
            lambda: client.models.generate_content(
                model=self.settings.gemini_model,
                contents=[prompt, types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type)],
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
