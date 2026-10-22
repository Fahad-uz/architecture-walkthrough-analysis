"""Deterministic lighting math shared with the standalone Blender process.

Keep this module standard-library-only: Blender loads the same source file
directly, without needing the server's Python environment or dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


def _bounded_number(
    values: Mapping[str, Any], key: str, default: float, lower: float, upper: float
) -> float:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"lighting {key} must be a number")
    number = float(value)
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"lighting {key} must be finite and between {lower} and {upper}")
    return number


def kelvin_rgb(temperature: float) -> tuple[float, float, float]:
    """Approximate a black-body white point in RGB for 1000–12000 kelvin."""
    scaled = temperature / 100.0
    if scaled <= 66:
        red = 255.0
        green = 99.4708025861 * math.log(scaled) - 161.1195681661
        blue = 0.0 if scaled <= 19 else 138.5177312231 * math.log(scaled - 10) - 305.0447927307
    else:
        red = 329.698727446 * (scaled - 60) ** -0.1332047592
        green = 288.1221695283 * (scaled - 60) ** -0.0755148492
        blue = 255.0
    rgb = tuple(max(0.0, min(255.0, value)) / 255.0 for value in (red, green, blue))
    return rgb[0], rgb[1], rgb[2]


def effective_lighting(values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve saved controls once, preserving the exact historical neutral look.

    UI tones map to 3000 K (warm), 4500 K (neutral) and 6500 K (cool).
    ``color_scale`` tints previously white sun/ambient/environment light.
    ``light_color`` calibrates the historical warm room lamps independently.
    Exposure is returned separately so baked power is never multiplied twice.
    """
    values = values or {}
    brightness = _bounded_number(values, "brightness", 1.0, 0.25, 2.0)
    intensity = _bounded_number(values, "intensity", 450.0, 0.0, 4500.0)
    exposure = _bounded_number(values, "exposure", 0.0, -5.0, 5.0)
    temperature = _bounded_number(values, "color_temperature", 4500.0, 1000.0, 12000.0)
    if not temperature.is_integer():
        raise ValueError("lighting color_temperature must be an integer")
    neutral = kelvin_rgb(4500.0)
    ratios = tuple(value / reference for value, reference in zip(kelvin_rgb(temperature), neutral))
    peak = max(ratios)
    return {
        "power_factor": brightness * (intensity / 450.0),
        "color_temperature": int(temperature),
        "color_scale": tuple(value / peak for value in ratios),
        "light_color": tuple(
            max(0.0, min(1.0, base * ratio))
            for base, ratio in zip((1.0, 0.90, 0.78), ratios)
        ),
        "exposure": exposure,
        "exposure_factor": 2.0 ** exposure,
    }
