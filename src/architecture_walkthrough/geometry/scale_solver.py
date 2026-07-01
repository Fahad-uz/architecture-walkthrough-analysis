from __future__ import annotations

from dataclasses import dataclass
import statistics

from architecture_walkthrough.geometry.models import ScaleConstraintRecord


@dataclass(frozen=True)
class ScaleConstraint:
    id: str
    source: str
    measured_px: tuple[float, float]
    expected_m: tuple[float, float]
    label: str | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class ScaleSolverResult:
    pixels_per_metre: float
    constraints_used: list[ScaleConstraintRecord]
    rejected_constraints: list[ScaleConstraintRecord]
    confidence: float
    source: str


def _constraint_ppm(constraint: ScaleConstraint) -> tuple[float, float, float]:
    measured = sorted([abs(constraint.measured_px[0]), abs(constraint.measured_px[1])], reverse=True)
    expected = sorted([abs(constraint.expected_m[0]), abs(constraint.expected_m[1])], reverse=True)
    ppm_a = measured[0] / expected[0]
    ppm_b = measured[1] / expected[1]
    residual = abs(ppm_a - ppm_b) / max((ppm_a + ppm_b) / 2, 1e-6)
    return (ppm_a + ppm_b) / 2, residual, max(0.05, constraint.weight)


def _record(constraint: ScaleConstraint, ppm: float, residual: float, accepted: bool, reason: str | None = None) -> ScaleConstraintRecord:
    return ScaleConstraintRecord(
        id=constraint.id,
        source=constraint.source,
        label=constraint.label,
        measured_px=constraint.measured_px,
        expected_m=constraint.expected_m,
        pixels_per_metre=ppm,
        residual=residual,
        weight=constraint.weight,
        accepted=accepted,
        reason=reason,
    )


def solve_scale(
    constraints: list[ScaleConstraint],
    manual_pixels_per_metre: float | None = None,
    min_pixels_per_metre: float = 10.0,
    max_pixels_per_metre: float = 1000.0,
    outlier_mad_factor: float = 2.8,
) -> ScaleSolverResult:
    if manual_pixels_per_metre is not None:
        if manual_pixels_per_metre <= 0:
            raise ValueError("manual pixels-per-metre must be positive")
        records = [
            _record(constraint, manual_pixels_per_metre, abs(_constraint_ppm(constraint)[0] - manual_pixels_per_metre), True)
            for constraint in constraints
        ]
        return ScaleSolverResult(
            pixels_per_metre=manual_pixels_per_metre,
            constraints_used=records,
            rejected_constraints=[],
            confidence=1.0,
            source="manual" if not constraints else "mixed",
        )

    candidates: list[tuple[ScaleConstraint, float, float, float]] = []
    rejected: list[ScaleConstraintRecord] = []
    for constraint in constraints:
        ppm, residual, weight = _constraint_ppm(constraint)
        if not min_pixels_per_metre <= ppm <= max_pixels_per_metre:
            rejected.append(_record(constraint, ppm, residual, False, "pixels_per_metre_out_of_range"))
            continue
        if residual > 0.35:
            rejected.append(_record(constraint, ppm, residual, False, "aspect_residual_too_large"))
            continue
        candidates.append((constraint, ppm, residual, weight))

    if not candidates:
        raise ValueError("scale cannot be established from available constraints")

    ppms = [candidate[1] for candidate in candidates]
    median = statistics.median(ppms)
    deviations = [abs(ppm - median) for ppm in ppms]
    mad = statistics.median(deviations) or max(median * 0.03, 1.0)
    accepted: list[tuple[ScaleConstraint, float, float, float]] = []
    for constraint, ppm, residual, weight in candidates:
        if abs(ppm - median) > mad * outlier_mad_factor:
            rejected.append(_record(constraint, ppm, residual, False, "median_absolute_deviation_outlier"))
        else:
            accepted.append((constraint, ppm, residual, weight))

    if not accepted:
        raise ValueError("all scale constraints were rejected as outliers")

    weighted_sum = sum(ppm * weight for _, ppm, _, weight in accepted)
    weight_sum = sum(weight for _, _, _, weight in accepted)
    final_ppm = weighted_sum / weight_sum
    used_records = [
        _record(constraint, ppm, abs(ppm - final_ppm) / max(final_ppm, 1e-6) + residual, True)
        for constraint, ppm, residual, _ in accepted
    ]
    residuals = [record.residual for record in used_records]
    confidence = max(0.05, min(1.0, len(used_records) / 5.0)) * max(0.0, 1.0 - min(0.8, statistics.mean(residuals)))
    return ScaleSolverResult(
        pixels_per_metre=final_ppm,
        constraints_used=used_records,
        rejected_constraints=rejected,
        confidence=confidence,
        source="automatic",
    )
