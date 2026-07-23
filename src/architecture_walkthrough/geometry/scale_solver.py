from __future__ import annotations

import statistics
from dataclasses import dataclass

from architecture_walkthrough.geometry.models import ScaleConstraintRecord

# Scale sources are tiered. The solver resolves within the single highest-
# priority tier that yields usable constraints; tiers are never averaged
# together, so an assumed wall thickness can never dilute a real dimension.
TIER_ORDER: tuple[str, ...] = (
    "manual",
    "dimension_annotation",  # dimension text associated with detected geometry
    "room_dimension",  # room-size labels matched to room polygons
    "plan_extent",  # last-resort typical residential plan span
    "door_width",  # unconfirmed repeated gaps, used only without plan extent
    "wall_thickness",  # assumed thickness - always low confidence
)

TIER_BASE_CONFIDENCE: dict[str, tuple[float, float]] = {
    # (single-constraint confidence, multi-consistent confidence)
    "manual": (1.0, 1.0),
    "dimension_annotation": (0.65, 0.9),
    "room_dimension": (0.6, 0.85),
    "door_width": (0.3, 0.45),
    "plan_extent": (0.18, 0.25),
    "wall_thickness": (0.15, 0.2),
}


@dataclass(frozen=True)
class ScaleConstraint:
    id: str
    source: str
    measured_px: tuple[float, float]
    expected_m: tuple[float, float]
    label: str | None = None
    weight: float = 1.0
    tier: str = "room_dimension"


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
        source=f"{constraint.tier}:{constraint.source}",
        label=constraint.label,
        measured_px=constraint.measured_px,
        expected_m=constraint.expected_m,
        pixels_per_metre=ppm,
        residual=residual,
        weight=constraint.weight,
        accepted=accepted,
        reason=reason,
    )


def _resolve_tier(
    tier: str,
    constraints: list[ScaleConstraint],
    min_pixels_per_metre: float,
    max_pixels_per_metre: float,
    outlier_mad_factor: float,
    rejected: list[ScaleConstraintRecord],
) -> tuple[float, list[ScaleConstraintRecord], float] | None:
    """Solve within one tier; returns (ppm, used_records, confidence) or None."""
    candidates: list[tuple[ScaleConstraint, float, float, float]] = []
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
        return None

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
        return None

    weighted_sum = sum(ppm * weight for _, ppm, _, weight in accepted)
    weight_sum = sum(weight for _, _, _, weight in accepted)
    final_ppm = weighted_sum / weight_sum
    used = [
        _record(constraint, ppm, abs(ppm - final_ppm) / max(final_ppm, 1e-6) + residual, True)
        for constraint, ppm, residual, _ in accepted
    ]
    single, multi = TIER_BASE_CONFIDENCE.get(tier, (0.3, 0.5))
    spread = statistics.mean(record.residual for record in used)
    base = multi if len(used) >= 2 else single
    confidence = max(0.05, base * max(0.4, 1.0 - min(0.6, spread)))
    return final_ppm, used, confidence


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
            source="manual",
        )

    by_tier: dict[str, list[ScaleConstraint]] = {}
    for constraint in constraints:
        by_tier.setdefault(constraint.tier if constraint.tier in TIER_ORDER else "room_dimension", []).append(constraint)

    rejected: list[ScaleConstraintRecord] = []
    for tier in TIER_ORDER:
        tier_constraints = by_tier.get(tier)
        if not tier_constraints:
            continue
        resolved = _resolve_tier(
            tier, tier_constraints, min_pixels_per_metre, max_pixels_per_metre, outlier_mad_factor, rejected
        )
        if resolved is None:
            continue
        final_ppm, used, confidence = resolved
        # Lower-tier constraints are recorded for the audit trail but play no
        # part in the estimate.
        for lower_tier in TIER_ORDER[TIER_ORDER.index(tier) + 1 :]:
            for constraint in by_tier.get(lower_tier, []):
                ppm, residual, _ = _constraint_ppm(constraint)
                rejected.append(_record(constraint, ppm, residual, False, f"superseded_by_{tier}_tier"))
        return ScaleSolverResult(
            pixels_per_metre=final_ppm,
            constraints_used=used,
            rejected_constraints=rejected,
            confidence=confidence,
            source=tier,
        )

    raise ValueError("scale cannot be established from available constraints")
