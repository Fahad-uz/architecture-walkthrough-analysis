# Colored structural wall recovery progress

## Stage 1: scope and inspection

- Owner: colored-wall subagent; root owns commits.
- Files: `src/architecture_walkthrough/vision/preprocessing.py`, `tests/test_colored_structural_walls.py`, this log.
- Usage at start: 72% five-hour allowance remaining, 96% weekly remaining. Stop new work below 25%.
- Read `CLAUDE.md`; its older black-wall-only input assumption is superseded by the user's current request for broader floor-plan accuracy. Geometry must still derive from local evidence.
- Development set only: 70 examples. Quarantine images/annotations will not be viewed or evaluated.
- Current preprocessing rejects every dark saturated pixel, so colored wall fills disappear. A safe patch must require geometric evidence of thin connected structural bands and reject room/furniture fills; hue- or image-specific rules are prohibited.
- Status: inspecting connected-component geometry and development outcomes before choosing an implementation.

## Stage 2: candidate patch and focused tests

- Added conservative recovery inside `structural_ink_mask`, shared by ROI and preprocessing. The recovery is hue independent and adds only observed pixels from connected saturated-dark components.
- Required evidence: extent >=20% of each image axis and >=40% of at least one; component fill <=25%; >=65% supported by long orthogonal bands, including >=10% support per orientation. Bands require >=3 pixels (scaled with image size). No gap closing is added.
- This intentionally misses isolated/short colored wall fragments. Large furniture drawn exactly like a structural network remains intrinsically ambiguous; this patch is not a semantic classifier.
- Visual development inspection confirmed the saturation failure in several green/red/blue wall styles; room-fill examples provide separate counterexamples. No quarantined data accessed.
- Focused validation: 21 tests passed (`test_colored_structural_walls`, `test_preprocessing`, `test_thin_line_plans`). New coverage includes three colors, two resolutions, retained doorway gaps, same-color furniture beside walls, filled rooms/counters, bold small furniture outlines, isolated bars, large thin grids, neutral-only invariance, and ROI/band integration.
- Status: measuring all development raw masks plus a focused AI-off pipeline subset before recommending the patch.

## Stage 3: measurements and handoff

- Source patch is complete; root owns review and commits. No dependencies, AI calls, models, or configuration knobs were added.
- Validation: 21 focused tests and 58 opening/wall-graph/outline/hybrid tests passed (79 total); focused Ruff check passed.
- All 70 development images were compared at raw structural-mask level. Exactly seven changed; added-pixel agreement with upstream wall polygons ranged from 90.79% to 99.99% exact precision, and 96.60% to 100% within the evaluator's 0.5%-of-diagonal tolerance. This is diagnostic agreement with unreviewed annotations, not certified accuracy. Unchanged raw masks do not alone guarantee unchanged downstream processing, which also denoises and crops.
- Raw measurements: `outputs/colored-wall-experiment/raw-mask-agreement.json`; runnable experiment script: `outputs/colored-wall-experiment/measure_masks.py`.
- Twelve deliberately selected development plans covered three dataset styles, multiple wall colors, and filled-room/furniture controls. Selection manifest: `outputs/colored-wall-experiment/development-subset.json`. These are development cases, not a random held-out evaluation.
- AI-off candidate pipeline: `outputs/colored-wall-pipeline-candidate/report.json`. Baseline subset references unchanged original baseline artifacts via `outputs/benchmark-development-baseline/colored-subset-report.json`.
- Baseline/candidate geometric measurements: `outputs/colored-wall-pipeline-baseline-metrics/report.json` and `outputs/colored-wall-pipeline-candidate-metrics/report.json`.

| Subset metric | Baseline | Candidate |
| --- | ---: | ---: |
| Mean wall polygon F1 | 0.282693 | 0.299473 |
| Mean wall polygon IoU | 0.185959 | 0.197574 |
| Mean tolerant wall polygon F1 | 0.407833 | 0.418058 |
| Mean room instance F1 | 0.117910 | 0.117910 |
| Mean room floor IoU | 0.184119 | 0.186418 |

- All scored per-image wall F1 values stayed equal or increased in this subset. The largest improvements were development `high_quality/11538` (0.159→0.233) and `colorful/619` (0.326→0.429).
- Twelve attempted; 11 analyses completed. `colorful/10706` still exits with native crash -11 both before and after, so colored-mask recovery does not fix its full pipeline failure. The evaluator assigns this failure zero metrics. `high_quality/8488` has the same upstream-geometry GEOS evaluation error before/after and is excluded from means; this leaves 11 scored entries including the analysis failure.
- Limits: isolated/short colored fragments remain excluded; thin colored wall strokes can remain excluded; bright colored walls above the existing dark threshold remain excluded. A large connected furniture network with wall-like geometry can still be ambiguous. This is a conservative preprocessing improvement, not a claim that rooms or complete plans are now generally accurate.
- No quarantine data were accessed, inspected, analyzed, or tuned against. No broad claim or final evaluation was performed.
- Recommended handoff: keep this as one focused commit after root review. Any full 70-image pipeline follow-up must remain development-only and should preserve failures and annotation errors in reporting.
- Final usage check: 21% five-hour allowance remains (79% consumed), 88% weekly remains. Stopped starting work at this check to preserve the root agent's reserve; no further experiments started.
