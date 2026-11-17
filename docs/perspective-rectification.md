# Perspective rectification foundation

The validated rectification core is implemented in
`vision/perspective_rectification.py`. It is not yet exposed by upload, CLI or
pipeline controls. Existing image preparation and legacy perspective helpers
remain unchanged. A complete upload/calibration workflow is still required.

Four photographed corners do not establish a rectangle's true aspect ratio.
The core requires the known width-to-height ratio of a real rectangle, such as
a sheet or reference. It never derives the true ratio from photographed edge
lengths. A fresh distance reference in the rectified frame supplies plan scale;
old source-image metres-per-pixel values cannot be reused after a projective warp.
User-provided ratios and measurements still depend on the user's knowledge.

## Core contract

`parse_perspective_spec` accepts exactly:

```json
{
  "corners": [[120, 80], [980, 105], [1060, 760], [70, 730]],
  "aspect_ratio": 1.4142135623730951,
  "output_width_px": 1200
}
```

Corners are TL/TR/BR/BL, clockwise in image coordinates, in the frame after
quarter-turns and straightening and before crop. Shapes, numeric types, bounds,
convexity, winding, degeneracy, normalized conditioning and projective poles
are validated. Target height rounds to the nearest integer, halves up. Actual
output dimensions are recorded. Defaults are 32 px minimum side, 6000 px maximum
width/height and 36 million pixels; callers can supply stricter limits.

`perspective_frame` composes source-to-oriented and oriented-to-rectified
homographies in double precision. `warp_perspective` samples the original raster
once using explicit pixel-edge/pixel-centre conversion. Inputs are preserved.
Metadata is detached JSON and matrices are read-only. Rectification metadata
alone does not claim a supported pipeline provenance version.

`validate_prepared_scale_reference` accepts exactly two rectified-frame points
and their known distance in metres. It returns measured pixels and pixels per
metre. Callers must invalidate the reference whenever preparation changes and
establish that the measurement belongs to the current image.

## Verification, 17 November 2026

119 new cases and existing preparation coverage passed: 220 tests. Full Ruff and
mypy across 88 source files passed. Independent probes verified interior
projective landmarks, safe poles outside the retained domain, image type/layout
handling and input preservation. Legacy helpers are byte-for-byte unchanged.

A controlled distortion of the existing private plan used an independently
constructed projective transform and known digital-page ratio. Rectification
returned the original 1072 × 1264 frame: normalized matrix round-trip error
2.6e-13, mean grayscale difference 1.60/255 after two interpolations. Edge
alignment F1 was 0.99983 at a preselected 2 px tolerance. This measures a known
synthetic image transform, not automatic reconstruction or real-photo accuracy.
Private local evidence: outputs/perspective-core-qa-2026-11-17/verification.json.
No new package, model, asset or dataset was downloaded.

## Remaining integration

- Strict upload validation, worker argument propagation, original-byte retention
  and rejection of inherited source-space manual scale.
- Four-corner preview, known-ratio input and fresh rectified-frame calibration.
- Whole-state preparation undo/redo and invalidation of old crops/references.
- End-to-end save, regeneration and walkthrough verification using permitted
  photographed plans, with structural accuracy measured separately.


## Projective provenance and evaluator

Commit 179ba1e adds `rectified_preparation_metadata(frame, roi, aw, ah)` and a
separate evaluator path for version3. The record includes source/rectified/
analysis dimensions, orientation, the validated rectangle facts, actual padded
ROI, exact per-axis resize composition and forward/inverse matrices. The ROI
must be nonempty and inside the rectified frame.

The evaluator reconstructs the canonical frame and rejects inconsistent facts,
ROIs, dimensions or matrices. It clips complete buffered wall footprints and
room polygons to the observed analysis raster before inverse projection, then
maps every vertex with homogeneous division and clips to source bounds. Holes
and multipart geometry are preserved. This avoids stretching wall thickness by
one misleading global projective scale. Legacy/v1/v2 evaluation is unchanged.

310 focused preparation/metric tests passed. The full Python suite passed:
992 tests, 9 optional skips, 2 existing dependency deprecations. Ruff and mypy
across 88 source files passed. Independent geometry and malformed-record probes
passed. Log: ../planstride-projective-tests-2026-11-17.log.

The application still does not emit v3 through upload or the pipeline. That
integration must require a known rectangle ratio and a fresh measurement,
propagate strict arguments through workers, preserve original bytes and reset
stale preparation references. For analysis scale, map the fresh measurement's
two endpoints through the actual per-axis crop/resize transform and recompute
pixels per metre. A nominal scalar resize multiplier can miss integer rounding.
