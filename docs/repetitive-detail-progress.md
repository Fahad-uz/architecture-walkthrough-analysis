# Repeated raster-detail suppression

Development case architectural/5016 exposed porch/bench strokes and dimension
rails being treated as structural walls. Thin slightly sloped strokes acquired
thick bounding boxes, so they exceeded the existing repeated-detail eligibility
threshold. Measuring the median occupied cross-section of each connected
component recovers their thin-ink evidence without changing wall bounding boxes,
centre lines, geometry thickness or fixture-pairing rules. Independently supported
regular runs are now recognized on either side of an interruption; no missing
course is bridged and no wall gap is closed.

## Evidence and checks,15October2026

A preselected15-case development subset covered the failure plus existing stair,
hatching and structural controls. All15 analyses completed;14scored and the same
9192annotation error remained explicit. The sole changed case5016 improved wall
F1 .182186→.241734 and room F1 .052632→.133333. Walls40→30;rooms36→13,against2
upstream annotatedrooms. Wall recall slightly declined .427646→.426730. Sauna
slats and dimension rails remain falsewalls. All14controls retained identical
structural geometry. This is one observed improvement,not broad accuracy.

The new full70run preserves every earlier report. All70 analyses completed;
66scored,4same annotation/geometry errors. No quarantine examples were accessed.
Failed or unscorable cases were not silently omitted from accounting.

| Mean development agreement | Before | Candidate |
| --- | ---: | ---: |
| Wall polygon F1 | .403484 | .404386 |
| Wall polygon IoU | .268075 | .268640 |
| Tolerant wall polygon F1 | .575542 | .576949 |
| Room instance F1 | .216164 | .217387 |
| Room-floor IoU | .335959 | .335959 |

Only5016 changed scoredwallF1;allother scoredwall values were unchanged within
1e-9. There is no scoredwallF1 regression in this run. Metrics measure agreement
with unreviewed upstream labels,not independently verified accuracy or the final
untouched evaluation. Overall structural performance is still poor.

Validation:706Python passes,9optional integration skips,Ruff and mypy83files
passed. Focused regressions cover both orientations,skewedhairlines,thick repeated
partitions,enclosing rails,sparse outline pairs,interrupted sequences,unchanged
component geometry and avoiding false stair classification for broad hatching.

Ignored local reports and overlays:
- outputs/repetitive-detail-subset-2026-10-15/selection.json
- outputs/repetitive-detail-subset-2026-10-15/paired-comparison.json
- outputs/benchmark-repetitive-detail-development-2026-10-15/report.json
- outputs/benchmark-repetitive-detail-metrics-2026-10-15/report.json
- outputs/benchmark-repetitive-detail-metrics-2026-10-15/paired-comparison.json

No dependencies,models or assets were downloaded. Source/style/building
independence,reviewed groundtruth,openings and scale evaluation remain pending.
