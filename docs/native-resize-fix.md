# Native image resize crash and development follow-up

The installed OpenCV 4.14.0.94 macOS ARM runtime crashed during RapidOCR's
bilinear detector resize. This reproduces without OCR or ONNX:

```python
cv2.resize(np.zeros((1026, 1580, 3), np.uint8), (1568, 1024))
```

The native stack enters KleidiCV's NEON resize path. Disabling optimizations or
threads did not avoid it. The verified OpenCV 4.13.0.92 completes that operation,
so both dependency manifests pin this version. A subprocess regression covers
this size and two odd-dimension resizes, checks pixels against independent
bilinear interpolation, limits runtime and disables core dumps.

## Full development comparison, 14 October 2026

Both runs use all 70 development plans with AI assistance disabled. The original
baseline and the pre-fix candidate are preserved. All 30 quarantined plans remain
untouched. Four identical annotation/geometry errors remain excluded from the
means, leaving 66 scored cases; failed analyses count as empty predictions when
annotations are usable. These are agreements with unreviewed upstream labels,
not independent accuracy certification or final evaluation.

| Diagnostic / mean metric | Before runtime fix | After runtime fix |
| --- | ---: | ---: |
| Completed analyses | 67 / 70 | 70 / 70 |
| Wall polygon F1 | 0.380963 | 0.403484 |
| Wall polygon IoU | 0.251971 | 0.268075 |
| Tolerant wall polygon F1 | 0.547857 | 0.575542 |
| Room instance F1 | 0.203121 | 0.216164 |
| Room floor IoU | 0.321791 | 0.335959 |

All three previously crashing plans complete: colorful/10706,
high_quality/10635 and high_quality/13719. These recovered cases account for
the overall improvement. One previously successful plan,
high_quality_architectural/7104, loses 0.003731 wall F1. Other scored wall F1
values are unchanged. This regression is retained in reporting; completion does
not imply accurate reconstruction or acceptable room/opening detection.

Local reports and overlays (ignored, restricted noncommercial evaluation data):
- Before: `outputs/benchmark-development-candidate-metrics-2026-10-14/report.json`
- After analysis: `outputs/benchmark-opencv413-development-2026-10-14/report.json`
- After metrics: `outputs/benchmark-opencv413-development-metrics-2026-10-14/report.json`

Reproduce only the metrics, using a fresh output directory:

```sh
.venv/bin/python tools/benchmark/evaluate_structure.py \
  --manifest outputs/benchmark-data/cubicasa5k/manifest.json \
  --baseline outputs/benchmark-opencv413-development-2026-10-14/report.json \
  --output outputs/benchmark-opencv413-development-metrics-new
```

The analysis report records source hashes, effective configuration, dependency
versions and the saved crop patch present at its start; source stayed static for
the full run. It starts from Git revision c3cc82b; the runtime fix was subsequently
published as 4e343b3. No corrected models contribute to these scores.

Validation before publication: existing full Python suite 447 passed / 9 optional
integration skips, three new native resize tests passed, Ruff and mypy passed,
and all 66 installed packages passed compatibility checking. The saved wheel was
installed with the existing uv tool; nothing additional was downloaded for this
follow-up. Source, license, size and SHA256 are recorded in CHECKPOINT.md and the
ignored `outputs/native-crash-diagnosis/opencv413/inventory.json`.
