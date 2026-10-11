# Automatic reconstruction baseline

This benchmark runs the original image through `analyze_image`. It never loads
reviewed or corrected models, supplies manual scale, or exports a forced GLB.
All external AI assistance and sanity calls are disabled. Detection counts and
the application's internal quality score are **not accuracy measurements**.

## Measured private regression, 3 October 2026 (Asia/Kolkata)

The user's unmodified source image was processed once, without tracing or edits.
The image, generated model and full report remain in ignored local outputs.

| Diagnostic | Observed value |
| --- | --- |
| Completion | Completed; review required |
| Worker elapsed time, including startup | 1.2587 seconds |
| Analysis elapsed time | 0.9987 seconds |
| Peak worker resident memory | 561,823,744 bytes (535.8 MiB) |
| Walls / room floors | 22 / 8 |
| Doors / windows | 0 / 0; unknown openings, not proof of absence |
| Furniture / special elements / camera waypoints | 4 / 1 / 17 |
| Estimated scale | 73.4732 pixels per metre |
| Internal quality score | 0.389927; not geometric accuracy |
| External AI / sanity calls attempted | No / no |
| Ground-truth accuracy | Not measured |

Warnings include an unresolved wall junction, ambiguous openings, inferred living
floor and stair ceiling boundaries, and unclosed wall gaps. This result does not
establish generalization to other drawings or a finished, accurate walkthrough.
The reviewed furnished reconstruction from the previous repair work is excluded.

The measured code revision was `ed2b21e7ce196453efd11ec13df362658f813a50`
(checkpoint/documentation changes after the repaired main revision). Concurrent
frontend edits did not affect this Python analysis. The report records dirty
working-tree state and hashes every Python source file and the runner used.

- Python source tree SHA-256:
  `d3991cf5194759d55099ab22c357904c3367ea2e9539f22e4f6f48e9026a3597`
- Effective configuration SHA-256:
  `e0ab0a810aec025f1d11827aaaaf950cf4f095b55ba36a655f53a9834887bc79`
- Private source SHA-256:
  `d0c2e8616e0cd62bb39a43a53b5de05331fa8d4c5827f6adc1708595142999b1`
- Local report: `outputs/benchmark-baseline/report.json`.
- Local private manifest: `outputs/benchmark-private-manifest.json`.

This is one run on the development Mac with installed OCR assets; it is not a
cold-download measurement, performance distribution or browser/export benchmark.

## Reproduction and interrupted runs

Use the project's Python environment and a manifest from the acquisition tool:

```sh
.venv/bin/python tools/benchmark/run_baseline.py \
  --manifest outputs/benchmark-data/cubicasa5k/manifest.json \
  --output outputs/benchmark-development-baseline \
  --split development --timeout 120
```

Each sample uses a separate worker with a hard timeout. The report is written
atomically after every result, including failures. Each worker records maximum
resident memory, dependency versions, model hash, issue list, geometry counts,
scale and internal quality state. Memory is unavailable if a worker times out or
crashes before recording its result. Runtime includes worker startup. The runner
currently targets macOS/Linux (`resource` is used for peak memory).

A fresh run refuses an existing output directory. To continue an interrupted
run, repeat the same command with `--resume`. It verifies the manifest, image
checksums, Python source, runner, configuration, selected IDs, runtime platform
and timeout before reusing results. Completed failures remain failures rather
than disappearing from statistics. Interrupted artifacts are preserved and only
unfinished samples rerun. Keep the same installed dependencies when resuming;
the report records their versions per worker but does not pin the environment.

For the private regression, use its local manifest with `--split
private_regression` and a new output directory. No private image is committed.
The original baseline was captured before resume support was added; a report
made with that earlier runner cannot be resumed using the changed runner.

Only `development` and `private_regression` splits are accepted. The collection's
`quarantine` split must remain unused until source/building independence and
annotations have been reviewed. Acquiring 100 examples from one dataset does
not establish diversity or certify the split as a final evaluation set.

## Evaluation protocol fixed before detector tuning

The following protocol is specified before new detector changes. Its metric
implementation, independent annotation review and final evaluation are **not yet
complete**. Do not turn these proposed thresholds into reported achievements.

1. Review ground truth separately from detector outputs. Map walls, rooms,
   openings and stairs into original image coordinates; mark uncertain or
   unobservable regions explicitly. Review metric scale independently where a
   reliable source measurement exists. Do not infer hidden dimensions as truth.
2. Separate buildings and source families before splitting. Keep exact and near
   duplicates, crops and transformed variants together. Reserve at least 30
   untouched examples out of at least 100 real plans; report unknown family
   identity as a split limitation, not certified independence.
3. Score wall centreline length precision/recall/F1 using a 0.15 m positional
   tolerance where independently known scale exists. For scale-free cases use
   0.5% of original image diagonal and report them separately. Also report
   median and 95th-percentile position error and wall-thickness error.
4. Match rooms one-to-one at polygon IoU >= 0.5, reporting room precision/recall,
   matched IoU and unmatched count. Score door/window matches by category with
   centre distance <= 0.20 m and wall interval IoU >= 0.5; use the scale-free
   positional tolerance when necessary, in a separate result group. Report
   room adjacency/connectivity accuracy only when annotated connections exist.
5. Report relative scale error only against independent measurements. Report
   missing or contradictory scale separately. Report unsupported/non-plan input
   rejection and all crashes/timeouts, never just successful reconstructions.
6. Publish per-image results and per-style aggregates, including sample counts,
   median/p95 runtime and peak memory. Measure correction effort separately
   using a fixed review task and recorded active correction time; corrected
   results must never replace automatic scores.
7. Initial improvement targets are wall F1 >= 0.95, matched mean room IoU >= 0.90,
   door/window F1 >= 0.90 and scale p95 error <= 5% on eligible labelled cases.
   All failures and unsupported families remain visible even if these targets
   are met. Report uncertainty intervals; do not claim universal perfection.
8. Validate exported geometry and walkthrough navigation independently of image
   recognition, including wall/door collisions, stairs, floor boundaries,
   furniture clearance, and lighting. This analysis runner does not exercise
   Blender export or browser rendering.

## Development annotation agreement, 3 October 2026

All 70 development analyses finished:67 completed and 3 failed. The 30 quarantined
examples remain untouched. Evaluation completed with 66 scored cases and 4 explicit
annotation/geometry errors. Failed analyses count as empty predictions when
annotations are usable. Unscorable cases are reported, but excluded from the
means below. This is upstream annotation agreement, not independently reviewed
accuracy, a final evaluation or an accuracy guarantee.

| Style | Scored/attempted | Wall footprint IoU | Room-floor IoU | Room instance F 1 |
| --- | --- | --- | --- | --- |
| colorful | 23/24 | 0.201 | 0.186 | 0.118 |
| high_quality | 20/23 | 0.291 | 0.322 | 0.227 |
| high_quality_architectural | 23/23 | 0.256 | 0.456 | 0.267 |
| Overall | 66/70 | 0.247 | 0.321 | 0.203 |

Overall wall footprint F 1 is 0.374; with 0.5% image-diagonal tolerance it is 0.542.
These are polygon-area scores, not the centreline F 1 target above. Rooms use
one-to-one matching at IoU>=0.5. Two SVG cases contain a polygon outside the image
(colorful/9192, high_quality/230); two have a GEOS overlay error (high_quality/8488,
high_quality/1429). Errors are recorded per sample without aborting evaluation,
silently changing labels, or reporting partial metrics. Eight focused tests pass,
including coordinate mapping, duplicate-room penalties and geometry-error recovery.

Visual alignment checked on colorful/10706, high_quality/932 and
high_quality_architectural/926: blue label boundaries align with drawing ink;
red predictions reveal missing colored walls and false walls on furniture,
door swings and dimension lines. This is an alignment spot check, not a complete
independent review of the annotations. Overlays include interior polygon rings.

Reproduce without rerunning recognition:

```sh
.venv/bin/python tools/benchmark/evaluate_structure.py \
  --manifest outputs/benchmark-data/cubicasa5k/manifest.json \
  --baseline outputs/benchmark-development-baseline/report.json \
  --output outputs/benchmark-development-metrics-new
```

Complete local reports: `outputs/benchmark-development-baseline/report.json` and
`outputs/benchmark-development-metrics-v3/report.json`. Source images, SVGs and
overlays remain ignored local data under CubiCasa5K CC BY-NC 4.0. No dependencies
or models were downloaded. Source-family independence, independent annotation
review, openings/scale/adjacency metrics and untouched final evaluation remain
outstanding. Next: measure cautious colored-wall recovery and report regressions.

Worker runtime including startup: median 1.481 s, 95th percentile 2.548 s on the development Mac.
