# Architecture Walkthrough Analysis

Accuracy-first floor-plan reconstruction for clean architectural PNGs.

The project now uses a deterministic hybrid pipeline:

```text
image -> ROI extraction -> layer preprocessing -> wall-band detection
      -> OCR + optional Gemini semantic hints -> scale solving
      -> vector reconstruction -> validation/overlay -> correction
      -> parametric GLB generation
```

Gemini is semantic-only. It may suggest labels and likely doors/windows, but it is never the source of final wall coordinates or room polygons. Structural geometry comes from local image evidence and geometric constraints.

## Supported Inputs

Best results come from clean black-and-white floor plans with thick dark walls, clear room labels, dimension text, and minimal skew. The ROI detector excludes title blocks, legends, compass marks, room schedules, and large page borders from structural wall detection where possible.

## Outputs

Each analysis job writes:

- `floorplan.raw.json`: raw ROI pixel-space wall-band detections.
- `floorplan.optimized.json`: authoritative optimized model for GLB generation.
- `floorplan.corrected.json`: optional human-edited model.
- `floorplan.json`: compatibility alias for optimized JSON.
- `validation_report.json`: quality score, scale constraints, and issues.
- `analysis_overlay.svg` and `analysis_overlay.png`: source-aligned reconstruction overlays.
- `building.glb`: generated model, when validation quality allows export.

The GLB builder chooses `floorplan.corrected.json`, then `floorplan.optimized.json`, then `floorplan.json`.

## Scale

Automatic scale is solved from multiple constraints: room polygon dimensions, OCR dimensions, and low-weight wall-thickness evidence. Manual scale overrides automatic solving:

```powershell
architecture-walkthrough --config configs/default.yaml analyze --input assets/input/plan.png --output outputs/job --manual-scale 0.01
```

`--manual-scale` is metres per pixel.

## Running

Analyze only:

```powershell
architecture-walkthrough --config configs/default.yaml analyze --input assets/input/third_floor.png --output outputs/third_floor --crop X Y W H
```

Generate a GLB from optimized or corrected JSON:

```powershell
architecture-walkthrough --config configs/default.yaml build-model --floorplan outputs/third_floor --output outputs/third_floor/building.glb
```

One-step image to GLB:

```powershell
python scripts/image_to_glb.py --input assets/input/third_floor.png --output outputs/third_floor/building.glb --work-dir outputs/third_floor --manual-scale 0.01
```

API:

```powershell
uvicorn architecture_walkthrough.api.app:create_app --factory --reload
```

Useful endpoints:

- `POST /jobs`: upload and analyze.
- `GET /jobs/{job_id}/raw-plan`
- `GET /jobs/{job_id}/optimized-plan`
- `GET /jobs/{job_id}/validation-report`
- `GET /jobs/{job_id}/overlay`
- `POST /jobs/{job_id}/corrections`
- `POST /jobs/{job_id}/validate-corrections`
- `POST /jobs/{job_id}/generate-model`

## Correction Workflow

Open `/jobs/{job_id}/edit` after upload. The browser editor shows the plan image under an SVG overlay and lets you:

- drag wall endpoints
- add and delete walls
- mark walls as external/internal
- add room rectangles and edit room names
- add doors and windows projected to the nearest wall
- correct pixels-per-metre scale
- save `floorplan.corrected.json`
- validate the corrected model
- generate `building.glb` from the corrected model

The API also accepts corrected `FloorPlanModel` JSON at `/jobs/{job_id}/corrections`. Saving a correction refreshes `validation_report.json`, regenerates the overlay when the ROI image is available, and makes future GLB generation prefer `floorplan.corrected.json`.

For near-human precision, use this order:

```text
upload -> inspect overlay -> correct 2D geometry -> save -> validate -> generate GLB
```

The automatic result is a starting point, not the final authority.

## Configuration

`configs/default.yaml` includes sections for ROI detection, preprocessing, wall-band extraction, OCR, scale solving, snapping, opening detection, overlay generation, quality thresholds, Gemini semantic assistance, and GLB generation.

Keep API keys in the environment. Never commit `.env` files or Gemini keys.

## Testing

```powershell
python -m pytest
python -m ruff check src tests scripts
python -m mypy src
```

Tests do not require Gemini or Blender. Gemini responses should be mocked in tests.

## Migration Notes

The old pipeline used Canny/Hough lines, assumed the image long side represented a configured number of metres, allowed Gemini wall coordinates to replace local geometry, and added fallback perimeter walls when detection failed.

The new pipeline detects the plan ROI, separates image layers, extracts thick wall regions, derives wall centrelines, solves scale from constraints, optimizes topology, extracts room polygons from wall topology, projects openings onto real walls, writes required audit artifacts, and blocks GLB generation for severe validation failures.

## Limitations

OCR quality depends on local Tesseract availability unless another backend is plugged in. Room extraction works best when walls form closed topology after snapping. The current correction UI is API-first rather than a full browser editor. Non-Manhattan plans have limited support; the supplied clean plan is treated as Manhattan-world.
