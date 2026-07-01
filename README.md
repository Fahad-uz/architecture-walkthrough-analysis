# Architecture Walkthrough Analysis

Accuracy-first floor-plan reconstruction for turning architectural plan images into editable analysis artifacts and GLB building models.

The current product is organized around three public modules:

```text
src/architecture_walkthrough/
  ui/             Browser upload and correction experience
  image_to_glb/   Image analysis plus GLB generation entry points
  walkthrough/    Camera/path planning and future walkthrough rendering

  ai/             Gemini semantic hints, never authoritative geometry
  api/            FastAPI routes used by the UI
  geometry/       Floor-plan models, validation, scale, rooms, topology
  scene/          GLB and Blender scene builders
  security/       Upload validation and subprocess safety
  vision/         ROI detection, preprocessing, OCR, walls, overlays
```

The automatic pipeline is deterministic where structure matters:

```text
image -> ROI extraction -> layer preprocessing -> wall-band detection
      -> OCR + optional Gemini semantic hints -> scale solving
      -> vector reconstruction -> validation/overlay -> correction
      -> parametric GLB generation
```

Gemini is semantic-only. It can suggest labels, furniture, doors, and windows, but final wall coordinates and room polygons come from local image evidence and geometric constraints.

## Setup

Use Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

For optional Gemini assistance, set `GEMINI_API_KEY` in your environment. Do not commit `.env` files or keys.

## Quick Start

Analyze a floor plan:

```powershell
architecture-walkthrough --config configs/default.yaml analyze --input assets/input/plan.png --output outputs/plan
```

Generate a GLB from optimized or corrected JSON:

```powershell
architecture-walkthrough --config configs/default.yaml build-model --floorplan outputs/plan --output outputs/plan/building.glb
```

Run the complete image-to-GLB flow:

```powershell
architecture-walkthrough --config configs/default.yaml image-to-glb --input assets/input/plan.png --output outputs/plan/building.glb --work-dir outputs/plan --manual-scale 0.01
```

You can also use the script wrapper:

```powershell
python scripts/image_to_glb.py --input assets/input/plan.png --output outputs/plan/building.glb --work-dir outputs/plan --manual-scale 0.01
```

`--manual-scale` is metres per pixel and overrides automatic scale solving.

## UI

Start the browser UI:

```powershell
uvicorn architecture_walkthrough.ui:create_app --factory --reload
```

Open `http://127.0.0.1:8000`, upload a plan image, inspect the generated artifacts, then open the correction editor. The editor lets you adjust walls, rooms, doors, windows, and scale before regenerating the GLB.

Useful API endpoints:

- `POST /jobs`
- `GET /jobs/{job_id}/edit`
- `GET /jobs/{job_id}/raw-plan`
- `GET /jobs/{job_id}/optimized-plan`
- `GET /jobs/{job_id}/validation-report`
- `GET /jobs/{job_id}/overlay`
- `POST /jobs/{job_id}/corrections`
- `POST /jobs/{job_id}/validate-corrections`
- `POST /jobs/{job_id}/generate-model`

## Outputs

Each analysis job writes:

- `floorplan.raw.json`: raw ROI pixel-space wall detections.
- `floorplan.optimized.json`: authoritative optimized model for GLB generation.
- `floorplan.corrected.json`: optional human-edited model.
- `floorplan.json`: compatibility alias for optimized JSON.
- `validation_report.json`: quality score, scale constraints, and issues.
- `analysis_overlay.svg` and `analysis_overlay.png`: source-aligned reconstruction overlays.
- `building.glb`: generated model when quality gates allow export.

GLB generation prefers `floorplan.corrected.json`, then `floorplan.optimized.json`, then `floorplan.json`.

## Walkthrough Status

The `walkthrough` module already contains path planning, collision checks, camera waypoint helpers, and FFmpeg encoding helpers. Full end-to-end walkthrough rendering is still early and should be treated as the next module to finish after the image-to-GLB and correction loop are stable.

## Repository Layout

```text
assets/
  input/       Local floor-plan images, ignored except .gitkeep
  models/      Optional local model assets, ignored except .gitkeep
  textures/    Optional material textures, ignored except .gitkeep
  hdri/        Optional lighting assets, ignored except .gitkeep
configs/       Runtime configuration
outputs/       Generated jobs and renders, ignored except .gitkeep
scripts/       Thin command-line wrappers
src/           Application package
tests/         Unit and focused integration tests
```

Generated artifacts, caches, virtual environments, and local tool downloads are intentionally ignored so the repository stays readable.

## Testing

```powershell
python -m pytest
python -m ruff check src tests scripts
python -m mypy src
```

Tests do not require Gemini or Blender. Gemini responses should be mocked in tests.

## Notes

Best results come from clean black-and-white floor plans with thick dark walls, clear room labels, dimension text, and minimal skew. Non-Manhattan plans have limited support.
