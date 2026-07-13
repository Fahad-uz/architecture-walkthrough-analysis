# Architecture Walkthrough Analysis

Turns a 2D floor-plan PNG into an accurate, well-lit 3D building (GLB) with an
interactive first-person browser walkthrough.

```text
image -> ROI extraction -> layer preprocessing -> wall detection (local CV,
      swappable for CubiCasa5k) -> local opening detection (gap + swing-arc +
      leaf + glazing-line evidence) -> tiered scale calibration -> wall-graph
      vectorization (rooms = graph faces) -> evidence-based validation gate
      -> correction editor (React/Konva) -> Blender generation + Cycles
      lightmap baking -> gltf-transform optimization -> R3F walkthrough
```

Design rules:

- Gemini is semantic-only (room labels, dimension text transcription, layout
  sanity-check warnings). Geometry always comes from local image evidence.
- Rooms are faces of the planar wall graph and regenerate on every edit;
  doorway gaps close during face enumeration only when a confirmed opening
  spans them — nothing is auto-invented.
- Openings are intervals along their wall (schema v3) with hinge/swing sides.
- Scale sources are strictly tiered (manual → dimension text → room dims →
  door width → assumed thickness) and never averaged across tiers.
- Validation is a gate: missing evidence lowers confidence, and low-quality
  scenes cannot export a GLB without an explicit override.

```text
src/architecture_walkthrough/
  ui/             FastAPI app factory (serves the built React frontend)
  image_to_glb/   Image analysis plus GLB generation entry points
  walkthrough/    Path planning helpers (browser walkthrough lives in frontend/)
  ai/             Gemini semantic roles, never authoritative geometry
  api/            FastAPI routes + background job runner
  geometry/       Schema, wall graph, rooms, scale tiers, validation gate
  scene/          Blender generator (bake modes), trimesh preview, optimizer
  security/       Upload validation and subprocess safety
  vision/         Preprocessing, wall bands, local opening detection, OCR
frontend/         React app: upload, Konva editor, preview, R3F walkthrough
tools/glb/        gltf-transform CLI for Draco/WebP GLB optimization
```

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

## Web app

Build the frontend once, then start the server:

```powershell
cd frontend; npm install; npm run build; cd ..
.venv\Scripts\uvicorn architecture_walkthrough.ui:create_app --factory --reload
```

Open `http://127.0.0.1:8000`: upload a plan → review/fix walls, openings and
scale in the correction editor (Gemini sanity warnings appear as markers; the
two-point scale tool sets a manual reference) → **Generate 3D** (bake mode
final/draft/none, quality-gate override checkbox) → preview with orbit/zoom
and GLB download → **Enter walkthrough** for first-person WASD exploration
with wall collision and an optional guided tour.

Useful API endpoints:

- `POST /jobs` (multipart upload; analysis runs in the background)
- `GET /jobs/{job_id}` (poll status/quality)
- `GET /jobs/{job_id}/edit-data`
- `POST /jobs/{job_id}/corrections` (regenerates rooms from the wall graph)
- `POST /jobs/{job_id}/validate-corrections`
- `POST /jobs/{job_id}/generate-model?force=&bake_mode=` (Blender build)
- `GET /jobs/{job_id}/artifacts/building.glb` and other artifacts

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
