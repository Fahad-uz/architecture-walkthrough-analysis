# Architecture Walkthrough Analysis

Convert a 2D architectural floor plan into an evidence-grounded, editable 3D
building and explore it in a browser. The project combines deterministic
computer vision, a human correction step, procedural Blender generation, GLB
optimization, and a first-person React walkthrough.

```text
floor-plan image
  -> plan-region extraction and layered preprocessing
  -> OCR-backed scale calibration
  -> wall, opening, room, and furniture reconstruction
  -> evidence and topology validation
  -> visual correction editor
  -> Blender scene generation and optional light baking
  -> GLB optimization
  -> orbit preview and collision-aware walkthrough
```

The main design rule is simple: local image evidence owns the geometry.
Optional Gemini assistance can interpret room labels, dimensions, furniture,
and suspicious regions, but it cannot silently invent walls or openings.
Low-confidence results stay reviewable and are blocked from final export unless
the user explicitly overrides the quality gate.

## What is included

- Automatic plan-region detection, deskewing, and text-aware wall filtering.
- Tiered scale solving from manual measurements, dimension annotations, room
  dimensions, door widths, or a documented fallback.
- Wall-graph cleanup, face-based room recovery, local door/window detection,
  label assignment, and furniture placement constrained to room polygons.
- A source-aligned SVG/PNG evidence overlay and JSON validation report.
- A React/Konva editor for correcting walls, openings, scale, and room data.
- Procedural floors, walls, ceilings, doors, windows, balconies, furniture,
  lighting, and cameras in both preview and Blender-generated scenes.
- Atomic model/correction updates that preserve the last good artifacts when a
  generation step fails.
- Responsive desktop/mobile walkthrough controls, collision handling, guided
  waypoints, model download, loading states, and error recovery.
- A production Docker image containing the frontend, Python service, Blender,
  FFmpeg, Node 22, and the native GLB optimization toolchain.

## Repository layout

```text
src/architecture_walkthrough/
  ai/             Optional semantic interpretation and sanity checks
  api/            FastAPI routes, durable job state, and artifact transactions
  geometry/       Models, scale, wall graph, rooms, furniture, and validation
  image_to_glb/   Public image-analysis and model-generation entry points
  scene/          Preview GLB, Blender scene, materials, baking, and validation
  security/       Upload, path, and subprocess safety
  ui/             FastAPI application factory for the built frontend
  vision/         ROI, preprocessing, OCR, walls, openings, and evidence filters
  walkthrough/    Route planning, camera animation, collision, and video helpers
frontend/         React, Three.js/R3F, and Konva browser application
tools/glb/        glTF Transform, Draco, texture, and WebP optimization
configs/          Runtime configuration
scripts/          Thin command-line wrappers
tests/            Unit, regression, and focused Blender integration tests
```

Generated jobs live under `outputs/`; local input plans and optional models,
textures, and HDRIs live under `assets/`. Those large or private artifacts are
ignored by Git while their directory placeholders remain tracked.

The small bundled CC0 surface library is an exception: plaster, wood, and tile
maps are included so new checkouts render textured surfaces without a download
or API key. See [asset sources and checksums](assets/THIRD_PARTY.md).

## Local setup

Requirements:

- Python 3.11+
- Node.js 22+ for the web frontend and GLB optimizer
- Blender for production-quality scene generation
- FFmpeg only for rendered walkthrough videos

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"

cd frontend
npm ci
npm run build
cd ..

cd tools\glb
npm ci
cd ..\..
```

Copy `.env.example` to `.env` if you need local overrides. Important variables
include `ARCH_WALK_CONFIG`, `ARCH_WALK_BLENDER_PATH`,
`ARCH_WALK_FFMPEG_PATH`, `ARCH_WALK_GEMINI_ENABLED`, and
`ARCH_WALK_GEMINI_MODEL`. Set `GEMINI_API_KEY` only if optional Gemini analysis
is enabled. Never commit `.env` files or credentials.

On macOS or Linux, use Python 3.12 and the Unix virtual-environment paths:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
npm --prefix frontend ci
npm --prefix frontend run build
npm --prefix tools/glb ci
.venv/bin/python scripts/download_materials.py --check
```

On macOS, set `ARCH_WALK_BLENDER_PATH` to the actual executable inside the app,
for example `/Applications/Blender.app/Contents/MacOS/Blender`. Blender needs its
bundled resources, so a shell wrapper that executes this path is preferable to
a symlink when adding it to `PATH`.

Start locally with `.venv/bin/uvicorn architecture_walkthrough.ui:create_app
--factory --host 127.0.0.1 --port 8001`. The Windows command below remains valid.

## Run the web application

```powershell
.\.venv\Scripts\uvicorn architecture_walkthrough.ui:create_app --factory --host 0.0.0.0 --port 8001
```

Open `http://127.0.0.1:8001`. The workflow is:

1. Upload a PNG or JPEG floor plan and optionally provide a manual scale.
2. Inspect the source-aligned reconstruction overlay and quality findings.
3. Correct walls, openings, rooms, or scale in the editor.
4. Generate the 3D model using `none`, `draft`, or `final` bake quality.
5. Orbit around the model, download the GLB, or enter the walkthrough.

Useful endpoints include:

- `POST /jobs` — stream and validate a new plan upload.
- `GET /jobs/{job_id}` — poll analysis or generation status.
- `GET /jobs/{job_id}/edit-data` — load the current editable model.
- `POST /jobs/{job_id}/corrections` — validate and atomically save an edit.
- `POST /jobs/{job_id}/validate-corrections` — rerun the quality gate.
- `POST /jobs/{job_id}/generate-model?force=&bake_mode=` — build a Blender GLB.
- `GET /jobs/{job_id}/artifacts/{artifact_name}` — download a safe artifact.

## Command line

Analyze a plan:

```powershell
architecture-walkthrough --config configs/default.yaml analyze `
  --input assets/input/plan.png `
  --output outputs/plan
```

Generate a GLB from optimized or corrected JSON:

```powershell
architecture-walkthrough --config configs/default.yaml build-model `
  --floorplan outputs/plan `
  --output outputs/plan/building.glb `
  --bake-mode draft
```

Run image analysis and model generation together:

```powershell
architecture-walkthrough --config configs/default.yaml image-to-glb `
  --input assets/input/plan.png `
  --output outputs/plan/building.glb `
  --work-dir outputs/plan `
  --manual-scale 0.01
```

`--manual-scale` is metres per pixel and has the highest calibration priority.
Use `--force` only after visually reviewing a reconstruction that did not pass
the normal quality gate.

## Output artifacts

Each job can contain:

- `floorplan.raw.json` — initial pixel-space detections.
- `floorplan.optimized.json` — reconstructed metric model.
- `floorplan.corrected.json` — optional human-reviewed replacement.
- `floorplan.json` — compatibility copy of the optimized model.
- `validation_report.json` — quality score, scale evidence, and issues.
- `analysis_overlay.svg` / `.png` — source-aligned visual reconstruction.
- `building.glb` — latest successful preview or Blender model.

Generation prefers corrected JSON, then optimized JSON, then the compatibility
copy. Related artifacts are replaced as a recoverable set, so a failed save or
Blender run does not overwrite the last good model.

## Docker

Build and run the complete production image:

```powershell
docker build -t architecture-walkthrough .
docker run --rm -p 8001:8000 -v "${PWD}\outputs:/app/outputs" architecture-walkthrough
```

The container health check calls `/health`. Mount a persistent output directory
if jobs should survive container replacement.

## Verification

The same core checks run in GitHub Actions:

```powershell
.\.venv\Scripts\python -m pip check
.\.venv\Scripts\python -m ruff check src tests scripts
.\.venv\Scripts\python -m mypy src
.\.venv\Scripts\python -m pytest

cd frontend
npm ci
npm audit --audit-level=high
npm run typecheck
npm test
npm run build

cd ..\tools\glb
npm ci
npm audit --audit-level=high
```

The test suite does not require a Gemini key. Focused Blender integration tests
skip automatically when Blender is unavailable.

## Input guidance

Clean, high-resolution plans with dark wall lines, readable dimensions, and
limited skew give the strongest results. Curved, heavily stylized, multi-level,
or severely occluded drawings may still need manual correction; the overlay and
quality report are intended to make that uncertainty visible rather than hide
it.
