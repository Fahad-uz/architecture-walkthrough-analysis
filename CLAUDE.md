# Architecture Walkthrough Analysis

Converts a 2D floor-plan PNG into an accurate, well-lit 3D model (GLB) with an
interactive first-person browser walkthrough. Geometry always comes from local
image evidence or the user's corrections — never from vision-LLM coordinates.

## Target pipeline

```
PNG → preprocessing → recognition (segmentation model + Gemini semantics)
    → vectorization & regularization (wall graph) → scale calibration
    → intermediate JSON scene → correction editor (React/Konva)
    → Blender 3D generation + Cycles lightmap baking → GLB optimization
    → Three.js (R3F) first-person walkthrough
```

Design rules that must not be violated:

- **Gemini is semantic-only.** It labels rooms, reads dimension text, classifies
  ambiguous opening candidates, and sanity-checks layouts. It must never be the
  source of wall/room/opening coordinates.
- **Rooms are faces of the wall graph**, not independently stored polygons.
  Editing a wall regenerates rooms; semantic data stays attached to persistent
  face IDs.
- **No blind gap-closing** beyond a small tolerance. Doorway gaps close during
  face enumeration only when they are confirmed openings.
- **Openings are intervals along a wall** (`wall_id`, start/end offsets, kind,
  hinge/swing, confidence, evidence) detected locally (gap evidence, swing
  arcs, leaf lines, parallel window lines) — not center points projected from
  AI hints.
- **Scale sources are tiered, never averaged across tiers**: manual two-point
  reference → plan dimension lines → consistent room dimensions → standard door
  width → assumed wall thickness (always flagged low-confidence).
- **Missing evidence lowers validation confidence** — zero detected doors in a
  real plan must read as "unknown/low", never as a perfect score. Low-quality
  scenes need an explicit user override to export GLB.
- Everything downstream of scale calibration is in **real-world metres**.
- Input assumption: clean CAD-style plans — bold solid black walls (mostly
  Manhattan), thin colored (e.g. orange) furniture linework, fixed line weight.
  Do not build robustness machinery for hand-drawn/skewed scans.

## Repository layout

```
src/architecture_walkthrough/
  pipeline.py     Orchestrates analyze_image / build_model / convert_image_to_glb
  config.py       Pydantic AppConfig loaded from configs/default.yaml + .env
  main.py         CLI (analyze | build-model | image-to-glb | walkthrough | run-all)
  vision/         ROI, preprocessing layers, wall bands, opening detection, OCR
  geometry/       FloorPlanModel schema, wall graph, rooms, scale, validation
  ai/             Gemini semantic services (hints only, mocked in tests)
  scene/          GLB/Blender builders (walls with opening cutouts, floors, …)
  api/            FastAPI app + job endpoints, serves the editor
  security/       Upload validation, path traversal guards, subprocess sandbox
  walkthrough/    Path planning / camera helpers (legacy MP4 path being replaced
                  by the browser walkthrough)
tests/            pytest; Gemini and Blender are always mocked/optional
configs/default.yaml   All tunables (thresholds, defaults, limits)
assets/           Registries + local inputs (binaries gitignored)
outputs/          Job artifacts (gitignored)
```

## Intermediate JSON (`FloorPlanModel`, geometry/models.py)

The single contract between recognition/editor and the Blender side. Current
`schema_version: "2.0"`; v3 (in progress) adds interval openings and persistent
room face IDs. Key fields:

- `walls[]`: `id`, `start`/`end` (centerline, metres), `thickness_m`,
  `height_m`, `external`, `confidence`, `evidence_source`
- `doors[]` / `windows[]`: `wall_id`, `offset_m` (v3: `start_offset_m`/
  `end_offset_m`, `hinge_side`, `swing_side`), `width_m`, `height_m`,
  `sill_height_m` (windows), `confidence`, `evidence_source`
- `rooms[]`: polygon `points`, `name`, `confidence` (v3: persistent `face_id`)
- `pixels_per_metre`, `scale_constraints[]` (audit trail of the scale solve)
- `validation_issues[]`, `reconstruction` (quality score/state, stage log)
- Coordinate system: metres, origin bottom-left, y-up (image y is flipped
  during pixel→metre conversion)

Artifacts per job: `floorplan.raw.json` (pixel-space detections),
`floorplan.optimized.json` (authoritative), `floorplan.corrected.json`
(human-edited, wins if present), `validation_report.json`,
`analysis_overlay.svg/png`, `building.glb`.

## Running things

Always use the project venv: `.venv\Scripts\python` (Windows).

```powershell
# Full image → GLB
.venv\Scripts\python -m architecture_walkthrough.main --config configs/default.yaml `
  image-to-glb --input assets/input/plan.png --output outputs/plan/building.glb `
  --work-dir outputs/plan --manual-scale 0.01   # metres per pixel, optional

# Analysis only
.venv\Scripts\python -m architecture_walkthrough.main analyze --input plan.png --output outputs/plan

# Web UI (upload + correction editor)
.venv\Scripts\uvicorn architecture_walkthrough.ui:create_app --factory --reload

# Tests / lint / types
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check src tests scripts
.venv\Scripts\python -m mypy src
```

## Environment

- Windows 11, Python 3.11 venv at `.venv/`
- Blender 5.1 on PATH (`blender`) — used headless for 3D generation/baking
- `GEMINI_API_KEY` from OS env or `.env` (gitignored; never commit keys)
- Node.js needed for the React frontend and gltf-transform GLB optimization

## Conventions

- Pydantic for all Python data models; TypeScript on the frontend
- Conventional-commit-style messages, one commit per logical unit
- Tests mock Gemini and never require Blender; geometry math must be unit-tested
- Furniture is a Phase-2 additive layer: everything must work with an empty
  `furniture` array, and no code may auto-inject furniture into a user-edited model
