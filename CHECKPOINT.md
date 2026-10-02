# PlanStride checkpoint

Updated: 2026-10-03 (Asia/Kolkata). Work is in progress, not complete.

## Resume instructions

Read this file, inspect `git status`, `git log`, and remote refs, then continue
the next unfinished step. Do not repeat completed work without a new reason.
Check the five-hour usage allowance before each major task and after every
commit. If remaining allowance is below 7%, stop starting tasks, save all files,
publish completed tested changes, record unfinished work here, and stop.
Do not guess usage if unavailable; keep saving checkpoints regardless.

Commit each completed logical change separately, with relevant checks first.
Push each commit and report its link. Merge tested milestones into `main` with
normal merge commits, never squash. Author: verified GitHub account Fahad-uz.

## Repository and baseline

- Repository: https://github.com/Fahad-uz/architecture-walkthrough-analysis
- Working branch: `agent/planstride-foundation`.
- Starting main commit: `f5a298f5b618ed7926d52badb46a367127f541c8`.
- Previous repairs are on main; all nine old branch tips are ancestors.
- Previous checks: 416 Python tests, 9 frontend navigation tests, lint, typing,
  build and all three GitHub checks passed. New changes need their own checks.
- Existing reviewed private reconstruction is NOT automatic-accuracy evidence.
- Original private image is preserved locally at
  `outputs/a731b8ed69234c5d84ba219e52fca930/source.jpeg`.
- Prior automatic job: `outputs/a03a9f4e550e4eb7a53bfec4f05525c9`.

## Completed milestones

- `eaf21226d03bb4725ceb3e0b7a4f2612ea04c638`: upgrade brief and initial checkpoint.
- `ed2b21e7ce196453efd11ec13df362658f813a50`: compatible PlanStride branding.
- `f9e330cf54256720e228f4ec7db843e62211ad80`: image preview, source-coordinate crop,
  precise bounds, reset, crop undo/redo and duplicate-submit guard.
- `a17db7eacfcc86263ba9df4d2d252637dcad3393`: editor undo/redo, grouped drags,
  five versioned local drafts, autosave/recovery and stale-response protection.
- `900a1e22b75085bc6e6faaf47efd21e59e58f0a6`: licensed dataset acquisition tool.
- Automatic baseline runner and protocol are now ready for publication. See
  `docs/baseline.md`; private image artifacts stay local, never committed.
- 100 CubiCasa5K plans acquired: 70 development and 30 quarantined, with images
  and SVG annotations. Manifest confirms complete. Total stored download bytes:
  39,876,235. Source/license/checksum inventory is in ignored
  `outputs/benchmark-data/cubicasa5k/manifest.json`; see `docs/benchmark-sources.md`.
- Tests: 425 Python passed, 9 Blender-dependent tests skipped (Blender not in test
  PATH), 21 frontend passed, frontend typecheck/build passed, Ruff passed.
  Full Python log: `../planstride-python-tests.log`.
- Private automatic baseline: 22 walls, 8 floor polygons, 0 identified doors or
  windows; requires review. Elapsed 1.2587s; peak worker memory 561,823,744 bytes.
  These counts and internal quality score are NOT geometric accuracy.

## Current work and next steps

1. Publish runner/tests/baseline document and checkpoint, inspect browser behavior,
   then merge PR #12 normally after checks. Preserve every focused commit.
2. Score development examples against source SVG annotations with validated
   coordinate mapping. Never evaluate/tune on the 30 quarantined examples.
3. Add actionable 2D editor warnings and persistent confirmations which do not
   raise confidence and invalidate when relevant geometry changes.
4. Finish rotation/straightening/perspective preparation, furnishing/material/
   lighting editing, 3D warning focus and broader end-to-end verification.

PR: https://github.com/Fahad-uz/architecture-walkthrough-analysis/pull/12
Working HEAD before this checkpoint: `900a1e22b75085bc6e6faaf47efd21e59e58f0a6`.
No user files overwritten. Only pending tracked work is the baseline runner,
its tests/documentation and this checkpoint. Subsequent agent edits must be
inspected using `git status` before continuing.

## Remaining full upgrade scope

- Improve general reconstruction using development examples and annotations.
- Visual crop, rotate, straighten and perspective correction, preserving source
  and coordinate transforms; image-preparation undo/redo.
- Clickable warnings in source and 3D, confirm/correct actions, stable IDs and
  confirmation invalidation after relevant edits.
- Furniture manipulation, collision/clearance checks, room finishes and lighting.
- End-to-end recovery and stale-result protection across all editing workflows.
- At least 100 diverse real plans; >=30 untouched final evaluation cases;
  independently reviewed ground truth and honest accuracy/performance metrics.
- Full upload-to-walkthrough checks, representative results and download inventory.

## Decisions, downloads and limitations

- No guaranteed perfect geometry from images lacking dimensions or elevations.
- No generated redraw may be treated as structural evidence.
- Restricted datasets stay in ignored local outputs, never distributable assets.
- Downloaded CubiCasa5K v1.0 selected images/annotations and provenance (39,876,235
  bytes) for noncommercial local evaluation, CC BY-NC 4.0. No new software or
  model weights. FloorPlanCAD downloads deferred because drawing rights are unclear.
- CLI GitHub authentication is unavailable; the connected GitHub tool can publish
  commits as the verified user. Public git fetch works. Never store credentials.

## Checkpoint policy

The commit containing this checkpoint is discoverable with
`git log -1 -- CHECKPOINT.md`. Refresh this file at every completed milestone,
not only at the usage threshold; concurrent work can consume the allowance quickly.
