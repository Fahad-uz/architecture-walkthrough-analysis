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
- Working branch: `agent/planstride-accuracy-review`.
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

1. PR #12 MERGED normally, preserving all six commits, at
   `1c7106d17c96c4ccc3d6800750f1624b1205996b`. All 3 GitHub checks passed.
   Baseline commit: `bbd2b29c5637d45689f55038f5b917fb120eff74`.
   Browser verification: upload/preview/crop undo, fresh automatic analysis,
   local edit autosave/manual draft, refresh recovery, restore and undo all passed.
   Fresh test job: `6e00fb89eafc435087f4b43c65296840`; temporary wall was undone
   and server model unchanged. Recovery screenshot: `../../outputs/PlanStride-editor-recovery.png`.
2. Score development examples against source SVG annotations with validated
   coordinate mapping. Never evaluate/tune on the 30 quarantined examples.
3. Add actionable 2D editor warnings and persistent confirmations which do not
   raise confidence and invalidate when relevant geometry changes.
4. Finish rotation/straightening/perspective preparation, furnishing/material/
   lighting editing, 3D warning focus and broader end-to-end verification.

PR: https://github.com/Fahad-uz/architecture-walkthrough-analysis/pull/12
Working HEAD before next milestone: `1c7106d17c96c4ccc3d6800750f1624b1205996b`.
No user files overwritten. Baseline tooling is committed. Pending work: development
metrics and actionable editor warnings. Inspect `docs/warning-progress.md`,
`docs/baseline.md` and `git status` before resuming any interrupted work.
The 70-example baseline is complete: 67 completed, 3 failed. All30 quarantined
examples remain untouched. Results: `outputs/benchmark-development-baseline/report.json`.
Counts are not accuracy; evaluate against annotations next. No detectorchanges yet.

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

## Usage checkpoint: unfinished work preserved locally

Resumed with99% allowance. Structural evaluator now finishes all70 cases:66
scored and4 errors retained. Eight metric tests passed; three source overlays
visually checked for alignment. Details and limitations: docs/baseline.md.
Main is functional at PR #12 merge `1c7106d17c96c4ccc3d6800750f1624b1205996b`.
All completed foundation commits are pushed and merged. The new branch
`agent/planstride-accuracy-review` starts from that merge.

Do not publish or claim these unfinished features as verified:

- `frontend/src/editorWarnings.ts` and changes to
  `frontend/src/pages/EditorPage.tsx`: actionable 2D warnings, acknowledgement
  fingerprints, generation-session guards. Read `docs/warning-progress.md` for
  exact state. Focused regression tests, typecheck/build and browser checks must
  pass before a feature commit. 3D warning focus is not implemented.
- `tools/benchmark/evaluate_structure.py` and8tests are complete and ready for
  their focused commit. Reports at outputs/benchmark-development-metrics-v3.
  Developing a cautious colored-wall recovery separately; inspect
  docs/colored-wall-progress.md before continuing that experiment.
- Any added tests reported by `git status` belong to these unfinished milestones.

Next resume: check allowance, read the two progress documents, inspect diff,
finish checks, commit each finished feature separately, then open a new PR and
merge normally after CI. Do not rerun the completed 70-plan baseline unless
code/config/input changes justify a new experiment. No new software downloads;
CubiCasa inventory remains exactly the manifest above.

## Checkpoint policy

The commit containing this checkpoint is discoverable with
`git log -1 -- CHECKPOINT.md`. Refresh this file at every completed milestone,
not only at the usage threshold; concurrent work can consume the allowance quickly.
