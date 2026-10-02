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

## Resume status

The previous session stopped at the usage limit before any feature implementation
or new baseline run was saved. This checkpoint and the complete upgrade brief
are now being published first. Starting allowance on resume: 82% remaining.

## Completed this session

- Published upgrade brief and initial checkpoint: `eaf21226d03bb4725ceb3e0b7a4f2612ea04c638`.
- Renamed the UI, browser title, API title, fallback page and README to PlanStride.
  Existing routes, package names and saved projects are unchanged.
  Checks: frontend typecheck/build and API upload tests.

## Current work

1. Establish reproducible automatic baseline, with AI disabled and no manual
   tracing. Report measured results and failures, not just successful examples.
2. Investigate licensed real-plan sources and acquire 100 local examples, with
   at least 30 untouched evaluation examples. Verify family/building separation
   and near duplicates before calling any set a valid final evaluation set.
3. Implement editor undo/redo and recoverable versioned local drafts.
4. Review and publish each tested milestone, then merge normally into main.

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
- No new downloads yet. Record source, version, license, bytes and purpose for each.
- CLI GitHub authentication is unavailable; the connected GitHub tool can publish
  commits as the verified user. Public git fetch works. Never store credentials.

## Next checkpoint contents

Replace this initial status with completed commits, check results, acquired data,
remaining files/tasks and exact next steps before stopping. The commit containing
this file is discoverable with `git log -1 -- CHECKPOINT.md`.
