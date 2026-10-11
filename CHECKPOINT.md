# PlanStride checkpoint

Updated 3 October 2026 (Asia/Kolkata). The full upgrade is NOT complete.

## Resume and publication rules

Read this file and docs/upgrade-brief.md, inspect git status and remote refs,
then resume the next unfinished step. Do not repeat completed work unnecessarily.
Check five-hour usage before major tasks and after commits. Below 7% remaining,
start no new work: save files, publish completed tested changes, update this file
and stop. Never guess unavailable usage. Keep checkpoints throughout the work.
Commit and push each completed logical change separately as Fahad-uz; report its
link. Merge tested milestones into main with normal merges, never squash.
No secrets, private plans or restricted dataset assets in Git.

## Published work

Repository: https://github.com/Fahad-uz/architecture-walkthrough-analysis
Working branch: agent/planstride-accuracy-review.
PR #12 merged at1c7106d17c96c4ccc3d6800750f1624b1205996b, preserving six commits:
branding, image preview/crop/reset/history, editor undo/redo and recoverable local
drafts, dataset acquisition, automatic baseline runner and initial brief.
All three GitHub checks passed. Browser crop/undo/draft recovery was verified.
Current PR: https://github.com/Fahad-uz/architecture-walkthrough-analysis/pull/13
Metric commit:31afb2adc0cede3c31546f9c5de558f8589a4958.
Use git log for subsequent feature/checkpoint commit hashes and merge status.
GitHub CLI has no authentication; connected GitHub tools publish commits.
Public git fetch works. Local identity matches the verified GitHub profile.

## Completed milestones pending final publication/merge verification

- Structural evaluation:70 development analyses,67 completed/3 failed;
 66 scorable,4 explicit evaluation errors. Wall footprint IoU0.247, room-floor
 IoU0.321; these are upstream-label agreement, not independently reviewed accuracy.
 Eight metric tests pass; source alignment checked on one example per style.
- Actionable 2D warnings:focus/highlight, correction guidance, acknowledgements,
 geometry-change invalidation, generation session guard.29 frontend tests,
 typecheck/build passed. Browser acknowledgement survived saving and refreshing,
 quality remained review_required39%. Full details:docs/warning-progress.md.
- Colored-wall recovery:conservative connected-band evidence, no hue/image hacks.
 79 focused tests+lint pass. Seven of70 raw masks changed. Twelve development
 pipeline cases show wall F1 .2827→.2995, no scored wall regression; roomF1 unchanged.
 Native crash colorful/10706 and annotation error8488 remain. Detailed evidence
 and limits:docs/colored-wall-progress.md. No new dependencies or models.
- Full current check logs:../planstride-current-tests.log,
 ../planstride-current-frontend.log, ../planstride-current-build.log.
 Inspect final exit/check results before saying the latest full suite passed.

## Data and reproducible outputs

100 CubiCasa5K v1.0 examples:70 development,30 untouched quarantine.
39,876,235 downloaded file bytes; source/license/SHA256 inventory:
outputs/benchmark-data/cubicasa5k/manifest.json (acquisition_complete:true).
Local noncommercial evaluation only, CC BY-NC4.0; no redistribution.
No exact/hash-near duplicates found, but building/source-family independence and
independent annotation review remain UNVERIFIED. Quarantine is not final eval.
FloorPlanCAD downloads deferred because drawing rights are unclear.

- Baseline:outputs/benchmark-development-baseline/report.json (complete70).
- Metrics:outputs/benchmark-development-metrics-v3/report.json (complete70).
- Candidate subset reports:outputs/colored-wall-pipeline-candidate/report.json
 and outputs/colored-wall-pipeline-candidate-metrics/report.json.
- Private original:outputs/a731b8ed69234c5d84ba219e52fca930/source.jpeg.
- Reviewed reference is never automatic-accuracy evidence.
- Browser test job:6e00fb89eafc435087f4b43c65296840, separate from reference.
- Runtime:http://127.0.0.1:8001; restart if health endpoint is unavailable.
 Use .venv Python, AI disabled; Blender at ../tools/Blender.app/Contents/MacOS/Blender.
 Browser source/recovery screenshot:../../outputs/PlanStride-editor-recovery.png.

## Exact next steps

1. Verify latest focused commits pushed, PR #13 CI and normal merge status.
   Preserve any unfinished working-tree files; inspect status before editing.
2. Run full70 candidate pipeline comparison in a NEW output directory; do not
   replace/rerun the preserved baseline. Keep all failures and annotation errors.
3. Investigate native crash and false walls on furniture/dimensions. Improve
   based on development evidence, with regressions checked across styles.
4. Complete rotation/straightening/perspective prep and transform tracking;
   furniture/material/lighting editing;3D warning focus and end-to-end checks.
5. Establish independent source/building groups, reviewed annotations and broader
   permitted sources before final untouched evaluation. Openings, adjacency,
   scale error and correction-effort measurements remain outstanding.

Never claim perfect reconstruction from every image. Missing measurements,
elevations or ambiguous geometry require explicit assumptions/review.

## Stop status

Allowance reached0%; no new tasks started. Full Python run447passed/9skipped;
color-wall dtype repair rechecked14tests; mypy82files passed. Frontend tests,
typecheck/build passed. Remaining work follows Exact next steps above.
Latest features are being pushed separately; PR13 merge awaits its final CI.
