# PlanStride checkpoint

Updated 12 October 2026 (Asia/Kolkata). Branch consolidation is complete.
The full PlanStride upgrade is NOT complete.

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
Working branch: main. Only main remains locally and on GitHub.
Prepared from commit 4762f72f19da9e723f25c6b202e799c15348530f.
The commit containing this checkpoint can be found with git log -1.
PR #12 merged at1c7106d17c96c4ccc3d6800750f1624b1205996b, preserving six commits:
branding, image preview/crop/reset/history, editor undo/redo and recoverable local
drafts, dataset acquisition, automatic baseline runner and initial brief.
All three GitHub checks passed. Browser crop/undo/draft recovery was verified.
PR #13 merged normally at bd56cdf59717df56966570a14cea4498ddf6054b:
https://github.com/Fahad-uz/architecture-walkthrough-analysis/pull/13
All five individual feature/checkpoint commits are preserved.
Local master history merged at b3f51445da26654ee809d75e95f19c99d6b3997b.
Its original snapshot is identical to upstream 4168810, already in main;
the merge preserved the original import commit without changing project files.
All 17 local/remote branch references were verified as ancestors of main.
At the user's request, 11 remote branches and four local branches were deleted
after checking their history; only main remains. Detailed inventory and checks:
docs/branch-consolidation.md.
Metric commit:31afb2adc0cede3c31546f9c5de558f8589a4958.
Use git log for subsequent feature/checkpoint commit hashes and merge status.
GitHub CLI is now authenticated as Fahad-uz. Local commit identity and GitHub
attribution were verified. Commits and both normal merges are pushed.

## Completed and merged milestones

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
 These record 447 Python passes/9 integration skips, 29 frontend passes,
 lint/type checks and the production build. All three PR #13 checks passed.
- Fresh main checks on 11 October: Python and frontend passed; the GLB audit
 found the newly published Sharp advisory GHSA-wq5f-xc86-pv6w.
 Commit 610602b upgrades both the direct dependency and override to Sharp 0.35.5.
 Local runtime audit reports zero vulnerabilities, SVG/PNG/WebP conversion passes,
 and furnished GLB optimization preserves all active nodes, geometry bounds and
 35,846 nondegenerate triangles. Twelve zero-area triangles are removed by Draco.
 All three GitHub checks passed on 610602b:
 https://github.com/Fahad-uz/architecture-walkthrough-analysis/actions/runs/38137172904

## Data and reproducible outputs

100 CubiCasa5K v1.0 examples:70 development,30 untouched quarantine.
39,876,235 downloaded file bytes; source/license/SHA256 inventory:
outputs/benchmark-data/cubicasa5k/manifest.json (acquisition_complete:true).
Local noncommercial evaluation only, CC BY-NC4.0; no redistribution.
No exact/hash-near duplicates found, but building/source-family independence and
independent annotation review remain UNVERIFIED. Quarantine is not final eval.
FloorPlanCAD downloads deferred because drawing rights are unclear.
Additional installed packages this session: Sharp 0.35.5, its macOS ARM64 native
binding 0.35.5 and libvips bundle 1.3.4. Sources, licenses and installed sizes are
listed in docs/branch-consolidation.md. No new plans, models or asset packs.

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

## API key check, 12 October 2026

Checked main at 4762f72f19da9e723f25c6b202e799c15348530f against the freshly
fetched GitHub main. Gitleaks 8.30.1 scanned 225 tracked files and all 128
reachable commits, including merge-parent diffs. Both scans found zero secrets.
Default scanner rules were used; repository ignore files and inline allow
comments were not allowed to suppress findings. No secret values were printed
or saved in the summary.

The tracked .env.example contains an empty GEMINI_API_KEY setting. No local
.env file was present. This checks repository content, not credentials held in
GitHub settings or elsewhere on the Mac.

Local summary: outputs/secret-check-2026-10-12/summary.json (ignored by Git).
Additional download: Gitleaks 8.30.1 for macOS ARM64, MIT license,
7,897,593 compressed bytes, used for this local secret check:
https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_darwin_arm64.tar.gz
The archive was verified against the release's SHA256 digest. It is stored at
../tools/gitleaks-8.30.1; its source, size and checksum inventory is recorded in
../gitleaks-download-inventory-2026-10-12.json. No runtime dependencies changed.

## Exact next steps

1. Run full70 candidate pipeline comparison in a NEW output directory; do not
   replace/rerun the preserved baseline. Keep all failures and annotation errors.
2. Investigate native crash and false walls on furniture/dimensions. Improve
   based on development evidence, with regressions checked across styles.
3. Complete rotation/straightening/perspective prep and transform tracking;
   furniture/material/lighting editing;3D warning focus and end-to-end checks.
4. Establish independent source/building groups, reviewed annotations and broader
   permitted sources before final untouched evaluation. Openings, adjacency,
   scale error and correction-effort measurements remain outstanding.

Never claim perfect reconstruction from every image. Missing measurements,
elevations or ambiguous geometry require explicit assumptions/review.

## Stop status

The requested branch consolidation and cleanup are complete. All feature,
dependency-fix and merge commits are pushed; all three GitHub checks pass on the
validated code commit 610602b. The accompanying documentation/checkpoint update
is a separate commit, with publication verified after pushing.
Five-hour allowance at the 11 October consolidation checkpoint: 29% remaining;
this session did not stop for quota. Preserve the untracked .DS_Store files in
the repository root and frontend; they are not project changes and were not pushed.
No unfinished code edits remain. The broader unfinished upgrade is listed above.
