# PlanStride checkpoint

Updated 14 October 2026 (Asia/Kolkata). Resumed at the user's request.
Branch consolidation is complete.
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
Prepared from commit 9bd67d52fa651a8a2089acf95efeb9815e7adf71.
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

## Saved progress, 14 October 2026

- Full70 candidate comparison completed before any Python edits:67 analyses
  completed,3 native crashes;66 scored and4 annotation/geometry errors.
  Wall polygon F1 .374279→.380963, IoU .247304→.251971; six scored plans improve,
  no scored wall F1 regression. Room instance F1 unchanged .203121. This is
  development agreement with unreviewed labels, not final verified accuracy.
  Full details:docs/colored-wall-progress.md. Existing baseline untouched.
- New immutable local reports:
  outputs/benchmark-development-candidate-2026-10-14/report.json and
  outputs/benchmark-development-candidate-metrics-2026-10-14/report.json.
- Native crash isolated to OpenCV4.14.0 macOS ARM KleidiCV resize, in RapidOCR
  detector preprocessing. Minimal reproduction without OCR/ONNX:
  cv2.resize(np.zeros((1026,1580,3),np.uint8),(1568,1024)).
  Stack:kleidicv::neon::kleidicv_resize_generic_stripe_u8<2,3>.
  Disabling optimizations/threads did not help; INTER_LINEAR_EXACT did.
- Isolated OpenCV4.13.0.92 passes the minimal resize and all three previously
  crashing plans:colorful/10706,high_quality/10635,high_quality/13719.
  These were bounded diagnostic probes, not a full candidate evaluation.
  On resume, shared .venv and both dependency manifests were updated to the
  verified OpenCV4.13.0.92. Fix pushed at 4e343b31be2ec990a7bb62654670ffaaeae55a77.
- Crash/probe logs, results and extracted isolated wheel:
  outputs/native-crash-diagnosis/ (ignored).
- Additional download:opencv-python4.13.0.92 macOS13 ARM64 wheel, Apache2.0,
  46,247,052 bytes. Purpose:isolated verification of native resize-crash fix.
  Source:https://files.pythonhosted.org/packages/fc/6f/5a28fef4c4a382be06afe3938c64cc168223016fa520c5abaf37e8862aa5/opencv_python-4.13.0.92-cp37-abi3-macosx_13_0_arm64.whl
  SHA256:caf60c071ec391ba51ed00a4a920f996d0b64e3e46068aac1f646b5de0326a19.
  Verified download inventory:outputs/native-crash-diagnosis/opencv413/inventory.json.

## Crop validation completed

API crop parsing, direct ROI validation and orientation-aware upload dimensions
are complete. Invalid/partial/malformed/out-of-bounds crops fail clearly before
analysis. Original upload bytes and valid padding behavior are preserved.
All8 EXIF orientations across JPEG/PNG/WebP decode consistently; only upload
validation needed an axis swap for orientations5–8. No decoder replacement.
175 focused tests passed, affected-file Ruff and mypy passed, diff check passed.
Files:api/app.py,vision/plan_roi.py,security/file_validation.py,
tests/test_crop_validation.py. These form one focused bug-fix commit.
No new download. Unrelated .DS_Store files remain untracked.

## Exact next steps

1. Read this checkpoint and inspect git diff. Do not rerun the completed full70
   pre-fix comparison or original baseline.
2. Post-fix geometric evaluation is complete. Do NOT rerun the analyses or
   completed metrics. Report:outputs/benchmark-opencv413-development-metrics-2026-10-14/report.json.
   All70 analyses complete;66scored/4same annotation errors. Wall F1 .380963→.403484,
   wall IoU .251971→.268075,room F1 .203121→.216164. One small wall F1 regression
   on architectural/7104 (-.003731); details:docs/native-resize-fix.md.
3. Crop validation is complete; inspect git log for its published commit.
   Restart server and verify a fresh upload after image preparation integration.
4. Continue generalized structural accuracy on development evidence; false
   walls on furniture/dimensions and weak room detection remain.
5. Image preparation:quarter-turn rotation first, then straightening and
   perspective. Preserve original bytes; record versioned source/prepared/
   analysis dimensions and forward/inverse3x3 mapping. Compose preparation,
   padded ROI and exact per-axis resize; retain editor/metre frame. Use one
   crop/rotation history. Preserve mapping as server-owned provenance on saves.
   Keep existing source-image endpoint serving analysis ROI; add original
   comparison separately. Extend evaluator to invert mapping before scoring.
   Perspective needs verified rectangle proportions; don't carry scalar source
   metres-per-pixel through a homography. Test EXIF and transformed equivalents.
6. Furniture/material/lighting editing,3D warning focus and full walkthrough
   checks remain. Establish independent source/building groups, reviewed labels
   and broader permitted sources before untouched final evaluation. Openings,
   adjacency,scale error and correction effort metrics remain outstanding.

Never claim perfect reconstruction from every image. Missing measurements,
elevations or ambiguous geometry require explicit assumptions/review.

## Prior pause status and verified completed fix

Paused at the user's request, not for quota. Five-hour allowance after the fix
commit:68% remaining. All agents and bounded benchmark processes have stopped.

Native resize crash fix committed and pushed separately:
4e343b31be2ec990a7bb62654670ffaaeae55a77
https://github.com/Fahad-uz/architecture-walkthrough-analysis/commit/4e343b31be2ec990a7bb62654670ffaaeae55a77
Both pyproject.toml and requirements.txt pin opencv-python4.13.0.92. The saved,
verified local wheel was installed via existing uv; no additional download.
Compatibility check:all66 installed packages compatible. New subprocess
regressions cover the known crash and two odd-dimension resizes with independent
bilinear pixel checks,20second timeout and disabled core dumps.3tests passed.
Existing full Python suite:447passed,9integration skips,2deprecation warnings.
Ruff passed; mypy passed all82source files; git diff --check passed.
Full-suite log:../planstride-opencv413-tests.log.

Post-fix development analyses:70/70 completed,0 failures. All three prior native
crashes now complete. This is completion evidence, NOT accuracy verification;
post-fix geometric metrics have NOT yet been evaluated. Pre-fix comparison and
original baseline remain intact. Completed report:
outputs/benchmark-opencv413-development-2026-10-14/report.json.
The report records starting HEADc3cc82b and the saved unchanged crop patch;
runtime Python source SHA256:2d9d018892e80530219b965eb2fc975e8e81f435f53bbf2fc3fb3ed92f9d77f5.
Those source files remained static for the full run. All workers used4.13.0.92.

The three unfinished crop files listed above are saved locally and excluded
from commits. No application restart or fresh browser validation was performed;
restart the server after finishing crop changes so it uses the corrected runtime.
No new models or assets were downloaded. Resume only on the user's request.

## Current resumed status

Post-fix development metrics completed and saved; no new download or rerun of
analysis. Crop tests established89passes/12failures:only validation dimensions
for EXIF orientations5–8 are wrong; JPEG/PNG/WebP decoded pixels already agree.
Crop author is applying a minimal orientation-aware validation fix plus boundary
and dimension-limit tests. Do not commit that unit until focused checks pass.
Usage on resume:62% remaining; checked again before the next major task.

The crop-validation unit has completed on resume. Next active milestone is
quarter-turn image preparation with explicit coordinate mapping; straightening
and perspective remain later work. No model/asset downloads are required.
