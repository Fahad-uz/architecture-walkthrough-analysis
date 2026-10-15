# PlanStride checkpoint

Updated 15 October 2026 (Asia/Kolkata). Paused at the user's explicit request.
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
Prepared from commit d348da4 (use git log for the full commit and this checkpoint commit).
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

## Completed rotation milestone, 14 October 2026

Quarter-turn preparation is implemented across frontend,API,pipeline and metric
mapping. Rotate left/right, clear crop, reset original, original comparison and
shared undo/redo preserve the original File. Rotation clears the old crop; undo
restores the prior crop and orientation. Multipart rotation_quarter_turns is
integer0..3 clockwise; crop coordinates use the prepared frame. Default0 keeps
legacy uploads working. Backend rotates before ROI without interpolation;
manual metres-per-pixel survives quarter-turns. Metadata image_preparation v1
records EXIF-oriented source/prepared/analysis dimensions and forward/inverse
3x3 matrices in continuous pixel-edge coordinates. Actual padded ROI and exact
per-axis resize are composed. Correction saves retain server-owned provenance
and PlanROI. /original-image serves exact uploaded bytes; /source-image remains
the analysis ROI. Evaluator now consumes nested metadata and retains legacy
fallback. It transforms whole wall footprints including thickness.

Validation:608Python tests passed,9optional integration skips,2deprecation
warnings; full log ../planstride-rotation-tests.log. The final51rotation/evaluator
tests also passed after the final integration fixture was saved. Ruff passed,
mypy passed83source files. Frontend32tests,typecheck and productionbuild passed.
A corrected asymmetric rotated fixture produces identical analysis pixels and
walls/openings/rooms/balconies/scale. No runtime dependency/model/asset downloads.

Runtime restarted with the project venv and AI disabled; server PID6141 saved in
../walkthrough-server.pid, log ../walkthrough-server.log. Health returnedok.
Fresh real-development-plan API smoke test uploaded a counterclockwise variant
of high_quality/932 with corrective clockwise quarter-turn and manual scale.
Job1790674bc15b457c982b3637f1bfe4f8 completed needs_review,39walls/7rooms/0doors/
2windows; original-image bytes matched exactly. This tests workflow/mapping,
not accuracy. Details:outputs/rotation-browser-smoke.json.
Editor:http://127.0.0.1:8001/#/jobs/1790674bc15b457c982b3637f1bfe4f8/edit

Browser visual QA is INCOMPLETE:initial tab opened before the server was ready
and showed connectionrefused. Subsequent binding was blocked on its data:error
page. No security bypass was attempted; a new ordinary localhost tab should be
opened on resume. Do not claim browser rotate/crop/undo or freshGLBwalkthrough
verification from the API smoke test.

## Completed on 15 October 2026

- Browser quarter-turn/crop/undo/redo/reset/original comparison and correction
  saving verified on the original privateplan. All GitHub CI checks passed on
  a6628fd. No repetition of the completed70-image analyses/metrics.
- Found and fixed ignored low-score export override in EditorPage.generate:
  b6abc77. Fresh Blender export, preview loading, walkthrough and guided/manual
  controls verified through the browser. Quality39% still requires review;
  forced export is not automatic-accuracy evidence.
- Fine straightening feature pushed at1b61df4:finite±15°clockwise, expandedwhite
  canvas, exact edge/index warp mapping, crop/quarterturn/angle history, preserved
  original and manualscale, version2 provenance/evaluator support. Zeroangle
  retains version1 and original lossless path. Browser invalid16°,+/−2.5°,crop
  undo/redo and save/provenance tests passed. A deliberatelytilted privateplan
  returned upright after correction;22walls/7rooms,not accuracy certification.
-700Python passes/9optional skips,34frontend passes,Ruff,mypy83files,typecheck,
  productionbuild pass. Logs:../planstride-straightening-tests.log and
  ../planstride-straightening-frontend.log. Independent coordinate review passed.
- Current server PID2521; ../walkthrough-server.pid/log. Healthok. AI disabled.
  Current browser/editor job:f7c01bb7af734d088de0b63c792ce565.
  http://127.0.0.1:8001/#/jobs/f7c01bb7af734d088de0b63c792ce565/edit
- Detailed supported-input/mapping/browser evidence:docs/image-preparation.md.
  Private artifacts:outputs/straightening-qa-2026-10-15/browser-verification.json.
  Screenshots are in parentworkspace outputs,not Git. No new downloads.

## Additional completed work on 15 October

- f293a9f:setup_cubicasa.py reports the unavailable inference provider before
  optional imports/downloads.10targeted tests+lint pass. f93fdd6 records primary
  model references/terms/runtime/availability in docs/segmentation-model-review.md.
  No pretrained model or new package was downloaded/adopted.
- d507bc6:skewed repeated-detail ink evidence and separate regular subsequences
  reduce false structural walls. d348da4 records the full development comparison
  in docs/repetitive-detail-progress.md.70analyses completed/0failures;66scored,
 4same annotation errors. MeanwallF1 .403484→.404386,roomF1 .216164→.217387.
 Onlyarchitectural/5016's scoredwallF1 changed;walls40→30,rooms36→13 against2
 annotatedrooms. Slightwallrecallloss remains recorded;sauna/dimension falsewalls
 persist. This is unreviewed-label agreement,not certifiedaccuracy.
- Full latest completed suite:706Pythonpasses/9optional skips,Ruff+mypy83pass.
  Log:../planstride-repetitive-detail-tests.log. No need to repeat the completed
  analysis/metric runs. Reports:
  outputs/benchmark-repetitive-detail-development-2026-10-15/report.json
  outputs/benchmark-repetitive-detail-metrics-2026-10-15/report.json
  outputs/benchmark-repetitive-detail-metrics-2026-10-15/paired-comparison.json.
- No downloads beyond previously recorded inventory. The browser-tested
  straightening/export/walkthrough evidence is in docs/image-preparation.md.

## Unfinished furniture work saved locally — NOT committed/pushed

Modified:
- frontend/src/pages/EditorPage.tsx
- frontend/src/types.ts
- src/architecture_walkthrough/geometry/models.py
- src/architecture_walkthrough/pipeline.py
New:
- frontend/src/editorFurniture.ts
- frontend/tests/editorFurniture.test.ts
- tests/test_furniture_edit_roundtrip.py

Frontend furniture footprints/select/drag/add/delete/position/rotation/dimensions
and provenance labels are implemented,using existing undo/autosave/save paths.
Helpers provide SAT wall/door-aperture placement hints. This does NOT establish
walking-route clearance. New userobjects have stableid/provenance=user; legacy
origins stayunknown. Backend optionalid/provenance fields and freshanalysis
source tagging are saved. Userarrangements/identities survive basic save/load
and walkthrough preparation without automatic furnitureinjection.

Checks completed for this unfinishedunit:frontend typecheck and8focused helper
tests pass;3backend roundtrip/empty-furnishing tests+focusedlint pass. Fullsuite,
productionbuild,browserediting and regeneration/export validation NOT run yet.
Important unresolved point:asset_placements may mirror furniture,so moving an
item might leave a duplicate asset in collect_scene_furniture. Inspect this
before claiming that arrangements survive 3D regeneration. Review helper and
editor diffs before publishing. The current server does not include these
uncommitted backend fields; restart the venvserver after this unit is complete.

## Exact next steps

1. Read checkpoint and inspect git status/diffs. Resume the saved furnitureunit;
   do not start over or rerun completed70-image analyses/metrics.
2. Inspect scene collect_scene_furniture and resolve any furniture/asset_placements
   mirror duplication on edit/delete. Preserve userarrangements/assetmetadata.
3. Review furniturehelpers/EditorPage,run the full relevant Python/frontend
   checks,build and restartserver using the project venv (don't resolve its
   Python symlink to the base interpreter). Browser test add/select/drag/
   inspector edits/undo/redo/save/reload and fresh3Dregeneration,then commit/push
   this logical feature separately. Original layout must remain unchanged.
4. Perspective preparation,roommaterials/lighting,3Dwarningfocus,walkroute checks
   and further structuralaccuracy remain. Independent source/buildinggroups,
   reviewedannotations and broader permitted sources are required before final
   untouched evaluation. Opening,adjacency,scaleerror and correctioneffort
   metrics remain outstanding.30quarantine examples remain untouched.

## Pause status

Paused because the user explicitly said "save everything and pause",not because
of quota.35% of the five-hour allowance remained at this pause check. All agents
and their helper stopped; all bounded benchmarks finished. Files are saved.
Completed fixes and reports are pushed to main;root verifies the accompanying
checkpoint push before reply. The seven unfinished furniture files remain local.
A redundant local WIP archive+tracked diff is saved under
outputs/checkpoints/2026-10-15-furniture-wip/ (ignored,not pushed).
Keep .DS_Store and frontend/.DS_Store untracked. ServerPID2521 remains running
with the last restarted image-preparation backend and current builtfrontend;
it has NOT loaded the latest wall/furniture source edits. Current userbrowser
is the saved editor jobf7c01bb7af734d088de0b63c792ce565. No additional downloads.
The full PlanStride upgrade remains incomplete. Resume only on userrequest.
