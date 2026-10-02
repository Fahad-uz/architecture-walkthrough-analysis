Upgrade my existing project, https://github.com/Fahad-uz/architecture-walkthrough-analysis, into **PlanStride**, a reliable floor-plan-to-3D walkthrough application.

Implement the changes, run the application, test it with real images and deliver working results. Do not stop at a plan or a polished demonstration.

The goal is for users to upload different floor plans and receive accurate, attractive, editable 3D walkthroughs with minimal correction. Prioritize structural accuracy, then usability, furniture and lighting. Never hide uncertainty or claim perfect reconstruction when the source image does not provide enough information.

### 1. Inspect the current project and establish a baseline

Start from the latest `main`, which contains the merged repair work. Read the repository instructions and inspect the complete upload, recognition, correction, export and walkthrough pipeline.

Preserve existing working features and user corrections. Identify which improvements are already implemented before adding replacements.

Measure the current automatic results before making changes. Keep manually traced or corrected models separate from automatic results. Do not use the previously reviewed reconstruction of my uploaded image as evidence of automatic accuracy.

### 2. Build a diverse internet reference and evaluation collection

Find and download suitable real floor-plan images from permitted internet sources and annotated datasets. Include different layouts, drawing conventions, resolutions, languages, measurement units, wall styles, furnished plans, scans and photographed plans.

Assess [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) and [FloorPlanCAD](https://floorplancad.github.io/) as research references. Check their actual terms before use: [CubiCasa5K’s license](https://raw.githubusercontent.com/CubiCasa/CubiCasa5k/master/LICENSE) and FloorPlanCAD’s published licensing include non-commercial restrictions. Choose sources compatible with the intended use and keep restricted data out of distributable project assets.

Build an initial benchmark of at least 100 suitable real plans, including at least 30 untouched evaluation examples. Use existing annotations where appropriate and independently review any new ground truth.

Separate development and evaluation data by building and source family. Keep duplicates, crops, rotations and other variants of the same plan in one split. Do not tune against the final evaluation set.

Record sources, licenses, attribution, checksums and category labels. Report results by drawing style, including failures.

### 3. Improve generalized reconstruction

Fix underlying detection and geometry problems rather than hardcoding particular images, filenames or room arrangements.

Improve:
- Wall boundaries, thicknesses, intersections and diagonal geometry.
- Doors, windows, stairs, balconies, room boundaries and room connectivity.
- Separation of structural walls from furniture, dimensions, text and decorative lines.
- OCR for room labels and metric/imperial dimensions.
- Scale estimation, contradictory measurements and missing dimensions.

Evaluate suitable segmentation or recognition models if the current approach cannot meet the benchmark. Verify their licensing, runtime cost and measured benefit before adopting them.

Ground structural geometry in image evidence and verified measurements. Do not use an image generator to redraw the plan and then treat invented details as accurate geometry. Keep optional decorative imagery separate from reconstruction.

Request one known measurement when reliable scale cannot be recovered. Clearly mark assumed heights, finishes and missing details. Handle unreadable or non-floor-plan images with a useful explanation instead of returning a misleading model.

### 4. Add visual image preparation

Before analysis, provide:
- Image preview and crop selection.
- Rotation, straightening and perspective correction.
- Reset-to-original and comparison with the original image.
- Clear feedback about resolution and readability.

Preserve the original upload. Track every transformation so source overlays, scale, detected geometry, corrections and the final 3D model remain aligned.

Test that differently cropped and rotated versions of the same plan produce equivalent geometry in the shared visible area.

### 5. Make uncertainty easy to correct

Turn warnings into actionable controls.

Clicking a warning should focus and highlight the affected wall, opening, room or stair area in the source overlay and corresponding 3D view.

Provide clear confirm/correct actions. Preserve confirmations across saving, and invalidate them when relevant geometry changes. Dismissing a warning must not falsely increase reconstruction confidence.

Add convenient comparison between the source drawing, detected layout and 3D model. Keep stable object identifiers through regeneration.

### 6. Add furniture, material and lighting editing

Allow users to select, move, rotate, resize and delete furniture directly in the editor. Provide suitable furniture choices for different room types.

Distinguish detected objects from suggested furnishing. Preserve user arrangements during regeneration. Check for furniture intersecting walls, blocking doorways or obstructing walking routes.

Add room-level floor/wall materials and understandable lighting controls, with attractive default presets.

Improve models and materials where useful, while preserving real-world scale and browser performance. Furniture placement and visual styling must not change the reconstructed building layout.

### 7. Add undo, redo and recoverable drafts

Implement undo/redo for image preparation and layout, opening, furniture, material and lighting edits.

Add versioned autosaved drafts, explicit save/load, unsaved-change feedback and recovery after refresh or interruption.

Prevent old background analysis or export results from overwriting newer edits. Test regeneration after corrections and restoration of saved projects.

### 8. Verify accuracy and the complete walkthrough

Measure structural accuracy separately from visual quality. Report:
- Wall detection and position accuracy.
- Room overlap, counts and adjacency.
- Door/window detection and placement.
- Scale and dimensional error where ground truth exists.
- Automatic completion and failure rates.
- User correction effort.
- Export time, memory use and walkthrough performance.

Define evaluation tolerances before tuning. Publish baseline and final results on the untouched evaluation set. Do not count corrected models as automatic successes.

Run complete upload → preparation → analysis → correction → save → export → walkthrough tests. Check reachable rooms, doorway passage, wall collisions, stair behavior, lighting, textures and exported geometry.

Add regression tests for failures found in real plans. Use synthetic tests as additional coverage, not as the sole proof of accuracy.

### 9. Rename, commit and deliver

Use **PlanStride** in the product interface and documentation. Preserve compatibility with existing saved projects and commands. Do not rename the GitHub repository or break existing URLs without approval.

Work in small, focused commits and push tested milestones regularly. Merge completed changes into `main` after relevant checks pass, preserving individual commits.

Download necessary tools, models and assets, but list every additional download with its source, version, license, size and purpose. Ask before paid services or billable computing.

Deliver:
- The working application and walkthrough links.
- Reproducible benchmark results and representative before/after examples.
- Tests, setup instructions and supported-input guidance.
- A complete download inventory.
- GitHub commit and merge links.
- An honest list of remaining limitations.

Aim for the most accurate walkthrough each image supports. When information is missing, provide a fast correction path to a verified result rather than silently inventing it.