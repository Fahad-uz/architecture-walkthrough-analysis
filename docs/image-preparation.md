# Preparing plan images

PlanStride preserves the uploaded file and prepares a separate analysis image.
Use **Rotate left/right** for quarter-turns, then enter a clockwise-positive
straightening angle between −15° and 15° and choose **Apply straightening**.
Choose a crop after setting the orientation. Changing orientation clears the
old crop; undo/redo restores the whole crop/orientation snapshot. Reset restores
the original, and Compare with original shows the untouched upload.

The preview reports the prepared dimensions and applied angle. Unapplied numeric
input does not change the preview or analysis. Keep wall lines and measurement
labels in the crop. Analysis adds the existing small crop margin.

Quarter-turns are lossless. Fine-angle adjustment uses one interpolation from
the original decoded image on an expanded white canvas; repeatedly adjusting
the preview never repeatedly resamples the upload. Fine lines can still lose
contrast through interpolation. These controls correct a known image orientation;
they do not infer missing walls, dimensions or room boundaries. Perspective
correction is not implemented yet.

## Coordinate and saved-project compatibility

Preparation happens before plan ROI detection. Crop coordinates refer to the
prepared display. The source frame is the upload after EXIF orientation; its
original bytes are retained. Zero straightening keeps version1 preparation
metadata and the previous lossless path. A nonzero angle emits version2 with
straighten_degrees, source/prepared/analysis dimensions and invertible3x3 matrices.
Continuous image-edge coordinates locate pixel centres at (x+.5,y+.5); the
OpenCV warp converts this convention explicitly. Recorded matrices include
quarter-turn, fine rotation, actual padded ROI and independent rounded resize
factors. Rotation preserves source metres-per-pixel; downstream resize handling
remains in force. Editor corrections cannot replace server-owned provenance.

The metric evaluator accepts legacy ROI-only records and preparation versions1
and2. It maps complete wall footprints and room polygons back into source
coordinates. This supports transformed-input evaluation without silently changing
labels. It does not certify the underlying detector or data split.

## Browser and runtime verification, 15 October 2026

At a6628fd, the original user plan passed preview rotation, precise crop,
undo/redo restoration, reset and original comparison. Rotated browser upload
completed; the source overlay aligned and correction saving retained its mapping
and exact original bytes. The same session exposed an export bug: the checked
quality override was dropped when no severe finding was present. Commitb6abc77
honours that explicit override for low-score models too. Repeating the browser
flow generated a fresh Blender GLB, displayed it in the preview, and opened the
walkthrough; guided tour and return to manual control worked. This forced export
is workflow evidence, not proof that the low-confidence layout is accurate.

Fine-angle browser checks reject16°, expand the canvas for2.5°, preserve crop
and angle together across undo/redo, reset to original and retain version2
provenance after correction save. A deliberately −2.5° tilted copy of the user's
plan was prepared with+2.5° and returned upright. It produced22walls/7rooms,
compared with22walls/8rooms in the earlier upright automatic example. This is
transformation/workflow testing on one private plan, not a generalization or
accuracy benchmark. The unmodified upright source remains preserved.

Private/ignored artifacts:
- outputs/straightening-qa-2026-10-15/browser-verification.json
- Browser jobs3e8e7342193b488da0c2c54579711f26,7cf4bf391fcb4015b41cd7014d9894e5,
  f7c01bb7af734d088de0b63c792ce565.
- Parent-workspace outputs/PlanStride-rotation-editor-2026-10-15.png,
  PlanStride-generated-preview-2026-10-15.png,
  PlanStride-walkthrough-2026-10-15.png and
  PlanStride-corrective-straightening-2026-10-15.png.

Validation for straightening:700Python tests passed,9optional integration skips,
34frontend tests passed; Ruff,mypy83source files,typecheck and productionbuild
passed. Tests cover asymmetric images, clockwise sign, half-pixel colour-ramp
landmarks, expanded bounds, exact zero-angle compatibility, crop/resize mapping,
manual scale, API validation, spawned-worker publication, saved provenance and
legacy/version2 evaluation. An independent read-only coordinate review found no
actionable bug. No dependencies,models or assets were downloaded.
