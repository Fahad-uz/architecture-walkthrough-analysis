# Saved lighting

The editor's Lighting panel offers Warm, Neutral and Cool light tones, a25–200%
brightness slider, and Reset lighting. These edits use the same undo/redo and
recoverable drafts as layout edits. Save changes updates the lightweight preview
and walkthrough. Generate3D rebuilds exported lights and baked lighting from the
saved settings. Existing baked files keep their previous lightmap until rebuilt.

The default neutral setting preserves the previous appearance. Legacy custom
Kelvin, intensity, exposure and preset values remain saved when using the tone
and brightness controls. A custom temperature appears explicitly in the menu.
Reset replaces the full lighting settings with defaults.

## Contract and rendering

FloorPlanModel.lighting stores brightness1.0, color_temperature4500,
intensity450.0, exposure0.0 and presetneutral_studio by default. The ordinary
tones map to3000/4500/6500K. Brightness is bounded0.25–2, temperature1000–12000K,
legacy intensity0–4500 and exposure−5–5; nonfinite values are rejected.

Shared dependency-free math drives production Blender and offline rendering.
Power is brightness×intensity/450. Exposure is a separate display adjustment.
The new controls do not change geometry or furnishing. The Blender exporter
applies power after realtime calibration clamps so dim and bright choices remain
distinct in none, draft and final modes.

Both browser views use the same saved settings for ambient/hemisphere lighting,
local environment reflections and display exposure. Imported GLB lights already
contain exported power/color and are not scaled again. The lightweight preview's
additional directional fill follows the same settings. No HDR download is needed.

## Verification,22October2026

-784Python tests passed,9optional integration skips; Ruff and mypy85files pass.
 61new lighting tests cover schema bounds, legacy compatibility, shared math,
 production lamp/world settings and mode-specific export clamps. Three additional
 API cases preserve reviewed/inferred open floors and provenance during lighting
 changes. Full log:../planstride-lighting-tests-2026-10-22.log.
-49frontend tests, typecheck and production build passed, including tone defaults,
 legacy preservation, color parity, draft validation and history behavior.
- Browser tested Warm/200%, undo/redo, reset/undo, save and refresh on a fresh
 automatic privateplan job. All walls, rooms, openings, floors, architectural
 elements, scale and furniture matched before/after saving.
- A fresh Blender no-bake export of that same job retained identical glTF meshes
 and geometry accessors. All16punctual light intensities doubled and became
 warmer. Preview and walkthrough loaded with no console errors.
- This verifies appearance controls and workflow,not automatic reconstruction
 accuracy. The plan remains review_required at40% with unresolved structural/
 opening/floor findings. A full final bake was not repeated for this UI milestone;
 all mode-specific light power paths are regression-tested.

Private local evidence:outputs/lighting-qa-2026-10-22/verification.json, with the
neutral and warm generatedglTF records in the same directory. Parentworkspace
screenshot:../../outputs/PlanStride-warm-lighting-preview-2026-10-22.png.
No new downloads or dependencies.
