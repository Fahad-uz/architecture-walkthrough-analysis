# Furniture editing

PlanStride lets you add a bed, sofa, table, chair, wardrobe, rug or plant in the
layout editor. Select a room first to add at an interior point, or add at the
visible plan centre. Drag its footprint to move it. The inspector edits position,
rotation, width, depth and height in metres. Clear height to use the model default.
Positive rotation is counterclockwise in world coordinates.

Select an object on the plan or through Furniture to edit. Delete selected removes
it; Undo and Redo restore the whole object. A drag and each focused numeric edit
form separate history steps. Existing local drafts include these changes. Save
changes writes the project; Generate 3D saves first and regenerates the model.

Source labels distinguish detected, suggested, user-added and unknown objects.
Editing an object preserves its original source. Legacy files do not acquire an
invented detection source. Empty furnishing stays empty.

## Regeneration and compatibility

Furniture has an optional persistent ID and source field. Detailed assets can
reference it through asset_placements[].furniture_id. The furniture geometry is
authoritative; moving or deleting it cannot leave the asset's old geometry or
collision obstacle behind. Unlinked assets remain independent.

Legacy category-family/footprint mirrors link on model load only when the
furniture match is unique. Anonymous matched objects receive a stable ID without
changing provenance. The old effective detailed height is retained at initial
legacy linkage; later furniture height edits win. Existing asset presets remain
saved. Renderers and route planning use the same backend resolver; the browser
uses the same linked-object rules.

## Checks and limits

Placement hints use rotated footprints against wall solids and anchored doorway
apertures. They do not yet certify door swing/approach clearance, furniture-to-
furniture clearance or walking-route accessibility. Missing doors or incomplete
room reconstruction can still prevent a useful walkthrough. Editing furniture
does not improve automatic structural accuracy or change the quality score.

On 22 October 2026, browser QA used a fresh automatic analysis of the private
reference plan with corrective straightening, AI disabled. This was workflow
evidence, not automatic-accuracy certification. Add, numeric edits, height reset,
drag, delete/undo, one-step drag undo/redo, save and refresh were verified. Walls,
openings, rooms, slabs, architectural elements and scale were identical before
and after saving. The custom sofa ID, source, position, dimensions and 37-degree
rotation reached the Blender input unchanged. A fresh no-bake Blender GLB loaded
in preview and walkthrough; guided/manual controls worked. Quality remained
review_required at about 40%, with unresolved floor/opening/layout findings.

Local evidence (ignored, private): outputs/furniture-qa-2026-10-22/verification.json
and job 4536f1c93118459ea18a624265bbadcf. Screenshot in the parent workspace:
../../outputs/PlanStride-furniture-preview-2026-10-22.png.

Validation: full Python suite: 718 passed and 9 optional skips, followed by final focused
coverage after two additional mirror cases (38 passed and 2 optional integration
cases deselected). Frontend: 43 tests, typecheck, production build, Ruff and
mypy across 84 source files passed. GitHub CI passed the asset-link fix 4d9c2c9.
No new package, model or asset downloads.
