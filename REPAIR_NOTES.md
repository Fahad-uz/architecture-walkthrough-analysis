# Walkthrough repair notes

This repair addresses measured layout errors and rendering inconsistencies in the floor-plan-to-walkthrough pipeline.

## Layout and openings

- Manual metres-per-pixel measurements now stay correct when a large input image is resized for analysis.
- Overlapping wall fragments no longer create false gaps. Confirmed opening gaps connect the measured endpoints, including small centerline offsets; a nearby door cannot close an unrelated gap.
- Room boundaries come from the measured wall graph. The image-morphology fallback that could invent room closures has been removed.
- Legacy wall identifiers preserve their door and window attachments. Reading a corrected model no longer silently resnaps authoritative geometry.
- Windows follow saved opening intervals, including a zero-height sill. Concave floors form one watertight solid.
- Outlined CAD walls no longer inherit dimension-extension tails, and exterior measurement baselines are excluded when paired wall faces establish the drawing style.
- Enclosed, shallow single-line cupboard/counter fronts are separated from structural walls. Their measured concave footprints become room-context fixture proposals without filling an L-shaped counter's open floor.
- Architectural feet/inches, fractions, and metric dimensions parse as complete tokens. Scale measurement uses text spans and clear wall faces to avoid measuring through a doorway into the next room; contradictory rectangles are excluded.
- Readable room labels without reconstructed floor boundaries are explicitly flagged for review. Stair stringers no longer split stair halls into false rooms, and rooms too small for a camera no longer crash route planning.

## Materials, lighting and furniture

- Nine bundled CC0 surface maps provide wood, plaster and tile colour, roughness and normals. Texture paths resolve relative to the registry and render into embedded GLB resources.
- Wall intersections are united into a continuous shell. Every lighting bake sees the original surface materials; already-baked groups no longer contaminate subsequent bakes.
- Exported light intensity accounts for Blender-to-glTF units. Room lights stay inside concave room polygons. Roof-hidden wall tops keep their surface material instead of displaying a black, ceiling-occluded bake.
- Explicit asset placements render alongside furniture without duplicating equivalent items, and retain specified height. Image-to-world conversion preserves rotated furniture footprints.
- Straight and dogleg stairs share footprint-faithful geometry across exporters. Explicit `metadata.ceiling_void_room_ids` keeps listed stair rooms open above while retaining the other ceilings.
- Toilets render as a pedestal, bowl, seat and cistern rather than a bathtub. Bookshelf and bookcase names resolve to cabinet geometry.

## Browser walkthrough

- Improved safe starting points, useful initial view direction, furniture clearance, roof visibility isolation and responsive camera framing.
- Model and plan refreshes stay paired. Controls wait for the matching collision mesh, including cached models.
- Mobile movement and simultaneous touch look, reset controls, bounded shadow maps and antialiasing are covered by the repair.
- Updated frontend dependency patches and added navigation regressions to CI.
- Local OCR disables ONNX telemetry before native initialization, avoiding a telemetry-worker shutdown crash observed with ONNX Runtime 1.29 on macOS.

## Validation scope

A clean synthetic 800 × 600 plan, reduced to 400 × 300 during analysis, recovers three rooms at 50 pixels per metre. Its wall-centerline footprint is 5.95 × 3.89 m against a 6 × 4 m drawn outline. The measured report and source-aligned overlay are included in the delivery folder.

Real Blender exports were checked in both unbaked and draft-baked modes. The browser preview, roof control, initial walkthrough view and guided tour were exercised. The compressed GLB retains geometry, embedded surface/light textures, wall collision names and door meshes.

The supplied real drawing was also tested locally. Its original analysis failed on an empty walkable room; that crash is fixed. Its outlined walls, feet/inch labels and cabinet fronts exposed the additional repairs above. Automatic scaling is within approximately 2.2% of the drawing's overall reference dimension. Automatic extraction still flags the broad, open living-room boundary for review; it is not represented as a fully solved layout.

A separate reviewed trace of that drawing contains seven named rooms and a stair hall, preserves its drawn wall openings, and uses the overall dimension as its scale reference. Loose furniture is proposed staging. Wall height, stair rise/direction and fixture heights remain documented assumptions; no windows absent from the drawing were added. This reviewed local artifact is distinct from automatic detection. The original image and its reconstruction are not committed to GitHub.

Validation at this revision: 358 Python tests (including real Blender exports), nine frontend navigation tests, Python lint/types, frontend types/build and runtime dependency audit passed. The actual-plan optimized GLB was decoded with the Draco decoder to verify full geometry, embedded textures and scene bounds. No external AI calls were made during validation.

See `assets/THIRD_PARTY.md` and `assets/download-manifest.json` for bundled material provenance. The delivery's `Downloads.md` and `Software-inventory.txt` list additional local tooling and dependencies.
