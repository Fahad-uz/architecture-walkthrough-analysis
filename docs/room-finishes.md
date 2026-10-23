# Room finishes

Choose a room from **Room to edit**, or select its outline in the plan. The
inspector offers independent floor and inside-wall finishes:

- Floors: Wood, Light stone, Grey tile.
- Walls: Warm plaster, Wood, Grey tile.
- Automatic removes the room override and restores its saved material rules.

These choices use existing local material presets. They describe a design
choice, not a finish detected in the source image. Unknown custom preset names
remain visible and are preserved until changed; unavailable registry entries
use the renderer's existing fallback and emit a material warning.

Each choice is an undoable edit, included in local drafts and **Save changes**.
Saved finishes update the lightweight preview and survive reload. **Generate
3D** rebuilds the full textured model with the choices. Existing saved projects
need no migration. Changes follow persistent room IDs through face matching.

## Surface behavior

The optional `floor_material_preset` and `wall_material_preset` fields belong to
the room. A direct floor choice takes precedence over legacy material
assignments, room-type defaults and the scene default. Clearing the override
restores those rules. Existing projects without overrides keep their rendering
path and appearance.

Floor choices partition the top of existing room floors, explicit slabs or
fallback foundations. They add no floor coverage or overlapping finish planes.
Inside-wall choices partition only wall-side spans bordering that room. This
supports different rooms on opposite sides of a shared wall and multiple rooms
along one long wall. Opening cutouts, end caps and reveals remain intact.
Conflicting overlapping room regions retain the original finish in the overlap
and report a material warning instead of selecting an arbitrary room.

Both renderers use the same dependency-free face partitioner. The instant
preview uses scalar colors; the production Blender path uses the registry's
existing texture maps and metre-based UV scale. Scalar previews therefore show
the finish family without the final texture detail.

## Verification, 24 October 2026

The full backend suite passed: 809 tests, 9 optional skips. Ruff and mypy across
87 source files passed. The frontend suite has 77 passing tests; type checking
and the production build passed. New tests cover persistence through actual
corrections, face matching, legacy precedence, shared/partial/diagonal walls,
concave regions, overlap conflicts, slab/fallback coverage, source materials,
stale renderer metadata, custom registries and actual GLB material primitives.
Independent randomized partition checks preserved area, volume, winding and
watertightness. A 36-room test bounds subdivision time and face count.

An actual Blender comparison of two adjacent rooms, a shared wall, a long wall,
a door and a window exported distinct materials with the same wall/floor bounds,
area and volume within GLB float precision. It used no-bake mode; a full final
bake was not repeated for this change.

The isolated real-plan project `4536f1c93118459ea18a624265bbadcf` verified two
independent room choices, undo/redo, automatic reset/undo, saving, reload, fresh
Blender generation, preview and walkthrough rendering. Saved geometry,
furniture, openings and scale were unchanged. Export input retained the exact
saved room choices. Local evidence is in
`outputs/room-finishes-qa-2026-10-24/` (ignored by Git).

This is workflow and rendering validation. The automatic reconstruction still
requires review at about 40% quality with missing openings and incomplete
floors. Material edits do not improve that quality score or certify geometry.
Large plans with many angled regions can require additional subdivision; the
current culling is per mesh. No new packages, texture packs or models were
downloaded.
