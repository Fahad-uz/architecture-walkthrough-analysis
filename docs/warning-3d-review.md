# Reviewing a finding in 3D

Select a finding in the editor to focus its source location. If that finding
references saved, identified geometry, **View in 3D** opens the preview at those
objects. Orange review lines remain visible through opaque walls. They identify
the referenced objects, not the exact defect within them.

Save edits before using the link. This keeps the source review and generated
preview on the same saved project. The preview links back to the same finding
in the editor. **Reset view** reframes that finding; **Clear finding** restores
the whole-model view and removes the review lines.

## What can be located

- Walls use their saved endpoints, thickness and height.
- Doors and windows need one identified host wall and a complete interval within
  that wall. Explicit start/end offsets take priority over the legacy midpoint
  and width. Cached opening centres do not override those positions. Windows
  retain their sill elevation.
- Rooms and balconies use the saved floor outline. Slab outlines extend down
  from the floor plane by their saved thickness.

The marker uses plan metres converted to the GLB coordinate system. Marker
heights have a finite reference fallback and a 20 m cap. These are review wires;
they do not modify source geometry, GLB materials, validation or quality scores.

Whole-plan findings, semantic image annotations without an identified object,
missing/ambiguous IDs, stale links, unsupported objects and invalid bounds get
an explanation instead of a fabricated 3D location. An ambiguous opening finding
that identifies only a wall highlights that wall, not an invented aperture.

## Verification, 24 October 2026

The isolated real-plan project `4536f1c93118459ea18a624265bbadcf` was used to
check a wall-crossing finding: source focus, 3D marker and camera framing,
reset, clear, whole-model framing, source backlink, dirty-edit blocking and
undo back to the saved state. A whole-plan finding had no 3D link. An obsolete
finding URL displayed an explicit explanation.

Tests cover diagonal walls, stale opening centres, legacy/explicit intervals,
missing references, duplicate IDs, invalid spans, source-only and stale findings,
slab direction, coordinate conversion and save/model gating. Eight asynchronous
tests cover concurrent writes, version changes, errors and unavailable models. Source opening
review tests also verify host-interval focus rather than a stale centre.

The published frontend suite has 73 passing tests; type checking and the
production build pass. Preview status is checked again after loading saved plan
data, retaining the last matched pair during concurrent writes. Artifact URLs
remain mutable, so this is a bounded read check rather than an immutable snapshot.

The plan still requires structural review at about 40% quality, with missing
openings and incomplete floors. This verifies the review workflow, not automatic
reconstruction accuracy. No new assets, models or packages were downloaded.
