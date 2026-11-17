# Courtyard floors

The lightweight exporter preserves interior holes when merging reconstructed
room floors. Previously it copied only the union's exterior boundary, which
filled an enclosed courtyard even though no room covered that area.

The fix cuts the existing floor triangulation along every hole edge, removes
only the fully divided void cells, and extrudes the remaining surface once.
This retains courtyard side walls, consistent winding and a closed floor solid.
Concave boundaries, multiple holes, rotated layouts and disconnected floor
islands are supported. Ordinary floors keep their existing triangulation.

Touching hole/exterior or hole/hole boundaries can form a pinched solid even
when the 2D polygon passes GEOS validation. They now fail with a clear error
instead of exporting broken geometry. No boundary is shifted or silently filled.

Explicit slabs remain authoritative. The existing low-room-coverage fallback
foundation policy is unchanged, so this fix does not infer courtyards from a
partially reconstructed plan or certify that a detected void is correct.

## Verification, 17 November 2026

66 focused floor, material, architectural and GLB export tests passed, including
the final two boundary-contact cases. Full Ruff and mypy across 87 source files
passed. Independent rotated/concave/multiple-hole, reversed-winding,
redundant-vertex and contact probes confirmed coverage, positive volume,
watertightness, winding and empty courtyard footprints.

The earlier actual GLB check remains applicable: a rotated synthetic layout has
76 m² of floor and a 4 m² courtyard, compared with the former incorrect 80 m²
exterior-only slab. Exported area agreed within float precision. Evidence:
`outputs/courtyard-qa-2026-10-24/verification.json` (local, ignored).
This is a geometry contract check, not automatic image-recognition accuracy.
No new package, model, texture or dataset was downloaded.
