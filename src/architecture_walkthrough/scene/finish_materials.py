"""Trimesh adapter for the shared, geometry-preserving finish partition."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import trimesh

from architecture_walkthrough.scene.finish_regions import partition_finish_mesh


def apply_finish_regions(
    mesh: trimesh.Trimesh,
    regions: list[dict[str, Any]],
    material_plan: dict[str, Any],
    *,
    wall: dict[str, Any] | None = None,
) -> trimesh.Trimesh:
    if not regions:
        return mesh
    vertices, faces, labels, sources = partition_finish_mesh(
        mesh.vertices.tolist(), mesh.faces.tolist(), regions, wall=wall,
    )
    # Preview finishes deliberately use scalar materials; maps are loaded once
    # in the production Blender renderer. Preserve every original face color.
    source_materials: Any
    if isinstance(mesh.visual, trimesh.visual.ColorVisuals):
        colors, source_materials = np.unique(np.asarray(mesh.visual.face_colors), axis=0,
                                            return_inverse=True)
        materials = [trimesh.visual.material.PBRMaterial(
            name=f"Original_Surface_{index}", baseColorFactor=color,
        ) for index, color in enumerate(colors)]
    elif isinstance(mesh.visual, trimesh.visual.TextureVisuals):
        original = mesh.visual.material
        materials = list(original.materials) if isinstance(
            original, trimesh.visual.material.MultiMaterial,
        ) else [original]
        source_materials = getattr(mesh.visual, "face_materials", None)
        if source_materials is None:
            source_materials = np.zeros(len(mesh.faces), dtype=int)
    else:
        materials = [trimesh.visual.material.PBRMaterial(name="Original_Surface")]
        source_materials = np.zeros(len(mesh.faces), dtype=int)
    presets = sorted({label for label in labels if label is not None})
    original_count = len(materials)
    for preset in presets:
        payload = material_plan["materials"][preset]
        materials.append(trimesh.visual.material.PBRMaterial(
            name=f"PBR_{preset}",
            baseColorFactor=[round(float(value) * 255) for value in payload["base_color"]],
            roughnessFactor=payload.get("roughness", 0.65),
            metallicFactor=payload.get("metallic", 0.0),
            doubleSided=payload.get("double_sided", False),
            alphaMode=payload.get("alpha_mode", "OPAQUE"),
        ))
    result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    coordinates = np.asarray(vertices)
    if wall is None:
        uv = coordinates[:, :2].copy()
    else:
        start, end = wall["start"], wall["end"]
        direction = np.array([end["x"] - start["x"], end["y"] - start["y"]])
        direction /= np.linalg.norm(direction)
        uv = np.column_stack((coordinates[:, :2] @ direction, coordinates[:, 2]))
    indices = {preset: index + original_count for index, preset in enumerate(presets)}
    result.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.MultiMaterial(materials),
        face_materials=np.array([indices[label] if label is not None else source_materials[source]
                                 for label, source in zip(labels, sources)]),
    )
    result.metadata.update(mesh.metadata)
    return result


def finish_primitives(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    """Trimesh's GLB writer exports one material per mesh; split face sets.

    These are disjoint subsets of the existing boundary triangles, never
    additional finish planes or separately extruded solids.
    """
    if not isinstance(mesh.visual, trimesh.visual.TextureVisuals) or not isinstance(
        mesh.visual.material, trimesh.visual.material.MultiMaterial,
    ):
        return [mesh]
    parts = []
    for index in sorted(set(mesh.visual.face_materials)):
        selected = np.flatnonzero(mesh.visual.face_materials == index)
        part = cast(trimesh.Trimesh, mesh.submesh([selected], append=True, repair=False))
        part.visual = trimesh.visual.TextureVisuals(
            uv=getattr(part.visual, "uv", None), material=mesh.visual.material.materials[index],
        )
        parts.append(part)
    return parts
