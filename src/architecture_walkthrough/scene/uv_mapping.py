from __future__ import annotations

import numpy as np
import trimesh


def apply_planar_uv(mesh: trimesh.Trimesh, scale_m: float = 1.0) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices)
    if len(vertices) == 0:
        return mesh
    uv = vertices[:, :2] / max(scale_m, 1e-6)
    mesh.visual.uv = uv
    return mesh
