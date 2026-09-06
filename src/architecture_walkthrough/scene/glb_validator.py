from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def validate_glb(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "file_size": path.stat().st_size if path.exists() else 0,
        "valid": False,
        "issues": [],
    }
    if not path.exists():
        report["issues"].append("file does not exist")
        return report
    if report["file_size"] <= 0:
        report["issues"].append("file is empty")
        return report
    try:
        loaded = trimesh.load(path, force="scene")
    except Exception as exc:
        report["issues"].append(f"trimesh failed to load GLB: {exc}")
        return report
    if not isinstance(loaded, trimesh.Scene):
        report["issues"].append("GLB did not load as a scene")
        return report
    scene = loaded
    bounds = scene.bounds
    report["mesh_count"] = len(scene.geometry)
    report["node_count"] = len(scene.graph.nodes_geometry)
    report["bounds"] = bounds.tolist() if bounds is not None else None
    if bounds is None or not np.isfinite(bounds).all():
        report["issues"].append("invalid or missing bounds")
    if len(scene.geometry) == 0:
        report["issues"].append("no geometry found")
    report["valid"] = not report["issues"]
    return report
