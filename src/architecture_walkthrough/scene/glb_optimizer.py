from __future__ import annotations

import logging
import shutil
from pathlib import Path

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.security.sandbox import run_subprocess

LOGGER = logging.getLogger(__name__)

TOOLS_DIR = Path(__file__).resolve().parents[3].parent / "tools" / "glb"


def _tools_dir() -> Path | None:
    root = Path(__file__).resolve()
    for parent in root.parents:
        candidate = parent / "tools" / "glb"
        if (candidate / "optimize.mjs").exists() and (candidate / "node_modules").exists():
            return candidate
    return None


def _node_path() -> Path | None:
    which = shutil.which("node")
    if which:
        return Path(which)
    default = Path("C:/Program Files/nodejs/node.exe")
    return default if default.exists() else None


def optimize_glb(
    input_glb: Path,
    output_glb: Path | None = None,
    config: AppConfig | None = None,
    texture_size: int = 2048,
    timeout_seconds: int = 600,
) -> Path:
    """Compress a GLB (Draco meshes + per-slot WebP textures).

    Baked lightmaps (emissive slot) are compressed losslessly at native
    resolution — lossy WebP bands their smooth gradients — while all other
    texture slots are compressed and resized aggressively (tools/glb/optimize.mjs).

    Optimization is best-effort: when the Node toolchain is unavailable the
    original GLB is returned untouched with a warning, so the pipeline never
    hard-depends on npm.
    """
    output_glb = output_glb or input_glb.with_suffix(".optimized.glb")
    tools = _tools_dir()
    node = _node_path()
    if tools is None or node is None:
        LOGGER.warning(
            "GLB optimizer unavailable (need Node.js and 'npm install' in tools/glb); "
            "skipping optimization."
        )
        if output_glb != input_glb:
            shutil.copyfile(input_glb, output_glb)
        return output_glb
    command = [
        str(node),
        str(tools / "optimize.mjs"),
        str(input_glb),
        str(output_glb),
        str(texture_size),
    ]
    result = run_subprocess(command, timeout_seconds=timeout_seconds)
    if not output_glb.exists():
        raise RuntimeError(f"gltf-transform did not produce output: {output_glb}\n{result.stdout[-1500:]}")
    before = input_glb.stat().st_size
    after = output_glb.stat().st_size
    LOGGER.info("optimized GLB %s: %.2f MB -> %.2f MB", input_glb.name, before / 1e6, after / 1e6)
    return output_glb
