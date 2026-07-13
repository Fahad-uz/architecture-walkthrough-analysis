from __future__ import annotations

import logging
import shutil
from pathlib import Path

from architecture_walkthrough.config import AppConfig
from architecture_walkthrough.security.sandbox import run_subprocess

LOGGER = logging.getLogger(__name__)

TOOLS_DIR = Path(__file__).resolve().parents[3].parent / "tools" / "glb"


def _cli_path() -> Path | None:
    """Locate the project-local gltf-transform CLI (Windows .cmd or POSIX bin)."""
    root = Path(__file__).resolve()
    for parent in root.parents:
        candidate_dir = parent / "tools" / "glb" / "node_modules" / ".bin"
        for name in ("gltf-transform.cmd", "gltf-transform"):
            candidate = candidate_dir / name
            if candidate.exists():
                return candidate
    which = shutil.which("gltf-transform")
    return Path(which) if which else None


def optimize_glb(
    input_glb: Path,
    output_glb: Path | None = None,
    config: AppConfig | None = None,
    texture_size: int = 2048,
    timeout_seconds: int = 600,
) -> Path:
    """Compress a GLB with gltf-transform (Draco meshes + WebP textures).

    Optimization is best-effort: when the Node toolchain is unavailable the
    original GLB is returned untouched with a warning, so the pipeline never
    hard-depends on npm.
    """
    output_glb = output_glb or input_glb.with_suffix(".optimized.glb")
    cli = _cli_path()
    if cli is None:
        LOGGER.warning(
            "gltf-transform CLI not found; skipping GLB optimization. "
            "Run 'npm install' in tools/glb to enable it."
        )
        if output_glb != input_glb:
            shutil.copyfile(input_glb, output_glb)
        return output_glb
    command = [
        str(cli),
        "optimize",
        str(input_glb),
        str(output_glb),
        "--compress",
        "draco",
        "--texture-compress",
        "webp",
        "--texture-size",
        str(texture_size),
    ]
    result = run_subprocess(command, timeout_seconds=timeout_seconds)
    if not output_glb.exists():
        raise RuntimeError(f"gltf-transform did not produce output: {output_glb}\n{result.stdout[-1500:]}")
    before = input_glb.stat().st_size
    after = output_glb.stat().st_size
    LOGGER.info("optimized GLB %s: %.2f MB -> %.2f MB", input_glb.name, before / 1e6, after / 1e6)
    return output_glb
