from __future__ import annotations

import argparse
from pathlib import Path

import trimesh

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()
path = Path(args.path)
if path.suffix.lower() != ".glb":
    raise SystemExit("expected a .glb file")
trimesh.load(path)
print(f"valid GLB: {path}")
