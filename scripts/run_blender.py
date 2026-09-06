from __future__ import annotations

import argparse
from pathlib import Path

from architecture_walkthrough.config import load_config
from architecture_walkthrough.scene.blender_runner import run_blender_script

parser = argparse.ArgumentParser()
parser.add_argument("script")
parser.add_argument("--config", default="configs/default.yaml")
args = parser.parse_args()
config = load_config(args.config)
run_blender_script(str(config.paths.blender_executable), Path(args.script), config.limits.subprocess_timeout_seconds)
