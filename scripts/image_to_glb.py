from __future__ import annotations

import argparse
from pathlib import Path

from architecture_walkthrough.config import load_config
from architecture_walkthrough.logging_config import configure_logging
from architecture_walkthrough.pipeline import convert_image_to_glb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a floor-plan image to an approximate GLB model.")
    parser.add_argument("--input", required=True, help="Path to a PNG, JPEG, or WebP floor-plan image")
    parser.add_argument("--output", required=True, help="Path to write the .glb file")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--work-dir", default=None, help="Directory for debug images and floorplan.json")
    parser.add_argument("--manual-scale", type=float, default=None, help="Metres per pixel; optional but more accurate")
    parser.add_argument("--use-blender", action="store_true", help="Use Blender for export instead of the default pure-Python exporter")
    parser.add_argument("--blender", default=None, help="Optional path to blender.exe when --use-blender is set")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.blender:
        config.paths.blender_executable = args.blender
    output = convert_image_to_glb(
        Path(args.input),
        Path(args.output),
        config,
        manual_scale=args.manual_scale,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        run_blender=args.use_blender,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
