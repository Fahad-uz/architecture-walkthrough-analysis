from __future__ import annotations

import argparse
from pathlib import Path

from architecture_walkthrough.config import load_config
from architecture_walkthrough.logging_config import configure_logging
from architecture_walkthrough.pipeline import analyze_image, build_model, render_walkthrough


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="architecture-walkthrough")
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--manual-scale", type=float, default=None, help="metres per pixel")

    build = sub.add_parser("build-model")
    build.add_argument("--floorplan", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--no-run-blender", action="store_true")

    walk = sub.add_parser("walkthrough")
    walk.add_argument("--floorplan", required=True)
    walk.add_argument("--output", required=True)
    walk.add_argument("--mode", choices=["preview", "final"], default="preview")

    run_all = sub.add_parser("run-all")
    run_all.add_argument("--input", required=True)
    run_all.add_argument("--output", required=True)
    run_all.add_argument("--manual-scale", type=float, default=None)
    run_all.add_argument("--no-run-blender", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "analyze":
        analyze_image(Path(args.input), Path(args.output), config, args.manual_scale)
        return 0
    if args.command == "build-model":
        build_model(Path(args.floorplan), Path(args.output), config, run_blender=not args.no_run_blender)
        return 0
    if args.command == "walkthrough":
        render_walkthrough(Path(args.floorplan), Path(args.output), config, args.mode)
        return 0
    if args.command == "run-all":
        out = Path(args.output)
        analyze_image(Path(args.input), out / "debug", config, args.manual_scale)
        build_model(out / "debug" / "floorplan.json", out / "models" / "building.glb", config, run_blender=not args.no_run_blender)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
