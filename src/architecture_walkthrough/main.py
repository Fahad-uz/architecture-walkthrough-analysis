from __future__ import annotations

import argparse
from pathlib import Path

from architecture_walkthrough.config import load_config
from architecture_walkthrough.logging_config import configure_logging
from architecture_walkthrough.pipeline import analyze_image, build_model, convert_image_to_glb, render_walkthrough
from architecture_walkthrough.scene.glb_validator import validate_glb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="architecture-walkthrough")
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--manual-scale", type=float, default=None, help="metres per pixel")
    analyze.add_argument("--require-gemini", action="store_true", help="Fail if Gemini vision hints cannot be generated")
    analyze.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="optional manual crop rectangle in source pixels")

    build = sub.add_parser("build-model")
    build.add_argument("--floorplan", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--use-blender", action="store_true", help="Use Blender instead of the default pure-Python GLB exporter")
    build.add_argument("--no-run-blender", action="store_true")

    image_to_glb = sub.add_parser("image-to-glb")
    image_to_glb.add_argument("--input", required=True)
    image_to_glb.add_argument("--output", required=True)
    image_to_glb.add_argument("--work-dir", default=None)
    image_to_glb.add_argument("--manual-scale", type=float, default=None, help="metres per pixel")
    image_to_glb.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="optional manual crop rectangle in source pixels")
    image_to_glb.add_argument("--use-blender", action="store_true", help="Use Blender instead of the default pure-Python GLB exporter")
    image_to_glb.add_argument("--use-gemini", action="store_true", help="Use Gemini vision hints when GEMINI_API_KEY is configured")
    image_to_glb.add_argument("--require-gemini", action="store_true", help="Fail if Gemini vision hints cannot be generated")

    walk = sub.add_parser("walkthrough")
    walk.add_argument("--floorplan", required=True)
    walk.add_argument("--output", required=True)
    walk.add_argument("--mode", choices=["preview", "final"], default="preview")

    validate = sub.add_parser("validate-glb")
    validate.add_argument("--input", required=True)
    validate.add_argument("--report", required=True)

    run_all = sub.add_parser("run-all")
    run_all.add_argument("--input", required=True)
    run_all.add_argument("--output", required=True)
    run_all.add_argument("--manual-scale", type=float, default=None)
    run_all.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="optional manual crop rectangle in source pixels")
    run_all.add_argument("--use-blender", action="store_true", help="Use Blender instead of the default pure-Python GLB exporter")
    run_all.add_argument("--use-gemini", action="store_true", help="Use Gemini vision hints when GEMINI_API_KEY is configured")
    run_all.add_argument("--require-gemini", action="store_true", help="Fail if Gemini vision hints cannot be generated")
    run_all.add_argument("--no-run-blender", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "analyze":
        analyze_image(
            Path(args.input),
            Path(args.output),
            config,
            args.manual_scale,
            require_ai_success=args.require_gemini,
            crop_rect=tuple(args.crop) if args.crop else None,
        )
        return 0
    if args.command == "build-model":
        build_model(Path(args.floorplan), Path(args.output), config, run_blender=args.use_blender)
        return 0
    if args.command == "image-to-glb":
        if args.use_gemini:
            config.ai.gemini_enabled = True
        convert_image_to_glb(
            Path(args.input),
            Path(args.output),
            config,
            manual_scale=args.manual_scale,
            work_dir=Path(args.work_dir) if args.work_dir else None,
            run_blender=args.use_blender,
            require_ai_success=args.require_gemini,
            crop_rect=tuple(args.crop) if args.crop else None,
        )
        return 0
    if args.command == "walkthrough":
        render_walkthrough(Path(args.floorplan), Path(args.output), config, args.mode)
        return 0
    if args.command == "validate-glb":
        report = validate_glb(Path(args.input))
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")
        return 0 if report["valid"] else 1
    if args.command == "run-all":
        if args.use_gemini:
            config.ai.gemini_enabled = True
        out = Path(args.output)
        analyze_image(
            Path(args.input),
            out,
            config,
            args.manual_scale,
            require_ai_success=args.require_gemini,
            crop_rect=tuple(args.crop) if args.crop else None,
        )
        build_model(out, out / "models" / "building.glb", config, run_blender=args.use_blender)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
