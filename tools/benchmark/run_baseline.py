"""Run automatic image analysis in isolated, time-bounded processes.

Inputs and generated artifacts remain local. Detection counts and the internal
quality score are diagnostics, never ground-truth accuracy measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
ALLOWED_SPLITS = {"development", "private_regression"}


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_samples(manifest_path: Path, split: str) -> list[dict[str, Any]]:
    if split not in ALLOWED_SPLITS:
        raise ValueError("Only development or private_regression may run; evaluation stays untouched")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema_version")
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Manifest samples must be a list")
    selected = []
    identifiers: set[str] = set()
    for entry in samples:
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError("Sample IDs must be nonempty and unique")
        identifiers.add(identifier)
        if entry.get("split") != split:
            continue
        image = entry.get("image", {})
        path = (manifest_path.parent / image["path"]).resolve()
        if not path.is_file():
            raise ValueError(f"Missing image for sample {identifier}")
        actual = checksum(path)
        if image.get("sha256") != actual:
            raise ValueError(f"Image checksum mismatch for sample {identifier}")
        selected.append({**entry, "input_path": str(path), "image_sha256": actual})
    if not selected:
        raise ValueError(f"No samples in split {split}")
    return selected


def effective_config(config_path: Path) -> dict[str, Any]:
    # Deliberately bypass .env/environment overrides so a developer's API key,
    # model checkpoint or provider setting cannot alter the benchmark.
    import yaml
    from architecture_walkthrough.config import AppConfig

    config = AppConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    config.ai.gemini_enabled = False
    config.ai.gemini_sanity_check_enabled = False
    if config.ai.segmentation_checkpoint is not None:
        raise ValueError("Baseline requires the local geometry provider, without external checkpoints")
    return config.model_dump(mode="json")


def git_value(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def provenance(config: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    code_hashes = {
        str(path.relative_to(REPO_ROOT)): checksum(path)
        for path in sorted((REPO_ROOT / "src/architecture_walkthrough").rglob("*.py"))
    }
    return {
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--porcelain"),
        "source_files_sha256": code_hashes,
        "source_tree_sha256": hashlib.sha256(
            json.dumps(code_hashes, sort_keys=True).encode()
        ).hexdigest(),
        "runner_sha256": checksum(Path(__file__)),
        "manifest_sha256": checksum(manifest_path),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "config": config,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def sample_directory(identifier: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_-]+", "-", identifier).strip("-")[:64] or "sample"
    return readable + "-" + hashlib.sha256(identifier.encode()).hexdigest()[:12]


def execute_worker(command: list[str], directory: Path, timeout: float) -> dict[str, Any]:
    """A completed result is accepted only after a clean worker exit."""
    started = time.perf_counter()
    env = {**os.environ, "ARCH_WALK_GEMINI_ENABLED": "false"}
    result_path = directory / "worker-result.json"
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
            env=env, start_new_session=os.name == "posix",
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            return {
                "status": "timeout", "runtime_seconds": round(time.perf_counter() - started, 4),
                "peak_memory_bytes": None, "error": f"Worker exceeded {timeout:g} seconds",
            }
        except BaseException:
            # An interrupted benchmark must not leave OCR workers running.
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            raise
    runtime = round(time.perf_counter() - started, 4)
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") in {"completed", "failed"} and (
                returncode == 0 or result.get("status") == "failed"
            ):
                return {**result, "runtime_seconds": runtime, "worker_exit_code": returncode}
        except (ValueError, AttributeError):
            pass
    return {
        "status": "failed", "runtime_seconds": runtime, "peak_memory_bytes": None,
        "worker_exit_code": returncode, "error": "Worker exited without a valid final result",
    }


def worker(request_path: Path) -> int:
    import resource
    from importlib.metadata import PackageNotFoundError, version
    from architecture_walkthrough.config import AppConfig
    from architecture_walkthrough.pipeline import analyze_image

    request = json.loads(request_path.read_text(encoding="utf-8"))
    directory = request_path.parent
    config = AppConfig.model_validate(request["config"])
    config.ai.gemini_enabled = False
    config.ai.gemini_sanity_check_enabled = False
    result: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        model = analyze_image(Path(request["input_path"]), directory / "analysis", config)
        result = {
            "status": "completed",
            "counts": {name: len(getattr(model, name)) for name in (
                "walls", "rooms", "doors", "windows", "furniture", "special_elements",
                "camera_waypoints",
            )},
            "quality_state": model.reconstruction.quality_state,
            "quality_score": model.reconstruction.quality_score,
            "pixels_per_metre": model.pixels_per_metre,
            "validation_issues": [issue.model_dump(mode="json") for issue in model.validation_issues],
            "ai_assist_attempted": model.metadata.get("ai_assist_attempted"),
            "sanity_check_attempted": model.metadata.get("sanity_check_attempted"),
            "model_sha256": checksum(directory / "analysis/floorplan.json"),
            "geometry_accuracy": None,
            "accuracy_unavailable_reason": "No independently reviewed ground-truth evaluation performed",
        }
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["peak_memory_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
        result["analysis_seconds"] = round(time.perf_counter() - started, 4)
        result["dependency_versions"] = {}
        for package in ("numpy", "opencv-python", "shapely", "rapidocr", "onnxruntime", "pydantic"):
            try:
                result["dependency_versions"][package] = version(package)
            except PackageNotFoundError:
                result["dependency_versions"][package] = None
        write_json(directory / "worker-result.json", result)
    return 0 if result["status"] == "completed" else 1


def run(manifest_path: Path, output_dir: Path, config_path: Path,
        split: str, timeout: float, limit: int | None = None,
        resume: bool = False) -> dict[str, Any]:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be positive and finite")
    if limit is not None and limit <= 0:
        raise ValueError("Limit must be positive")
    samples = load_samples(manifest_path, split)
    if limit is not None:
        samples = samples[:limit]
    config = effective_config(config_path)
    identity = provenance(config, manifest_path)
    # New runs require fresh directories. Explicit resume verifies experiment
    # identity and preserves completed results and interrupted artifacts.
    output_dir.mkdir(parents=True, exist_ok=resume)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "mode": "automatic_unedited",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": identity, "split": split, "timeout_seconds": timeout,
        "selected_sample_count": len(samples), "sample_ids": [item["id"] for item in samples],
        "results": [],
        "accuracy_claim": "None: counts and internal quality scores are not accuracy measurements",
        "peak_memory_definition": "Worker maximum resident set size; unavailable on timeout/crash",
        "status": "running",
    }
    report_path = output_dir / "report.json"
    if resume and report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        for key in ("source_tree_sha256", "config_sha256", "manifest_sha256", "runner_sha256",
                    "python_version", "platform"):
            if previous.get("provenance", {}).get(key) != identity[key]:
                raise ValueError(f"Cannot resume a changed experiment: {key}")
        for key in ("schema_version", "split", "timeout_seconds", "sample_ids"):
            if previous.get(key) != report[key]:
                raise ValueError(f"Cannot resume a changed experiment: {key}")
        report = previous
        report["status"] = "running"
        report.setdefault("resumed_at", []).append(datetime.now(timezone.utc).isoformat())
    write_json(report_path, report)
    finished_ids = {item["id"] for item in report["results"]}
    for sample in samples:
        if sample["id"] in finished_ids:
            continue
        base = sample_directory(sample["id"])
        directory = output_dir / base
        attempt = 1
        while directory.exists():
            attempt += 1
            directory = output_dir / f"{base}-attempt-{attempt}"
        directory.mkdir()
        write_json(directory / "request.json", {"input_path": sample["input_path"], "config": config})
        outcome = execute_worker(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(directory / "request.json")],
            directory, timeout,
        )
        report["results"].append({
            "id": sample["id"], "split": sample["split"],
            "image_sha256": sample["image_sha256"],
            "ground_truth_reviewed": sample.get("ground_truth_reviewed", False),
            "artifact_directory": directory.name, **outcome,
        })
        report["summary"] = dict(Counter(item["status"] for item in report["results"]))
        write_json(output_dir / "report.json", report)
        print(f"{sample['id']}: {outcome['status']} ({outcome['runtime_seconds']:.2f}s)", flush=True)
    report["status"] = "finished"
    write_json(output_dir / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/default.yaml")
    parser.add_argument("--split", choices=sorted(ALLOWED_SPLITS), default="development")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="Resume matching partial report")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if args.manifest is None or args.output is None:
        parser.error("--manifest and --output are required")
    report = run(args.manifest.resolve(), args.output.resolve(), args.config.resolve(),
                 args.split, args.timeout, args.limit, args.resume)
    return 0 if all(item["status"] == "completed" for item in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
