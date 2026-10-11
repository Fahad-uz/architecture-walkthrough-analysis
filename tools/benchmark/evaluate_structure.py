"""Development-only geometric agreement with CubiCasa's upstream SVG labels.

SVG coordinates are F1_scaled pixel coordinates, as in CubiCasa's official
loader. This does not rescale annotations to the SVG viewBox or register them
against predictions. Upstream annotations have not been independently reviewed.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from shapely import make_valid
from shapely.affinity import affine_transform
from shapely.errors import GEOSException
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from run_baseline import checksum, load_samples, write_json

PROTOCOL = {
    "version": 1,
    "wall_metric": "polygon area precision/recall/F1/IoU; not centreline F1",
    "wall_tolerance": "0.005 * original image diagonal, additionally reported separately",
    "room_match_min_iou": 0.5,
    "room_matching": "maximum cardinality bipartite; deterministic IoU-descending search",
    "excluded_room_class": "Outdoor",
    "alignment": "SVG polygons directly in F1_scaled pixels; no viewBox rescale or registration",
    "accuracy_scope": "development agreement with unreviewed upstream annotations",
}


def svg_transform(text: str) -> np.ndarray:
    matrix = np.eye(3)
    token = re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")
    if token.sub("", text).strip(" ,\t\r\n"):
        raise ValueError("Unsupported SVG transform syntax")
    for name, arguments in token.findall(text):
        values = [float(v) for v in re.split(r"[\s,]+", arguments.strip()) if v]
        item = np.eye(3)
        if name == "matrix" and len(values) == 6:
            a, b, c, d, e, f = values
            item = np.array([[a, c, e], [b, d, f], [0, 0, 1]])
        elif name == "translate" and len(values) in (1, 2):
            item[0, 2] = values[0]
            item[1, 2] = values[-1] if len(values) == 2 else 0
        elif name == "scale" and len(values) in (1, 2):
            item[0, 0], item[1, 1] = values[0], values[-1]
        elif name == "rotate" and len(values) in (1, 3):
            angle = math.radians(values[0])
            item[:2, :2] = [[math.cos(angle), -math.sin(angle)],
                            [math.sin(angle), math.cos(angle)]]
            if len(values) == 3:
                centre = np.array(values[1:])
                item[:2, 2] = centre - item[:2, :2] @ centre
        else:
            raise ValueError(f"Unsupported SVG transform: {name}")
        matrix = matrix @ item
    return matrix


def load_svg(path: Path, width: int, height: int) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    bounds = box(0, 0, width, height)
    output: dict[str, Any] = {"walls": [], "rooms": [], "outdoor_rooms": [], "repaired": 0}

    def walk(element, inherited):
        matrix = inherited @ svg_transform(element.get("transform", ""))
        classes = element.get("class", "").split()
        category = "walls" if "Wall" in classes else "rooms" if "Space" in classes else None
        if category:
            if "Outdoor" in classes:
                category = "outdoor_rooms"
            polygons = [child for child in element if child.tag.rsplit("}", 1)[-1] == "polygon"]
            if len(polygons) != 1:
                raise ValueError("Expected one direct structural polygon")
            child = polygons[0]
            values = [float(v) for v in re.split(r"[\s,]+", child.get("points", "").strip()) if v]
            if len(values) < 6 or len(values) % 2:
                raise ValueError("Invalid structural polygon coordinates")
            points = np.array(values).reshape(-1, 2)
            transformed = np.c_[points, np.ones(len(points))] @ (
                matrix @ svg_transform(child.get("transform", ""))
            ).T
            polygon = Polygon(transformed[:, :2])
            if not polygon.is_valid:
                output["repaired"] += 1
                polygon = make_valid(polygon)
            polygon = polygon.intersection(bounds)
            if polygon.is_empty or polygon.area <= 0:
                raise ValueError("Structural polygon has no area in the image")
            output[category].append(polygon)
        for child in element:
            walk(child, matrix)

    walk(root, np.eye(3))
    if not output["walls"] or not output["rooms"]:
        raise ValueError("SVG has no usable walls or indoor rooms")
    return output


def prediction_geometry(model: dict[str, Any], analysis_size: tuple[int, int],
                        image_size: tuple[int, int]) -> dict[str, Any]:
    roi = model["plan_roi"]["rect"]
    ppm = float(model["pixels_per_metre"])
    aw, ah = analysis_size
    if min(aw, ah, ppm, roi["width"], roi["height"]) <= 0:
        raise ValueError("Invalid prediction coordinate transform")
    # The pipeline works in metres with Y up. Undo the actual resized ROI
    # dimensions independently on each axis to handle integer resize rounding.
    factors = [ppm * roi["width"] / aw, 0, 0, -ppm * roi["height"] / ah,
               roi["x"], roi["y"] + roi["height"]]
    bounds = box(0, 0, *image_size)
    walls = []
    for wall in model["walls"]:
        endpoints = [(wall[key]["x"], wall[key]["y"]) for key in ("start", "end")]
        footprint = LineString(endpoints).buffer(wall["thickness_m"] / 2, cap_style=2)
        walls.append(affine_transform(footprint, factors).intersection(bounds))
    rooms = []
    for room in model["rooms"]:
        polygon = make_valid(Polygon([(p["x"], p["y"]) for p in room["points"]]))
        rooms.append(affine_transform(polygon, factors).intersection(bounds))
    return {"walls": walls, "rooms": rooms}


def area_scores(prediction, truth, tolerance: float = 0) -> dict[str, float]:
    precision = (prediction.intersection(truth.buffer(tolerance)).area / prediction.area
                 if prediction.area else 0.0)
    recall = (truth.intersection(prediction.buffer(tolerance)).area / truth.area
              if truth.area else 0.0)
    union = prediction.union(truth).area
    return {
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "iou": prediction.intersection(truth).area / union if union else 0.0,
    }


def match_rooms(predicted: list[Any], truth: list[Any], minimum: float = 0.5) -> dict[str, Any]:
    ious = [[area_scores(p, t)["iou"] for t in truth] for p in predicted]
    edges = [sorted((j for j, score in enumerate(row) if score >= minimum),
                    key=lambda j: (-row[j], j)) for row in ious]
    truth_to_prediction: dict[int, int] = {}

    def augment(i: int, seen: set[int]) -> bool:
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in truth_to_prediction or augment(truth_to_prediction[j], seen):
                truth_to_prediction[j] = i
                return True
        return False

    for index in range(len(predicted)):
        augment(index, set())
    matched = [ious[i][j] for j, i in truth_to_prediction.items()]
    tp = len(matched)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(truth) if truth else 0.0
    return {
        "matched": tp, "predicted": len(predicted), "truth": len(truth),
        "false_positive": len(predicted) - tp, "false_negative": len(truth) - tp,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "matched_mean_iou": float(np.mean(matched)) if matched else None,
    }


def diagnostic_overlay(image_path: Path, output_path: Path, predicted, truth) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for geometry, colour in ((unary_union(truth["walls"]), (0, 100, 255)),
                             (unary_union(predicted["walls"]), (255, 40, 0))):
        polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
        for polygon in polygons:
            if polygon.geom_type == "Polygon":
                draw.line(list(polygon.exterior.coords), fill=colour, width=2)
                for interior in polygon.interiors:
                    draw.line(list(interior.coords), fill=colour, width=2)
    image.thumbnail((1400, 1400))
    image.save(output_path)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in results if r.get("metrics")]
    return {
        "images": len(results), "scored": len(scored),
        "analysis_failures": sum(r["analysis_status"] != "completed" for r in results),
        "annotation_or_artifact_errors": sum("evaluation_error" in r for r in results),
        "wall_area_iou_mean": float(np.mean([r["metrics"]["wall"]["iou"] for r in scored]))
        if scored else None,
        "wall_area_f1_mean": float(np.mean([r["metrics"]["wall"]["f1"] for r in scored]))
        if scored else None,
        "wall_tolerant_area_f1_mean": float(np.mean([
            r["metrics"]["wall_tolerant"]["f1"] for r in scored])) if scored else None,
        "room_instance_f1_mean": float(np.mean([r["metrics"]["rooms"]["f1"] for r in scored]))
        if scored else None,
        "room_floor_iou_mean": float(np.mean([r["metrics"]["room_floor"]["iou"] for r in scored]))
        if scored else None,
    }


def evaluate(manifest_path: Path, report_path: Path, output: Path) -> dict[str, Any]:
    samples = load_samples(manifest_path, "development")
    baseline = json.loads(report_path.read_text())
    if baseline.get("split") != "development" or baseline.get("mode") != "automatic_unedited":
        raise ValueError("Only automatic development reports may be evaluated")
    if baseline.get("status") != "finished" or len(baseline["results"]) != len(samples):
        raise ValueError("Finish all development analyses before evaluation")
    by_id = {entry["id"]: entry for entry in baseline["results"]}
    if set(by_id) != {sample["id"] for sample in samples}:
        raise ValueError("Development sample IDs do not match the baseline")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "protocol": PROTOCOL, "manifest_sha256": checksum(manifest_path),
              "baseline_report_sha256": checksum(report_path), "evaluator_sha256": checksum(Path(__file__)),
              "baseline_git_commit": baseline["provenance"]["git_commit"],
              "ground_truth_independently_reviewed": False, "split": "development", "results": []}
    for sample in samples:
        result = by_id[sample["id"]]
        row = {"id": sample["id"], "style": sample.get("style", "unknown"),
               "analysis_status": result["status"], "image_sha256": sample["image_sha256"]}
        try:
            if result["image_sha256"] != sample["image_sha256"]:
                raise ValueError("Baseline image hash mismatch")
            annotation = manifest_path.parent / sample["annotation"]["path"]
            if checksum(annotation) != sample["annotation"]["sha256"]:
                raise ValueError("Annotation checksum mismatch")
            image_path = Path(sample["input_path"])
            with Image.open(image_path) as source:
                image_size = source.size
            truth = load_svg(annotation, *image_size)
            predicted = {"walls": [], "rooms": []}
            if result["status"] == "completed":
                directory = report_path.parent / result["artifact_directory"] / "analysis"
                model_path = directory / "floorplan.json"
                if checksum(model_path) != result["model_sha256"]:
                    raise ValueError("Model checksum mismatch")
                model = json.loads(model_path.read_text())
                with Image.open(directory / "debug/01_original_roi.png") as roi_image:
                    analysis_size = roi_image.size
                predicted = prediction_geometry(model, analysis_size, image_size)
            tolerance = 0.005 * math.hypot(*image_size)
            wall_p, wall_t = unary_union(predicted["walls"]), unary_union(truth["walls"])
            row["metrics"] = {
                "wall": area_scores(wall_p, wall_t),
                "wall_tolerant": area_scores(wall_p, wall_t, tolerance),
                "tolerance_pixels": tolerance,
                "rooms": match_rooms(predicted["rooms"], truth["rooms"]),
                "room_floor": area_scores(unary_union(predicted["rooms"]), unary_union(truth["rooms"])),
                "gt_polygon_repairs": truth["repaired"],
                "outdoor_rooms_excluded": len(truth["outdoor_rooms"]),
            }
            row["annotation_sha256"] = checksum(annotation)
            diagnostic_overlay(image_path, output / (result["artifact_directory"] + ".png"),
                               predicted, truth)
        except (ValueError, KeyError, OSError, ET.ParseError, GEOSException) as exc:
            # Invalid upstream geometry must remain visible as a failed sample,
            # rather than aborting the collection or silently changing labels.
            row.pop("metrics", None)
            row["evaluation_error"] = f"{type(exc).__name__}: {exc}"
        report["results"].append(row)
        write_json(output / "report.json", report)
    report["summary"] = summarize(report["results"])
    report["by_style"] = {style: summarize([r for r in report["results"] if r["style"] == style])
                          for style in sorted({r["style"] for r in report["results"]})}
    report["status"] = "finished"
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = evaluate(arguments.manifest.resolve(), arguments.baseline.resolve(), arguments.output.resolve())
    print(json.dumps(result["summary"], indent=2))
