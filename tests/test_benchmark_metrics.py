from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from shapely.geometry import box

MODULE = Path(__file__).resolve().parents[1] / "tools/benchmark/evaluate_structure.py"
sys.path.insert(0, str(MODULE.parent))
spec = importlib.util.spec_from_file_location("benchmark_metrics", MODULE)
assert spec and spec.loader
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
sys.path.pop(0)


def test_svg_coordinates_do_not_stretch_to_viewbox(tmp_path: Path) -> None:
    svg = tmp_path / "annotation.svg"
    svg.write_text('''<svg width="50" height="50" viewBox="0 0 50 50">
      <g class="Wall"><polygon points="10,10 20,10 20,40 10,40"/></g>
      <g class="Space Bedroom"><polygon points="20,10 40,10 40,40 20,40"/></g>
      <g class="Space Outdoor"><polygon points="0,0 5,0 5,5 0,5"/></g>
    </svg>''')
    actual = metrics.load_svg(svg, 100, 100)
    assert actual["walls"][0].bounds == (10, 10, 20, 40)
    assert len(actual["rooms"]) == len(actual["outdoor_rooms"]) == 1


def test_svg_inherited_transforms_and_polygon_transform(tmp_path: Path) -> None:
    svg = tmp_path / "annotation.svg"
    svg.write_text('''<svg><g transform="translate(10 20)">
      <g class="Wall" transform="scale(2)"><polygon points="0,0 5,0 5,10 0,10"/></g>
      <g class="Space Bedroom"><polygon transform="translate(5,0)"
       points="0,0 10,0 10,10 0,10"/></g></g></svg>''')
    actual = metrics.load_svg(svg, 100, 100)
    assert actual["walls"][0].bounds == (10, 20, 20, 40)
    assert actual["rooms"][0].bounds == (15, 20, 25, 30)


def test_prediction_inverse_crop_resize_and_y_flip() -> None:
    model = {"plan_roi": {"rect": {"x": 100, "y": 200, "width": 400, "height": 200}},
             "pixels_per_metre": 10, "walls": [
                 {"start": {"x": 0, "y": 0}, "end": {"x": 10, "y": 0}, "thickness_m": 1}
             ], "rooms": [{"points": [{"x": 0, "y": 0}, {"x": 10, "y": 0},
                                         {"x": 10, "y": 5}, {"x": 0, "y": 5}]}]}
    result = metrics.prediction_geometry(model, (200, 100), (1000, 1000))
    assert result["walls"][0].bounds == (100, 390, 300, 410)
    assert result["rooms"][0].bounds == (100, 300, 300, 400)


def test_metric_does_not_hide_shift_or_missed_wall() -> None:
    truth = box(0, 0, 10, 10)
    shifted = box(5, 0, 15, 10)
    score = metrics.area_scores(shifted, truth)
    assert score["precision"] == score["recall"] == score["f1"] == 0.5
    assert score["iou"] == pytest.approx(1 / 3)
    assert metrics.area_scores(box(30, 30, 40, 40), truth)["f1"] == 0
    tolerant = metrics.area_scores(shifted, truth, 5)
    assert tolerant["f1"] == 1
    assert tolerant["iou"] == score["iou"]


def test_duplicate_room_predictions_cannot_double_count_match() -> None:
    room = box(0, 0, 10, 10)
    result = metrics.match_rooms([room, room], [room])
    assert result["matched"] == 1
    assert result["false_positive"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 1


def test_merged_rooms_and_missing_predictions_are_penalized() -> None:
    truth = [box(0, 0, 10, 10), box(10, 0, 20, 10)]
    result = metrics.match_rooms([box(0, 0, 20, 10)], truth)
    assert result["matched"] == 1
    assert result["false_negative"] == 1
    assert result["matched_mean_iou"] == 0.5
    empty = metrics.match_rooms([], truth)
    assert empty["f1"] == 0
    assert empty["matched_mean_iou"] is None


def test_non_development_reports_are_rejected_without_reading_predictions(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(metrics, "load_samples", lambda *args: [])
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"split": "quarantine", "mode": "automatic_unedited"}))
    with pytest.raises(ValueError, match="Only automatic development"):
        metrics.evaluate(tmp_path / "manifest.json", report, tmp_path / "output")


def test_geos_error_records_failed_sample_and_continues_collection(tmp_path, monkeypatch):
    import json
    from PIL import Image
    from shapely.errors import GEOSException

    source = tmp_path / "plan.png"
    Image.new("RGB", (100, 100), "white").save(source)
    svg = tmp_path / "model.svg"
    svg.write_text('<svg><g class="Wall"><polygon points="0,0 10,0 10,100 0,100"/></g>'
                   '<g class="Space"><polygon points="10,0 100,0 100,100 10,100"/></g></svg>')
    samples = [{"id": name, "input_path": str(source), "image_sha256": "sourcehash",
                "annotation": {"path": "model.svg", "sha256": metrics.checksum(svg)}}
               for name in ("bad-overlay", "next-plan")]
    monkeypatch.setattr(metrics, "load_samples", lambda *args: samples)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "split": "development", "mode": "automatic_unedited", "status": "finished",
        "provenance": {"git_commit": "test"}, "results": [
            {"id": item["id"], "status": "failed", "image_sha256": "sourcehash",
             "artifact_directory": item["id"]} for item in samples],
    }))
    original = metrics.area_scores
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise GEOSException("AssertionFailedException: simulated invalid overlay")
        return original(*args, **kwargs)

    monkeypatch.setattr(metrics, "area_scores", fail_once)
    report = metrics.evaluate(manifest, baseline, tmp_path / "evaluation")
    assert report["status"] == "finished"
    assert len(report["results"]) == 2
    assert "GEOSException" in report["results"][0]["evaluation_error"]
    assert "metrics" not in report["results"][0]
    assert report["results"][1]["metrics"]["wall"]["f1"] == 0
    assert report["summary"]["analysis_failures"] == 2
    assert report["summary"]["annotation_or_artifact_errors"] == 1
    assert report["summary"]["scored"] == 1
