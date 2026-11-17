from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from architecture_walkthrough.geometry.models import BoundingBox, FloorPlanModel, PlanROI
from architecture_walkthrough.vision.image_preparation import preparation_metadata
from architecture_walkthrough.vision.perspective_rectification import (
    perspective_frame,
    rectified_preparation_metadata,
)

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


@pytest.mark.parametrize("metadata", [None, {}, {"image_preparation": None}])
def test_prediction_inverse_crop_resize_and_y_flip(metadata) -> None:
    model = {"plan_roi": {"rect": {"x": 100, "y": 200, "width": 400, "height": 200}},
             "pixels_per_metre": 10, "walls": [
                 {"start": {"x": 0, "y": 0}, "end": {"x": 10, "y": 0}, "thickness_m": 1}
             ], "rooms": [{"points": [{"x": 0, "y": 0}, {"x": 10, "y": 0},
                                         {"x": 10, "y": 5}, {"x": 0, "y": 5}]}]}
    if metadata is not None:
        model["metadata"] = metadata
    result = metrics.prediction_geometry(model, (200, 100), (1000, 1000))
    assert result["walls"][0].bounds == (100, 390, 300, 410)
    assert result["rooms"][0].bounds == (100, 300, 300, 400)


def prepared_prediction() -> dict:
    # A 200×120 source rotated clockwise, cropped at (10, 20) to 91×151,
    # then resized to 60×99. Integer rounding gives unequal axis scales.
    model = {
        "metadata": {"image_preparation": {
            "version": 1, "rotation_quarter_turns": 1,
            "source_width_px": 200, "source_height_px": 120,
            "prepared_width_px": 120, "prepared_height_px": 200,
            "analysis_width_px": 60, "analysis_height_px": 99,
            "source_to_analysis": [[0, -60 / 91, 110 * 60 / 91],
                                   [99 / 151, 0, -20 * 99 / 151], [0, 0, 1]],
            "analysis_to_source": [[0, 151 / 99, 20], [-91 / 60, 0, 110], [0, 0, 1]],
        }},
        "plan_roi": {"rect": {"x": 10, "y": 20, "width": 91, "height": 151},
                     "confidence": 1.0, "source": "manual"},
        "pixels_per_metre": 10,
        "walls": [{"start": {"x": 1, "y": 2}, "end": {"x": 4, "y": 6},
                   "thickness_m": 0.5}],
        "rooms": [{"points": [{"x": 1, "y": 1}, {"x": 4, "y": 1},
                               {"x": 4, "y": 6}, {"x": 1, "y": 6}]}],
    }
    return FloorPlanModel.model_validate(model).model_dump(mode="json")


def test_prediction_inverse_rotation_crop_actual_resize_and_y_flip() -> None:
    result = metrics.prediction_geometry(prepared_prediction(), (60, 99), (200, 120))

    def source_point(x, y):
        return 171 - y * 1510 / 99, 110 - x * 91 / 6

    # A diagonal wall makes transforming its entire buffered footprint
    # observably different from buffering its transformed centreline.
    expected_wall = Polygon([source_point(x, y) for x, y in
                             [(0.8, 2.15), (3.8, 6.15), (4.2, 5.85), (1.2, 1.85)]])
    expected_room = Polygon([source_point(x, y) for x, y in
                             [(1, 1), (4, 1), (4, 6), (1, 6)]])
    assert result["walls"][0].symmetric_difference(expected_wall).area == pytest.approx(0, abs=1e-10)
    assert result["rooms"][0].symmetric_difference(expected_room).area == pytest.approx(0, abs=1e-10)


def straightened_prediction(degrees: float = 7.5) -> dict:
    model = prepared_prediction()
    model["metadata"]["image_preparation"] = preparation_metadata(
        200, 120, 1, FloorPlanModel.model_validate(model).plan_roi, 60, 99, degrees,
    )
    return model


@pytest.mark.parametrize("degrees", [-15, -7.5, 7.5, 15])
def test_prediction_inverse_fine_rotation_preserves_complete_wall_footprint(degrees) -> None:
    model = straightened_prediction(degrees)
    result = metrics.prediction_geometry(model, (60, 99), (200, 120))
    metadata = model["metadata"]["image_preparation"]
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)

    def source_point(x, y):
        # Undo model-Y, actual per-axis resize, crop, expanded fine rotation,
        # then the quarter turn. This is independent of the recorded matrices.
        px, py = 10 + x * 10 * 91 / 60, 20 + (99 - y * 10) * 151 / 99
        dx = px - metadata["prepared_width_px"] / 2
        dy = py - metadata["prepared_height_px"] / 2
        qx = cosine * dx + sine * dy + 60
        qy = -sine * dx + cosine * dy + 100
        return qy, 120 - qx

    bounds = box(0, 0, 200, 120)
    expected_wall = Polygon([source_point(x, y) for x, y in
                             [(0.8, 2.15), (3.8, 6.15), (4.2, 5.85), (1.2, 1.85)]]).intersection(bounds)
    expected_room = Polygon([source_point(x, y) for x, y in
                             [(1, 1), (4, 1), (4, 6), (1, 6)]]).intersection(bounds)
    assert result["walls"][0].symmetric_difference(expected_wall).area == pytest.approx(0, abs=1e-10)
    assert result["rooms"][0].symmetric_difference(expected_room).area == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("degrees", [True, "5", None, -15.01, 15.01,
                                         float("nan"), float("inf")])
def test_prediction_rejects_invalid_fine_angle(degrees) -> None:
    model = straightened_prediction()
    model["metadata"]["image_preparation"]["straighten_degrees"] = degrees
    with pytest.raises(ValueError, match="Invalid image preparation transform"):
        metrics.prediction_geometry(model, (60, 99), (200, 120))


def test_prediction_rejects_missing_fine_angle_and_wrong_expanded_dimensions() -> None:
    model = straightened_prediction()
    del model["metadata"]["image_preparation"]["straighten_degrees"]
    with pytest.raises(ValueError, match="Invalid image preparation transform"):
        metrics.prediction_geometry(model, (60, 99), (200, 120))
    model = straightened_prediction()
    model["metadata"]["image_preparation"]["prepared_width_px"] = 120
    with pytest.raises(ValueError, match="Invalid image preparation transform"):
        metrics.prediction_geometry(model, (60, 99), (200, 120))


@pytest.mark.parametrize(("field", "value"), [
    ("version", 3),
    ("version", True),
    ("rotation_quarter_turns", 4),
    ("rotation_quarter_turns", True),
    ("source_width_px", 201),
    ("source_height_px", 0),
    ("prepared_width_px", 121),
    ("analysis_height_px", 100),
    ("analysis_width_px", 60.0),
    ("source_to_analysis", [[1, 0], [0, 1]]),
    ("source_to_analysis", [[1, 0, float("nan")], [0, 1, 0], [0, 0, 1]]),
    ("source_to_analysis", [[1, 0, 0], [0, 1, 0], [0.01, 0, 1]]),
    ("source_to_analysis", [[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
    ("source_to_analysis", "invalid"),
    ("analysis_to_source", [[0, 151 / 99, 20], [-91 / 60, 0, float("inf")], [0, 0, 1]]),
    ("analysis_to_source", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
])
def test_prediction_rejects_malformed_preparation(field, value) -> None:
    model = prepared_prediction()
    model["metadata"]["image_preparation"][field] = value
    with pytest.raises(ValueError, match="Invalid image preparation transform"):
        metrics.prediction_geometry(model, (60, 99), (200, 120))


def test_prediction_rejects_incomplete_preparation() -> None:
    model = prepared_prediction()
    del model["metadata"]["image_preparation"]["analysis_to_source"]
    with pytest.raises(ValueError, match="Invalid image preparation transform"):
        metrics.prediction_geometry(model, (60, 99), (200, 120))


def projective_prediction(degrees=7.5):
    cosine, sine = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    ow = math.ceil(120 * cosine + 200 * abs(sine) - 1e-9)
    oh = math.ceil(120 * abs(sine) + 200 * cosine - 1e-9)
    corners = []
    for u, v in [(0, 0), (1, 0), (1, 1), (0, 1)]:
        denominator = .1 * u + .2 * v + 1
        corners.append([ow * (.60 * u + .06 * v + .12) / denominator,
                        oh * (.04 * u + .60 * v + .12) / denominator])
    frame = perspective_frame(200, 120, {"corners": corners, "aspect_ratio": 1.5,
                                          "output_width_px": 120},
                              rotation_quarter_turns=1, straighten_degrees=degrees)
    roi = PlanROI(rect=BoundingBox(x=8, y=6, width=95, height=65), confidence=1,
                  padding_px=2, full_image_width_px=120, full_image_height_px=80)
    model = {"pixels_per_metre": 10, "plan_roi": roi.model_dump(mode="json"),
             "walls": [{"start": {"x": .7, "y": .6}, "end": {"x": 3.1, "y": 2.4},
                        "thickness_m": .4}],
             "rooms": [{"points": [{"x": x, "y": y} for x, y in
                                     [(.5, .3), (4, .3), (4, 3.4), (.5, 3.4)]]}],
             "metadata": {"image_preparation": rectified_preparation_metadata(frame, roi, 60, 41)}}
    return model


def independent_projective_source(x, y, degrees=7.5):
    # Model -> analysis -> actual padded ROI -> rectified unit rectangle.
    u, v = (8 + x * 10 * 95 / 60) / 120, (6 + (41 - y * 10) * 65 / 41) / 80
    cosine, sine = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    ow = math.ceil(120 * cosine + 200 * abs(sine) - 1e-9)
    oh = math.ceil(120 * abs(sine) + 200 * cosine - 1e-9)
    denominator = .1 * u + .2 * v + 1
    ox = ow * (.60 * u + .06 * v + .12) / denominator
    oy = oh * (.04 * u + .60 * v + .12) / denominator
    # Undo expanded straightening and clockwise quarter turn.
    qx = cosine * (ox - ow / 2) + sine * (oy - oh / 2) + 60
    qy = -sine * (ox - ow / 2) + cosine * (oy - oh / 2) + 100
    return qy, 120 - qx


@pytest.mark.parametrize("degrees", [-7.5, 0, 7.5])
def test_v3_maps_complete_projective_wall_footprint_and_room_after_crop_resize(degrees):
    model = projective_prediction(degrees)
    result = metrics.prediction_geometry(model, (60, 41), (200, 120))
    def source(x, y):
        return independent_projective_source(x, y, degrees)
    bounds = box(0, 0, 200, 120)
    # The normal to this 2.4 x 1.8 diagonal is (-.6,.8); half-thickness is .2.
    expected_wall = Polygon([source(x, y) for x, y in
                             [(.58, .76), (2.98, 2.56), (3.22, 2.24), (.82, .44)]]).intersection(bounds)
    expected_room = Polygon([source(x, y) for x, y in
                             [(.5, .3), (4, .3), (4, 3.4), (.5, 3.4)]]).intersection(bounds)
    assert result["walls"][0].symmetric_difference(expected_wall).area < 1e-9
    assert result["rooms"][0].symmetric_difference(expected_room).area < 1e-9


def test_v3_maps_room_holes_and_multipart_repaired_polygons():
    model = projective_prediction()
    outer = [(.5, .3), (4, .3), (4, 3.4), (.5, 3.4), (.5, .3)]
    hole = [(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)]
    # A zero-width bridge encodes a hole in the existing point-only room model;
    # make_valid retains the interior ring and an area-free bridge line.
    bridged = outer + hole + [outer[0]]
    bowtie = [(.5, .5), (4, 3), (4, .5), (.5, 3)]
    model["rooms"] = [{"points": [{"x": x, "y": y} for x, y in ring]}
                      for ring in [bridged, bowtie]]
    result = metrics.prediction_geometry(model, (60, 41), (200, 120))
    expected = Polygon([independent_projective_source(x, y) for x, y in outer],
                       [[independent_projective_source(x, y) for x, y in hole]])
    assert result["rooms"][0].symmetric_difference(expected).area < 1e-9
    assert result["rooms"][1].geom_type == "MultiPolygon"
    assert len(result["rooms"][1].geoms) == 2


def test_v3_clips_to_analysis_bounds_before_inverse_and_discards_unobserved_area():
    model = projective_prediction()
    model["rooms"] = [{"points": [{"x": x, "y": y} for x, y in
                                 [(-1e6, -1e6), (1e6, -1e6), (1e6, 1e6), (-1e6, 1e6)]]}]
    model["walls"] = [{"start": {"x": -100, "y": -100}, "end": {"x": -99, "y": -99},
                       "thickness_m": 1}]
    result = metrics.prediction_geometry(model, (60, 41), (200, 120))
    expected = Polygon([independent_projective_source(x, y) for x, y in
                        [(0, 0), (6, 0), (6, 4.1), (0, 4.1)]])
    assert result["rooms"][0].symmetric_difference(expected).area < 1e-9
    assert result["walls"][0].is_empty


@pytest.mark.parametrize("path,value", [
    (("source_width_px",), 201), (("analysis_width_px",), 61),
    (("prepared_width_px",), 121), (("rotation_quarter_turns",), True),
    (("straighten_degrees",), float("nan")), (("straighten_degrees",), "7.5"),
    (("rectification", "aspect_ratio"), 1.6),
    (("rectification", "aspect_ratio_source"), "inferred_from_photo"),
    (("rectification", "oriented_width_px"), 1),
    (("rectification", "oriented_width_px"), True),
    (("rectification", "corner_coordinate_frame"), "source_edges"),
    (("rectification", "corners"), [[0, 0]] * 4),
    (("rectification", "oriented_to_prepared"), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    (("prepared_roi", "x"), -1), (("prepared_roi", "width"), 0),
    (("prepared_roi", "height"), float("inf")),
    (("source_to_analysis",), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    (("source_to_analysis",), [[True, 0, 0], [0, 1, 0], [0, 0, 1]]),
    (("source_to_analysis",), [["1", 0, 0], [0, 1, 0], [0, 0, 1]]),
    (("analysis_to_source",), [[1, 0, 0], [0, 1, 0], [1, 0, -30]]),
    (("analysis_to_source",), [[1, 0, 0], [0, 1, 0], [0, 0, float("nan")]]),
])
def test_v3_rejects_inconsistent_malformed_and_pole_records(path, value):
    model = projective_prediction()
    target = model["metadata"]["image_preparation"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="Invalid image preparation transform"):
        metrics.prediction_geometry(model, (60, 41), (200, 120))


@pytest.mark.parametrize("field", ["rectification", "prepared_roi", "straighten_degrees",
                                    "source_to_analysis", "analysis_to_source"])
def test_v3_rejects_missing_required_facts(field):
    model = projective_prediction()
    del model["metadata"]["image_preparation"][field]
    with pytest.raises(ValueError, match="Invalid image preparation transform"):
        metrics.prediction_geometry(model, (60, 41), (200, 120))


def test_v3_rejects_roi_model_mismatch_and_extra_rectification_facts():
    model = projective_prediction()
    model["plan_roi"]["rect"]["x"] += 1
    with pytest.raises(ValueError, match="Model ROI"):
        metrics.prediction_geometry(model, (60, 41), (200, 120))
    model = projective_prediction()
    model["metadata"]["image_preparation"]["rectification"]["unexpected"] = 1
    with pytest.raises(ValueError, match="unexpected"):
        metrics.prediction_geometry(model, (60, 41), (200, 120))


def test_projective_geometry_rejects_poles_before_division():
    matrix = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, -.5]])
    with pytest.raises(ValueError, match="pole"):
        metrics._projective_geometry(box(0, 0, 1, 1), matrix)


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
