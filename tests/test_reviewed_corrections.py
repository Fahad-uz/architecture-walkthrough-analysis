from __future__ import annotations

from pathlib import Path

import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image
from shapely.geometry import Polygon

import architecture_walkthrough.api.app as api_app
from architecture_walkthrough.api.app import JobRecord, create_app
from architecture_walkthrough.config import AppConfig, PathSettings
from architecture_walkthrough.geometry.models import (
    ArchitecturalElement,
    DoorOpening,
    FloorPlanModel,
    FurniturePlacement,
    Point2D,
    RoomPolygon,
    ValidationIssue,
    WallSegment,
)
from architecture_walkthrough.geometry.stair_voids import infer_stair_ceiling_voids
from architecture_walkthrough.geometry.wall_graph import enumerate_faces


def _open_living_plan() -> FloorPlanModel:
    """A closed bedroom beside a reviewed zone with a wide, unclosed edge."""
    walls = [
        ((0, 0), (3, 0)),
        ((3, 0), (3, 3)),
        ((3, 3), (0, 3)),
        ((0, 3), (0, 0)),
        ((3, 0), (8, 0)),
        ((8, 0), (8, 6)),
    ]
    return FloorPlanModel(
        walls=[
            WallSegment(
                id=f"wall_{index}",
                start=Point2D(x=start[0], y=start[1]),
                end=Point2D(x=end[0], y=end[1]),
            )
            for index, (start, end) in enumerate(walls)
        ],
        rooms=[
            RoomPolygon(
                id="bedroom",
                name="Bedroom",
                points=[Point2D(x=x, y=y) for x, y in [(0, 0), (3, 0), (3, 3), (0, 3)]],
            ),
            RoomPolygon(
                id="living",
                name="Living and dining",
                points=[Point2D(x=x, y=y) for x, y in [(3, 0), (8, 0), (8, 6), (3, 6)]],
            ),
        ],
        metadata={"geometry_origin": "reviewed_source_trace"},
    )


def _save_reviewed_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: FloorPlanModel,
    *, first_save: bool = False,
) -> tuple[str, Path]:
    monkeypatch.setattr(
        api_app,
        "load_config",
        lambda: AppConfig(paths=PathSettings(work_root=tmp_path)),
    )
    job_id = "reviewed-open-plan"
    job_dir = tmp_path / job_id
    (job_dir / "debug").mkdir(parents=True)
    Image.new("RGB", (320, 240), "white").save(job_dir / "debug" / "01_original_roi.png")
    (job_dir / "job.json").write_text(
        JobRecord(job_id=job_id, status="needs_review").model_dump_json(), encoding="utf-8"
    )
    model.save_json(job_dir / (
        "floorplan.optimized.json" if first_save else "floorplan.corrected.json"
    ))
    return job_id, job_dir


@pytest.mark.parametrize("edit", ["label", "furniture"])
def test_non_geometric_save_preserves_reviewed_open_living_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edit: str
) -> None:
    original = _open_living_plan()
    # The regression only matters when graph enumeration cannot recover this zone.
    assert len(enumerate_faces(original.walls).faces) == 1
    job_id, job_dir = _save_reviewed_job(tmp_path, monkeypatch, original)
    edited = original.model_copy(deep=True)
    if edit == "label":
        edited.rooms[1].name = "Family room"
    else:
        edited.furniture.append(
            FurniturePlacement(
                category="sofa", center=Point2D(x=6, y=2), width_m=1.8, depth_m=0.8
            )
        )
    # Array ordering and presentation metadata do not constitute a geometry change.
    edited.walls.reverse()
    edited.metadata = {}

    with TestClient(create_app()) as client:
        response = client.post(f"/jobs/{job_id}/corrections", json=edited.model_dump(mode="json"))
        assert response.status_code == 200, response.text
        corrected = FloorPlanModel.model_validate(response.json()["model"])
        persisted = FloorPlanModel.load_json(job_dir / "floorplan.corrected.json")
        assert corrected == persisted
        assert [(room.id, room.points) for room in corrected.rooms] == [
            (room.id, room.points) for room in original.rooms
        ]
        assert corrected.rooms[1].name == edited.rooms[1].name
        assert corrected.furniture == edited.furniture
        assert corrected.metadata["geometry_origin"] == "reviewed_source_trace"
        assert not corrected.metadata.get("reviewed_source_trace_invalidated")
        assert "reviewed_trace_invalidated" not in {
            issue["code"] for issue in response.json()["issues"]
        }
        assert client.get(f"/jobs/{job_id}/edit-data").json()["model"] == response.json()["model"]

    preview = trimesh.load(job_dir / "building.glb", force="scene")
    floor_volume = sum(
        mesh.volume for name, mesh in preview.geometry.items() if name.startswith("Floor_")
    )
    # 9 m² bedroom + 30 m² living zone at 0.1 m thickness; no bounding-box fallback slab.
    assert floor_volume == pytest.approx(3.9, rel=1e-6)


def test_moving_walls_rebuilds_rooms_and_invalidates_previous_source_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _open_living_plan()
    job_id, job_dir = _save_reviewed_job(tmp_path, monkeypatch, original)
    edited = original.model_copy(deep=True)
    # Move the bedroom's left side one metre inward, maintaining its closed topology.
    for wall in edited.walls:
        for point in (wall.start, wall.end):
            if point.x == 0:
                point.x = 1

    with TestClient(create_app()) as client:
        response = client.post(f"/jobs/{job_id}/corrections", json=edited.model_dump(mode="json"))
        validation = client.post(f"/jobs/{job_id}/validate-corrections")

    assert response.status_code == 200, response.text
    corrected = FloorPlanModel.load_json(job_dir / "floorplan.corrected.json")
    assert "geometry_origin" not in corrected.metadata
    assert corrected.metadata["reviewed_source_trace_invalidated"] is True
    assert [room.id for room in corrected.rooms] == ["bedroom"]
    bedroom = Polygon([(point.x, point.y) for point in corrected.rooms[0].points])
    assert bedroom.bounds == pytest.approx((1, 0, 3, 3))
    assert bedroom.area == pytest.approx(6)
    assert "reviewed_trace_invalidated" in {issue["code"] for issue in response.json()["issues"]}
    assert validation.status_code == 200
    assert "reviewed_trace_invalidated" in {issue["code"] for issue in validation.json()["issues"]}


def _inferred_open_living_plan() -> FloorPlanModel:
    model = _open_living_plan()
    model.rooms[1].evidence_source = "inferred_open_floor_boundary"
    model.rooms[1].confidence = 0.65
    model.metadata = {
        "floor_boundary_inferences": [{
            "room_id": "living", "room_name": "Living and dining", "area_m2": 30,
            "wall_a": "wall_2", "wall_b": "wall_5", "gap_length_m": 5,
            "boundary_start": [3, 6], "boundary_end": [8, 6],
            "physical_wall_added": False,
        }],
    }
    return model


@pytest.mark.parametrize("first_save", [False, True])
@pytest.mark.parametrize("edit", ["label", "furniture"])
def test_non_geometric_save_preserves_inferred_floor_and_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_save: bool, edit: str
) -> None:
    original = _inferred_open_living_plan()
    job_id, job_dir = _save_reviewed_job(
        tmp_path, monkeypatch, original, first_save=first_save
    )
    edited = original.model_copy(deep=True)
    if edit == "label":
        edited.rooms[1].name = "Family room"
    else:
        edited.furniture.append(
            FurniturePlacement(
                category="sofa", center=Point2D(x=6, y=2), width_m=1.8, depth_m=0.8
            )
        )
    edited.metadata = {"geometry_origin": "reviewed_source_trace"}
    edited.rooms[1].evidence_source = "reviewed_source_trace"
    edited.rooms[1].confidence = 1.0
    edited.validation_issues = []
    with TestClient(create_app()) as client:
        response = client.post(f"/jobs/{job_id}/corrections", json=edited.model_dump(mode="json"))
        validation = client.post(f"/jobs/{job_id}/validate-corrections")

    assert response.status_code == 200, response.text
    corrected = FloorPlanModel.load_json(job_dir / "floorplan.corrected.json")
    assert [room.points for room in corrected.rooms] == [room.points for room in original.rooms]
    assert corrected.rooms[1].name == edited.rooms[1].name
    assert corrected.furniture == edited.furniture
    assert corrected.rooms[1].evidence_source == "inferred_open_floor_boundary"
    assert corrected.rooms[1].confidence == 0.65
    assert corrected.metadata["floor_boundary_inferences"] == original.metadata["floor_boundary_inferences"]
    assert "geometry_origin" not in corrected.metadata
    for result in (response, validation):
        assert result.status_code == 200, result.text
        warnings = [issue for issue in result.json()["issues"] if issue["code"] == "inferred_open_floor_boundary"]
        assert len(warnings) == 1
        assert warnings[0]["element_id"] == "living"
        assert edited.rooms[1].name in warnings[0]["message"]
    preview = trimesh.load(job_dir / "building.glb", force="scene")
    floor_volume = sum(
        mesh.volume for name, mesh in preview.geometry.items() if name.startswith("Floor_")
    )
    assert floor_volume == pytest.approx(3.9, rel=1e-6)


@pytest.mark.parametrize("change", ["wall", "room", "opening"])
def test_geometry_change_discards_inferred_floor_and_keeps_review_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    original = _inferred_open_living_plan()
    job_id, job_dir = _save_reviewed_job(tmp_path, monkeypatch, original, first_save=True)
    edited = original.model_copy(deep=True)
    if change == "wall":
        for wall in edited.walls:
            for point in (wall.start, wall.end):
                if point.x == 0:
                    point.x = 1
    elif change == "room":
        edited.rooms[1].points[2].y = 5
    else:
        edited.doors.append(DoorOpening(
            wall_id="wall_0", center=Point2D(x=1.5, y=0), width_m=0.8,
        ))
    with TestClient(create_app()) as client:
        response = client.post(f"/jobs/{job_id}/corrections", json=edited.model_dump(mode="json"))
        validation = client.post(f"/jobs/{job_id}/validate-corrections")
        assert response.status_code == 200, response.text
        corrected = FloorPlanModel.load_json(job_dir / "floorplan.corrected.json")
        assert [room.id for room in corrected.rooms] == ["bedroom"]
        assert "floor_boundary_inferences" not in corrected.metadata
        assert corrected.metadata["floor_boundary_inferences_invalidated"] is True
        # Even a later client save which omits all provenance must keep the warning.
        corrected.metadata = {}
        subsequent = client.post(
            f"/jobs/{job_id}/corrections", json=corrected.model_dump(mode="json")
        )
    for result in (response, validation, subsequent):
        assert result.status_code == 200, result.text
        codes = {issue["code"] for issue in result.json()["issues"]}
        assert "inferred_open_floor_boundary_invalidated" in codes
        assert "inferred_open_floor_boundary" not in codes


@pytest.mark.parametrize("claimed_provenance", ["reviewed", "inferred"])
def test_request_cannot_promote_unreviewed_geometry_to_an_explicit_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, claimed_provenance: str
) -> None:
    original = _open_living_plan()
    original.metadata = {}
    job_id, job_dir = _save_reviewed_job(tmp_path, monkeypatch, original)
    claimed_metadata = (
        {"geometry_origin": "reviewed_source_trace"}
        if claimed_provenance == "reviewed" else _inferred_open_living_plan().metadata
    )
    submitted = original.model_copy(update={"metadata": claimed_metadata})

    with TestClient(create_app()) as client:
        response = client.post(f"/jobs/{job_id}/corrections", json=submitted.model_dump(mode="json"))

    assert response.status_code == 200, response.text
    corrected = FloorPlanModel.load_json(job_dir / "floorplan.corrected.json")
    assert [room.id for room in corrected.rooms] == ["bedroom"]
    assert "geometry_origin" not in corrected.metadata
    assert "floor_boundary_inferences" not in corrected.metadata


@pytest.mark.parametrize("first_save", [False, True])
def test_missing_source_room_warning_survives_saves_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_save: bool,
) -> None:
    original = _open_living_plan()
    missing_room = ValidationIssue(
        code="unreconstructed_labeled_room",
        severity="warning",
        message=(
            'Room "Study" is visible in the image but its floor boundary was not '
            "reconstructed. Review its wall gaps and draw the room boundary before export."
        ),
    )
    original.validation_issues = [missing_room]
    job_id, job_dir = _save_reviewed_job(
        tmp_path, monkeypatch, original, first_save=first_save,
    )
    edited = original.model_copy(deep=True)
    edited.rooms[0].name = "Main bedroom"
    edited.validation_issues = []
    edited.metadata = {}

    def assert_missing_room_warning(result: dict) -> None:
        assert [
            issue for issue in result["issues"]
            if issue["code"] == "unreconstructed_labeled_room"
        ] == [missing_room.model_dump(mode="json")]

    with TestClient(create_app()) as client:
        # Validate must also preserve the original analysis warning before a save.
        validation = client.post(f"/jobs/{job_id}/validate-corrections")
        assert validation.status_code == 200, validation.text
        assert_missing_room_warning(validation.json())

        for move_wall in (False, True):
            if move_wall:
                # A geometry edit cannot be taken as evidence that the missing
                # source label is now enclosed, since its OCR position is absent.
                for wall in edited.walls:
                    for point in (wall.start, wall.end):
                        if point.x == 0:
                            point.x = 1
            response = client.post(
                f"/jobs/{job_id}/corrections", json=edited.model_dump(mode="json"),
            )
            validation = client.post(f"/jobs/{job_id}/validate-corrections")
            for result in (response, validation):
                assert result.status_code == 200, result.text
                assert_missing_room_warning(result.json())
            persisted = FloorPlanModel.load_json(job_dir / "floorplan.corrected.json")
            assert [
                issue for issue in persisted.validation_issues
                if issue.code == "unreconstructed_labeled_room"
            ] == [missing_room]
            edited = persisted.model_copy(deep=True)
            edited.validation_issues = []
            edited.metadata = {}


@pytest.mark.parametrize("change", ["wall", "stair", "label"])
def test_saves_recompute_stair_voids_without_removing_manual_voids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    original = _open_living_plan()
    original.rooms[0].name = "Stair hall"
    original.special_elements = [ArchitecturalElement(
        id="stairs", kind="staircase", evidence_source="repetitive_parallel_treads",
        confidence=0.9,
        polygon=[Point2D(x=x, y=y) for x, y in [(0.2, 0.2), (2.8, 0.2), (2.8, 2.8), (0.2, 2.8)]],
    )]
    inferred = infer_stair_ceiling_voids(original.rooms, original.special_elements)
    assert inferred.room_ids == ["bedroom"]
    original.metadata.update({
        "ceiling_void_room_ids": ["living", *inferred.room_ids],
        "ceiling_void_inferences": inferred.inferences,
    })
    original.validation_issues = inferred.issues
    job_id, job_dir = _save_reviewed_job(tmp_path, monkeypatch, original, first_save=True)
    edited = original.model_copy(deep=True)
    edited.metadata = {}
    edited.validation_issues = []

    with TestClient(create_app()) as client:
        response = client.post(
            f"/jobs/{job_id}/corrections", json=edited.model_dump(mode="json"),
        )
        validation = client.post(f"/jobs/{job_id}/validate-corrections")
        for result in (response, validation):
            assert result.status_code == 200, result.text
            assert len([
                issue for issue in result.json()["issues"]
                if issue["code"] == "inferred_stair_ceiling_void"
            ]) == 1
        persisted = FloorPlanModel.load_json(job_dir / "floorplan.corrected.json")
        assert persisted.metadata["ceiling_void_room_ids"] == ["living", "bedroom"]

        edited = persisted.model_copy(deep=True)
        edited.metadata = {}
        edited.validation_issues = []
        if change == "wall":
            for wall in edited.walls:
                for point in (wall.start, wall.end):
                    if point.x == 0:
                        point.x = 1
        elif change == "stair":
            for point in edited.special_elements[0].polygon:
                point.x += 4
        else:
            edited.rooms[0].name = "Bedroom"
        response = client.post(
            f"/jobs/{job_id}/corrections", json=edited.model_dump(mode="json"),
        )
        validation = client.post(f"/jobs/{job_id}/validate-corrections")
        for result in (response, validation):
            assert result.status_code == 200, result.text
            assert "inferred_stair_ceiling_void" not in {
                issue["code"] for issue in result.json()["issues"]
            }
    corrected = FloorPlanModel.load_json(job_dir / "floorplan.corrected.json")
    assert corrected.metadata["ceiling_void_room_ids"] == ["living"]
    assert corrected.metadata["ceiling_void_inferences"] == []
