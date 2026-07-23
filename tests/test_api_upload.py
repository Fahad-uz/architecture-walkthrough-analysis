from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

import architecture_walkthrough.api.app as api_app
from architecture_walkthrough.api.app import JobRecord, LocalJobRunner, create_app
from architecture_walkthrough.config import AISettings, AppConfig, LimitSettings, PathSettings
from architecture_walkthrough.geometry.models import FloorPlanModel
from architecture_walkthrough.pipeline import convert_image_to_glb


def test_image_upload_pipeline_generates_glb_artifact(tmp_path: Path) -> None:
    image_path = tmp_path / "plan.png"
    image = Image.new("RGB", (240, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 220, 160), outline="black", width=8)
    image.save(image_path)

    output = convert_image_to_glb(
        image_path,
        tmp_path / "building.glb",
        AppConfig(ai=AISettings(gemini_enabled=False)),
        work_dir=tmp_path / "work",
    )
    assert output.exists()
    assert output.stat().st_size > 0


def test_app_exposes_upload_and_download_routes() -> None:
    app = create_app()
    routes = {getattr(route, "path", "") for route in app.routes}
    assert "/jobs" in routes
    assert "/jobs/{job_id}/edit-data" in routes
    assert "/jobs/{job_id}/source-image" in routes
    assert "/jobs/{job_id}/artifacts/{artifact_name}" in routes
    assert "/jobs/{job_id}/generate-model" in routes
    assert "/gemini-status" in routes


def test_built_frontend_supports_direct_navigation_to_client_route(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frontend_dist = tmp_path / "dist"
    frontend_dist.mkdir()
    (frontend_dist / "index.html").write_text('<div id="root"></div>', encoding="utf-8")
    monkeypatch.setattr(api_app, "FRONTEND_DIST", frontend_dist)

    response = TestClient(create_app()).get("/jobs/example/edit")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<div id="root"></div>' in response.text
    assert TestClient(create_app()).get("/assets/missing.js").status_code == 404


def test_oversized_upload_is_stopped_while_streaming(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "load_config",
        lambda: AppConfig(
            paths=PathSettings(work_root=tmp_path),
            limits=LimitSettings(max_upload_mb=1),
        ),
    )

    response = TestClient(create_app()).post(
        "/jobs",
        files={"file": ("too-large.png", b"x" * (1024 * 1024 + 1), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "upload exceeds configured size limit"
    assert not list(tmp_path.glob("*/upload*"))


def test_total_multipart_body_is_capped_before_extra_files_are_parsed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        api_app,
        "load_config",
        lambda: AppConfig(
            paths=PathSettings(work_root=tmp_path),
            limits=LimitSettings(max_upload_mb=1),
        ),
    )

    response = TestClient(create_app()).post(
        "/jobs",
        files={
            "file": ("plan.png", b"x", "image/png"),
            "ignored": ("payload.bin", b"x" * (2 * 1024 * 1024), "application/octet-stream"),
        },
    )

    assert response.status_code == 413
    assert not list(tmp_path.iterdir())


def test_unexpected_multipart_fields_are_rejected_before_job_creation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        api_app,
        "load_config",
        lambda: AppConfig(paths=PathSettings(work_root=tmp_path)),
    )

    response = TestClient(create_app()).post(
        "/jobs",
        files={
            "file": ("plan.png", b"x", "image/png"),
            "ignored": ("payload.bin", b"x", "application/octet-stream"),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "unexpected or duplicate multipart field"
    assert not list(tmp_path.iterdir())


def test_artifact_set_rolls_back_every_target_when_commit_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first_target = tmp_path / "first.json"
    second_target = tmp_path / "second.glb"
    first_temporary = tmp_path / ".first.new"
    second_temporary = tmp_path / ".second.new"
    first_target.write_bytes(b"old first")
    second_target.write_bytes(b"old second")
    first_temporary.write_bytes(b"new first")
    second_temporary.write_bytes(b"new second")
    original_replace = Path.replace

    def fail_second_replace(path: Path, target: Path):
        if path == second_temporary:
            raise PermissionError("simulated locked target")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_second_replace)

    with pytest.raises(PermissionError, match="simulated locked target"):
        api_app._commit_artifact_set(
            [
                (first_temporary, first_target),
                (second_temporary, second_target),
            ],
            "rollback",
        )

    assert first_target.read_bytes() == b"old first"
    assert second_target.read_bytes() == b"old second"
    assert not list(tmp_path.glob(".*.rollback.bak"))


def test_backup_cleanup_failure_does_not_turn_a_committed_artifact_into_failure(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    target = tmp_path / "building.glb"
    temporary = tmp_path / ".building.new.glb"
    target.write_bytes(b"old model")
    temporary.write_bytes(b"new model")
    original_unlink = Path.unlink

    def lock_backup(path: Path, *args, **kwargs):
        if path.name.endswith(".cleanup.bak"):
            raise PermissionError("simulated antivirus lock")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", lock_backup)

    api_app._commit_artifact_set([(temporary, target)], "cleanup")

    assert target.read_bytes() == b"new model"
    assert (tmp_path / ".building.glb.cleanup.bak").read_bytes() == b"old model"
    assert "cleanup incomplete after commit" in caplog.text


def test_saving_corrections_replaces_stale_blender_model_with_current_preview(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        api_app,
        "load_config",
        lambda: AppConfig(paths=PathSettings(work_root=tmp_path)),
    )
    job_id = "correction-job"
    job_dir = tmp_path / job_id
    (job_dir / "debug").mkdir(parents=True)
    Image.new("RGB", (320, 240), "white").save(job_dir / "debug" / "01_original_roi.png")
    (job_dir / "building.glb").write_bytes(b"stale blender bytes")
    (job_dir / "job.json").write_text(
        JobRecord(
            job_id=job_id,
            status="model_generated",
            glb_source="blender",
            glb_version=7,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json"))

    client = TestClient(create_app())
    response = client.post(
        f"/jobs/{job_id}/corrections",
        json=model.model_dump(mode="json"),
    )

    assert response.status_code == 200, response.text
    record = client.get(f"/jobs/{job_id}").json()
    assert record["status"] == "needs_review"
    assert record["glb_source"] == "preview"
    assert record["glb_version"] > 7
    assert (job_dir / "building.glb").read_bytes()[:4] == b"glTF"


def test_invalid_correction_does_not_overwrite_last_good_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "load_config",
        lambda: AppConfig(paths=PathSettings(work_root=tmp_path)),
    )
    job_id = "invalid-correction-job"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        JobRecord(job_id=job_id, status="needs_review").model_dump_json(indent=2),
        encoding="utf-8",
    )
    good_path = job_dir / "floorplan.corrected.json"
    good_path.write_text('{"last_good": true}', encoding="utf-8")

    response = TestClient(create_app()).post(
        f"/jobs/{job_id}/corrections",
        json={"coordinate_system": "metres", "unknown": True},
    )

    assert response.status_code == 422
    assert good_path.read_text(encoding="utf-8") == '{"last_good": true}'


def test_unrenderable_correction_keeps_last_good_json_and_glb(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "load_config",
        lambda: AppConfig(paths=PathSettings(work_root=tmp_path)),
    )
    job_id = "unrenderable-correction-job"
    job_dir = tmp_path / job_id
    (job_dir / "debug").mkdir(parents=True)
    Image.new("RGB", (320, 240), "white").save(job_dir / "debug" / "01_original_roi.png")
    (job_dir / "job.json").write_text(
        JobRecord(job_id=job_id, status="needs_review").model_dump_json(indent=2),
        encoding="utf-8",
    )
    good_json = job_dir / "floorplan.corrected.json"
    good_json.write_text('{"last_good": true}', encoding="utf-8")
    good_glb = job_dir / "building.glb"
    good_glb.write_bytes(b"last good model")

    def fail_export(*_args, **_kwargs):
        raise ValueError("preview geometry is not renderable")

    monkeypatch.setattr(api_app, "export_simple_glb", fail_export)
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json"))
    response = TestClient(create_app()).post(
        f"/jobs/{job_id}/corrections",
        json=model.model_dump(mode="json"),
    )

    assert response.status_code == 422
    assert good_json.read_text(encoding="utf-8") == '{"last_good": true}'
    assert good_glb.read_bytes() == b"last good model"
    assert not list(job_dir.glob(".*.*.glb"))


def test_failed_blender_generation_keeps_last_good_preview(tmp_path: Path, monkeypatch) -> None:
    config = AppConfig(paths=PathSettings(work_root=tmp_path))
    runner = LocalJobRunner(config)
    record = JobRecord(job_id="generation-job", status="generating", glb_version=3)
    job_dir = tmp_path / record.job_id
    job_dir.mkdir(parents=True)
    good_glb = job_dir / "building.glb"
    good_glb.write_bytes(b"last good preview")

    def fail_after_partial_write(_floorplan, output_glb, *_args, **_kwargs):
        output_glb.write_bytes(b"partial blender output")
        raise RuntimeError("Blender crashed")

    monkeypatch.setattr(api_app, "build_model", fail_after_partial_write)
    runner._run_generation(record, force=False, bake_mode="none")

    assert good_glb.read_bytes() == b"last good preview"
    assert record.status == "generation_failed"
    assert record.glb_version == 3
    assert not list(job_dir.glob(".building.*"))


def test_concurrent_generation_requests_claim_job_once(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "load_config",
        lambda: AppConfig(paths=PathSettings(work_root=tmp_path)),
    )
    job_id = "generation-race-job"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        JobRecord(job_id=job_id, status="needs_review").model_dump_json(indent=2),
        encoding="utf-8",
    )
    release = threading.Event()
    monkeypatch.setattr(
        LocalJobRunner,
        "_run_generation",
        lambda *_args, **_kwargs: release.wait(timeout=5),
    )
    client = TestClient(create_app())

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(
            executor.map(
                lambda _index: client.post(f"/jobs/{job_id}/generate-model?bake_mode=none"),
                range(2),
            )
        )
    release.set()

    assert sorted(response.status_code for response in responses) == [200, 409]


def test_blender_generation_capacity_is_bounded_across_jobs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "load_config",
        lambda: AppConfig(
            paths=PathSettings(work_root=tmp_path),
            limits=LimitSettings(max_concurrent_generations=1),
        ),
    )
    for job_id in ("first-generation-job", "second-generation-job"):
        job_dir = tmp_path / job_id
        job_dir.mkdir()
        (job_dir / "job.json").write_text(
            JobRecord(job_id=job_id, status="needs_review").model_dump_json(indent=2),
            encoding="utf-8",
        )
    release = threading.Event()
    monkeypatch.setattr(
        LocalJobRunner,
        "_run_generation",
        lambda *_args, **_kwargs: release.wait(timeout=5),
    )
    client = TestClient(create_app())

    first = client.post("/jobs/first-generation-job/generate-model?bake_mode=none")
    second = client.post("/jobs/second-generation-job/generate-model?bake_mode=none")
    release.set()

    assert first.status_code == 200
    assert second.status_code == 429
    assert client.get("/jobs/second-generation-job").json()["status"] == "needs_review"


def test_interrupted_jobs_recover_to_safe_persisted_states(tmp_path: Path) -> None:
    expected = {
        "processing": "failed",
        "generating": "generation_failed",
        "saving_corrections": "needs_review",
    }

    for index, (interrupted, recovered) in enumerate(expected.items()):
        job_id = f"restart-job-{index}"
        job_dir = tmp_path / job_id
        job_dir.mkdir()
        record_path = job_dir / "job.json"
        record_path.write_text(
            JobRecord(job_id=job_id, status=interrupted).model_dump_json(indent=2),
            encoding="utf-8",
        )

        loaded = LocalJobRunner(AppConfig(paths=PathSettings(work_root=tmp_path))).get(job_id)
        persisted = JobRecord.model_validate_json(record_path.read_text(encoding="utf-8"))

        assert loaded.status == recovered
        assert persisted.status == recovered
        assert "interrupted" in loaded.message


def test_encoded_parent_job_id_cannot_escape_work_root(tmp_path: Path, monkeypatch) -> None:
    work_root = tmp_path / "outputs"
    work_root.mkdir()
    (tmp_path / "job.json").write_text(
        JobRecord(job_id="parent", status="needs_review").model_dump_json(indent=2),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        api_app,
        "load_config",
        lambda: AppConfig(paths=PathSettings(work_root=work_root)),
    )

    response = TestClient(create_app()).get("/jobs/%2E%2E")

    assert response.status_code == 404


def test_job_id_has_one_case_sensitive_canonical_spelling(tmp_path: Path, monkeypatch) -> None:
    job_id = "abcdef0123456789abcdef0123456789"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / "job.json").write_text(
        JobRecord(job_id=job_id, status="needs_review").model_dump_json(indent=2),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        api_app,
        "load_config",
        lambda: AppConfig(paths=PathSettings(work_root=tmp_path)),
    )
    client = TestClient(create_app())

    assert client.get(f"/jobs/{job_id}").status_code == 200
    assert client.get(f"/jobs/{job_id.upper()}").status_code == 404


def test_correction_that_opens_every_room_drops_stale_room_polygons(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "load_config",
        lambda: AppConfig(paths=PathSettings(work_root=tmp_path)),
    )
    job_id = "open-room-correction-job"
    job_dir = tmp_path / job_id
    (job_dir / "debug").mkdir(parents=True)
    Image.new("RGB", (320, 240), "white").save(job_dir / "debug" / "01_original_roi.png")
    (job_dir / "job.json").write_text(
        JobRecord(job_id=job_id, status="needs_review").model_dump_json(indent=2),
        encoding="utf-8",
    )
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json"))
    model = model.model_copy(update={"walls": model.walls[:1]})

    response = TestClient(create_app()).post(
        f"/jobs/{job_id}/corrections",
        json=model.model_dump(mode="json"),
    )

    assert response.status_code == 200, response.text
    assert response.json()["model"]["rooms"] == []


def test_corrections_are_rejected_while_analysis_is_running(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "load_config",
        lambda: AppConfig(paths=PathSettings(work_root=tmp_path)),
    )
    job_id = "processing-job"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        JobRecord(job_id=job_id, status="processing").model_dump_json(indent=2),
        encoding="utf-8",
    )
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json"))

    response = TestClient(create_app()).post(
        f"/jobs/{job_id}/corrections",
        json=model.model_dump(mode="json"),
    )

    assert response.status_code == 409
