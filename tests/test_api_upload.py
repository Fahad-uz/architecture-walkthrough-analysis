from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from fastapi import HTTPException
from fastapi.testclient import TestClient

import architecture_walkthrough.api.app as api_app
from architecture_walkthrough.api.app import JobRecord, LocalJobRunner, create_app
from architecture_walkthrough.config import AISettings, AppConfig, LimitSettings, PathSettings
from architecture_walkthrough.geometry.models import FloorPlanModel
from architecture_walkthrough.pipeline import convert_image_to_glb


class _FakeAnalysisProcess:
    def __init__(
        self,
        *,
        release: threading.Event | None = None,
        exit_code: int = 1,
        timeout_immediately: bool = False,
        complete_on_start: bool = False,
        start_error: Exception | None = None,
    ) -> None:
        self.release = release
        self.planned_exit_code = exit_code
        self.timeout_immediately = timeout_immediately
        self.complete_on_start = complete_on_start
        self.start_error = start_error
        self.exitcode: int | None = None
        self.alive = False
        self.join_timeouts: list[float | None] = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.alive = not self.complete_on_start
        if self.complete_on_start:
            self.exitcode = self.planned_exit_code

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)
        if not self.alive:
            return
        if self.timeout_immediately and self.terminate_calls == 0:
            return
        if self.release is not None and self.release.wait(timeout=timeout):
            self.alive = False
            self.exitcode = self.planned_exit_code

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.alive = False
        self.exitcode = -15
        if self.release is not None:
            self.release.set()

    def kill(self) -> None:
        self.kill_calls += 1
        self.alive = False
        self.exitcode = -9
        if self.release is not None:
            self.release.set()


def _wait_for_job_status(
    runner: LocalJobRunner,
    job_id: str,
    expected: str,
    timeout: float = 2.0,
) -> JobRecord:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = runner.snapshot(job_id)
        if record.status == expected:
            return record
        time.sleep(0.01)
    raise AssertionError(
        f"job {job_id} did not reach {expected}; last status was "
        f"{runner.snapshot(job_id).status}"
    )


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
            JobRecord(
                job_id=job_id,
                status="needs_review",
                message="ready for generation",
            ).model_dump_json(indent=2),
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
    restored = client.get("/jobs/second-generation-job").json()
    assert restored["status"] == "needs_review"
    assert restored["message"] == "ready for generation"
    persisted = JobRecord.model_validate_json(
        (tmp_path / "second-generation-job" / "job.json").read_text(encoding="utf-8")
    )
    assert persisted.status == "needs_review"
    assert persisted.message == "ready for generation"


def test_analysis_capacity_is_bounded_without_leaving_an_orphan_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        api_app,
        "load_config",
        lambda: AppConfig(
            paths=PathSettings(work_root=tmp_path),
            limits=LimitSettings(max_concurrent_analyses=1),
        ),
    )
    release = threading.Event()
    fake = _FakeAnalysisProcess(release=release)
    app = create_app()
    monkeypatch.setattr(
        app.state.runner,
        "_new_analysis_process",
        lambda *_args, **_kwargs: fake,
    )
    image_path = tmp_path / "plan.png"
    Image.new("RGB", (120, 90), "white").save(image_path)
    image_bytes = image_path.read_bytes()
    image_path.unlink()

    with TestClient(app) as client:
        first = client.post(
            "/jobs",
            files={"file": ("plan.png", image_bytes, "image/png")},
        )
        directories_after_first = {path.name for path in tmp_path.iterdir()}
        second = client.post(
            "/jobs",
            files={"file": ("plan.png", image_bytes, "image/png")},
        )
        directories_after_second = {path.name for path in tmp_path.iterdir()}

        assert first.status_code == 200
        assert first.json()["status"] == "processing"
        assert second.status_code == 429
        assert second.headers["retry-after"] == "5"
        assert directories_after_second == directories_after_first

        release.set()
        first_job = first.json()["job_id"]
        _wait_for_job_status(app.state.runner, first_job, "failed")


def test_analysis_timeout_stops_worker_before_capacity_is_reused(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = LocalJobRunner(
        AppConfig(
            paths=PathSettings(work_root=tmp_path),
            limits=LimitSettings(
                processing_timeout_seconds=1,
                max_concurrent_analyses=1,
            ),
        )
    )
    timed_out = _FakeAnalysisProcess(timeout_immediately=True)
    next_worker = _FakeAnalysisProcess(complete_on_start=True)
    workers = iter((timed_out, next_worker))
    monkeypatch.setattr(
        runner,
        "_new_analysis_process",
        lambda *_args, **_kwargs: next(workers),
    )
    first = runner.create_job()
    first_image = tmp_path / first.job_id / "plan.png"
    first_image.write_bytes(b"image")

    runner.start_analysis(first, first_image, runner.config, None, None)
    terminal = _wait_for_job_status(runner, first.job_id, "failed")

    assert terminal.message == "analysis timed out after 1 seconds"
    assert timed_out.join_timeouts[0] == 1
    assert timed_out.terminate_calls == 1
    assert timed_out.kill_calls == 0
    assert not list((tmp_path / first.job_id).glob(".analysis.*"))

    second = runner.create_job()
    second_image = tmp_path / second.job_id / "plan.png"
    second_image.write_bytes(b"image")
    runner.start_analysis(second, second_image, runner.config, None, None)
    _wait_for_job_status(runner, second.job_id, "failed")
    runner.close()


def test_analysis_stop_failure_is_reported_without_escaping(tmp_path: Path) -> None:
    runner = LocalJobRunner(AppConfig(paths=PathSettings(work_root=tmp_path)))
    worker = _FakeAnalysisProcess()
    worker.start()

    def fail_terminate() -> None:
        raise OSError("process handle became unavailable")

    worker.terminate = fail_terminate  # type: ignore[method-assign]

    assert runner._stop_analysis_process(worker) is False


def test_analysis_capacity_stays_reserved_until_an_unstoppable_worker_exits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = LocalJobRunner(
        AppConfig(
            paths=PathSettings(work_root=tmp_path),
            limits=LimitSettings(
                processing_timeout_seconds=1,
                max_concurrent_analyses=1,
            ),
        )
    )
    release = threading.Event()
    worker = _FakeAnalysisProcess(
        release=release,
        timeout_immediately=True,
    )
    monkeypatch.setattr(
        runner,
        "_new_analysis_process",
        lambda *_args, **_kwargs: worker,
    )

    def cannot_stop(_process) -> bool:
        worker.timeout_immediately = False
        return False

    monkeypatch.setattr(runner, "_stop_analysis_process", cannot_stop)
    first = runner.create_job()
    first_image = tmp_path / first.job_id / "plan.png"
    first_image.write_bytes(b"image")
    runner.start_analysis(first, first_image, runner.config, None, None)

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        active = runner.snapshot(first.job_id)
        if "capacity remains reserved" in active.message:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("analysis monitor did not report the retained worker")

    second = runner.create_job()
    second_image = tmp_path / second.job_id / "plan.png"
    second_image.write_bytes(b"image")
    with pytest.raises(HTTPException) as raised:
        runner.start_analysis(second, second_image, runner.config, None, None)
    assert raised.value.status_code == 429
    assert runner.snapshot(first.job_id).status == "processing"

    release.set()
    _wait_for_job_status(runner, first.job_id, "failed", timeout=3)

    next_worker = _FakeAnalysisProcess(complete_on_start=True)
    monkeypatch.setattr(
        runner,
        "_new_analysis_process",
        lambda *_args, **_kwargs: next_worker,
    )
    runner.start_analysis(second, second_image, runner.config, None, None)
    _wait_for_job_status(runner, second.job_id, "failed")
    runner.close()


def test_analysis_worker_start_failure_restores_job_and_capacity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = LocalJobRunner(
        AppConfig(
            paths=PathSettings(work_root=tmp_path),
            limits=LimitSettings(max_concurrent_analyses=1),
        )
    )
    start_failure = _FakeAnalysisProcess(start_error=OSError("spawn unavailable"))
    next_worker = _FakeAnalysisProcess(complete_on_start=True)
    workers = iter((start_failure, next_worker))
    monkeypatch.setattr(
        runner,
        "_new_analysis_process",
        lambda *_args, **_kwargs: next(workers),
    )
    first = runner.create_job()
    first_image = tmp_path / first.job_id / "plan.png"
    first_image.write_bytes(b"image")

    with pytest.raises(HTTPException) as raised:
        runner.start_analysis(first, first_image, runner.config, None, None)

    assert raised.value.status_code == 503
    assert runner.snapshot(first.job_id).status == "created"
    assert not list((tmp_path / first.job_id).glob(".analysis.*"))

    second = runner.create_job()
    second_image = tmp_path / second.job_id / "plan.png"
    second_image.write_bytes(b"image")
    runner.start_analysis(second, second_image, runner.config, None, None)
    _wait_for_job_status(runner, second.job_id, "failed")
    runner.close()


def test_monitor_start_failure_tracks_and_reaps_a_worker_that_cannot_stop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = LocalJobRunner(
        AppConfig(
            paths=PathSettings(work_root=tmp_path),
            limits=LimitSettings(max_concurrent_analyses=1),
        )
    )
    release = threading.Event()
    worker = _FakeAnalysisProcess(release=release)
    monkeypatch.setattr(
        runner,
        "_new_analysis_process",
        lambda *_args, **_kwargs: worker,
    )
    monkeypatch.setattr(runner, "_stop_analysis_process", lambda _process: False)
    original_monitor = runner._monitor_analysis
    fallback_saw_tracked_operation = False

    def checked_monitor(record: JobRecord, operation) -> None:
        nonlocal fallback_saw_tracked_operation
        assert runner._analysis_operations.get(record.job_id) is operation
        fallback_saw_tracked_operation = True
        original_monitor(record, operation)

    monkeypatch.setattr(runner, "_monitor_analysis", checked_monitor)
    original_thread_start = threading.Thread.start

    def fail_analysis_monitor_start(thread: threading.Thread) -> None:
        if thread.name.startswith("analysis-monitor-"):
            raise RuntimeError("thread resources unavailable")
        original_thread_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_analysis_monitor_start)
    record = runner.create_job()
    image = tmp_path / record.job_id / "plan.png"
    image.write_bytes(b"image")

    runner.start_analysis(record, image, runner.config, None, None)

    active = runner.snapshot(record.job_id)
    assert active.status == "processing"
    assert "capacity remains reserved" in active.message
    assert runner._analysis_operations.get(record.job_id) is not None

    monkeypatch.setattr(threading.Thread, "start", original_thread_start)
    second = runner.create_job()
    second_image = tmp_path / second.job_id / "plan.png"
    second_image.write_bytes(b"image")
    with pytest.raises(HTTPException) as raised:
        runner.start_analysis(second, second_image, runner.config, None, None)
    assert raised.value.status_code == 429

    release.set()
    terminal = _wait_for_job_status(runner, record.job_id, "failed")

    assert fallback_saw_tracked_operation
    assert "monitoring could not start" in terminal.message
    assert not runner._analysis_operations
    assert not list((tmp_path / record.job_id).glob(".analysis.*"))

    next_worker = _FakeAnalysisProcess(complete_on_start=True)
    monkeypatch.setattr(
        runner,
        "_new_analysis_process",
        lambda *_args, **_kwargs: next_worker,
    )
    runner.start_analysis(second, second_image, runner.config, None, None)
    _wait_for_job_status(runner, second.job_id, "failed")
    runner.close()


def test_runner_shutdown_terminates_active_analysis_and_persists_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = LocalJobRunner(
        AppConfig(
            paths=PathSettings(work_root=tmp_path),
            limits=LimitSettings(max_concurrent_analyses=1),
        )
    )
    release = threading.Event()
    worker = _FakeAnalysisProcess(release=release)
    monkeypatch.setattr(
        runner,
        "_new_analysis_process",
        lambda *_args, **_kwargs: worker,
    )
    record = runner.create_job()
    image = tmp_path / record.job_id / "plan.png"
    image.write_bytes(b"image")
    runner.start_analysis(record, image, runner.config, None, None)

    runner.close()

    terminal = runner.snapshot(record.job_id)
    persisted = JobRecord.model_validate_json(
        (tmp_path / record.job_id / "job.json").read_text(encoding="utf-8")
    )
    assert worker.terminate_calls == 1
    assert terminal.status == "failed"
    assert terminal.message == (
        "analysis was interrupted by server shutdown; upload the plan again"
    )
    assert persisted == terminal
    assert not runner._analysis_operations
    assert not list((tmp_path / record.job_id).glob(".analysis.*"))


def test_completed_analysis_promotes_debug_and_artifacts_from_staging(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    staging = job_dir / ".analysis.revision"
    debug = staging / "debug"
    debug.mkdir(parents=True)
    (debug / "01_original_roi.png").write_bytes(b"source")
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json"))
    model = model.model_copy(
        update={
            "metadata": {
                **model.metadata,
                "roi_image": str(debug / "01_original_roi.png"),
            }
        }
    )
    for name in ("floorplan.json", "floorplan.raw.json", "floorplan.optimized.json"):
        model.save_json(staging / name)
    (staging / "validation_report.json").write_text(
        model.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (staging / "building.glb").write_bytes(b"glTF")
    (staging / "analysis_overlay.svg").write_text("<svg/>", encoding="utf-8")
    (staging / "analysis_overlay.png").write_bytes(b"png")

    promoted = api_app._promote_analysis_artifacts(staging, job_dir, "revision")

    assert (job_dir / "debug" / "01_original_roi.png").read_bytes() == b"source"
    assert (job_dir / "building.glb").read_bytes() == b"glTF"
    assert promoted.metadata["roi_image"] == str(
        job_dir.resolve() / "debug" / "01_original_roi.png"
    )
    assert not (staging / "debug").exists()


def test_terminal_state_persistence_failure_removes_promoted_analysis_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = LocalJobRunner(
        AppConfig(
            paths=PathSettings(work_root=tmp_path),
            limits=LimitSettings(max_concurrent_analyses=1),
        )
    )
    worker = _FakeAnalysisProcess(exit_code=0, complete_on_start=True)
    monkeypatch.setattr(
        runner,
        "_new_analysis_process",
        lambda *_args, **_kwargs: worker,
    )
    model = FloorPlanModel.load_json(Path("tests/fixtures/sample_floorplan.json"))

    def promote(_staging: Path, job_dir: Path, _revision: str) -> FloorPlanModel:
        (job_dir / "debug").mkdir()
        (job_dir / "debug" / "01_original_roi.png").write_bytes(b"debug")
        for name in api_app.ANALYSIS_PUBLIC_ARTIFACTS:
            (job_dir / name).write_bytes(b"promoted")
        return model

    monkeypatch.setattr(api_app, "_promote_analysis_artifacts", promote)
    original_persist = runner._persist_locked
    failed_once = False

    def fail_success_publication(updated: JobRecord) -> None:
        nonlocal failed_once
        if updated.status == "needs_review" and not failed_once:
            failed_once = True
            raise OSError("job state disk failure")
        original_persist(updated)

    monkeypatch.setattr(runner, "_persist_locked", fail_success_publication)
    record = runner.create_job()
    image = tmp_path / record.job_id / "plan.png"
    image.write_bytes(b"image")

    runner.start_analysis(record, image, runner.config, None, None)
    terminal = _wait_for_job_status(runner, record.job_id, "failed")

    assert "analysis publication failed" in terminal.message
    assert not (tmp_path / record.job_id / "debug").exists()
    for name in api_app.ANALYSIS_PUBLIC_ARTIFACTS:
        assert not (tmp_path / record.job_id / name).exists()
    persisted = JobRecord.model_validate_json(
        (tmp_path / record.job_id / "job.json").read_text(encoding="utf-8")
    )
    assert persisted.status == "failed"
    runner.close()


def test_spawned_analysis_process_publishes_a_complete_job(tmp_path: Path) -> None:
    config = AppConfig(
        paths=PathSettings(work_root=tmp_path),
        limits=LimitSettings(
            processing_timeout_seconds=30,
            max_concurrent_analyses=1,
        ),
        ai=AISettings(gemini_enabled=False),
    )
    config.ocr.enabled = False
    runner = LocalJobRunner(config)
    record = runner.create_job()
    job_dir = tmp_path / record.job_id
    image_path = job_dir / "plan.png"
    image = Image.new("RGB", (240, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 220, 160), outline="black", width=8)
    image.save(image_path)

    runner.start_analysis(record, image_path, config, None, None)
    terminal = _wait_for_job_status(
        runner,
        record.job_id,
        "needs_review",
        timeout=30,
    )

    assert terminal.glb_source == "preview"
    assert (job_dir / "building.glb").read_bytes()[:4] == b"glTF"
    assert (job_dir / "floorplan.optimized.json").is_file()
    assert (job_dir / "debug" / "01_original_roi.png").is_file()
    assert not list(job_dir.glob(".analysis.*"))
    runner.close()


def test_poll_snapshot_is_detached_from_cached_record(tmp_path: Path) -> None:
    runner = LocalJobRunner(AppConfig(paths=PathSettings(work_root=tmp_path)))
    record = runner.create_job()
    before = runner.snapshot(record.job_id)

    runner.publish(
        record,
        expected_status="created",
        status="processing",
        message="analysis started",
    )

    after = runner.snapshot(record.job_id)
    assert before.status == "created"
    assert before.message == ""
    assert after.status == "processing"
    assert after.message == "analysis started"


def test_poll_waits_for_complete_terminal_bundle_to_persist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = LocalJobRunner(AppConfig(paths=PathSettings(work_root=tmp_path)))
    record = runner.create_job()
    entered_persistence = threading.Event()
    release_persistence = threading.Event()
    original_persist = runner._persist_locked

    def blocked_persist(updated: JobRecord) -> None:
        if updated.status == "needs_review":
            entered_persistence.set()
            release_persistence.wait(timeout=5)
        original_persist(updated)

    monkeypatch.setattr(runner, "_persist_locked", blocked_persist)
    with ThreadPoolExecutor(max_workers=2) as executor:
        publishing = executor.submit(
            runner.publish,
            record,
            expected_status="created",
            status="needs_review",
            message="analysis complete",
            bump_glb_version=True,
            glb_url=f"/jobs/{record.job_id}/artifacts/building.glb",
            glb_source="preview",
        )
        assert entered_persistence.wait(timeout=1)
        polling = executor.submit(runner.snapshot, record.job_id)
        with pytest.raises(FutureTimeout):
            polling.result(timeout=0.05)
        release_persistence.set()
        publishing.result(timeout=2)
        snapshot = polling.result(timeout=2)

    assert snapshot.status == "needs_review"
    assert snapshot.message == "analysis complete"
    assert snapshot.glb_source == "preview"
    assert snapshot.glb_url and snapshot.glb_url.endswith("/building.glb")
    assert snapshot.glb_version > 0


def test_job_json_replace_failure_restores_memory_and_disk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = LocalJobRunner(AppConfig(paths=PathSettings(work_root=tmp_path)))
    record = runner.create_job()
    record_path = tmp_path / record.job_id / "job.json"
    original_bytes = record_path.read_bytes()
    original_replace = api_app.os.replace

    def fail_job_replace(source, target) -> None:
        if Path(target) == record_path:
            raise PermissionError("simulated locked job.json")
        original_replace(source, target)

    monkeypatch.setattr(api_app.os, "replace", fail_job_replace)

    with pytest.raises(PermissionError, match="simulated locked job.json"):
        runner.publish(
            record,
            expected_status="created",
            status="processing",
            message="analysis started",
        )

    assert runner.snapshot(record.job_id).status == "created"
    assert runner.snapshot(record.job_id).message == ""
    assert record_path.read_bytes() == original_bytes
    assert not list(record_path.parent.glob(".job.*.tmp"))


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
        stale_analysis = job_dir / ".analysis.abandoned"
        stale_analysis.mkdir()
        (stale_analysis / "partial.json").write_text("{}", encoding="utf-8")

        loaded = LocalJobRunner(AppConfig(paths=PathSettings(work_root=tmp_path))).get(job_id)
        persisted = JobRecord.model_validate_json(record_path.read_text(encoding="utf-8"))

        assert loaded.status == recovered
        assert persisted.status == recovered
        assert "interrupted" in loaded.message
        assert not stale_analysis.exists()


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
