from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from architecture_walkthrough.config import AppConfig, load_config
from architecture_walkthrough.geometry.floorplan import load_corrected_floorplan
from architecture_walkthrough.geometry.models import FloorPlanModel
from architecture_walkthrough.geometry.validation import (
    evaluate_quality,
    load_source_evidence,
    validate_reconstruction,
)
from architecture_walkthrough.geometry.wall_graph import enumerate_faces, match_faces_to_rooms
from architecture_walkthrough.pipeline import analyze_image, build_model, prepare_walkthrough_floorplan
from architecture_walkthrough.scene.simple_glb import export_simple_glb
from architecture_walkthrough.vision.overlay import write_analysis_overlay
from architecture_walkthrough.security.file_validation import create_job_dir, ensure_within_directory, validate_image_file

LOGGER = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"


def _bake_summary(report_path: Path) -> str:
    """One-line human summary of the Blender build report."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "GLB ready"
    if report.get("mode") == "none":
        return f"GLB ready — no bake (real-time lights), {report.get('duration_seconds', '?')}s"
    return (
        f"GLB ready — {report.get('mode')} bake: {report.get('samples')} spp, "
        f"{report.get('lightmap_px')}px lightmaps on {report.get('device', '?').upper()} "
        f"in {report.get('duration_seconds', '?')}s"
    )

ARTIFACT_WHITELIST = {
    "building.glb",
    "floorplan.json",
    "floorplan.raw.json",
    "floorplan.optimized.json",
    "floorplan.corrected.json",
    "validation_report.json",
    "analysis_overlay.svg",
    "analysis_overlay.png",
}


class JobRecord(BaseModel):
    job_id: str
    status: str
    message: str = ""
    glb_url: str | None = None
    glb_source: str | None = None  # "preview" (trimesh) | "blender"
    # Bumped whenever a new GLB is written; viewers append it to the artifact
    # URL so browser and loader caches can never serve stale model bytes.
    glb_version: int = 0
    optimized_json_url: str | None = None
    overlay_url: str | None = None
    validation_report_url: str | None = None
    quality_state: str | None = None
    quality_score: float | None = None
    ai_assist_attempted: bool = False
    ai_assist_succeeded: bool = False
    ai_assist_error: str | None = None


class LocalJobRunner:
    """Single-process job store; long work runs on daemon threads and the
    frontend polls GET /jobs/{id}. Swap for a queue if this outgrows one user."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create_job(self) -> JobRecord:
        job_dir = create_job_dir(self.config.paths.work_root)
        record = JobRecord(job_id=job_dir.name, status="created")
        with self._lock:
            self.jobs[record.job_id] = record
        self.save(record)
        return record

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id in self.jobs:
                return self.jobs[job_id]
        record_path = self.config.paths.work_root / job_id / "job.json"
        if record_path.exists():
            record = JobRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
            with self._lock:
                self.jobs[job_id] = record
            return record
        raise HTTPException(status_code=404, detail="job not found")

    def save(self, record: JobRecord) -> None:
        job_dir = self.config.paths.work_root / record.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")

    # -- background stages ---------------------------------------------------

    def start_analysis(
        self,
        record: JobRecord,
        image_path: Path,
        job_config: AppConfig,
        manual_scale: float | None,
        crop_rect: tuple[int, int, int, int] | None,
    ) -> None:
        record.status = "processing"
        self.save(record)
        thread = threading.Thread(
            target=self._run_analysis,
            args=(record, image_path, job_config, manual_scale, crop_rect),
            daemon=True,
        )
        thread.start()

    def _run_analysis(
        self,
        record: JobRecord,
        image_path: Path,
        job_config: AppConfig,
        manual_scale: float | None,
        crop_rect: tuple[int, int, int, int] | None,
    ) -> None:
        job_dir = self.config.paths.work_root / record.job_id
        try:
            model = analyze_image(
                image_path,
                job_dir,
                job_config,
                manual_scale=manual_scale,
                require_ai_success=False,
                crop_rect=crop_rect,
            )
            # Instant untextured preview so the model page works before the
            # Blender quality build is requested.
            export_simple_glb(model, job_dir / "building.glb")
            metadata = model.metadata
            record.ai_assist_attempted = bool(metadata.get("ai_assist_attempted"))
            record.ai_assist_succeeded = bool(metadata.get("ai_assist_succeeded"))
            record.ai_assist_error = metadata.get("ai_assist_error")
            record.quality_state = model.reconstruction.quality_state
            record.quality_score = model.reconstruction.quality_score
            record.status = "needs_review"
            record.message = "analysis complete; review the layout in the editor"
            record.glb_url = f"/jobs/{record.job_id}/artifacts/building.glb"
            record.glb_source = "preview"
            record.glb_version = int(time.time())
            record.optimized_json_url = f"/jobs/{record.job_id}/artifacts/floorplan.optimized.json"
            record.overlay_url = f"/jobs/{record.job_id}/artifacts/analysis_overlay.svg"
            record.validation_report_url = f"/jobs/{record.job_id}/artifacts/validation_report.json"
        except Exception as exc:  # surfaced via polling, never a 500 later
            LOGGER.exception("analysis failed for job %s", record.job_id)
            record.status = "failed"
            record.message = str(exc)
        self.save(record)

    def start_generation(self, record: JobRecord, force: bool, bake_mode: str | None) -> None:
        mode = bake_mode or self.config.bake.mode
        if mode == "none":
            preset = "no bake, real-time lights"
        else:
            samples = self.config.bake.final_samples if mode == "final" else self.config.bake.draft_samples
            lightmap = self.config.bake.final_lightmap_px if mode == "final" else self.config.bake.draft_lightmap_px
            preset = f"{samples} spp, {lightmap}px lightmaps"
        record.status = "generating"
        record.message = f"Blender build running — {mode} ({preset})"
        self.save(record)
        thread = threading.Thread(target=self._run_generation, args=(record, force, bake_mode), daemon=True)
        thread.start()

    def _run_generation(self, record: JobRecord, force: bool, bake_mode: str | None) -> None:
        job_dir = self.config.paths.work_root / record.job_id
        try:
            build_model(
                job_dir,
                job_dir / "building.glb",
                self.config,
                run_blender=True,
                force=force,
                bake_mode=bake_mode,
            )
            record.status = "model_generated"
            record.message = _bake_summary(job_dir / "building.bake.json")
            record.glb_url = f"/jobs/{record.job_id}/artifacts/building.glb"
            record.glb_source = "blender"
            record.glb_version = int(time.time())
        except ValueError as exc:  # quality gate
            record.status = "blocked"
            record.message = str(exc)
        except Exception as exc:
            LOGGER.exception("generation failed for job %s", record.job_id)
            record.status = "generation_failed"
            record.message = str(exc)
        self.save(record)


def create_app() -> FastAPI:
    config = load_config()
    runner = LocalJobRunner(config)
    app = FastAPI(title="Architecture Walkthrough Analysis")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/gemini-status")
    def gemini_status() -> dict[str, object]:
        return {
            "enabled": config.ai.gemini_enabled,
            "api_key_present": bool(os.getenv("GEMINI_API_KEY")),
            "model": config.ai.gemini_model,
            "available": config.ai.gemini_enabled and bool(os.getenv("GEMINI_API_KEY")),
        }

    @app.post("/jobs", response_model=JobRecord)
    async def create_job(
        file: UploadFile = File(...),
        use_gemini: bool = Form(True),
        manual_scale: float | None = Form(None),
        crop_x: int | None = Form(None),
        crop_y: int | None = Form(None),
        crop_width: int | None = Form(None),
        crop_height: int | None = Form(None),
    ) -> JobRecord:
        record = runner.create_job()
        job_dir = config.paths.work_root / record.job_id
        suffix = Path(file.filename or "").suffix.lower()
        upload_path = job_dir / f"upload{suffix}"
        with upload_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        try:
            validated = validate_image_file(upload_path, config.limits, file.content_type)
        except ValueError as exc:
            record.status = "rejected"
            record.message = str(exc)
            runner.save(record)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        safe_path = job_dir / validated.safe_filename
        upload_path.replace(safe_path)
        job_config = config.model_copy(deep=True)
        job_config.ai.gemini_enabled = use_gemini
        crop_rect = None
        if crop_x is not None and crop_y is not None and crop_width is not None and crop_height is not None:
            crop_rect = (crop_x, crop_y, crop_width, crop_height)
        runner.start_analysis(record, safe_path, job_config, manual_scale, crop_rect)
        return record

    @app.get("/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        return runner.get(job_id)

    def current_floorplan_path(job_dir: Path) -> Path:
        for name in ("floorplan.corrected.json", "floorplan.optimized.json", "floorplan.json"):
            path = job_dir / name
            if path.exists():
                return path
        raise HTTPException(status_code=404, detail="floorplan artifact not found")

    def source_image_path(job_dir: Path) -> Path:
        for floorplan_name in ("floorplan.corrected.json", "floorplan.optimized.json", "floorplan.json"):
            floorplan = job_dir / floorplan_name
            if not floorplan.exists():
                continue
            try:
                data = json.loads(floorplan.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            roi_image = data.get("metadata", {}).get("roi_image")
            if roi_image:
                roi_path = Path(roi_image)
                if roi_path.exists():
                    try:
                        return ensure_within_directory(job_dir, roi_path)
                    except ValueError:
                        continue
        for candidate in (job_dir / "debug" / "01_original_roi.png", job_dir / "debug" / "plan_roi.png"):
            if candidate.exists():
                return candidate
        for candidate in job_dir.glob("*.png"):
            return candidate
        raise HTTPException(status_code=404, detail="source image artifact not found")

    @app.get("/jobs/{job_id}/edit-data")
    def edit_data(job_id: str) -> dict[str, object]:
        runner.get(job_id)
        job_dir = ensure_within_directory(config.paths.work_root, config.paths.work_root / job_id)
        floorplan = current_floorplan_path(job_dir)
        model = load_corrected_floorplan(floorplan)
        return {
            "job_id": job_id,
            "corrected": floorplan.name == "floorplan.corrected.json",
            "image_url": f"/jobs/{job_id}/source-image",
            "model": model.model_dump(mode="json"),
        }

    @app.get("/jobs/{job_id}/source-image")
    def source_image(job_id: str) -> FileResponse:
        runner.get(job_id)
        job_dir = ensure_within_directory(config.paths.work_root, config.paths.work_root / job_id)
        path = source_image_path(job_dir)
        media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return FileResponse(path, media_type=media_type, filename=path.name)

    @app.post("/jobs/{job_id}/corrections")
    def corrections(job_id: str, correction: dict) -> dict[str, object]:
        record = runner.get(job_id)
        job_dir = config.paths.work_root / job_id
        path = job_dir / "floorplan.corrected.json"
        path.write_text(json.dumps(correction, indent=2), encoding="utf-8")
        model = load_corrected_floorplan(path)
        # Rooms are faces of the wall graph: regenerate from the edited walls
        # and openings so polygons never go stale; semantics carry via face
        # matching, and edits that orphan a room drop it.
        face_result = enumerate_faces(model.walls, model.doors, model.windows)
        if face_result.faces:
            regenerated = match_faces_to_rooms(face_result.faces, model.rooms)
            model = model.model_copy(update={"rooms": regenerated})
        evidence = load_source_evidence(job_dir, model)
        issues = validate_reconstruction(model, evidence)
        quality = evaluate_quality(model.model_copy(update={"validation_issues": issues}), evidence)
        model = model.model_copy(
            update={
                "validation_issues": issues,
                "reconstruction": model.reconstruction.model_copy(
                    update={"quality_score": quality.score, "quality_state": quality.state}
                ),
                "metadata": {
                    **model.metadata,
                    "correction_source": model.metadata.get("correction_source", "browser_editor"),
                    "quality_components": quality.components,
                },
            }
        )
        model.save_json(path)
        record.quality_state = quality.state
        record.quality_score = quality.score
        runner.save(record)
        report = {
            "quality_state": quality.state,
            "quality_score": quality.score,
            "components": quality.components,
            "issues": [issue.model_dump() for issue in issues],
        }
        (job_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        source = source_image_path(job_dir)
        if source.exists():
            write_analysis_overlay(source, model, job_dir / "analysis_overlay.svg", job_dir / "analysis_overlay.png")
        return {"status": "accepted", "model": model.model_dump(mode="json"), **report}

    @app.post("/jobs/{job_id}/validate-corrections")
    def validate_corrections(job_id: str) -> dict[str, object]:
        runner.get(job_id)
        job_dir = config.paths.work_root / job_id
        floorplan = job_dir / "floorplan.corrected.json"
        if not floorplan.exists():
            floorplan = job_dir / "floorplan.optimized.json"
        model = load_corrected_floorplan(floorplan)
        evidence = load_source_evidence(job_dir, model)
        issues = validate_reconstruction(model, evidence)
        quality = evaluate_quality(model.model_copy(update={"validation_issues": issues}), evidence)
        return {
            "quality_state": quality.state,
            "quality_score": quality.score,
            "components": quality.components,
            "issues": [issue.model_dump() for issue in issues],
        }

    @app.post("/jobs/{job_id}/generate-model", response_model=JobRecord)
    def generate_model(job_id: str, force: bool = False, bake_mode: str | None = None) -> JobRecord:
        record = runner.get(job_id)
        if record.status == "generating":
            raise HTTPException(status_code=409, detail="a Blender build is already running for this job")
        runner.start_generation(record, force=force, bake_mode=bake_mode)
        return record

    @app.post("/jobs/{job_id}/generate-walkthrough")
    def generate_walkthrough(job_id: str) -> dict[str, str]:
        runner.get(job_id)
        job_dir = config.paths.work_root / job_id
        output = job_dir / "walkthrough.floorplan.json"
        prepare_walkthrough_floorplan(current_floorplan_path(job_dir), output)
        return {"artifact": str(output)}

    @app.get("/jobs/{job_id}/artifacts")
    def artifacts(job_id: str) -> dict[str, list[str]]:
        runner.get(job_id)
        job_dir = config.paths.work_root / job_id
        return {"artifacts": [path.name for path in job_dir.glob("*") if path.is_file()]}

    @app.get("/jobs/{job_id}/raw-plan")
    def raw_plan(job_id: str) -> FileResponse:
        return download_artifact(job_id, "floorplan.raw.json")

    @app.get("/jobs/{job_id}/optimized-plan")
    def optimized_plan(job_id: str) -> FileResponse:
        return download_artifact(job_id, "floorplan.optimized.json")

    @app.get("/jobs/{job_id}/validation-report")
    def validation_report(job_id: str) -> FileResponse:
        return download_artifact(job_id, "validation_report.json")

    @app.get("/jobs/{job_id}/overlay")
    def overlay(job_id: str) -> FileResponse:
        return download_artifact(job_id, "analysis_overlay.svg")

    @app.get("/jobs/{job_id}/artifacts/{artifact_name}")
    def download_artifact(job_id: str, artifact_name: str) -> FileResponse:
        runner.get(job_id)
        if artifact_name not in ARTIFACT_WHITELIST:
            raise HTTPException(status_code=404, detail="artifact not found")
        job_dir = ensure_within_directory(config.paths.work_root, config.paths.work_root / job_id)
        artifact_path = ensure_within_directory(job_dir, job_dir / artifact_name)
        if not artifact_path.exists() or not artifact_path.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        media_type = "model/gltf-binary" if artifact_path.suffix.lower() == ".glb" else "application/json"
        if artifact_path.suffix.lower() == ".svg":
            media_type = "image/svg+xml"
        if artifact_path.suffix.lower() == ".png":
            media_type = "image/png"
        # Artifacts are replaced in place (building.glb); force revalidation so
        # a regenerated model is never served from the browser cache.
        return FileResponse(
            artifact_path,
            media_type=media_type,
            filename=artifact_name,
            headers={"Cache-Control": "no-cache"},
        )

    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    else:

        @app.get("/", response_class=HTMLResponse)
        def frontend_missing() -> str:
            return (
                "<h1>Architecture Walkthrough Analysis</h1>"
                "<p>The web app is not built yet. Run <code>npm install && npm run build</code> "
                "in <code>frontend/</code>, then restart the server. The JSON API is live.</p>"
            )

    return app


__all__ = ["FloorPlanModel", "JobRecord", "LocalJobRunner", "create_app"]
