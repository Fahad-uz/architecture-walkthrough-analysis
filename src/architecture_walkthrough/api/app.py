from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from architecture_walkthrough.config import AppConfig, load_config
from architecture_walkthrough.geometry.floorplan import load_corrected_floorplan
from architecture_walkthrough.geometry.models import FloorPlanModel, ValidationIssue
from architecture_walkthrough.geometry.validation import (
    evaluate_quality,
    load_source_evidence,
    validate_reconstruction,
)
from architecture_walkthrough.geometry.wall_graph import enumerate_faces, match_faces_to_rooms
from architecture_walkthrough.pipeline import analyze_image, build_model, prepare_walkthrough_floorplan
from architecture_walkthrough.scene.simple_glb import export_simple_glb
from architecture_walkthrough.vision.overlay import write_analysis_overlay
from architecture_walkthrough.security.file_validation import (
    create_job_dir,
    ensure_within_directory,
    validate_image_file,
    validate_job_id,
)
from architecture_walkthrough.walkthrough.camera_animation import waypoints_from_points
from architecture_walkthrough.walkthrough.path_planner import manual_or_auto_waypoints

LOGGER = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"
MULTIPART_OVERHEAD_BYTES = 256 * 1024


class RequestBodyLimitMiddleware:
    """Reject oversized mutation bodies before multipart parsing or spooling."""

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                content_length = int(raw_length)
            except ValueError:
                content_length = 0
            if content_length > self.max_body_bytes:
                await Response("request body exceeds configured limit", status_code=413)(
                    scope, receive, send
                )
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > self.max_body_bytes:
                await Response("request body exceeds configured limit", status_code=413)(
                    scope, receive, send
                )
                return
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, replay, send)


class SPAStaticFiles(StaticFiles):
    """Serve the React entry point for client-side routes, but not missing assets."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404 or Path(path).suffix:
                raise
        else:
            if response.status_code != 404 or Path(path).suffix:
                return response
        return await super().get_response("index.html", scope)


def _next_glb_version(previous: int = 0) -> int:
    """Return a browser-safe, strictly increasing model revision."""

    return max(int(time.time() * 1000), previous + 1)


async def _save_upload_with_limit(upload: UploadFile, destination: Path, max_bytes: int) -> None:
    """Stream an upload to disk without allowing it to exceed the configured limit."""

    written = 0
    try:
        with destination.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError("upload exceeds configured size limit")
                handle.write(chunk)
    finally:
        await upload.close()


def _semantic_review_issues(model: FloorPlanModel) -> list[ValidationIssue]:
    """Keep AI review prompts when deterministic validation is recomputed."""

    existing = [issue for issue in model.validation_issues if issue.code.startswith("gemini_")]
    if existing:
        return existing
    recovered: list[ValidationIssue] = []
    for warning in model.metadata.get("sanity_warnings", []):
        if not isinstance(warning, dict):
            continue
        kind = str(warning.get("kind") or "review")
        description = str(warning.get("description") or "Review this area against the source plan")
        x, y = warning.get("x"), warning.get("y")
        location = f" (at {float(x):.2f}, {float(y):.2f} normalized)" if isinstance(x, (int, float)) and isinstance(y, (int, float)) else ""
        recovered.append(
            ValidationIssue(
                code=f"gemini_{kind}",
                severity="warning",
                message=f"{description}{location}",
            )
        )
    return recovered


def _carried_local_review_issues(model: FloorPlanModel) -> list[ValidationIssue]:
    """Keep image-derived ambiguity that cannot be recomputed from JSON alone."""

    return [issue for issue in model.validation_issues if issue.code == "ambiguous_opening"]


def _deduplicate_issues(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    deduplicated: list[ValidationIssue] = []
    seen: set[tuple[str, str]] = set()
    for issue in issues:
        key = (issue.code, issue.message)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(issue)
    return deduplicated


def _unclosed_gap_issue(gap: dict[str, object]) -> ValidationIssue:
    raw_length = gap.get("length_m")
    length_m = float(raw_length) if isinstance(raw_length, (int, float)) else 0.0
    return ValidationIssue(
        code="unclosed_wall_gap",
        severity="warning",
        message=(
            f"unclosed gap of {length_m:.2f} m between walls "
            f"{gap.get('wall_a', '?')} and {gap.get('wall_b', '?')}"
        ),
    )


def _best_effort_unlink(path: Path, context: str) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        LOGGER.warning("could not remove %s after %s: %s", path, context, exc)


def _commit_artifact_set(replacements: list[tuple[Path, Path]], revision: str) -> None:
    """Replace related artifacts as one recoverable set.

    Filesystems do not provide a multi-file atomic rename. Keep byte-for-byte
    backups until every replacement succeeds, then roll the entire set back if
    any commit-stage operation fails.
    """

    backups: dict[Path, Path] = {}
    replaced: set[Path] = set()
    retained_backups: set[Path] = set()
    committed = False
    try:
        for _temporary, target in replacements:
            if not target.exists():
                continue
            new_backup = target.with_name(f".{target.name}.{revision}.bak")
            shutil.copy2(target, new_backup)
            backups[target] = new_backup
        for temporary, target in replacements:
            temporary.replace(target)
            replaced.add(target)
        committed = True
    except Exception as original:
        rollback_errors: list[str] = []
        for _temporary, target in reversed(replacements):
            existing_backup = backups.get(target)
            if existing_backup is not None and existing_backup.exists():
                try:
                    shutil.copy2(existing_backup, target)
                except OSError as exc:
                    retained_backups.add(existing_backup)
                    rollback_errors.append(f"{target.name}: {exc}")
            elif target in replaced:
                try:
                    target.unlink(missing_ok=True)
                except OSError as exc:
                    rollback_errors.append(f"{target.name}: {exc}")
        if rollback_errors:
            raise RuntimeError(
                "artifact commit failed and rollback was incomplete; retained backups: "
                f"{', '.join(str(path) for path in retained_backups)}; "
                f"errors: {'; '.join(rollback_errors)}"
            ) from original
        raise
    finally:
        cleanup_errors: list[str] = []
        for temporary, _target in replacements:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_errors.append(f"{temporary}: {exc}")
        for backup in backups.values():
            if backup not in retained_backups:
                try:
                    backup.unlink(missing_ok=True)
                except OSError as exc:
                    retained_backups.add(backup)
                    cleanup_errors.append(f"{backup}: {exc}")
        if cleanup_errors:
            LOGGER.warning(
                "artifact transaction %s cleanup incomplete after %s: %s",
                revision,
                "commit" if committed else "rollback",
                "; ".join(cleanup_errors),
            )


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

EDITABLE_JOB_STATES = {"needs_review", "model_generated", "blocked", "generation_failed"}


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
        self._generation_slots = threading.BoundedSemaphore(
            config.limits.max_concurrent_generations
        )

    def create_job(self) -> JobRecord:
        job_dir = create_job_dir(self.config.paths.work_root)
        record = JobRecord(job_id=job_dir.name, status="created")
        with self._lock:
            self.jobs[record.job_id] = record
        self.save(record)
        return record

    def _job_dir(self, job_id: str) -> Path:
        try:
            canonical_id = validate_job_id(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        if canonical_id != job_id:
            raise HTTPException(status_code=404, detail="job not found")
        try:
            return ensure_within_directory(
                self.config.paths.work_root,
                self.config.paths.work_root / canonical_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    def get(self, job_id: str) -> JobRecord:
        # Keep the cache miss and insertion in one critical section. Without
        # this, two first requests after a process restart can deserialize two
        # independent records and both successfully claim the same mutation.
        with self._lock:
            if job_id in self.jobs:
                return self.jobs[job_id]
            record_path = self._job_dir(job_id) / "job.json"
            if record_path.exists():
                record = JobRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
                if record.job_id != job_id:
                    raise HTTPException(status_code=404, detail="job not found")
                original_status = record.status
                if record.status == "processing":
                    record.status = "failed"
                    record.message = "analysis was interrupted by a server restart; upload the plan again"
                elif record.status == "generating":
                    record.status = "generation_failed"
                    record.message = "model generation was interrupted; the last good model is preserved"
                elif record.status == "saving_corrections":
                    record.status = "needs_review"
                    record.message = "saving corrections was interrupted; review and save the layout again"
                if record.status != original_status:
                    record_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
                self.jobs[job_id] = record
                return record
        raise HTTPException(status_code=404, detail="job not found")

    def save(self, record: JobRecord) -> None:
        job_dir = self._job_dir(record.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")

    def claim_mutation(self, record: JobRecord, status: str, message: str) -> str:
        """Atomically reserve a job for one state-changing operation."""

        with self._lock:
            if record.status not in EDITABLE_JOB_STATES:
                raise HTTPException(
                    status_code=409,
                    detail=f"job is {record.status}; wait for the active operation to finish",
                )
            previous = record.status
            record.status = status
            record.message = message
        self.save(record)
        return previous

    def restore_mutation(self, record: JobRecord, previous_status: str) -> None:
        with self._lock:
            record.status = previous_status
        self.save(record)

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
            record.glb_version = _next_glb_version(record.glb_version)
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
        previous_status = self.claim_mutation(
            record,
            "generating",
            f"Blender build running — {mode} ({preset})",
        )
        if not self._generation_slots.acquire(blocking=False):
            self.restore_mutation(record, previous_status)
            raise HTTPException(
                status_code=429,
                detail="all Blender generation slots are busy; retry shortly",
            )
        thread = threading.Thread(
            target=self._run_generation_with_slot,
            args=(record, force, bake_mode),
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            self._generation_slots.release()
            self.restore_mutation(record, previous_status)
            raise

    def _run_generation_with_slot(
        self,
        record: JobRecord,
        force: bool,
        bake_mode: str | None,
    ) -> None:
        try:
            self._run_generation(record, force, bake_mode)
        finally:
            self._generation_slots.release()

    def _run_generation(self, record: JobRecord, force: bool, bake_mode: str | None) -> None:
        job_dir = self.config.paths.work_root / record.job_id
        revision = uuid.uuid4().hex
        temporary_glb = job_dir / f".building.{revision}.glb"
        try:
            build_model(
                job_dir,
                temporary_glb,
                self.config,
                run_blender=True,
                force=force,
                bake_mode=bake_mode,
            )
            replacements = [(temporary_glb, job_dir / "building.glb")]
            for suffix in (".blender_input.json", ".bake.json"):
                companion = temporary_glb.with_suffix(suffix)
                if companion.exists():
                    replacements.append((companion, job_dir / f"building{suffix}"))
            _commit_artifact_set(replacements, revision)
            record.status = "model_generated"
            record.message = _bake_summary(job_dir / "building.bake.json")
            record.glb_url = f"/jobs/{record.job_id}/artifacts/building.glb"
            record.glb_source = "blender"
            record.glb_version = _next_glb_version(record.glb_version)
        except ValueError as exc:  # quality gate
            record.status = "blocked"
            record.message = str(exc)
        except Exception as exc:
            LOGGER.exception("generation failed for job %s", record.job_id)
            record.status = "generation_failed"
            record.message = str(exc)
        finally:
            for temporary in job_dir.glob(f"{temporary_glb.stem}*"):
                if temporary.is_file():
                    _best_effort_unlink(temporary, f"generation job {record.job_id}")
        self.save(record)


def _apply_correction_revision(
    model: FloorPlanModel,
    record: JobRecord,
    runner: LocalJobRunner,
    job_dir: Path,
    source_image: Path,
) -> dict[str, object]:
    """Rebuild and commit every correction artifact while the job is claimed."""

    face_result = enumerate_faces(
        model.walls,
        model.doors,
        model.windows,
        unconfirmed_opening_range_m=(0.55, 1.40),
    )
    regenerated = match_faces_to_rooms(face_result.faces, model.rooms)
    model = model.model_copy(update={"rooms": regenerated, "camera_waypoints": []})
    try:
        route = manual_or_auto_waypoints(model)
        model = model.model_copy(update={"camera_waypoints": waypoints_from_points(route)})
    except ValueError:
        # Manual walking remains available when no collision-safe guided route exists.
        pass

    evidence = load_source_evidence(job_dir, model)
    gap_issues = [_unclosed_gap_issue(gap) for gap in face_result.unclosed_gaps]
    issues = _deduplicate_issues(
        [
            *validate_reconstruction(model, evidence),
            *_carried_local_review_issues(model),
            *gap_issues,
            *_semantic_review_issues(model),
        ]
    )
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
    report = {
        "quality_state": quality.state,
        "quality_score": quality.score,
        "components": quality.components,
        "issues": [issue.model_dump() for issue in issues],
    }

    revision = uuid.uuid4().hex
    temporary_json = job_dir / f".floorplan.corrected.{revision}.json"
    temporary_glb = job_dir / f".building.preview.{revision}.glb"
    temporary_report = job_dir / f".validation_report.{revision}.json"
    temporary_svg = job_dir / f".analysis_overlay.{revision}.svg"
    temporary_png = job_dir / f".analysis_overlay.{revision}.png"
    replacements = [
        (temporary_glb, job_dir / "building.glb"),
        (temporary_json, job_dir / "floorplan.corrected.json"),
        (temporary_report, job_dir / "validation_report.json"),
        (temporary_svg, job_dir / "analysis_overlay.svg"),
        (temporary_png, job_dir / "analysis_overlay.png"),
    ]
    try:
        model.save_json(temporary_json)
        export_simple_glb(model, temporary_glb)
        temporary_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_analysis_overlay(source_image, model, temporary_svg, temporary_png)
        _commit_artifact_set(replacements, revision)
    finally:
        for temporary, _target in replacements:
            _best_effort_unlink(temporary, f"correction revision {revision}")

    record.quality_state = quality.state
    record.quality_score = quality.score
    record.status = "needs_review"
    record.message = "corrections saved; lightweight 3D preview updated"
    record.glb_url = f"/jobs/{record.job_id}/artifacts/building.glb"
    record.glb_source = "preview"
    record.glb_version = _next_glb_version(record.glb_version)
    runner.save(record)
    return {"status": "accepted", "model": model.model_dump(mode="json"), **report}


def create_app() -> FastAPI:
    config = load_config()
    runner = LocalJobRunner(config)
    app = FastAPI(title="Architecture Walkthrough Analysis")
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=config.limits.max_upload_mb * 1024 * 1024 + MULTIPART_OVERHEAD_BYTES,
    )

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
        request: Request,
        file: UploadFile = File(...),
        use_gemini: bool = Form(True),
        manual_scale: float | None = Form(None),
        crop_x: int | None = Form(None),
        crop_y: int | None = Form(None),
        crop_width: int | None = Form(None),
        crop_height: int | None = Form(None),
    ) -> JobRecord:
        form = await request.form()
        allowed_fields = {
            "file",
            "use_gemini",
            "manual_scale",
            "crop_x",
            "crop_y",
            "crop_width",
            "crop_height",
        }
        form_items = list(form.multi_items())
        field_names = [name for name, _value in form_items]
        if set(field_names) - allowed_fields or len(field_names) != len(set(field_names)):
            for _name, value in form_items:
                if isinstance(value, StarletteUploadFile):
                    await value.close()
            raise HTTPException(status_code=400, detail="unexpected or duplicate multipart field")

        record = runner.create_job()
        job_dir = config.paths.work_root / record.job_id
        suffix = Path(file.filename or "").suffix.lower()
        upload_path = job_dir / f"upload{suffix}"
        try:
            await _save_upload_with_limit(
                file,
                upload_path,
                config.limits.max_upload_mb * 1024 * 1024,
            )
            validated = validate_image_file(upload_path, config.limits, file.content_type)
        except ValueError as exc:
            upload_path.unlink(missing_ok=True)
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
        # Validate the payload in memory first.  The old flow overwrote the
        # last good correction file and only then attempted to parse it, so a
        # malformed request could permanently corrupt the job.
        try:
            model = FloorPlanModel.model_validate(correction)
        except PydanticValidationError as exc:
            raise HTTPException(status_code=422, detail=f"invalid floorplan correction: {exc}") from exc
        if model.coordinate_system.value != "metres":
            raise HTTPException(status_code=422, detail="editor corrections must use metre coordinates")
        if not model.walls:
            raise HTTPException(status_code=422, detail="a corrected floorplan must contain at least one wall")
        previous_status = runner.claim_mutation(
            record,
            "saving_corrections",
            "validating and rebuilding the corrected model",
        )
        try:
            source = source_image_path(job_dir)
            return _apply_correction_revision(model, record, runner, job_dir, source)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"correction cannot be rendered safely: {exc}",
            ) from exc
        finally:
            if record.status == "saving_corrections":
                runner.restore_mutation(record, previous_status)

    @app.post("/jobs/{job_id}/validate-corrections")
    def validate_corrections(job_id: str) -> dict[str, object]:
        runner.get(job_id)
        job_dir = config.paths.work_root / job_id
        floorplan = job_dir / "floorplan.corrected.json"
        if not floorplan.exists():
            floorplan = job_dir / "floorplan.optimized.json"
        model = load_corrected_floorplan(floorplan)
        evidence = load_source_evidence(job_dir, model)
        face_result = enumerate_faces(
            model.walls,
            model.doors,
            model.windows,
            unconfirmed_opening_range_m=(0.55, 1.40),
        )
        gap_issues = [_unclosed_gap_issue(gap) for gap in face_result.unclosed_gaps]
        issues = _deduplicate_issues(
            [
                *validate_reconstruction(model, evidence),
                *_carried_local_review_issues(model),
                *gap_issues,
                *_semantic_review_issues(model),
            ]
        )
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
        app.mount("/", SPAStaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
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
