from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from architecture_walkthrough.config import AppConfig, load_config
from architecture_walkthrough.geometry.floorplan import load_corrected_floorplan
from architecture_walkthrough.pipeline import analyze_image, build_model, prepare_walkthrough_floorplan
from architecture_walkthrough.security.file_validation import create_job_dir, validate_image_file


class JobRecord(BaseModel):
    job_id: str
    status: str
    message: str = ""


class LocalJobRunner:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.jobs: dict[str, JobRecord] = {}

    def create_job(self) -> JobRecord:
        job_dir = create_job_dir(self.config.paths.work_root)
        record = JobRecord(job_id=job_dir.name, status="created")
        self.jobs[record.job_id] = record
        return record

    def get(self, job_id: str) -> JobRecord:
        try:
            return self.jobs[job_id]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc


def create_app() -> FastAPI:
    config = load_config()
    runner = LocalJobRunner(config)
    app = FastAPI(title="Architecture Walkthrough Analysis")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/jobs", response_model=JobRecord)
    async def create_job(file: UploadFile = File(...)) -> JobRecord:
        record = runner.create_job()
        job_dir = config.paths.work_root / record.job_id
        suffix = Path(file.filename or "").suffix.lower()
        upload_path = job_dir / f"upload{suffix}"
        with upload_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        try:
            validated = validate_image_file(upload_path, config.limits, file.content_type)
            safe_path = job_dir / validated.safe_filename
            upload_path.replace(safe_path)
            analyze_image(safe_path, job_dir, config)
        except ValueError as exc:
            record.status = "rejected"
            record.message = str(exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        record.status = "analyzed"
        return record

    @app.get("/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        return runner.get(job_id)

    @app.post("/jobs/{job_id}/corrections")
    def corrections(job_id: str, correction: dict) -> dict[str, str]:
        runner.get(job_id)
        path = config.paths.work_root / job_id / "floorplan.corrected.json"
        path.write_text(__import__("json").dumps(correction, indent=2), encoding="utf-8")
        load_corrected_floorplan(path)
        return {"status": "accepted"}

    @app.post("/jobs/{job_id}/generate-model")
    def generate_model(job_id: str) -> dict[str, str]:
        runner.get(job_id)
        job_dir = config.paths.work_root / job_id
        floorplan = job_dir / "floorplan.corrected.json"
        if not floorplan.exists():
            floorplan = job_dir / "floorplan.json"
        output = job_dir / "building.glb"
        build_model(floorplan, output, config)
        return {"artifact": str(output)}

    @app.post("/jobs/{job_id}/generate-walkthrough")
    def generate_walkthrough(job_id: str) -> dict[str, str]:
        runner.get(job_id)
        job_dir = config.paths.work_root / job_id
        output = job_dir / "walkthrough.floorplan.json"
        prepare_walkthrough_floorplan(job_dir / "floorplan.json", output)
        return {"artifact": str(output), "message": "render execution is intentionally out-of-request"}

    @app.get("/jobs/{job_id}/artifacts")
    def artifacts(job_id: str) -> dict[str, list[str]]:
        runner.get(job_id)
        job_dir = config.paths.work_root / job_id
        return {"artifacts": [str(path) for path in job_dir.glob("*") if path.is_file()]}

    return app
