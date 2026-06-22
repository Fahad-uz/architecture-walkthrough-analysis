from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from architecture_walkthrough.config import AppConfig, load_config
from architecture_walkthrough.geometry.floorplan import load_corrected_floorplan
from architecture_walkthrough.pipeline import analyze_image, build_model, prepare_walkthrough_floorplan
from architecture_walkthrough.security.file_validation import create_job_dir, ensure_within_directory, validate_image_file


class JobRecord(BaseModel):
    job_id: str
    status: str
    message: str = ""
    glb_url: str | None = None


class LocalJobRunner:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.jobs: dict[str, JobRecord] = {}

    def create_job(self) -> JobRecord:
        job_dir = create_job_dir(self.config.paths.work_root)
        record = JobRecord(job_id=job_dir.name, status="created")
        self.jobs[record.job_id] = record
        self.save(record)
        return record

    def get(self, job_id: str) -> JobRecord:
        if job_id in self.jobs:
            return self.jobs[job_id]
        record_path = self.config.paths.work_root / job_id / "job.json"
        if record_path.exists():
            record = JobRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
            self.jobs[job_id] = record
            return record
        raise HTTPException(status_code=404, detail="job not found")

    def save(self, record: JobRecord) -> None:
        job_dir = self.config.paths.work_root / record.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")


UPLOAD_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Architecture Walkthrough Analysis</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 48px auto; padding: 0 18px; }
    form { display: grid; gap: 14px; border: 1px solid #ddd; padding: 18px; border-radius: 8px; }
    button { width: fit-content; padding: 8px 14px; }
    pre { white-space: pre-wrap; background: #f6f6f6; padding: 14px; border-radius: 8px; }
  </style>
</head>
<body>
  <h1>Architecture Walkthrough Analysis</h1>
  <form id="upload-form">
    <input name="file" type="file" accept="image/png,image/jpeg,image/webp" required>
    <label><input name="use_openai" type="checkbox" value="true" checked> Use OpenAI vision assist</label>
    <button type="submit">Create GLB</button>
  </form>
  <p id="download"></p>
  <pre id="result"></pre>
  <script>
    const form = document.getElementById("upload-form");
    const result = document.getElementById("result");
    const download = document.getElementById("download");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      result.textContent = "Generating GLB...";
      download.textContent = "";
      const response = await fetch("/jobs", { method: "POST", body: new FormData(form) });
      const data = await response.json();
      result.textContent = JSON.stringify(data, null, 2);
      if (response.ok && data.glb_url) {
        download.innerHTML = `<a href="${data.glb_url}">Download building.glb</a>`;
      }
    });
  </script>
</body>
</html>
"""


def create_app() -> FastAPI:
    config = load_config()
    runner = LocalJobRunner(config)
    app = FastAPI(title="Architecture Walkthrough Analysis")

    @app.get("/", response_class=HTMLResponse)
    def upload_page() -> str:
        return UPLOAD_PAGE

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/jobs", response_model=JobRecord)
    async def create_job(file: UploadFile = File(...), use_openai: bool = Form(True)) -> JobRecord:
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
            config.ai.openai_enabled = use_openai or config.ai.openai_enabled
            analyze_image(safe_path, job_dir, config)
            build_model(job_dir / "floorplan.json", job_dir / "building.glb", config, run_blender=False)
        except ValueError as exc:
            record.status = "rejected"
            record.message = str(exc)
            runner.save(record)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        record.status = "model_generated"
        record.glb_url = f"/jobs/{record.job_id}/artifacts/building.glb"
        runner.save(record)
        return record

    @app.get("/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        return runner.get(job_id)

    @app.post("/jobs/{job_id}/corrections")
    def corrections(job_id: str, correction: dict) -> dict[str, str]:
        runner.get(job_id)
        path = config.paths.work_root / job_id / "floorplan.corrected.json"
        path.write_text(json.dumps(correction, indent=2), encoding="utf-8")
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
        build_model(floorplan, output, config, run_blender=False)
        record = runner.get(job_id)
        record.status = "model_generated"
        record.glb_url = f"/jobs/{job_id}/artifacts/building.glb"
        runner.save(record)
        return {"artifact": str(output), "glb_url": record.glb_url}

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

    @app.get("/jobs/{job_id}/artifacts/{artifact_name}")
    def download_artifact(job_id: str, artifact_name: str) -> FileResponse:
        runner.get(job_id)
        if artifact_name not in {"building.glb", "floorplan.json", "floorplan.corrected.json"}:
            raise HTTPException(status_code=404, detail="artifact not found")
        job_dir = ensure_within_directory(config.paths.work_root, config.paths.work_root / job_id)
        artifact_path = ensure_within_directory(job_dir, job_dir / artifact_name)
        if not artifact_path.exists() or not artifact_path.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        media_type = "model/gltf-binary" if artifact_path.suffix.lower() == ".glb" else "application/json"
        return FileResponse(artifact_path, media_type=media_type, filename=artifact_name)

    return app
