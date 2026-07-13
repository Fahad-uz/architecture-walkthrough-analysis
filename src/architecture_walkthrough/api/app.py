from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from architecture_walkthrough.ai.floorplan_vision import GeminiFloorPlanVisionError
from architecture_walkthrough.config import AppConfig, load_config
from architecture_walkthrough.geometry.floorplan import load_corrected_floorplan
from architecture_walkthrough.geometry.validation import score_quality, validate_reconstruction
from architecture_walkthrough.pipeline import analyze_image, build_model, prepare_walkthrough_floorplan
from architecture_walkthrough.vision.overlay import write_analysis_overlay
from architecture_walkthrough.security.file_validation import create_job_dir, ensure_within_directory, validate_image_file


class JobRecord(BaseModel):
    job_id: str
    status: str
    message: str = ""
    glb_url: str | None = None
    optimized_json_url: str | None = None
    overlay_url: str | None = None
    validation_report_url: str | None = None
    ai_assist_attempted: bool = False
    ai_assist_succeeded: bool = False
    ai_assist_error: str | None = None


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
    <label><input name="use_gemini" type="checkbox" value="true" checked> Use Gemini vision assist</label>
    <label>Manual scale (metres per pixel, optional)<input name="manual_scale" type="number" min="0" step="0.0001"></label>
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
      if (response.ok) {
        download.innerHTML = `<a href="/jobs/${data.job_id}/edit">Open correction editor</a>`;
        if (data.glb_url) {
          download.innerHTML += ` · <a href="${data.glb_url}">Download building.glb</a>`;
        }
      }
    });
    fetch("/gemini-status").then(r => r.json()).then(data => {
      result.textContent = `Gemini configured for this server: ${data.available}`;
    }).catch(() => {});
  </script>
</body>
</html>
"""


EDITOR_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Floor Plan Correction Editor</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, sans-serif; background: #f5f5f3; color: #1f2529; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 12px 18px; border-bottom: 1px solid #d8d8d2; background: #fff; }
    main { display: grid; grid-template-columns: 1fr 320px; height: calc(100vh - 58px); }
    button, input, select { font: inherit; }
    button { border: 1px solid #b8b8b0; background: #fff; border-radius: 6px; padding: 7px 10px; cursor: pointer; }
    button.active { background: #173f5f; color: #fff; border-color: #173f5f; }
    button.primary { background: #1f7a4d; color: #fff; border-color: #1f7a4d; }
    button.danger { border-color: #b8483c; color: #9b2d23; }
    #stageWrap { position: relative; overflow: auto; background: #161616; }
    #stage { position: relative; margin: 20px; width: max-content; height: max-content; }
    #planImage { display: block; max-width: none; user-select: none; }
    #overlay { position: absolute; inset: 0; overflow: visible; touch-action: none; }
    .wall { stroke: #0072bc; stroke-linecap: round; cursor: pointer; }
    .wall.external { stroke-width: 8; }
    .wall.internal { stroke-width: 5; }
    .wall.selected { stroke: #ff9f1c; }
    .handle { fill: #fff; stroke: #222; stroke-width: 2; cursor: grab; }
    .room { fill: rgba(46, 160, 67, 0.13); stroke: #2ea043; stroke-width: 2; cursor: pointer; }
    .room.selected { fill: rgba(255, 159, 28, 0.16); stroke: #ff9f1c; }
    .door { fill: #be6a2f; stroke: #6d3614; stroke-width: 2; cursor: move; }
    .window { fill: #58c4dd; stroke: #176b7c; stroke-width: 2; cursor: move; }
    .ghost { stroke: #ff9f1c; stroke-width: 3; stroke-dasharray: 8 6; fill: none; pointer-events: none; }
    aside { overflow: auto; padding: 14px; border-left: 1px solid #d8d8d2; background: #fbfbfa; }
    .panel { display: grid; gap: 10px; margin-bottom: 16px; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    label { display: grid; gap: 4px; font-size: 13px; color: #3c4449; }
    input[type="text"], input[type="number"], select { width: 100%; border: 1px solid #c7c7bf; border-radius: 6px; padding: 7px 8px; background: #fff; }
    #status { white-space: pre-wrap; background: #efefeb; border-radius: 6px; padding: 10px; min-height: 72px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }
    .small { font-size: 12px; color: #667078; }
  </style>
</head>
<body>
  <header>
    <strong>Floor Plan Correction Editor</strong>
    <div class="row">
      <button id="saveBtn" class="primary">Save Correction</button>
      <button id="validateBtn">Validate</button>
      <button id="buildBtn">Generate GLB</button>
      <a id="downloadGlb" href="#" hidden>Download GLB</a>
    </div>
  </header>
  <main>
    <section id="stageWrap">
      <div id="stage">
        <img id="planImage" alt="">
        <svg id="overlay"></svg>
      </div>
    </section>
    <aside>
      <div class="panel">
        <div class="row">
          <button data-tool="select" class="active">Select</button>
          <button data-tool="wall">Wall</button>
          <button data-tool="room">Room</button>
          <button data-tool="door">Door</button>
          <button data-tool="window">Window</button>
        </div>
        <button id="deleteBtn" class="danger">Delete Selected</button>
      </div>
      <div class="panel">
        <label>Selected<input id="selectedInfo" type="text" readonly></label>
        <label>Name / Type<input id="nameInput" type="text"></label>
        <label>Scale px/m<input id="scaleInput" type="number" min="1" step="0.01"></label>
        <label>Wall thickness m<input id="thicknessInput" type="number" min="0.01" step="0.01"></label>
        <label>Opening width m<input id="openingWidthInput" type="number" min="0.1" step="0.05"></label>
        <label>Opening height m<input id="openingHeightInput" type="number" min="0.1" step="0.05"></label>
        <label><span><input id="externalInput" type="checkbox"> External wall</span></label>
        <div class="row">
          <button id="applyBtn">Apply</button>
          <button id="snapBtn">Snap 90°</button>
          <button id="fillWindowBtn">Fit Window</button>
        </div>
      </div>
      <div class="panel">
        <div class="small">Workflow: adjust walls and room rectangles, place openings on walls, save, then generate GLB.</div>
        <pre id="status">Loading...</pre>
      </div>
    </aside>
  </main>
  <script>
    const jobId = "__JOB_ID__";
    const image = document.getElementById("planImage");
    const svg = document.getElementById("overlay");
    const statusBox = document.getElementById("status");
    const selectedInfo = document.getElementById("selectedInfo");
    const nameInput = document.getElementById("nameInput");
    const scaleInput = document.getElementById("scaleInput");
    const thicknessInput = document.getElementById("thicknessInput");
    const openingWidthInput = document.getElementById("openingWidthInput");
    const openingHeightInput = document.getElementById("openingHeightInput");
    const externalInput = document.getElementById("externalInput");
    const downloadGlb = document.getElementById("downloadGlb");
    let model = null;
    let tool = "select";
    let selected = null;
    let drag = null;
    let firstPoint = null;
    let naturalWidth = 1;
    let naturalHeight = 1;

    function setStatus(value) {
      statusBox.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    }
    function ppm() { return Number(model.pixels_per_metre || 1); }
    function toPx(point) { return { x: point.x * ppm(), y: naturalHeight - point.y * ppm() }; }
    function toM(evt) {
      const rect = svg.getBoundingClientRect();
      const x = (evt.clientX - rect.left) * naturalWidth / rect.width;
      const y = (evt.clientY - rect.top) * naturalHeight / rect.height;
      return { x: x / ppm(), y: (naturalHeight - y) / ppm() };
    }
    function pointString(points) {
      return points.map((p) => {
        const px = toPx(p);
        return `${px.x},${px.y}`;
      }).join(" ");
    }
    function nearestWall(point) {
      let best = null;
      for (const wall of model.walls) {
        const ax = wall.start.x, ay = wall.start.y, bx = wall.end.x, by = wall.end.y;
        const dx = bx - ax, dy = by - ay;
        const len2 = dx * dx + dy * dy || 1;
        const t = Math.max(0, Math.min(1, ((point.x - ax) * dx + (point.y - ay) * dy) / len2));
        const px = ax + t * dx, py = ay + t * dy;
        const dist = Math.hypot(point.x - px, point.y - py);
        if (!best || dist < best.dist) best = { wall, offset: Math.sqrt(len2) * t, point: { x: px, y: py }, dist };
      }
      return best;
    }
    function wallById(id) {
      return model.walls.find((wall) => wall.id === id) || null;
    }
    function openingEndpoints(item) {
      const wall = wallById(item.wall_id) || nearestWall(item.center)?.wall;
      const center = toPx(item.center);
      if (!wall) return { a: { x: center.x - 8, y: center.y }, b: { x: center.x + 8, y: center.y } };
      const start = toPx(wall.start);
      const end = toPx(wall.end);
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const length = Math.hypot(dx, dy) || 1;
      const ux = dx / length;
      const uy = dy / length;
      const half = ((item.width_m || 1) * ppm()) / 2;
      return {
        a: { x: center.x - ux * half, y: center.y - uy * half },
        b: { x: center.x + ux * half, y: center.y + uy * half }
      };
    }
    function select(type, index) {
      selected = type ? { type, index } : null;
      updateInspector();
      render();
    }
    function updateInspector() {
      if (!selected) {
        selectedInfo.value = "";
        nameInput.value = "";
        thicknessInput.value = "";
        openingWidthInput.value = "";
        openingHeightInput.value = "";
        externalInput.checked = false;
        return;
      }
      const item = model[selected.type][selected.index];
      selectedInfo.value = `${selected.type} ${item.id || selected.index}`;
      nameInput.value = item.name || item.opening_type || item.category || "";
      thicknessInput.value = item.thickness_m || "";
      openingWidthInput.value = item.width_m || "";
      openingHeightInput.value = item.height_m || "";
      externalInput.checked = Boolean(item.external);
    }
    function render() {
      svg.innerHTML = "";
      svg.setAttribute("viewBox", `0 0 ${naturalWidth} ${naturalHeight}`);
      svg.style.width = image.width + "px";
      svg.style.height = image.height + "px";
      for (const [index, room] of model.rooms.entries()) {
        const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
        poly.setAttribute("points", pointString(room.points));
        poly.setAttribute("class", `room ${selected?.type === "rooms" && selected.index === index ? "selected" : ""}`);
        poly.addEventListener("pointerdown", (evt) => { evt.stopPropagation(); select("rooms", index); });
        svg.appendChild(poly);
      }
      for (const [index, wall] of model.walls.entries()) {
        const a = toPx(wall.start), b = toPx(wall.end);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", a.x); line.setAttribute("y1", a.y);
        line.setAttribute("x2", b.x); line.setAttribute("y2", b.y);
        line.setAttribute("class", `wall ${wall.external ? "external" : "internal"} ${selected?.type === "walls" && selected.index === index ? "selected" : ""}`);
        line.addEventListener("pointerdown", (evt) => { evt.stopPropagation(); select("walls", index); });
        svg.appendChild(line);
        if (selected?.type === "walls" && selected.index === index) {
          [["start", a], ["end", b]].forEach(([key, p]) => {
            const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            c.setAttribute("cx", p.x); c.setAttribute("cy", p.y); c.setAttribute("r", 8);
            c.setAttribute("class", "handle");
            c.addEventListener("pointerdown", (evt) => { evt.stopPropagation(); drag = { type: "wallPoint", index, key }; svg.setPointerCapture(evt.pointerId); });
            svg.appendChild(c);
          });
        }
      }
      for (const [index, door] of model.doors.entries()) drawOpening("doors", index, door, "door");
      for (const [index, win] of model.windows.entries()) drawOpening("windows", index, win, "window");
      if (firstPoint) {
        const p = toPx(firstPoint);
        const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        c.setAttribute("cx", p.x); c.setAttribute("cy", p.y); c.setAttribute("r", 6);
        c.setAttribute("class", "ghost");
        svg.appendChild(c);
      }
    }
    function drawOpening(type, index, item, cls) {
      const endpoints = openingEndpoints(item);
      const el = document.createElementNS("http://www.w3.org/2000/svg", "line");
      el.setAttribute("x1", endpoints.a.x);
      el.setAttribute("y1", endpoints.a.y);
      el.setAttribute("x2", endpoints.b.x);
      el.setAttribute("y2", endpoints.b.y);
      el.setAttribute("stroke-width", cls === "door" ? 9 : 7);
      el.setAttribute("stroke-linecap", "square");
      el.setAttribute("class", cls);
      el.addEventListener("pointerdown", (evt) => { evt.stopPropagation(); select(type, index); drag = { type: "opening", collection: type, index }; svg.setPointerCapture(evt.pointerId); });
      svg.appendChild(el);
    }
    function addWall(point) {
      if (!firstPoint) { firstPoint = point; render(); return; }
      model.walls.push({
        id: `cw${Date.now()}`,
        start: firstPoint,
        end: point,
        thickness_m: Number(thicknessInput.value || 0.12),
        height_m: 3,
        external: externalInput.checked,
        wall_type: externalInput.checked ? "external" : "internal",
        confidence: 1,
        evidence_source: "manual_correction"
      });
      firstPoint = null;
      select("walls", model.walls.length - 1);
    }
    function addRoom(point) {
      if (!firstPoint) { firstPoint = point; render(); return; }
      const x0 = Math.min(firstPoint.x, point.x), x1 = Math.max(firstPoint.x, point.x);
      const y0 = Math.min(firstPoint.y, point.y), y1 = Math.max(firstPoint.y, point.y);
      model.rooms.push({
        id: `room_manual_${Date.now()}`,
        name: "Room",
        points: [{x:x0,y:y0},{x:x1,y:y0},{x:x1,y:y1},{x:x0,y:y1}],
        confidence: 1,
        evidence_source: "manual_correction"
      });
      firstPoint = null;
      select("rooms", model.rooms.length - 1);
    }
    function addOpening(collection, point) {
      const match = nearestWall(point);
      const center = match ? match.point : point;
      const item = collection === "doors"
        ? { id: `door_manual_${Date.now()}`, center, width_m: 0.9, height_m: 2.1, wall_id: match?.wall.id || null, offset_m: match?.offset || null, opening_type: "single_leaf", confidence: 1, evidence_source: "manual_correction" }
        : { id: `window_manual_${Date.now()}`, center, width_m: 1.2, height_m: 1.2, sill_height_m: 0.9, wall_id: match?.wall.id || null, offset_m: match?.offset || null, opening_type: "fixed", confidence: 1, evidence_source: "manual_correction" };
      model[collection].push(item);
      select(collection, model[collection].length - 1);
    }
    svg.addEventListener("pointerdown", (evt) => {
      const point = toM(evt);
      if (tool === "select") { select(null, -1); return; }
      if (tool === "wall") addWall(point);
      if (tool === "room") addRoom(point);
      if (tool === "door") addOpening("doors", point);
      if (tool === "window") addOpening("windows", point);
    });
    svg.addEventListener("pointermove", (evt) => {
      if (!drag) return;
      const point = toM(evt);
      if (drag.type === "wallPoint") {
        model.walls[drag.index][drag.key] = point;
      } else if (drag.type === "opening") {
        const match = nearestWall(point);
        const item = model[drag.collection][drag.index];
        item.center = match ? match.point : point;
        item.wall_id = match?.wall.id || null;
        item.offset_m = match?.offset || null;
      }
      render();
    });
    svg.addEventListener("pointerup", () => { drag = null; });
    document.querySelectorAll("[data-tool]").forEach((button) => {
      button.addEventListener("click", () => {
        tool = button.dataset.tool;
        firstPoint = null;
        document.querySelectorAll("[data-tool]").forEach((b) => b.classList.toggle("active", b === button));
        render();
      });
    });
    document.getElementById("applyBtn").addEventListener("click", () => {
      model.pixels_per_metre = Number(scaleInput.value || model.pixels_per_metre);
      if (selected) {
        const item = model[selected.type][selected.index];
        if (selected.type === "rooms") item.name = nameInput.value || null;
        if (selected.type === "walls") {
          item.thickness_m = Number(thicknessInput.value || item.thickness_m || 0.12);
          item.external = externalInput.checked;
          item.wall_type = item.external ? "external" : "internal";
        }
        if (selected.type === "doors" || selected.type === "windows") {
          item.opening_type = nameInput.value || item.opening_type;
          item.width_m = Number(openingWidthInput.value || item.width_m || 1);
          item.height_m = Number(openingHeightInput.value || item.height_m || 1);
          const match = nearestWall(item.center);
          item.wall_id = match?.wall.id || item.wall_id || null;
          item.offset_m = match?.offset || item.offset_m || null;
          item.center = match?.point || item.center;
        }
      }
      render();
    });
    document.getElementById("fillWindowBtn").addEventListener("click", () => {
      if (selected?.type !== "windows") return;
      const item = model.windows[selected.index];
      const wall = wallById(item.wall_id) || nearestWall(item.center)?.wall;
      if (!wall) return;
      const length = Math.hypot(wall.end.x - wall.start.x, wall.end.y - wall.start.y);
      item.width_m = Math.max(0.3, length - 0.16);
      openingWidthInput.value = item.width_m.toFixed(2);
      render();
    });
    document.getElementById("snapBtn").addEventListener("click", () => {
      if (selected?.type !== "walls") return;
      const wall = model.walls[selected.index];
      if (Math.abs(wall.end.x - wall.start.x) > Math.abs(wall.end.y - wall.start.y)) {
        const y = (wall.start.y + wall.end.y) / 2; wall.start.y = y; wall.end.y = y;
      } else {
        const x = (wall.start.x + wall.end.x) / 2; wall.start.x = x; wall.end.x = x;
      }
      render();
    });
    document.getElementById("deleteBtn").addEventListener("click", () => {
      if (!selected) return;
      model[selected.type].splice(selected.index, 1);
      select(null, -1);
    });
    async function saveCorrection() {
      model.coordinate_system = "metres";
      model.schema_version = model.schema_version || "2.0";
      model.metadata = model.metadata || {};
      model.metadata.correction_source = "browser_editor";
      const response = await fetch(`/jobs/${jobId}/corrections`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(model)
      });
      const data = await response.json();
      setStatus(data);
      if (!response.ok) throw new Error(JSON.stringify(data));
    }
    document.getElementById("saveBtn").addEventListener("click", () => saveCorrection().catch((error) => setStatus(String(error))));
    document.getElementById("validateBtn").addEventListener("click", async () => {
      await saveCorrection();
      const response = await fetch(`/jobs/${jobId}/validate-corrections`, { method: "POST" });
      setStatus(await response.json());
    });
    document.getElementById("buildBtn").addEventListener("click", async () => {
      await saveCorrection();
      const response = await fetch(`/jobs/${jobId}/generate-model`, { method: "POST" });
      const data = await response.json();
      setStatus(data);
      if (data.glb_url) { downloadGlb.href = data.glb_url; downloadGlb.hidden = false; }
    });
    async function load() {
      const response = await fetch(`/jobs/${jobId}/edit-data`);
      const data = await response.json();
      if (!response.ok) throw new Error(JSON.stringify(data));
      model = data.model;
      scaleInput.value = model.pixels_per_metre || 1;
      image.onload = () => {
        naturalWidth = image.naturalWidth;
        naturalHeight = image.naturalHeight;
        svg.style.width = image.width + "px";
        svg.style.height = image.height + "px";
        render();
      };
      image.src = data.image_url;
      setStatus({ job_id: jobId, quality: model.reconstruction?.quality_state, walls: model.walls.length, rooms: model.rooms.length });
    }
    load().catch((error) => setStatus(String(error)));
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
            safe_path = job_dir / validated.safe_filename
            upload_path.replace(safe_path)
            # Copy per request: mutating the shared config would leak this
            # request's Gemini toggle into concurrent jobs.
            job_config = config.model_copy(deep=True)
            job_config.ai.gemini_enabled = use_gemini
            crop_rect = None
            if crop_x is not None and crop_y is not None and crop_width is not None and crop_height is not None:
                crop_rect = (crop_x, crop_y, crop_width, crop_height)
            model = analyze_image(
                safe_path,
                job_dir,
                job_config,
                manual_scale=manual_scale,
                require_ai_success=False,
                crop_rect=crop_rect,
            )
            build_model(job_dir, job_dir / "building.glb", job_config, run_blender=False)
            metadata = model.metadata
            record.ai_assist_attempted = bool(metadata.get("ai_assist_attempted"))
            record.ai_assist_succeeded = bool(metadata.get("ai_assist_succeeded"))
            record.ai_assist_error = metadata.get("ai_assist_error")
        except ValueError as exc:
            record.status = "rejected"
            record.message = str(exc)
            runner.save(record)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GeminiFloorPlanVisionError as exc:
            record.status = "gemini_failed"
            record.message = str(exc)
            record.ai_assist_attempted = True
            record.ai_assist_succeeded = False
            record.ai_assist_error = str(exc)
            runner.save(record)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        record.status = "model_generated"
        record.glb_url = f"/jobs/{record.job_id}/artifacts/building.glb"
        record.optimized_json_url = f"/jobs/{record.job_id}/artifacts/floorplan.optimized.json"
        record.overlay_url = f"/jobs/{record.job_id}/artifacts/analysis_overlay.svg"
        record.validation_report_url = f"/jobs/{record.job_id}/artifacts/validation_report.json"
        runner.save(record)
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

    @app.get("/jobs/{job_id}/edit", response_class=HTMLResponse)
    def edit_page(job_id: str) -> str:
        runner.get(job_id)
        return EDITOR_PAGE.replace("__JOB_ID__", job_id)

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
    def corrections(job_id: str, correction: dict) -> dict[str, str]:
        runner.get(job_id)
        job_dir = config.paths.work_root / job_id
        path = config.paths.work_root / job_id / "floorplan.corrected.json"
        path.write_text(json.dumps(correction, indent=2), encoding="utf-8")
        model = load_corrected_floorplan(path)
        issues = validate_reconstruction(model)
        score, state = score_quality(model.model_copy(update={"validation_issues": issues}))
        model = model.model_copy(
            update={
                "validation_issues": issues,
                "reconstruction": model.reconstruction.model_copy(update={"quality_score": score, "quality_state": state}),
                "metadata": {
                    **model.metadata,
                    "correction_source": model.metadata.get("correction_source", "browser_editor"),
                },
            }
        )
        model.save_json(path)
        report = {
            "quality_state": state,
            "quality_score": score,
            "issues": [issue.model_dump() for issue in issues],
        }
        (job_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        source_image = source_image_path(job_dir)
        if source_image.exists():
            write_analysis_overlay(source_image, model, job_dir / "analysis_overlay.svg", job_dir / "analysis_overlay.png")
        return {"status": "accepted"}

    @app.post("/jobs/{job_id}/validate-corrections")
    def validate_corrections(job_id: str) -> dict[str, object]:
        runner.get(job_id)
        job_dir = config.paths.work_root / job_id
        floorplan = job_dir / "floorplan.corrected.json"
        if not floorplan.exists():
            floorplan = job_dir / "floorplan.optimized.json"
        model = load_corrected_floorplan(floorplan)
        issues = validate_reconstruction(model)
        score, state = score_quality(model.model_copy(update={"validation_issues": issues}))
        return {"quality_state": state, "quality_score": score, "issues": [issue.model_dump() for issue in issues]}

    @app.post("/jobs/{job_id}/generate-model")
    def generate_model(job_id: str) -> dict[str, str]:
        runner.get(job_id)
        job_dir = config.paths.work_root / job_id
        output = job_dir / "building.glb"
        build_model(job_dir, output, config, run_blender=False)
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
        if artifact_name not in {
            "building.glb",
            "floorplan.json",
            "floorplan.raw.json",
            "floorplan.optimized.json",
            "floorplan.corrected.json",
            "validation_report.json",
            "analysis_overlay.svg",
            "analysis_overlay.png",
        }:
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
        return FileResponse(artifact_path, media_type=media_type, filename=artifact_name)

    return app
