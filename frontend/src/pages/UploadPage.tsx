import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { getJob, uploadPlan } from "../api";
import type { JobRecord } from "../types";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [useGemini, setUseGemini] = useState(true);
  const [manualScale, setManualScale] = useState("");
  const [job, setJob] = useState<JobRecord | null>(null);
  const [error, setError] = useState("");
  const pollRef = useRef<number | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (pollRef.current !== null) window.clearTimeout(pollRef.current);
    };
  }, []);

  const startPolling = (jobId: string) => {
    if (pollRef.current !== null) window.clearTimeout(pollRef.current);
    const poll = async () => {
      try {
        const latest = await getJob(jobId);
        if (!mountedRef.current) return;
        setJob(latest);
        setError("");
        if (!["created", "processing", "generating"].includes(latest.status)) {
          pollRef.current = null;
          return;
        }
      } catch (exc) {
        if (!mountedRef.current) return;
        setError(String(exc));
      }
      if (mountedRef.current) pollRef.current = window.setTimeout(() => void poll(), 1500);
    };
    pollRef.current = window.setTimeout(() => void poll(), 500);
  };

  const submit = async () => {
    if (!file) return;
    setError("");
    const parsedScale = manualScale.trim() ? Number(manualScale) : null;
    if (parsedScale !== null && (!Number.isFinite(parsedScale) || parsedScale <= 0)) {
      setError("Manual scale must be a positive number of metres per pixel.");
      return;
    }
    try {
      const record = await uploadPlan(file, useGemini, manualScale);
      if (!mountedRef.current) return;
      setJob(record);
      startPolling(record.job_id);
    } catch (exc) {
      if (mountedRef.current) setError(String(exc));
    }
  };

  const busy = job !== null && ["created", "processing", "generating"].includes(job.status);

  return (
    <div className="upload-card">
      <h1>Upload a floor plan</h1>
      <p style={{ margin: 0, fontSize: 14, color: "#555" }}>
        PNG/JPEG/WebP of a CAD-style plan. Analysis detects walls, openings and rooms from the
        image; you review and fix everything in the editor before generating the 3D model.
      </p>
      <label>
        Plan image
        <input type="file" accept="image/png,image/jpeg,image/webp" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </label>
      <label className="row" style={{ display: "flex" }}>
        <input type="checkbox" checked={useGemini} onChange={(e) => setUseGemini(e.target.checked)} />
        Use Gemini for room labels / dimension text / sanity check (never geometry)
      </label>
      <label>
        Manual scale — metres per pixel (optional; you can also set scale in the editor)
        <input
          type="number"
          min="0.000001"
          step="any"
          inputMode="decimal"
          value={manualScale}
          onChange={(e) => setManualScale(e.target.value)}
          placeholder="e.g. 0.01"
        />
      </label>
      <div className="row">
        <button type="button" className="primary" onClick={() => void submit()} disabled={!file || busy}>
          {busy ? "Analyzing…" : "Analyze plan"}
        </button>
        {job && ["needs_review", "model_generated", "blocked"].includes(job.status) && (
          <Link className="button-link" to={`/jobs/${job.job_id}/edit`}>
            Open correction editor →
          </Link>
        )}
      </div>
      {(job || error) && (
        <div className="status-box" role={error ? "alert" : "status"} aria-live="polite">
          {error
            ? error
            : `job ${job!.job_id}\nstatus: ${job!.status}\n${job!.message}` +
              (job!.quality_score != null
                ? `\nquality: ${(job!.quality_score * 100).toFixed(0)}% (${job!.quality_state})`
                : "") +
              (job!.ai_assist_error ? `\ngemini: ${job!.ai_assist_error}` : "")}
        </div>
      )}
    </div>
  );
}
