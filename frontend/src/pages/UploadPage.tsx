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

  useEffect(() => {
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, []);

  const startPolling = (jobId: string) => {
    if (pollRef.current) window.clearInterval(pollRef.current);
    pollRef.current = window.setInterval(async () => {
      try {
        const latest = await getJob(jobId);
        setJob(latest);
        if (!["created", "processing", "generating"].includes(latest.status) && pollRef.current) {
          window.clearInterval(pollRef.current);
        }
      } catch (exc) {
        setError(String(exc));
      }
    }, 1500);
  };

  const submit = async () => {
    if (!file) return;
    setError("");
    try {
      const record = await uploadPlan(file, useGemini, manualScale);
      setJob(record);
      startPolling(record.job_id);
    } catch (exc) {
      setError(String(exc));
    }
  };

  const busy = job !== null && ["created", "processing"].includes(job.status);

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
        <input value={manualScale} onChange={(e) => setManualScale(e.target.value)} placeholder="e.g. 0.01" />
      </label>
      <div className="row">
        <button className="primary" onClick={submit} disabled={!file || busy}>
          {busy ? "Analyzing…" : "Analyze plan"}
        </button>
        {job && ["needs_review", "model_generated", "blocked"].includes(job.status) && (
          <Link to={`/jobs/${job.job_id}/edit`}>
            <button>Open correction editor →</button>
          </Link>
        )}
      </div>
      {(job || error) && (
        <div className="status-box">
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
