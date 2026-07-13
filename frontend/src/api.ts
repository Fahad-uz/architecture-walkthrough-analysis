import type { FloorPlanModel, JobRecord, QualityReport } from "./types";

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status}: ${body.slice(0, 500)}`);
  }
  return (await response.json()) as T;
}

export async function uploadPlan(file: File, useGemini: boolean, manualScale: string): Promise<JobRecord> {
  const form = new FormData();
  form.append("file", file);
  form.append("use_gemini", String(useGemini));
  if (manualScale.trim()) form.append("manual_scale", manualScale.trim());
  return json(await fetch("/jobs", { method: "POST", body: form }));
}

export async function getJob(jobId: string): Promise<JobRecord> {
  return json(await fetch(`/jobs/${jobId}`));
}

export async function getEditData(jobId: string): Promise<{ image_url: string; model: FloorPlanModel }> {
  return json(await fetch(`/jobs/${jobId}/edit-data`));
}

export async function saveCorrections(
  jobId: string,
  model: FloorPlanModel,
): Promise<QualityReport & { model: FloorPlanModel }> {
  return json(
    await fetch(`/jobs/${jobId}/corrections`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(model),
    }),
  );
}

export async function validateCorrections(jobId: string): Promise<QualityReport> {
  return json(await fetch(`/jobs/${jobId}/validate-corrections`, { method: "POST" }));
}

export async function generateModel(jobId: string, force: boolean, bakeMode: string): Promise<JobRecord> {
  const params = new URLSearchParams({ force: String(force), bake_mode: bakeMode });
  return json(await fetch(`/jobs/${jobId}/generate-model?${params}`, { method: "POST" }));
}

export function glbUrl(jobId: string): string {
  return `/jobs/${jobId}/artifacts/building.glb`;
}
