import type { FloorPlanModel, JobRecord } from "./types";

export interface SavedPreview {
  job: JobRecord;
  plan: FloorPlanModel | null;
}

interface PreviewReads {
  getJob: () => Promise<JobRecord>;
  getPlan: () => Promise<{ model: FloorPlanModel }>;
}

function planIsChanging(job: JobRecord): boolean {
  // The backend reserves these states before replacing plan/preview artifacts.
  // Blender generation uses the existing saved plan, so it may keep displaying.
  return job.status === "saving_corrections" || job.status === "processing";
}

function sameArtifact(first: JobRecord, second: JobRecord): boolean {
  return first.job_id === second.job_id && first.glb_url === second.glb_url &&
    first.glb_version === second.glb_version;
}

/** Read a plan only between matching, settled GLB revisions. A changed or
 * in-progress revision leaves the prior pair intact until the next poll. */
export async function readSavedPreview(previous: SavedPreview | null, reads: PreviewReads): Promise<{
  snapshot: SavedPreview | null;
  latestJob: JobRecord;
}> {
  const first = await reads.getJob();
  if (planIsChanging(first)) return { snapshot: previous, latestJob: first };
  if (!first.glb_url) return { snapshot: { job: first, plan: null }, latestJob: first };
  if (previous?.plan && sameArtifact(previous.job, first)) {
    return { snapshot: { job: first, plan: previous.plan }, latestJob: first };
  }

  const { model } = await reads.getPlan();
  const verified = await reads.getJob();
  if (planIsChanging(verified) || !sameArtifact(first, verified)) {
    return { snapshot: previous, latestJob: verified };
  }
  return { snapshot: { job: verified, plan: model }, latestJob: verified };
}
