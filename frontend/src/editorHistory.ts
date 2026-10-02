import type { FloorPlanModel } from "./types";

export const HISTORY_LIMIT = 50;
export const DRAFT_VERSION_LIMIT = 5;
const MAX_DRAFT_BYTES = 8_000_000;

export interface EditHistory<T> {
  past: T[];
  present: T;
  future: T[];
}

export function createHistory<T>(value: T): EditHistory<T> {
  return { past: [], present: structuredClone(value), future: [] };
}

/** Coalescing makes an entire pointer drag one undoable edit. */
export function recordEdit<T>(history: EditHistory<T>, value: T, coalesce = false): EditHistory<T> {
  if (JSON.stringify(value) === JSON.stringify(history.present)) return history;
  return {
    past: coalesce ? history.past : [...history.past, history.present].slice(-HISTORY_LIMIT),
    present: structuredClone(value),
    future: [],
  };
}

export function undoEdit<T>(history: EditHistory<T>): EditHistory<T> {
  if (!history.past.length) return history;
  return {
    past: history.past.slice(0, -1),
    present: history.past[history.past.length - 1],
    future: [history.present, ...history.future],
  };
}

export function redoEdit<T>(history: EditHistory<T>): EditHistory<T> {
  if (!history.future.length) return history;
  return {
    past: [...history.past, history.present].slice(-HISTORY_LIMIT),
    present: history.future[0],
    future: history.future.slice(1),
  };
}

export interface DraftVersion {
  id: string;
  savedAt: string;
  source: "automatic" | "manual";
  model: FloorPlanModel;
}

export interface DraftEnvelope {
  version: 1;
  jobId: string;
  versions: DraftVersion[];
}

export function draftStorageKey(jobId: string): string {
  return `planstride:editor-draft:v1:${encodeURIComponent(jobId)}`;
}

type ObjectValue = Record<string, unknown>;
const object = (value: unknown): value is ObjectValue =>
  value !== null && typeof value === "object" && !Array.isArray(value);
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const point = (value: unknown): boolean => object(value) && finite(value.x) && finite(value.y);
const polygon = (value: unknown): boolean =>
  object(value) && Array.isArray(value.points) && value.points.length >= 3 && value.points.every(point);
const optionalArray = (value: unknown, check: (item: unknown) => boolean): boolean =>
  value === undefined || (Array.isArray(value) && value.every(check));
const placement = (value: unknown): boolean =>
  object(value) && typeof value.category === "string" && point(value.center) &&
  finite(value.width_m) && finite(value.depth_m) &&
  (value.rotation_deg === undefined || finite(value.rotation_deg));

/** Local storage is untrusted and can outlive schema changes. Validate fields the
 * editor and renderer dereference before offering recovery; the server remains
 * authoritative for geometric validity when the user explicitly saves. */
export function isDraftModel(value: unknown): value is FloorPlanModel {
  if (!object(value) || typeof value.schema_version !== "string" || typeof value.coordinate_system !== "string") return false;
  if (value.pixels_per_metre !== null && (!finite(value.pixels_per_metre) || value.pixels_per_metre <= 0)) return false;
  if (!object(value.metadata)) return false;
  const wall = (item: unknown) => object(item) && point(item.start) && point(item.end) &&
    finite(item.thickness_m) && finite(item.height_m) && typeof item.external === "boolean";
  const opening = (item: unknown) => object(item) && point(item.center) && finite(item.width_m) && finite(item.height_m);
  if (!Array.isArray(value.walls) || !value.walls.every(wall) ||
      !Array.isArray(value.doors) || !value.doors.every(opening) ||
      !Array.isArray(value.windows) || !value.windows.every(opening) ||
      !Array.isArray(value.rooms) || !value.rooms.every(polygon)) return false;
  if (!optionalArray(value.balconies, polygon) || !optionalArray(value.slabs, polygon) ||
      !optionalArray(value.furniture, placement) || !optionalArray(value.asset_placements, placement)) return false;
  if (!optionalArray(value.camera_waypoints, (item) => object(item) && point(item.position) &&
      (item.look_at == null || point(item.look_at)))) return false;
  if (!optionalArray(value.special_elements, (item) => object(item) && typeof item.kind === "string" &&
      (item.center == null || point(item.center)) && optionalArray(item.polygon, point))) return false;
  if (value.entrance != null && !point(value.entrance)) return false;
  if (!optionalArray(value.metadata.sanity_warnings, (item) => object(item) &&
      typeof item.kind === "string" && typeof item.description === "string" && finite(item.x) && finite(item.y))) return false;
  if (!optionalArray(value.validation_issues, (item) => object(item) &&
      typeof item.code === "string" && typeof item.severity === "string" && typeof item.message === "string")) return false;
  return true;
}

export function parseDraft(raw: string | null, jobId: string, schemaVersion: string): DraftEnvelope | null {
  if (raw === null) return null;
  if (raw.length > MAX_DRAFT_BYTES) throw new Error("The local draft is too large to recover safely.");
  let parsed: unknown;
  try { parsed = JSON.parse(raw); } catch { throw new Error("The local draft is damaged and could not be read."); }
  if (!object(parsed) || parsed.version !== 1 || parsed.jobId !== jobId || !Array.isArray(parsed.versions) ||
      !parsed.versions.length || parsed.versions.length > DRAFT_VERSION_LIMIT ||
      !parsed.versions.every((item: unknown) => object(item) && typeof item.id === "string" &&
        typeof item.savedAt === "string" && Number.isFinite(Date.parse(item.savedAt)) &&
        ["automatic", "manual"].includes(String(item.source)) &&
        isDraftModel(item.model) && item.model.schema_version === schemaVersion)) {
    throw new Error("The local draft does not match this plan or its supported format.");
  }
  return parsed as unknown as DraftEnvelope;
}

export function appendDraft(
  previous: DraftEnvelope | null,
  jobId: string,
  model: FloorPlanModel,
  source: DraftVersion["source"],
  now = new Date(),
): DraftEnvelope {
  if (previous && previous.jobId !== jobId) throw new Error("Cannot save a draft under a different plan.");
  if (!isDraftModel(model)) throw new Error("The current layout cannot be safely stored as a local draft.");
  const versions = previous?.versions ?? [];
  if (versions.length && JSON.stringify(versions[0].model) === JSON.stringify(model) && source === "automatic") {
    return previous!;
  }
  const savedAt = now.toISOString();
  return {
    version: 1,
    jobId,
    versions: [{ id: `${now.getTime()}-${globalThis.crypto.randomUUID()}`, savedAt, source, model: structuredClone(model) },
      ...versions].slice(0, DRAFT_VERSION_LIMIT),
  };
}

export interface DraftStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

/** Compare before writing so an older tab cannot silently overwrite another
 * tab's draft. Keep the last valid stored value if serialization/quota fails. */
export function writeDraft(storage: DraftStorage, draft: DraftEnvelope, expectedRaw: string | null): string {
  const key = draftStorageKey(draft.jobId);
  if (storage.getItem(key) !== expectedRaw) {
    throw new Error("Another tab changed this plan's local draft. Reload to review it before saving locally.");
  }
  const raw = JSON.stringify(draft);
  if (raw.length > MAX_DRAFT_BYTES) throw new Error("This local draft is too large. Save changes to the project instead.");
  storage.setItem(key, raw);
  return raw;
}
