import type { FloorPlanModel, Point2D } from "./types";

export type TargetKind = "walls" | "doors" | "windows" | "rooms" | "balconies" | "slabs" | "special_elements";
export interface ReviewTarget { kind: TargetKind; index: number; id: string }
export interface ReviewIssue {
  id: string;
  code: string;
  severity: string;
  message: string;
  targets: ReviewTarget[];
  location?: Point2D; // normalized source-image coordinates, only when explicitly supplied
}
interface IssueInput { code: string; severity: string; message: string; element_id?: string | null }
interface Acknowledgement { fingerprint: string; confirmed_at: string }
const ACK_KEY = "editor_review_acknowledgements";
const kinds: TargetKind[] = ["walls", "doors", "windows", "rooms", "balconies", "slabs", "special_elements"];
const isObject = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);

/** Exact, order-independent object serialization avoids hash collisions and
 * makes acknowledgement checks insensitive to JSON property ordering. */
function stable(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  if (isObject(value)) return `{${Object.keys(value).sort().filter((key) => value[key] !== undefined)
    .map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(",")}}`;
  return JSON.stringify(value) ?? "null";
}

/** Compact change detector, not an authentication or security digest. Two
 * independent 32-bit accumulators avoid storing a full floor plan per issue. */
function fingerprint(value: unknown): string {
  const text = stable(value);
  let first = 2166136261;
  let second = 3339675911;
  for (let index = 0; index < text.length; index += 1) {
    first = Math.imul(first ^ text.charCodeAt(index), 16777619);
    second = Math.imul(second ^ text.charCodeAt(index), 2246822519);
  }
  return `v1:${(first >>> 0).toString(16).padStart(8, "0")}${(second >>> 0).toString(16).padStart(8, "0")}:${text.length}`;
}

function findTargets(model: FloorPlanModel, ids: string[]): ReviewTarget[] {
  return kinds.flatMap((kind) => (model[kind] ?? []).flatMap((item, index) =>
    typeof item.id === "string" && ids.includes(item.id) ? [{ kind, index, id: item.id }] : []));
}

function referencedIds(issue: IssueInput): string[] {
  const ids = issue.element_id ? [issue.element_id] : [];
  // Legacy backend issue formats carry explicit IDs in text. Do not infer a
  // nearest object or extract arbitrary words from unrelated issue messages.
  const gap = issue.code === "unclosed_wall_gap"
    ? /^unclosed gap of [\d.]+ m between walls (\S+) and (\S+)$/.exec(issue.message) : null;
  const opening = issue.code === "ambiguous_opening"
    ? /^ambiguous on (\S+) at [\d.]+-[\d.]+ m \(width heuristic only\)$/.exec(issue.message) : null;
  const crossing = issue.code === "wall_crossing_without_junction"
    ? /^walls (\S+) and (\S+) cross without a shared junction$/.exec(issue.message) : null;
  if (gap) ids.push(gap[1], gap[2]);
  if (opening) ids.push(opening[1]);
  if (crossing) ids.push(crossing[1], crossing[2]);
  return [...new Set(ids)].sort();
}

export function collectReviewIssues(model: FloorPlanModel, issues: IssueInput[] = model.validation_issues ?? []): ReviewIssue[] {
  const notes = Array.isArray(model.metadata.sanity_warnings) ? model.metadata.sanity_warnings.filter((item) =>
    isObject(item) && typeof item.kind === "string" && typeof item.description === "string") : [];
  const result: ReviewIssue[] = [];
  const seen = new Set<string>();
  const add = (issue: IssueInput, location?: Point2D) => {
    const ids = referencedIds(issue);
    const id = `review-v1:${encodeURIComponent(stable([issue.code, ids, issue.message, location ?? null]))}`;
    if (seen.has(id)) return;
    seen.add(id);
    result.push({ id, code: issue.code, severity: issue.severity, message: issue.message,
      targets: findTargets(model, ids), ...(location ? { location } : {}) });
  };
  for (const issue of issues) {
    // Semantic findings are also persisted as normalized image annotations.
    const note = notes.find((item) => issue.code === `gemini_${item.kind}` &&
      (issue.message === item.description || issue.message.startsWith(`${item.description} (at `)));
    const location = note && validLocation(note) ? { x: note.x, y: note.y } : undefined;
    add(issue, location);
  }
  for (const note of notes) {
    if (!isObject(note) || typeof note.kind !== "string" || typeof note.description !== "string") continue;
    if (issues.some((issue) => issue.code === `gemini_${note.kind}` &&
        (issue.message === note.description || issue.message.startsWith(`${note.description} (at `)))) continue;
    add({ code: `gemini_${note.kind}`, severity: "warning", message: note.description },
      validLocation(note) ? { x: note.x, y: note.y } : undefined);
  }
  return result;
}

function validLocation(value: { x?: unknown; y?: unknown }): value is Point2D {
  return typeof value.x === "number" && Number.isFinite(value.x) && value.x >= 0 && value.x <= 1 &&
    typeof value.y === "number" && Number.isFinite(value.y) && value.y >= 0 && value.y <= 1;
}

function targetGeometry(model: FloorPlanModel, target: ReviewTarget): unknown {
  // Resolve by ID again: stale indices must not acknowledge a different object
  // after a preceding element was deleted or the server reordered elements.
  const item = (model[target.kind] ?? []).find((entry) => entry.id === target.id);
  if (!item) return { missing: target.id };
  const record = item as unknown as Record<string, unknown>;
  const fields = ["id", "start", "end", "points", "polygon", "center", "wall_id", "width_m", "depth_m", "height_m",
    "thickness_m", "rotation_deg", "sill_height_m", "offset_m", "start_offset_m", "end_offset_m", "external", "wall_type", "name", "kind"];
  return Object.fromEntries(fields.filter((key) => record[key] !== undefined).map((key) => [key, record[key]]));
}

function allGeometry(model: FloorPlanModel): unknown {
  return kinds.map((kind) => ({ kind, items: (model[kind] ?? []).map((item, index) => {
    if (typeof item.id === "string") return targetGeometry(model, { kind, index, id: item.id });
    return item;
  }).map(stable).sort() }));
}

export function issueFingerprint(model: FloorPlanModel, issue: ReviewIssue): string {
  // Relationships (crossing, overlap, room enclosure) can change when any
  // neighboring wall moves. Room/source/whole-plan findings therefore cover
  // all structural geometry. A directly targeted opening covers its host wall
  // and sibling openings, without invalidating for unrelated furniture edits.
  const openingOnly = issue.targets.length > 0 && issue.targets.every((target) => target.kind === "doors" || target.kind === "windows");
  let geometry: unknown = allGeometry(model);
  if (openingOnly) {
    const hostIds = new Set(issue.targets.flatMap((target) => {
      const opening = model[target.kind as "doors" | "windows"].find((item) => item.id === target.id);
      return typeof opening?.wall_id === "string" ? [opening.wall_id] : [];
    }));
    const dependencies = [...issue.targets,
      ...findTargets(model, [...hostIds]),
      ...(["doors", "windows"] as const).flatMap((kind) => model[kind].flatMap((item, index) =>
        item.id && item.wall_id && hostIds.has(item.wall_id) ? [{ kind, index, id: item.id }] : [])),
    ];
    geometry = dependencies.map((target) => stable([target.kind, target.id, targetGeometry(model, target)])).sort();
  }
  return fingerprint({ issue: [issue.code, issue.message, issue.location ?? null], scale: model.pixels_per_metre,
    coordinateSystem: model.coordinate_system, geometry,
    fixtures: issue.targets.length === 0 || /furniture|collision|clearance|camera/.test(issue.code)
      ? [model.furniture, model.asset_placements] : undefined,
    // These alter review meaning even when wall positions do not change.
    ceilingVoids: model.metadata.ceiling_void_room_ids,
    floorInferences: model.metadata.floor_boundary_inferences,
  });
}

function acknowledgements(model: FloorPlanModel): Record<string, Acknowledgement> {
  const raw = model.metadata[ACK_KEY];
  if (!isObject(raw)) return {};
  return Object.fromEntries(Object.entries(raw).filter((entry): entry is [string, Acknowledgement] =>
    isObject(entry[1]) && typeof entry[1].fingerprint === "string" &&
    typeof entry[1].confirmed_at === "string" && Number.isFinite(Date.parse(entry[1].confirmed_at))));
}

export function acknowledgementState(model: FloorPlanModel, issue: ReviewIssue): "unreviewed" | "confirmed" | "stale" {
  const acknowledgement = acknowledgements(model)[issue.id];
  if (!acknowledgement) return "unreviewed";
  return acknowledgement.fingerprint === issueFingerprint(model, issue) ? "confirmed" : "stale";
}

export function acknowledgeIssue(model: FloorPlanModel, issue: ReviewIssue, confirmed: boolean, now = new Date()): FloorPlanModel {
  const entries = acknowledgements(model);
  if (confirmed) entries[issue.id] = { fingerprint: issueFingerprint(model, issue), confirmed_at: now.toISOString() };
  else delete entries[issue.id];
  // Preserve issues and quality exactly; acknowledging cannot bypass validation.
  return { ...model, metadata: { ...model.metadata, [ACK_KEY]: entries } };
}

export function reviewTargetPoints(model: FloorPlanModel, target: ReviewTarget): Point2D[] {
  const item = (model[target.kind] ?? []).find((entry) => entry.id === target.id) as Record<string, unknown> | undefined;
  if (!item) return [];
  const point = (value: unknown): value is Point2D => isObject(value) &&
    typeof value.x === "number" && Number.isFinite(value.x) && typeof value.y === "number" && Number.isFinite(value.y);
  if (point(item.start) && point(item.end)) return [item.start, item.end];
  if (Array.isArray(item.points)) return item.points.filter(point);
  if (Array.isArray(item.polygon) && item.polygon.length) return item.polygon.filter(point);
  return point(item.center) ? [item.center] : [];
}

export function issueGuidance(issue: ReviewIssue): string {
  if (issue.code.includes("scale")) return "Use scale (2 pts), choose two known points, and enter their measured distance. Then save and validate.";
  if (issue.code.includes("opening") || issue.code.includes("gap")) return "Compare the source plan. Select an opening to move it or change its width; use the wall, door, or window tool only where the source supports it. Save and validate afterward.";
  if (issue.code.includes("wall") || issue.code.includes("junction")) return "Select the highlighted wall and drag its endpoint handles, or delete and redraw it from the source. Then save and validate.";
  if (issue.code.includes("room") || issue.code.includes("floor") || issue.code.includes("ceiling")) return "Compare the highlighted boundary with the source. Correct enclosing walls if needed, then save to rebuild rooms. Ceiling or inferred-floor assumptions still require review before export.";
  return "Compare this finding with the source plan, correct the supported geometry using the editor tools, then save and validate. Confirm only after reviewing the uncertainty.";
}
