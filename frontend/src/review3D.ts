import { collectReviewIssues, reviewTargetPoints, type ReviewIssue, type ReviewTarget } from "./editorWarnings.ts";
import type { FloorPlanModel, Opening, Point2D, WallSegment } from "./types";

export type ReviewWorldPoint = [number, number, number];
export interface Review3DShape {
  target: ReviewTarget;
  paths: ReviewWorldPoint[][];
}
export interface Review3DFocus {
  issue?: ReviewIssue;
  shapes: Review3DShape[];
  bounds?: { min: ReviewWorldPoint; max: ReviewWorldPoint };
  reason?: string;
}

export function reviewWorldPoint(point: Point2D, height = 0): ReviewWorldPoint {
  return [point.x, height, -point.y];
}

export function findingPath(jobId: string, page: "edit" | "preview", findingId: string): string {
  return `/jobs/${encodeURIComponent(jobId)}/${page}?${new URLSearchParams({ finding: findingId })}`;
}

const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const positive = (value: unknown): value is number => finite(value) && value > 0;
const height = (value: unknown, fallback: number, ceiling = 20): number =>
  Math.min(positive(value) ? value : fallback, ceiling);

function prism(points: Point2D[], bottom: number, top: number): ReviewWorldPoint[][] {
  const lower = points.map((point) => reviewWorldPoint(point, bottom));
  const upper = points.map((point) => reviewWorldPoint(point, top));
  return [[...lower, lower[0]], [...upper, upper[0]], ...lower.map((point, index) => [point, upper[index]])];
}

function wallBand(wall: WallSegment, start: number, end: number): Point2D[] | null {
  if (![wall.start.x, wall.start.y, wall.end.x, wall.end.y].every(finite)) return null;
  const length = Math.hypot(wall.end.x - wall.start.x, wall.end.y - wall.start.y);
  if (!positive(length) || !positive(wall.thickness_m) || !finite(start) || !finite(end) || end <= start) return null;
  const x = (wall.end.x - wall.start.x) / length;
  const y = (wall.end.y - wall.start.y) / length;
  const half = wall.thickness_m / 2;
  return [
    { x: wall.start.x + x * start - y * half, y: wall.start.y + y * start + x * half },
    { x: wall.start.x + x * end - y * half, y: wall.start.y + y * end + x * half },
    { x: wall.start.x + x * end + y * half, y: wall.start.y + y * end - x * half },
    { x: wall.start.x + x * start + y * half, y: wall.start.y + y * start - x * half },
  ];
}

function openingSpan(opening: Opening, wall: WallSegment): [number, number] | null {
  const length = Math.hypot(wall.end.x - wall.start.x, wall.end.y - wall.start.y);
  if (!positive(length)) return null;
  let span: [number, number];
  if (opening.start_offset_m != null || opening.end_offset_m != null) {
    if (!finite(opening.start_offset_m) || !finite(opening.end_offset_m)) return null;
    span = [opening.start_offset_m, opening.end_offset_m];
  } else {
    if (!positive(opening.width_m) || !finite(opening.offset_m)) return null;
    span = [opening.offset_m - opening.width_m / 2, opening.offset_m + opening.width_m / 2];
  }
  // A rendered aperture needs a verified interval inside its identified host.
  // Neither a stale centre nor clipping an invalid span establishes that aperture.
  return finite(span[0]) && finite(span[1]) && span[0] >= 0 && span[1] <= length && span[1] > span[0] ? span : null;
}

function targetShape(model: FloorPlanModel, target: ReviewTarget, modelHeight: number): Review3DShape | null {
  const matches = (model[target.kind] ?? []).filter((item) => item.id === target.id);
  if (matches.length !== 1) return null;
  const points = reviewTargetPoints(model, target);
  if (!points.length && target.kind !== "doors" && target.kind !== "windows") return null;
  const item = matches[0];
  let paths: ReviewWorldPoint[][];
  if (target.kind === "walls") {
    if (points.length !== 2) return null;
    const wall = item as WallSegment;
    const polygon = wallBand(wall, 0, Math.hypot(wall.end.x - wall.start.x, wall.end.y - wall.start.y));
    if (!polygon) return null;
    paths = prism(polygon, 0, height(wall.height_m, modelHeight));
  } else if (target.kind === "doors" || target.kind === "windows") {
    const opening = item as Opening;
    const walls = model.walls.filter((wall) => !!opening.wall_id && wall.id === opening.wall_id);
    // A centre alone cannot establish an aperture's orientation. Never guess a host.
    if (walls.length !== 1 || reviewTargetPoints(model, { kind: "walls", id: opening.wall_id!, index: 0 }).length !== 2) return null;
    const wall = walls[0];
    const span = openingSpan(opening, wall);
    const polygon = span && wallBand(wall, span[0], span[1]);
    if (!polygon) return null;
    const wallHeight = height(wall.height_m, modelHeight);
    const sill = target.kind === "windows" ? opening.sill_height_m ?? 0.9 : 0;
    if (!finite(sill) || sill < 0 || sill >= wallHeight) return null;
    const top = Math.min(wallHeight, sill + height(opening.height_m, target.kind === "doors" ? 2.1 : 1.2, wallHeight));
    paths = prism(polygon, sill, top);
  } else if (["rooms", "balconies", "slabs"].includes(target.kind)) {
    const polygon = item as { points?: unknown; thickness_m?: unknown };
    if (!Array.isArray(polygon.points) || points.length !== polygon.points.length || points.length < 3 ||
        new Set(points.map((point) => `${point.x},${point.y}`)).size < 3) return null;
    if (target.kind === "slabs") {
      // Both renderers build slabs downward from the plan's ground plane.
      paths = prism(points, -height(polygon.thickness_m, 0.1, modelHeight), 0);
    } else {
      const outline = points.map((point) => reviewWorldPoint(point));
      paths = [[...outline, outline[0]]];
    }
  } else return null;
  if (paths.some((path) => path.some((point) => !point.every(finite)))) return null;
  return { target, paths };
}

/** Only saved, identified objects may create 3D review markers. Source-image
 * annotations and whole-plan findings do not establish a 3D location. */
export function reviewIssue3DFocus(model: FloorPlanModel, issue: ReviewIssue): Review3DFocus {
  const unavailable = (reason: string): Review3DFocus => ({ issue, shapes: [], reason });
  if (!["metres", "cartesian_metres"].includes(model.coordinate_system)) return unavailable("This plan has no verified metre coordinates for a 3D location.");
  if (issue.referencedIds?.some((id) => !issue.targets.some((target) => target.id === id))) return unavailable("A referenced object is no longer present in the saved plan. Review the current finding in the source editor.");
  if (!issue.targets.length) return unavailable(issue.location
    ? "This finding marks the source image only; no verified 3D object is linked."
    : "This finding has no verified 3D object location.");
  if (new Set(issue.targets.map((target) => target.id)).size !== issue.targets.length) return unavailable("The referenced object IDs are ambiguous; no 3D marker is available.");
  const wallHeights = model.walls.map((wall) => wall.height_m).filter(positive);
  const modelHeight = height(wallHeights.length ? Math.max(...wallHeights) : undefined, 3);
  const shapes = issue.targets.flatMap((target) => {
    const shape = targetShape(model, target, modelHeight);
    return shape ? [shape] : [];
  });
  if (shapes.length !== issue.targets.length) return unavailable("A referenced object is missing, unsupported, or has no verified 3D bounds. Review it in the source editor.");
  const points = shapes.flatMap((shape) => shape.paths.flat());
  const min = [0, 1, 2].map((axis) => Math.min(...points.map((point) => point[axis]))) as ReviewWorldPoint;
  const max = [0, 1, 2].map((axis) => Math.max(...points.map((point) => point[axis]))) as ReviewWorldPoint;
  if (!min.every(finite) || !max.every(finite)) return unavailable("The referenced object has invalid 3D bounds.");
  return { issue, shapes, bounds: { min, max } };
}

export function resolveReview3DFocus(model: FloorPlanModel, findingId: string): Review3DFocus {
  const issue = collectReviewIssues(model).find((candidate) => candidate.id === findingId);
  return issue ? reviewIssue3DFocus(model, issue) : {
    shapes: [], reason: "This finding is no longer present in the saved plan. Return to the editor to review the current findings.",
  };
}

export function review3DLinkState(focus: Review3DFocus | null, state: { dirty: boolean; saving: boolean; modelAvailable: boolean }): { enabled: boolean; reason?: string } {
  if (!focus?.bounds) return { enabled: false, reason: focus?.reason ?? "This finding has no verified 3D object location." };
  if (state.dirty || state.saving) return { enabled: false, reason: "Save changes first to view this finding in 3D." };
  if (!state.modelAvailable) return { enabled: false, reason: "A saved 3D model is unavailable. Save changes or generate a model first." };
  return { enabled: true };
}
