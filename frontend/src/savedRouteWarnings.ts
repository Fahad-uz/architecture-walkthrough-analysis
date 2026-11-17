import { CAPSULE_RADIUS, distanceToSegment, isWalkableFurniture } from "./navigation.ts";
import type { FloorPlanModel, FurniturePlacement, Point2D } from "./types";

export interface SavedRouteWarning {
  kind: "saved_route" | "saved_route_stale";
  message: string;
}

const object = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const point = (value: unknown): value is Point2D => object(value) && finite(value.x) && finite(value.y);
const metres = (value: string): boolean => ["metres", "cartesian_metres"].includes(value);

/** Compare complete geometry values, rather than a rounded or collision-prone
 * hash. Element order and rendering/semantic properties do not change the frame. */
function stable(value: unknown): string {
  if (typeof value === "number" && !finite(value)) throw new Error("Non-finite geometry");
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  if (object(value)) return `{${Object.keys(value).filter((key) => value[key] !== undefined).sort()
    .map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(",")}}`;
  return JSON.stringify(value) ?? "null";
}

function geometrySignature(model: FloorPlanModel): string | null {
  if (model.pixels_per_metre != null && (!finite(model.pixels_per_metre) || model.pixels_per_metre <= 0)) return null;
  const fields = ["id", "face_id", "start", "end", "points", "polygon", "center", "wall_id", "width_m", "depth_m",
    "height_m", "thickness_m", "rotation_deg", "sill_height_m", "offset_m", "start_offset_m", "end_offset_m",
    "external", "wall_type", "opening_type", "hinge_side", "swing_side", "kind", "metadata"];
  try {
    const geometry = ["walls", "doors", "windows", "rooms", "balconies", "slabs", "special_elements"].map((kind) => {
      const items = model[kind] ?? [];
      if (!Array.isArray(items) || !items.every(object)) throw new Error("Invalid geometry");
      return [kind, items.map((item) => stable(Object.fromEntries(fields.filter((field) => item[field] !== undefined)
        .map((field) => [field, item[field]])))).sort()];
    });
    return stable({ geometry, coordinateSystem: model.coordinate_system, schemaVersion: model.schema_version,
      scale: model.pixels_per_metre ?? null, roi: model.plan_roi, entrance: model.entrance ?? null,
      ceilingVoids: model.metadata?.ceiling_void_room_ids, floorInferences: model.metadata?.floor_boundary_inferences });
  } catch {
    return null;
  }
}

/** Only adjacent, finite saved points establish a route segment. An invalid
 * waypoint breaks adjacency; neither entrances nor room centres fill gaps. */
function savedSegments(model: FloorPlanModel): [Point2D, Point2D][] {
  const segments: [Point2D, Point2D][] = [];
  if (!Array.isArray(model.camera_waypoints)) return segments;
  let previous: Point2D | null = null;
  for (const waypoint of model.camera_waypoints) {
    const position = object(waypoint) && point(waypoint.position) ? waypoint.position : null;
    if (previous && position && (previous.x !== position.x || previous.y !== position.y)) segments.push([previous, position]);
    previous = position;
  }
  return segments;
}

/** Exact distance to the oriented rectangle: transform the whole segment to
 * local furniture coordinates, clip it against the rectangle, then compare
 * endpoints and corners. This retains rounded capsule clearance at corners. */
function segmentFootprintDistance(a: Point2D, b: Point2D, item: FurniturePlacement): number {
  const angle = -((item.rotation_deg ?? 0) % 360) * Math.PI / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const local = (p: Point2D): Point2D => ({
    x: (p.x - item.center.x) * cos - (p.y - item.center.y) * sin,
    y: (p.x - item.center.x) * sin + (p.y - item.center.y) * cos,
  });
  const start = local(a);
  const end = local(b);
  if (!point(start) || !point(end)) return Infinity;
  const half = { x: item.width_m / 2, y: item.depth_m / 2 };
  let first = 0;
  let last = 1;
  for (const axis of ["x", "y"] as const) {
    const delta = end[axis] - start[axis];
    if (delta === 0) {
      if (Math.abs(start[axis]) > half[axis]) { first = Infinity; break; }
    } else {
      const enter = (-half[axis] - start[axis]) / delta;
      const leave = (half[axis] - start[axis]) / delta;
      first = Math.max(first, Math.min(enter, leave));
      last = Math.min(last, Math.max(enter, leave));
      if (first > last) break;
    }
  }
  if (first <= last) return 0;
  const corners = [
    { x: -half.x, y: -half.y }, { x: half.x, y: -half.y },
    { x: half.x, y: half.y }, { x: -half.x, y: half.y },
  ];
  let distance = Infinity;
  corners.forEach((corner, index) => {
    const next = corners[(index + 1) % corners.length];
    distance = Math.min(distance, distanceToSegment(corner, start, end),
      distanceToSegment(start, corner, next), distanceToSegment(end, corner, next));
  });
  return distance;
}

/** A local placement hint against an existing saved tour, not a route planner
 * or a connectivity/route-validity/door-swing check. The caller must supply the
 * saved source snapshot, never the live edited/scaled camera_waypoints array. */
export function savedRouteWarnings(
  current: FloorPlanModel,
  item: FurniturePlacement,
  saved: FloorPlanModel | null,
): SavedRouteWarning[] {
  if (!saved || !metres(saved.coordinate_system) || typeof item.category !== "string" ||
      isWalkableFurniture(item.category) || !point(item.center) || !finite(item.width_m) || item.width_m <= 0 ||
      !finite(item.depth_m) || item.depth_m <= 0 || (item.rotation_deg !== undefined && !finite(item.rotation_deg))) return [];
  const segments = savedSegments(saved);
  if (!segments.length) return [];
  const sourceGeometry = geometrySignature(saved);
  if (!metres(current.coordinate_system) || sourceGeometry === null || sourceGeometry !== geometrySignature(current)) {
    return [{ kind: "saved_route_stale", message: "The layout or scale changed since the saved tour. Save changes to regenerate and review the route before checking furniture placement." }];
  }
  // A nanometre tolerance keeps floating-point roundoff from missing contact.
  return segments.some(([a, b]) => segmentFootprintDistance(a, b, item) <= CAPSULE_RADIUS + 1e-9)
    ? [{ kind: "saved_route", message: "This furniture overlaps the saved tour path. Move it or save to regenerate and review the route." }]
    : [];
}
