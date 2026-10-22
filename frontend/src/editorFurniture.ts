import type { FloorPlanModel, FurniturePlacement, Opening, Point2D, RoomPolygon, WallSegment } from "./types";

/** These families have corresponding furniture builders in both exporters. */
export const FURNITURE_PRESETS = [
  { category: "bed", label: "Bed", width_m: 1.6, depth_m: 2, height_m: 0.55 },
  { category: "sofa", label: "Sofa", width_m: 2.1, depth_m: 0.9, height_m: 0.85 },
  { category: "table", label: "Table", width_m: 1.4, depth_m: 0.8, height_m: 0.75 },
  { category: "chair", label: "Chair", width_m: 0.5, depth_m: 0.5, height_m: 0.85 },
  { category: "wardrobe", label: "Wardrobe", width_m: 1.8, depth_m: 0.6, height_m: 2.1 },
  { category: "rug", label: "Rug", width_m: 2, depth_m: 1.5, height_m: 0.03 },
  { category: "plant", label: "Plant", width_m: 0.45, depth_m: 0.45, height_m: 0.9 },
] as const;

export function furnitureLabel(category: string): string {
  return FURNITURE_PRESETS.find((preset) => preset.category === category.toLowerCase())?.label ??
    category.replace(/[_-]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function createFurniture(category: string, center: Point2D): FurniturePlacement {
  const preset = FURNITURE_PRESETS.find((candidate) => candidate.category === category);
  if (!preset) throw new Error("Choose a supported furniture type.");
  return {
    id: globalThis.crypto.randomUUID(), provenance: "user", category: preset.category,
    center: { ...center }, width_m: preset.width_m, depth_m: preset.depth_m,
    height_m: preset.height_m, rotation_deg: 0,
  };
}

/** Local width is x, local depth is y; positive rotation is CCW in metres. */
export function furnitureFootprint(item: FurniturePlacement): Point2D[] {
  const angle = ((item.rotation_deg ?? 0) * Math.PI) / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [
    { x: -item.width_m / 2, y: -item.depth_m / 2 },
    { x: item.width_m / 2, y: -item.depth_m / 2 },
    { x: item.width_m / 2, y: item.depth_m / 2 },
    { x: -item.width_m / 2, y: item.depth_m / 2 },
  ].map((point) => ({
    x: item.center.x + point.x * cos - point.y * sin,
    y: item.center.y + point.x * sin + point.y * cos,
  }));
}

function inside(point: Point2D, polygon: Point2D[]): boolean {
  let result = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const a = polygon[i];
    const b = polygon[j];
    if ((a.y > point.y) !== (b.y > point.y) &&
        point.x < a.x + ((point.y - a.y) * (b.x - a.x)) / (b.y - a.y)) result = !result;
  }
  return result;
}

function roomCentre(room: RoomPolygon): Point2D | null {
  const points = room.points;
  if (points.length < 3 || points.some((point) => !Number.isFinite(point.x) || !Number.isFinite(point.y))) return null;
  let twiceArea = 0;
  let xSum = 0;
  let ySum = 0;
  for (let index = 0; index < points.length; index += 1) {
    const a = points[index];
    const b = points[(index + 1) % points.length];
    const cross = a.x * b.y - b.x * a.y;
    twiceArea += cross;
    xSum += (a.x + b.x) * cross;
    ySum += (a.y + b.y) * cross;
  }
  if (Math.abs(twiceArea) > Number.EPSILON) {
    const centroid = { x: xSum / (3 * twiceArea), y: ySum / (3 * twiceArea) };
    if (inside(centroid, points)) return centroid;
  }

  // A concave polygon's centroid can be outside it. Each strip between vertex
  // heights has a stable set of crossings; midpoints of paired crossings are
  // strictly inside, even in a room too narrow for a fixed sampling grid.
  const heights = [...new Set(points.map((point) => point.y))].sort((a, b) => a - b);
  let best: Point2D | null = null;
  let bestWidth = 0;
  for (let index = 1; index < heights.length; index += 1) {
    const y = (heights[index - 1] + heights[index]) / 2;
    const crossings: number[] = [];
    for (let edge = 0; edge < points.length; edge += 1) {
      const a = points[edge];
      const b = points[(edge + 1) % points.length];
      if ((a.y > y) !== (b.y > y)) crossings.push(a.x + ((y - a.y) * (b.x - a.x)) / (b.y - a.y));
    }
    crossings.sort((a, b) => a - b);
    for (let crossing = 0; crossing + 1 < crossings.length; crossing += 2) {
      const width = crossings[crossing + 1] - crossings[crossing];
      const candidate = { x: (crossings[crossing] + crossings[crossing + 1]) / 2, y };
      if (width > bestWidth && inside(candidate, points)) {
        best = candidate;
        bestWidth = width;
      }
    }
  }
  return best;
}

/** The caller supplies the visible centre in metres, already clamped to the
 * source image. This places only the item's centre; its size may still overlap. */
export function addPosition(_model: FloorPlanModel, selectedRoom: RoomPolygon | null | undefined, visibleCentre: Point2D): Point2D {
  return (selectedRoom && roomCentre(selectedRoom)) || { ...visibleCentre };
}

export interface FurnitureOverlapWarning {
  kind: "wall" | "door";
  message: string;
  targetId?: string | null;
}

const EPSILON = 1e-8;

/** Separating axis test uses the actual rotated footprint, not its outer box.
 * Mere edge contact has no area and does not count as overlap. */
function intersects(a: Point2D[], b: Point2D[]): boolean {
  for (const polygon of [a, b]) {
    for (let index = 0; index < polygon.length; index += 1) {
      const first = polygon[index];
      const second = polygon[(index + 1) % polygon.length];
      const axis = { x: -(second.y - first.y), y: second.x - first.x };
      const axisLength = Math.hypot(axis.x, axis.y);
      if (axisLength <= EPSILON) continue;
      const aValues = a.map((point) => (point.x * axis.x + point.y * axis.y) / axisLength);
      const bValues = b.map((point) => (point.x * axis.x + point.y * axis.y) / axisLength);
      if (Math.min(Math.max(...aValues), Math.max(...bValues)) - Math.max(Math.min(...aValues), Math.min(...bValues)) <= EPSILON) return false;
    }
  }
  return true;
}

function wallBand(wall: WallSegment, start: number, end: number): Point2D[] {
  const length = Math.hypot(wall.end.x - wall.start.x, wall.end.y - wall.start.y);
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

function doorInterval(door: Opening, wall: WallSegment, length: number): [number, number] | null {
  let start: number;
  let end: number;
  if (door.start_offset_m != null && door.end_offset_m != null) {
    start = door.start_offset_m;
    end = door.end_offset_m;
  } else {
    const midpoint = door.offset_m ?? ((door.center.x - wall.start.x) * (wall.end.x - wall.start.x) +
      (door.center.y - wall.start.y) * (wall.end.y - wall.start.y)) / length;
    start = midpoint - door.width_m / 2;
    end = midpoint + door.width_m / 2;
  }
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  start = Math.max(0, start);
  end = Math.min(length, end);
  return end - start > EPSILON ? [start, end] : null;
}

/** Local placement hints only: checks wall solids and anchored door apertures.
 * It does not evaluate a walking route, door swing, or approach clearance. */
export function overlapWarnings(model: FloorPlanModel, item: FurniturePlacement): FurnitureOverlapWarning[] {
  const footprint = furnitureFootprint(item);
  if (item.width_m <= 0 || item.depth_m <= 0 || footprint.some((point) => !Number.isFinite(point.x) || !Number.isFinite(point.y))) return [];
  const warnings: FurnitureOverlapWarning[] = [];
  for (const wall of model.walls) {
    const length = Math.hypot(wall.end.x - wall.start.x, wall.end.y - wall.start.y);
    if (!Number.isFinite(length) || length <= EPSILON || !Number.isFinite(wall.thickness_m) || wall.thickness_m <= 0) continue;
    const openings = model.doors.flatMap((door) => {
      if (!wall.id || door.wall_id !== wall.id) return [];
      const interval = doorInterval(door, wall, length);
      return interval ? [{ door, interval }] : [];
    }).sort((a, b) => a.interval[0] - b.interval[0]);
    let cursor = 0;
    let overlapsWall = false;
    for (const { door, interval: [start, end] } of openings) {
      if (start > cursor + EPSILON && intersects(footprint, wallBand(wall, cursor, start))) overlapsWall = true;
      if (intersects(footprint, wallBand(wall, start, end))) {
        warnings.push({ kind: "door", message: "Overlaps a door opening.", targetId: door.id });
      }
      cursor = Math.max(cursor, end);
    }
    if (cursor < length - EPSILON && intersects(footprint, wallBand(wall, cursor, length))) overlapsWall = true;
    if (overlapsWall) warnings.push({ kind: "wall", message: "Overlaps a wall.", targetId: wall.id });
  }
  return warnings;
}
