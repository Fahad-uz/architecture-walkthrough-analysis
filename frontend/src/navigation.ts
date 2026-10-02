import type { FurniturePlacement, Point2D, RoomPolygon, WallSegment } from "./types";

export const CAPSULE_RADIUS = 0.3;

/** The asset list may be the detailed representation of an existing item. */
export function resolveFurniture(furniture: FurniturePlacement[], assets: FurniturePlacement[]): FurniturePlacement[] {
  const familyNames = new Set(["bed", "sofa", "couch", "table", "chair", "plant", "rug", "carpet", "wardrobe", "cabinet", "sink", "stove"]);
  const key = (item: FurniturePlacement) => {
    let family = item.category.toLowerCase().split(/[^a-z0-9]+/).find((token) => familyNames.has(token)) ?? item.category.toLowerCase();
    if (family === "couch") family = "sofa";
    if (family === "carpet") family = "rug";
    return [family, ...[item.center.x, item.center.y, item.width_m, item.depth_m, (((item.rotation_deg ?? 0) % 360) + 360) % 360].map((value) => value.toFixed(5))].join(":");
  };
  const result = [...furniture];
  const occupied = new Map(result.map((item, index) => [key(item), index]));
  for (const item of assets) {
    const identity = key(item);
    const index = occupied.get(identity);
    if (index == null) {
      occupied.set(identity, result.length);
      result.push(item);
    } else if (item.height_m != null) {
      result[index] = { ...result[index], height_m: item.height_m };
    }
  }
  return result;
}

export function pointInPolygon(point: Point2D, polygon: Point2D[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const a = polygon[i];
    const b = polygon[j];
    if (
      (a.y > point.y) !== (b.y > point.y) &&
      point.x < ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y || Number.EPSILON) + a.x
    ) {
      inside = !inside;
    }
  }
  return inside;
}

export function distanceToSegment(point: Point2D, a: Point2D, b: Point2D): number {
  const lengthSquared = (b.x - a.x) ** 2 + (b.y - a.y) ** 2;
  if (lengthSquared === 0) return Math.hypot(point.x - a.x, point.y - a.y);
  const t = Math.max(0, Math.min(1, ((point.x - a.x) * (b.x - a.x) + (point.y - a.y) * (b.y - a.y)) / lengthSquared));
  return Math.hypot(point.x - (a.x + t * (b.x - a.x)), point.y - (a.y + t * (b.y - a.y)));
}

export function clearanceFromWalls(point: Point2D, walls: WallSegment[]): number {
  let clearance = Number.POSITIVE_INFINITY;
  for (const wall of walls) {
    clearance = Math.min(
      clearance,
      distanceToSegment(point, wall.start, wall.end) - wall.thickness_m / 2,
    );
  }
  return clearance;
}

export function roomArea(room: RoomPolygon): number {
  let twiceArea = 0;
  for (let i = 0; i < room.points.length; i += 1) {
    const a = room.points[i];
    const b = room.points[(i + 1) % room.points.length];
    twiceArea += a.x * b.y - b.x * a.y;
  }
  return Math.abs(twiceArea) / 2;
}

/** Coarse polylabel: choose an interior point with the greatest sampled
 * clearance from the room boundary. Unlike a vertex average, this remains
 * inside concave rooms and avoids spawning the camera in a wall. */
export function insideFurniture(point: Point2D, item: FurniturePlacement, clearance = 0.35): boolean {
  if (isWalkableFurniture(item.category)) return false;
  const angle = (-(item.rotation_deg ?? 0) * Math.PI) / 180;
  const dx = point.x - item.center.x;
  const dy = point.y - item.center.y;
  const localX = dx * Math.cos(angle) - dy * Math.sin(angle);
  const localY = dx * Math.sin(angle) + dy * Math.cos(angle);
  return (
    Math.abs(localX) <= item.width_m / 2 + clearance &&
    Math.abs(localY) <= item.depth_m / 2 + clearance
  );
}

export function safeRoomPoint(
  room: RoomPolygon,
  furniture: FurniturePlacement[],
  walls: WallSegment[],
): Point2D | null {
  if (room.points.length < 3) return null;
  const xs = room.points.map((point) => point.x);
  const ys = room.points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  let best: Point2D | null = null;
  let bestClearance = -1;
  const samplesX = Math.min(64, Math.max(16, Math.ceil((maxX - minX) / 0.18)));
  const samplesY = Math.min(64, Math.max(16, Math.ceil((maxY - minY) / 0.18)));
  for (let xi = 0; xi < samplesX; xi += 1) {
    for (let yi = 0; yi < samplesY; yi += 1) {
      const candidate = {
        x: minX + ((xi + 0.5) / samplesX) * (maxX - minX),
        y: minY + ((yi + 0.5) / samplesY) * (maxY - minY),
      };
      if (!pointInPolygon(candidate, room.points)) continue;
      if (furniture.some((item) => insideFurniture(candidate, item))) continue;
      let clearance = Number.POSITIVE_INFINITY;
      for (let index = 0; index < room.points.length; index += 1) {
        clearance = Math.min(
          clearance,
          distanceToSegment(candidate, room.points[index], room.points[(index + 1) % room.points.length]),
        );
      }
      clearance = Math.min(clearance, clearanceFromWalls(candidate, walls));
      if (clearance > bestClearance) {
        best = candidate;
        bestClearance = clearance;
      }
    }
  }
  return bestClearance >= CAPSULE_RADIUS + 0.08 ? best : null;
}

export function isSafePlanPoint(
  point: Point2D,
  rooms: RoomPolygon[],
  furniture: FurniturePlacement[],
  walls: WallSegment[],
): boolean {
  const room = rooms.find((candidate) => pointInPolygon(point, candidate.points));
  if (!room || furniture.some((item) => insideFurniture(point, item))) return false;
  let clearance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < room.points.length; index += 1) {
    clearance = Math.min(
      clearance,
      distanceToSegment(point, room.points[index], room.points[(index + 1) % room.points.length]),
    );
  }
  clearance = Math.min(clearance, clearanceFromWalls(point, walls));
  return clearance >= CAPSULE_RADIUS + 0.08;
}

/** Find a useful first view direction without assuming that world -Z faces
 * into the room. Rays stop as soon as they approach a wall or furniture. */
export function openViewTarget(
  origin: Point2D,
  rooms: RoomPolygon[],
  furniture: FurniturePlacement[],
  walls: WallSegment[],
): Point2D | null {
  let best: Point2D | null = null;
  let bestReach = 0;
  const directionCount = 16;
  for (let index = 0; index < directionCount; index += 1) {
    const angle = (index / directionCount) * Math.PI * 2;
    for (let reach = 0.25; reach <= 2.5; reach += 0.25) {
      const candidate = {
        x: origin.x + Math.cos(angle) * reach,
        y: origin.y + Math.sin(angle) * reach,
      };
      if (!isSafePlanPoint(candidate, rooms, furniture, walls)) break;
      if (reach > bestReach) {
        best = candidate;
        bestReach = reach;
      }
    }
  }
  return best;
}

/** Prefer starting with a recognizable room feature in view. A clear wall is
 * technically collision-safe, but it makes the first impression feel broken. */
export function interestingViewTarget(
  origin: Point2D,
  rooms: RoomPolygon[],
  furniture: FurniturePlacement[],
  walls: WallSegment[],
): Point2D | null {
  const room = rooms.find((candidate) => pointInPolygon(origin, candidate.points));
  if (!room) return null;
  let best: { target: Point2D; score: number } | null = null;
  for (const item of furniture) {
    if (isWalkableFurniture(item.category) || !pointInPolygon(item.center, room.points)) continue;
    const distance = Math.hypot(item.center.x - origin.x, item.center.y - origin.y);
    if (distance < 0.8 || distance > 6) continue;
    let sightlineClear = true;
    const dx = item.center.x - origin.x;
    const dy = item.center.y - origin.y;
    const sideX = (-dy / distance) * 0.18;
    const sideY = (dx / distance) * 0.18;
    const sampleCount = Math.max(10, Math.ceil(distance / 0.08));
    for (let step = 1; step < sampleCount; step += 1) {
      const amount = step / sampleCount;
      const center = {
        x: origin.x + dx * amount,
        y: origin.y + dy * amount,
      };
      const samples = [
        center,
        { x: center.x + sideX, y: center.y + sideY },
        { x: center.x - sideX, y: center.y - sideY },
      ];
      if (
        samples.some((sample) => clearanceFromWalls(sample, walls) < 0.03) ||
        furniture.some(
          (blocker) =>
            blocker !== item &&
            !isWalkableFurniture(blocker.category) &&
            insideFurniture(center, blocker, 0.02),
        )
      ) {
        sightlineClear = false;
        break;
      }
    }
    if (!sightlineClear) continue;
    const category = item.category.toLowerCase();
    const interestPenalty = /(sofa|couch|chair|table|bed)/.test(category)
      ? 0
      : /(counter|sink|stove|plant)/.test(category)
        ? 0.35
        : 0.7;
    const score = Math.abs(distance - 3) + interestPenalty;
    if (!best || score < best.score) best = { target: item.center, score };
  }
  return best?.target ?? null;
}

export function isStructuralColliderName(name: string): boolean {
  return (
    (/^Wall_/i.test(name) && !/^Wall_Cap_/i.test(name)) ||
    /^Walls_Joined(?:_|$)/i.test(name) ||
    /^DoorLeaf_/i.test(name) ||
    /^Window(?:Glass|_)/i.test(name) ||
    /^Special_/i.test(name)
  );
}

export function furnitureProxyHeight(category: string): number {
  const normalized = category.toLowerCase().replace(/[- ]/g, "_");
  if (/(wardrobe|cabinet|closet|appliance)/.test(normalized)) return 1.8;
  if (/(counter|sink|stove|cooktop)/.test(normalized)) return 1.0;
  if (/(bed|sofa|couch|chair|table|plant)/.test(normalized)) return 0.9;
  return 1.0;
}

export function isWalkableFurniture(category: string): boolean {
  const normalized = category.toLowerCase().replace(/[- ]/g, "_");
  const tokens = new Set(normalized.split("_").filter(Boolean));
  return (
    normalized.includes("floor_patch") ||
    tokens.has("rug") ||
    tokens.has("carpet") ||
    tokens.has("door")
  );
}
