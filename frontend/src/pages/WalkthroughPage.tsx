import { PointerLockControls, useGLTF, useProgress } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { EffectComposer, N8AO, SMAA, ToneMapping } from "@react-three/postprocessing";
import nipplejs from "nipplejs";
import { ToneMappingMode } from "postprocessing";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ACESFilmicToneMapping,
  Box3,
  BoxGeometry,
  Curve,
  CurvePath,
  Euler,
  Group,
  Line3,
  LineCurve3,
  MathUtils,
  Matrix4,
  Mesh,
  SRGBColorSpace,
  Vector3,
} from "three";
import { MeshBVH, StaticGeometryGenerator } from "three-mesh-bvh";
import { getEditData, getJob, versionedGlbUrl } from "../api";
import NeutralEnvironment from "../components/NeutralEnvironment";
import SceneErrorBoundary from "../components/SceneErrorBoundary";
import type {
  FloorPlanModel,
  FurniturePlacement,
  JobRecord,
  Point2D,
  RoomPolygon,
  WallSegment,
} from "../types";
import useCoarsePointer from "../useCoarsePointer";

const EYE_HEIGHT = 1.6;
const WALK_SPEED = 1.4; // m/s
const CAPSULE_RADIUS = 0.3;

/** The GLB is Y-up (metres); plan coords (x, y) map to world (x, -y). */
function planToWorld(x: number, y: number, height = EYE_HEIGHT): Vector3 {
  return new Vector3(x, height, -y);
}

function pointInPolygon(point: Point2D, polygon: Point2D[]): boolean {
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

function distanceToSegment(point: Point2D, a: Point2D, b: Point2D): number {
  const lengthSquared = (b.x - a.x) ** 2 + (b.y - a.y) ** 2;
  if (lengthSquared === 0) return Math.hypot(point.x - a.x, point.y - a.y);
  const t = Math.max(0, Math.min(1, ((point.x - a.x) * (b.x - a.x) + (point.y - a.y) * (b.y - a.y)) / lengthSquared));
  return Math.hypot(point.x - (a.x + t * (b.x - a.x)), point.y - (a.y + t * (b.y - a.y)));
}

function clearanceFromWalls(point: Point2D, walls: WallSegment[]): number {
  let clearance = Number.POSITIVE_INFINITY;
  for (const wall of walls) {
    clearance = Math.min(
      clearance,
      distanceToSegment(point, wall.start, wall.end) - wall.thickness_m / 2,
    );
  }
  return clearance;
}

function roomArea(room: RoomPolygon): number {
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
function insideFurniture(point: Point2D, item: FurniturePlacement, clearance = 0.35): boolean {
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

function safeRoomPoint(
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

function isSafePlanPoint(
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
function openViewTarget(
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
function interestingViewTarget(
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

function useKeys() {
  const keys = useRef<Record<string, boolean>>({});
  useEffect(() => {
    const down = (e: KeyboardEvent) => (keys.current[e.code] = true);
    const up = (e: KeyboardEvent) => (keys.current[e.code] = false);
    const clear = () => {
      keys.current = {};
    };
    const visibility = () => {
      if (document.hidden) clear();
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", clear);
    document.addEventListener("visibilitychange", visibility);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", clear);
      document.removeEventListener("visibilitychange", visibility);
    };
  }, []);
  return keys;
}

/** First-person drag look for touch screens. The left side remains reserved
 * for the movement joystick; dragging the right side controls yaw and pitch. */
function TouchLookControls({ enabled }: { enabled: boolean }) {
  const camera = useThree((state) => state.camera);
  const gl = useThree((state) => state.gl);
  const drag = useRef<{ pointerId: number; x: number; y: number } | null>(null);
  const rotation = useMemo(() => new Euler(0, 0, 0, "YXZ"), []);

  useEffect(() => {
    if (!enabled) return;
    const element = gl.domElement;
    const previousTouchAction = element.style.touchAction;
    element.style.touchAction = "none";

    const down = (event: PointerEvent) => {
      if (event.pointerType !== "touch" || !event.isPrimary) return;
      const bounds = element.getBoundingClientRect();
      if (event.clientX < bounds.left + bounds.width * 0.45) return;
      drag.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
      element.setPointerCapture(event.pointerId);
      event.preventDefault();
    };
    const move = (event: PointerEvent) => {
      if (!drag.current || drag.current.pointerId !== event.pointerId) return;
      const dx = event.clientX - drag.current.x;
      const dy = event.clientY - drag.current.y;
      drag.current.x = event.clientX;
      drag.current.y = event.clientY;
      rotation.setFromQuaternion(camera.quaternion);
      rotation.y -= dx * 0.004;
      rotation.x = MathUtils.clamp(rotation.x - dy * 0.004, -Math.PI * 0.48, Math.PI * 0.48);
      rotation.z = 0;
      camera.quaternion.setFromEuler(rotation);
      event.preventDefault();
    };
    const end = (event: PointerEvent) => {
      if (drag.current?.pointerId !== event.pointerId) return;
      drag.current = null;
      if (element.hasPointerCapture(event.pointerId)) element.releasePointerCapture(event.pointerId);
    };

    element.addEventListener("pointerdown", down, { passive: false });
    element.addEventListener("pointermove", move, { passive: false });
    element.addEventListener("pointerup", end);
    element.addEventListener("pointercancel", end);
    return () => {
      drag.current = null;
      element.style.touchAction = previousTouchAction;
      element.removeEventListener("pointerdown", down);
      element.removeEventListener("pointermove", move);
      element.removeEventListener("pointerup", end);
      element.removeEventListener("pointercancel", end);
    };
  }, [camera, enabled, gl, rotation]);

  return null;
}

interface PlayerProps {
  collider: Mesh | null;
  start: Vector3;
  initialLookAt: Vector3 | null;
  joystick: React.MutableRefObject<{ x: number; y: number }>;
  tour: Curve<Vector3> | null;
  tourActive: boolean;
  onLeaveTour: () => void;
}

function Player({
  collider,
  start,
  initialLookAt,
  joystick,
  tour,
  tourActive,
  onLeaveTour,
}: PlayerProps) {
  const camera = useThree((s) => s.camera);
  const keys = useKeys();
  const position = useRef(start.clone());
  const tourT = useRef(0);
  const previousTourPoint = useRef<Vector3 | null>(null);
  const tourJoinTarget = useRef<Vector3 | null>(null);
  const tourJoinLastDistance = useRef(Number.POSITIVE_INFINITY);
  const tourJoinStalledSeconds = useRef(0);
  const temp = useMemo(
    () => ({
      box: new Box3(),
      segment: new Line3(),
      vector: new Vector3(),
      vector2: new Vector3(),
      matrix: new Matrix4(),
      forward: new Vector3(),
      right: new Vector3(),
      move: new Vector3(),
      ahead: new Vector3(),
      tourExpected: new Vector3(),
    }),
    [],
  );

  useEffect(() => {
    position.current.copy(start);
    camera.position.copy(start);
    if (initialLookAt && initialLookAt.distanceToSquared(start) > 1e-6) {
      camera.lookAt(initialLookAt);
    }
  }, [camera, initialLookAt, start]);

  useEffect(() => {
    if (tourActive && tour) {
      // Join the route at its nearest point instead of teleporting back to
      // waypoint zero. This keeps a collision-safe manual spawn intact.
      let closestT = 0;
      let closestDistance = Number.POSITIVE_INFINITY;
      let closestPoint = tour.getPointAt(0);
      for (let index = 0; index <= 100; index += 1) {
        const amount = index / 100;
        const candidate = tour.getPointAt(amount);
        const distance = candidate.distanceToSquared(position.current);
        if (distance < closestDistance) {
          closestT = amount;
          closestDistance = distance;
          closestPoint = candidate;
        }
      }
      tourT.current = closestT;
      const joinDistance = closestPoint.distanceTo(position.current);
      if (joinDistance <= 0.03) {
        position.current.copy(closestPoint);
        previousTourPoint.current = closestPoint;
        tourJoinTarget.current = null;
      } else {
        previousTourPoint.current = null;
        tourJoinTarget.current = closestPoint.clone();
      }
      tourJoinLastDistance.current = joinDistance;
      tourJoinStalledSeconds.current = 0;
    } else {
      previousTourPoint.current = null;
      tourJoinTarget.current = null;
      tourJoinLastDistance.current = Number.POSITIVE_INFINITY;
      tourJoinStalledSeconds.current = 0;
    }
  }, [tourActive, tour]);

  useFrame((_, rawDelta) => {
    const delta = Math.min(rawDelta, 0.05);
    let finishTour = false;
    let orientTour = false;
    if (tourActive && tour) {
      const joinTarget = tourJoinTarget.current;
      if (joinTarget) {
        temp.move.copy(joinTarget).sub(position.current);
        const remaining = temp.move.length();
        if (remaining <= 0.03) {
          position.current.copy(joinTarget);
          previousTourPoint.current = joinTarget.clone();
          tourJoinTarget.current = null;
        } else {
          temp.ahead.copy(temp.move);
          temp.move.multiplyScalar(Math.min(1, (WALK_SPEED * 0.75 * delta) / remaining));
          position.current.add(temp.move);
          orientTour = true;
        }
      } else {
        const duration = Math.max(tour.getLength() / 0.8, 1);
        tourT.current = Math.min(1, tourT.current + delta / duration);
        const point = tour.getPointAt(tourT.current);
        const ahead = tour.getPointAt(Math.min(1, tourT.current + 0.04));
        if (previousTourPoint.current) {
          temp.move.copy(point).sub(previousTourPoint.current);
          position.current.add(temp.move);
        } else {
          position.current.copy(point);
        }
        previousTourPoint.current = point;
        temp.ahead.copy(ahead).sub(point);
        orientTour = tourT.current < 1;
        finishTour = tourT.current >= 1;
      }
      finishTour =
        finishTour ||
        keys.current.KeyW ||
        keys.current.KeyS ||
        keys.current.KeyA ||
        keys.current.KeyD ||
        Math.abs(joystick.current.x) > 0.05 ||
        Math.abs(joystick.current.y) > 0.05;
    } else {
      // Movement input: WASD + mobile joystick, camera-relative on the ground plane.
      camera.getWorldDirection(temp.forward);
      temp.forward.y = 0;
      temp.forward.normalize();
      temp.right.crossVectors(temp.forward, new Vector3(0, 1, 0));
      temp.move.set(0, 0, 0);
      const fw = (keys.current.KeyW ? 1 : 0) - (keys.current.KeyS ? 1 : 0) + joystick.current.y;
      const side =
        (keys.current.KeyD ? 1 : 0) - (keys.current.KeyA ? 1 : 0) + joystick.current.x;
      temp.move.addScaledVector(temp.forward, fw).addScaledVector(temp.right, side);
      if (temp.move.lengthSq() > 0) {
        temp.move
          .normalize()
          .multiplyScalar(WALK_SPEED * (keys.current.ShiftLeft ? 2.0 : 1.0) * delta);
        position.current.add(temp.move);
      }
    }
    position.current.y = EYE_HEIGHT;
    if (tourActive && tour) temp.tourExpected.copy(position.current);

    // Capsule-vs-BVH: push the capsule out of every intersecting triangle,
    // which naturally slides the player along walls.
    if (collider) {
      const bvh = (collider.geometry as any).boundsTree as MeshBVH | undefined;
      if (bvh) {
        temp.segment.start.copy(position.current);
        temp.segment.end.copy(position.current);
        temp.segment.start.y = position.current.y - EYE_HEIGHT + CAPSULE_RADIUS + 0.05;
        temp.segment.end.y = position.current.y;
        temp.box.makeEmpty();
        temp.box.expandByPoint(temp.segment.start);
        temp.box.expandByPoint(temp.segment.end);
        temp.box.min.addScalar(-CAPSULE_RADIUS);
        temp.box.max.addScalar(CAPSULE_RADIUS);
        bvh.shapecast({
          intersectsBounds: (box) => box.intersectsBox(temp.box),
          intersectsTriangle: (tri) => {
            const closestOnTri = temp.vector;
            const closestOnSegment = temp.vector2;
            const distance = tri.closestPointToSegment(temp.segment, closestOnTri, closestOnSegment);
            if (distance < CAPSULE_RADIUS) {
              const depth = CAPSULE_RADIUS - distance;
              const direction = closestOnSegment.sub(closestOnTri);
              direction.y = 0; // walls only; stay on the floor plane
              if (direction.lengthSq() > 1e-10) {
                direction.normalize();
                temp.segment.start.addScaledVector(direction, depth);
                temp.segment.end.addScaledVector(direction, depth);
              }
            }
            return false;
          },
        });
        position.current.x = temp.segment.end.x;
        position.current.z = temp.segment.end.z;
      }
    }
    if (
      tourActive &&
      tour &&
      Math.hypot(
        position.current.x - temp.tourExpected.x,
        position.current.z - temp.tourExpected.z,
      ) > 0.15
    ) {
      // A generated/legacy route has hit real geometry. Hand control back at
      // the resolved safe position instead of forcing the camera through it.
      finishTour = true;
    }
    if (tourActive && tourJoinTarget.current && !finishTour) {
      const joinDistance = position.current.distanceTo(tourJoinTarget.current);
      if (joinDistance < tourJoinLastDistance.current - 0.002) {
        tourJoinStalledSeconds.current = 0;
      } else {
        tourJoinStalledSeconds.current += delta;
      }
      tourJoinLastDistance.current = joinDistance;
      if (tourJoinStalledSeconds.current > 0.75) finishTour = true;
    }
    camera.position.copy(position.current);
    if (orientTour && temp.ahead.lengthSq() > 1e-8) {
      camera.lookAt(temp.vector.copy(position.current).add(temp.ahead));
    }
    if (finishTour) {
      tourT.current = 0;
      previousTourPoint.current = null;
      tourJoinTarget.current = null;
      tourJoinLastDistance.current = Number.POSITIVE_INFINITY;
      tourJoinStalledSeconds.current = 0;
      onLeaveTour();
    }
  });
  return null;
}

function isStructuralColliderName(name: string): boolean {
  return (
    (/^Wall_/i.test(name) && !/^Wall_Cap_/i.test(name)) ||
    /^Walls_Joined(?:_|$)/i.test(name) ||
    /^DoorLeaf_/i.test(name) ||
    /^Window(?:Glass|_)/i.test(name) ||
    /^Special_/i.test(name)
  );
}

function furnitureProxyHeight(category: string): number {
  const normalized = category.toLowerCase().replace(/[- ]/g, "_");
  if (/(wardrobe|cabinet|closet|appliance)/.test(normalized)) return 1.8;
  if (/(counter|sink|stove|cooktop)/.test(normalized)) return 1.0;
  if (/(bed|sofa|couch|chair|table|plant)/.test(normalized)) return 0.9;
  return 1.0;
}

function isWalkableFurniture(category: string): boolean {
  const normalized = category.toLowerCase().replace(/[- ]/g, "_");
  const tokens = new Set(normalized.split("_").filter(Boolean));
  return (
    normalized.includes("floor_patch") ||
    tokens.has("rug") ||
    tokens.has("carpet") ||
    tokens.has("door")
  );
}

function Scene({
  url,
  furniture,
  onCollider,
}: {
  url: string;
  furniture: FurniturePlacement[];
  onCollider: (mesh: Mesh | null) => void;
}) {
  const { scene } = useGLTF(url, "/draco/");
  // Never mount drei's cached scene directly: previews and Strict Mode mounts
  // must not share mutable node visibility or transforms.
  const model = useMemo(() => scene.clone(true) as Group, [scene]);
  useEffect(() => {
    model.updateMatrixWorld(true);
    // Keep only architectural barriers from the render model. Detailed
    // furniture meshes (cushions, handles, feet, pillows) create expensive,
    // snag-prone BVHs, so furniture uses one plan-grounded proxy per item.
    const staticMeshes: Mesh[] = [];
    model.traverse((node) => {
      if (
        (node as Mesh).isMesh &&
        node.visible &&
        isStructuralColliderName(node.name)
      ) {
        staticMeshes.push(node as Mesh);
      }
    });
    const proxies = furniture
      .filter((item) => !isWalkableFurniture(item.category))
      .map((item, index) => {
        const height = furnitureProxyHeight(item.category);
        const proxy = new Mesh(new BoxGeometry(item.width_m, height, item.depth_m));
        proxy.name = `FurnitureProxy_${index.toString().padStart(3, "0")}`;
        proxy.position.set(item.center.x, height / 2, -item.center.y);
        proxy.rotation.y = MathUtils.degToRad(item.rotation_deg ?? 0);
        proxy.updateMatrixWorld(true);
        staticMeshes.push(proxy);
        return proxy;
      });
    if (staticMeshes.length === 0) {
      for (const proxy of proxies) {
        proxy.geometry.dispose();
        if (!Array.isArray(proxy.material)) proxy.material.dispose();
      }
      onCollider(null);
      return;
    }
    // Generate directly from the original meshes. Reparenting clones loses
    // transforms inherited from GLTF node groups and misaligns collisions.
    const generator = new StaticGeometryGenerator(staticMeshes);
    generator.applyWorldTransforms = true;
    generator.attributes = ["position"];
    let merged;
    try {
      merged = generator.generate();
    } finally {
      for (const proxy of proxies) {
        proxy.geometry.dispose();
        if (Array.isArray(proxy.material)) {
          for (const material of proxy.material) material.dispose();
        } else {
          proxy.material.dispose();
        }
      }
    }
    (merged as any).boundsTree = new MeshBVH(merged);
    const collider = new Mesh(merged);
    collider.visible = false;
    onCollider(collider);
    return () => {
      onCollider(null);
      (merged as any).boundsTree?.dispose?.();
      merged.dispose();
    };
  }, [furniture, model, onCollider]);
  return <primitive object={model} />;
}

function LoadingOverlay({ ready }: { ready: boolean }) {
  const { progress, active } = useProgress();
  if (ready) return null;
  return (
    <div className="loading-overlay" role="status" aria-live="polite">
      {active ? `Loading model… ${progress.toFixed(0)}%` : "Preparing collision-safe walkthrough…"}
    </div>
  );
}

export default function WalkthroughPage() {
  const { jobId = "" } = useParams();
  const [collider, setCollider] = useState<Mesh | null>(null);
  const [plan, setPlan] = useState<FloorPlanModel | null>(null);
  const [job, setJob] = useState<JobRecord | null>(null);
  const [tourActive, setTourActive] = useState(false);
  const [locked, setLocked] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [sceneError, setSceneError] = useState<string | null>(null);
  const [sceneRetry, setSceneRetry] = useState(0);
  const joystick = useRef({ x: 0, y: 0 });
  const coarsePointer = useCoarsePointer();
  // Versioned URL: never render a stale cached GLB after a Blender rebuild.
  const url = job?.glb_url ? versionedGlbUrl(jobId, job.glb_version) : null;
  const sceneResetKey = url ? `${url}:${sceneRetry}` : null;
  // New jobs publish the actual bake preset. Fall back to the legacy source
  // marker only for records created before bake-mode metadata existed.
  const baked =
    job?.glb_bake_mode == null
      ? job?.glb_source === "blender"
      : job.glb_bake_mode !== "none";

  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;
    let knownVersion: number | undefined;
    let planLoaded = false;
    const refresh = async () => {
      try {
        const nextJob = await getJob(jobId);
        if (cancelled) return;
        setJob(nextJob);
        setRefreshError(null);
        if (!planLoaded || nextJob.glb_version !== knownVersion) {
          try {
            const data = await getEditData(jobId);
            if (!cancelled) {
              setPlan(data.model);
              knownVersion = nextJob.glb_version;
              planLoaded = true;
            }
          } catch (error) {
            if (!cancelled) {
              setRefreshError(`Could not refresh the walkthrough layout: ${String(error)}`);
            }
          }
        }
      } catch (error) {
        if (!cancelled) setRefreshError(`Could not refresh model status: ${String(error)}`);
      } finally {
        if (!cancelled) timer = window.setTimeout(() => void refresh(), 2000);
      }
    };
    void refresh();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [jobId]);

  useEffect(() => {
    setCollider(null);
    setTourActive(false);
    setSceneError(null);
    setSceneRetry(0);
  }, [url]);

  useEffect(() => {
    if (!coarsePointer) return;
    const zone = document.getElementById("joystick-zone");
    if (!zone) return;
    const manager = nipplejs.create({ zone, mode: "static", position: { left: "80px", bottom: "80px" } });
    manager.on("move", (_, data) => {
      const radians = data.angle?.radian ?? 0;
      const force = Math.min(data.force ?? 0, 1);
      joystick.current = { x: Math.cos(radians) * force, y: Math.sin(radians) * force };
    });
    manager.on("end", () => (joystick.current = { x: 0, y: 0 }));
    return () => {
      joystick.current = { x: 0, y: 0 };
      manager.destroy();
    };
  }, [coarsePointer]);

  const furnitureObstacles = useMemo(
    () => [...(plan?.furniture ?? []), ...(plan?.asset_placements ?? [])],
    [plan],
  );
  const start = useMemo(() => {
    const firstWaypoint = plan?.camera_waypoints?.[0]?.position;
    const rooms = plan?.rooms ?? [];
    const walls = plan?.walls ?? [];
    if (firstWaypoint && isSafePlanPoint(firstWaypoint, rooms, furnitureObstacles, walls)) {
      return planToWorld(firstWaypoint.x, firstWaypoint.y);
    }
    const roomsByArea = [...(plan?.rooms ?? [])].sort((a, b) => roomArea(b) - roomArea(a));
    for (const room of roomsByArea) {
      const point = safeRoomPoint(room, furnitureObstacles, walls);
      if (point) return planToWorld(point.x, point.y);
    }
    if (
      plan?.entrance &&
      ((rooms.length === 0 && !furnitureObstacles.some((item) => insideFurniture(plan.entrance!, item))) ||
        isSafePlanPoint(plan.entrance, rooms, furnitureObstacles, walls))
    ) {
      return planToWorld(plan.entrance.x, plan.entrance.y);
    }
    // Production waypoints are generated from the same capsule clearance as
    // the player. If the conservative browser-side room test cannot classify
    // a legacy/doorway point, it is still a better collision-resolved fallback
    // than an unrelated hard-coded coordinate.
    if (firstWaypoint) return planToWorld(firstWaypoint.x, firstWaypoint.y);
    return new Vector3(2, EYE_HEIGHT, -2);
  }, [furnitureObstacles, plan]);

  const tour = useMemo(() => {
    const waypoints = plan?.camera_waypoints ?? [];
    if (waypoints.length < 2) return null;
    const path = new CurvePath<Vector3>();
    for (let index = 1; index < waypoints.length; index += 1) {
      const from = waypoints[index - 1].position;
      const to = waypoints[index].position;
      path.add(new LineCurve3(planToWorld(from.x, from.y), planToWorld(to.x, to.y)));
    }
    return path;
  }, [plan]);

  const initialLookAt = useMemo(() => {
    const origin = { x: start.x, y: -start.z };
    const waypoints = plan?.camera_waypoints ?? [];
    const matchingIndex = waypoints.findIndex(
      (waypoint) =>
        Math.hypot(waypoint.position.x - origin.x, waypoint.position.y - origin.y) < 0.25,
    );
    if (matchingIndex >= 0) {
      const waypoint = waypoints[matchingIndex];
      const explicitTarget = waypoint.look_at;
      if (
        explicitTarget &&
        Math.hypot(explicitTarget.x - origin.x, explicitTarget.y - origin.y) >= 0.25
      ) {
        return planToWorld(explicitTarget.x, explicitTarget.y);
      }
    }
    const rooms = plan?.rooms ?? [];
    const walls = plan?.walls ?? [];
    const featureTarget = interestingViewTarget(origin, rooms, furnitureObstacles, walls);
    if (featureTarget) return planToWorld(featureTarget.x, featureTarget.y, 0.8);
    const openTarget = openViewTarget(origin, rooms, furnitureObstacles, walls);
    if (openTarget) return planToWorld(openTarget.x, openTarget.y);
    for (let index = Math.max(0, matchingIndex + 1); index < waypoints.length; index += 1) {
      const target = waypoints[index].position;
      if (Math.hypot(target.x - origin.x, target.y - origin.y) >= 0.25) {
        return planToWorld(target.x, target.y);
      }
    }
    return null;
  }, [furnitureObstacles, plan, start]);

  useEffect(() => {
    if (!tour) setTourActive(false);
  }, [tour]);

  return (
    <div className="walkthrough-root">
      <div className="hud">
        <Link to={`/jobs/${jobId}/preview`} className="walkthrough-exit">
          ← exit walkthrough
        </Link>
        <div className="row">
          {tour && (
            <button
              type="button"
              aria-pressed={tourActive}
              disabled={!collider}
              onClick={() => setTourActive((v) => !v)}
            >
              {tourActive ? "take manual control" : "guided tour"}
            </button>
          )}
        </div>
      </div>
      {!locked && !tourActive && collider && (
        <div className="hint" role="status">
          {coarsePointer
            ? "Use the left joystick to walk; drag the right side to look around"
            : "Click to look around — WASD to walk, Shift to hurry, Esc to release"}
        </div>
      )}
      {url ? (
        <SceneErrorBoundary
          resetKey={sceneResetKey}
          fallback={
            <div className="walkthrough-error" role="alert">
              <span>This model could not be loaded for walkthrough.</span>
              <button
                type="button"
                onClick={() => {
                  useGLTF.clear(url);
                  setSceneError(null);
                  setSceneRetry((value) => value + 1);
                }}
              >
                Retry model
              </button>
            </div>
          }
          onError={(error) => setSceneError(error.message)}
        >
          <Canvas
            aria-label="First-person architectural walkthrough"
            camera={{ position: start.toArray(), fov: coarsePointer ? 68 : 62, near: 0.05 }}
            dpr={coarsePointer ? [1, 1.25] : [1, 1.75]}
            gl={{ toneMapping: ACESFilmicToneMapping, outputColorSpace: SRGBColorSpace }}
          >
            <color attach="background" args={["#c9ced3"]} />
            <NeutralEnvironment />
            <ambientLight intensity={0.32} />
            <hemisphereLight intensity={0.34} color="#ffffff" groundColor="#8f9498" />
            {job?.glb_source === "preview" && (
              <directionalLight position={[6, 9, 4]} intensity={0.9} />
            )}
            <Suspense fallback={null}>
              <Scene
                key={sceneResetKey}
                url={url}
                furniture={plan?.furniture ?? []}
                onCollider={setCollider}
              />
            </Suspense>
            {collider && plan && (
              <Player
                collider={collider}
                start={start}
                initialLookAt={initialLookAt}
                joystick={joystick}
                tour={tour}
                tourActive={tourActive}
                onLeaveTour={() => setTourActive(false)}
              />
            )}
            {collider && !tourActive && !coarsePointer && (
              <PointerLockControls onLock={() => setLocked(true)} onUnlock={() => setLocked(false)} />
            )}
            <TouchLookControls enabled={Boolean(collider && coarsePointer && !tourActive)} />
            {!baked && (
              <EffectComposer multisampling={0}>
                <N8AO
                  aoRadius={0.35}
                  intensity={1.8}
                  distanceFalloff={1}
                  quality={coarsePointer ? "performance" : "medium"}
                  halfRes={coarsePointer}
                  depthAwareUpsampling
                />
                <ToneMapping mode={ToneMappingMode.ACES_FILMIC} />
                <SMAA />
              </EffectComposer>
            )}
          </Canvas>
        </SceneErrorBoundary>
      ) : (
        <div className="walkthrough-error" role="status">
          {refreshError ?? "Waiting for a generated 3D model…"}
        </div>
      )}
      {coarsePointer && collider && !tourActive && <div id="joystick-zone" aria-label="Movement joystick" />}
      {url && !sceneError && <LoadingOverlay ready={Boolean(collider && plan)} />}
      {refreshError && job && <div className="walkthrough-notice">{refreshError}</div>}
    </div>
  );
}
