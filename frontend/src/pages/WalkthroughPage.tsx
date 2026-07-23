import { PointerLockControls, useGLTF, useProgress } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { EffectComposer, N8AO, SMAA } from "@react-three/postprocessing";
import nipplejs from "nipplejs";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ACESFilmicToneMapping,
  Box3,
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
import SceneErrorBoundary from "../components/SceneErrorBoundary";
import type { FloorPlanModel, FurniturePlacement, JobRecord, Point2D, RoomPolygon } from "../types";

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

function safeRoomPoint(room: RoomPolygon, furniture: FurniturePlacement[]): Point2D | null {
  if (room.points.length < 3) return null;
  const xs = room.points.map((point) => point.x);
  const ys = room.points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  let best: Point2D | null = null;
  let bestClearance = -1;
  const samples = 16;
  for (let xi = 0; xi < samples; xi += 1) {
    for (let yi = 0; yi < samples; yi += 1) {
      const candidate = {
        x: minX + ((xi + 0.5) / samples) * (maxX - minX),
        y: minY + ((yi + 0.5) / samples) * (maxY - minY),
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
  return clearance >= CAPSULE_RADIUS + 0.08;
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

function useCoarsePointer(): boolean {
  const [coarse, setCoarse] = useState(() => window.matchMedia("(pointer: coarse)").matches);
  useEffect(() => {
    const query = window.matchMedia("(pointer: coarse)");
    const update = () => setCoarse(query.matches);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  return coarse;
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
  joystick: React.MutableRefObject<{ x: number; y: number }>;
  tour: Curve<Vector3> | null;
  tourActive: boolean;
  onLeaveTour: () => void;
}

function Player({ collider, start, joystick, tour, tourActive, onLeaveTour }: PlayerProps) {
  const camera = useThree((s) => s.camera);
  const keys = useKeys();
  const position = useRef(start.clone());
  const tourT = useRef(0);
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
    }),
    [],
  );

  useEffect(() => {
    position.current.copy(start);
    camera.position.copy(start);
  }, [start, camera]);

  useEffect(() => {
    if (tourActive) tourT.current = 0;
  }, [tourActive, tour]);

  useFrame((_, rawDelta) => {
    const delta = Math.min(rawDelta, 0.05);
    if (tourActive && tour) {
      const duration = Math.max(tour.getLength() / 0.8, 1);
      tourT.current = Math.min(1, tourT.current + delta / duration);
      const point = tour.getPointAt(tourT.current);
      const ahead = tour.getPointAt(Math.min(1, tourT.current + 0.015));
      camera.position.copy(point);
      if (tourT.current < 1) camera.lookAt(ahead);
      position.current.copy(camera.position);
      if (
        tourT.current >= 1 ||
        keys.current.KeyW ||
        keys.current.KeyS ||
        keys.current.KeyA ||
        keys.current.KeyD ||
        Math.abs(joystick.current.x) > 0.05 ||
        Math.abs(joystick.current.y) > 0.05
      ) {
        tourT.current = 0;
        onLeaveTour();
      }
      return;
    }

    // Movement input: WASD + mobile joystick, camera-relative on the ground plane.
    camera.getWorldDirection(temp.forward);
    temp.forward.y = 0;
    temp.forward.normalize();
    temp.right.crossVectors(temp.forward, new Vector3(0, 1, 0));
    temp.move.set(0, 0, 0);
    const fw = (keys.current.KeyW ? 1 : 0) - (keys.current.KeyS ? 1 : 0) + joystick.current.y;
    const side = (keys.current.KeyD ? 1 : 0) - (keys.current.KeyA ? 1 : 0) + joystick.current.x;
    temp.move.addScaledVector(temp.forward, fw).addScaledVector(temp.right, side);
    if (temp.move.lengthSq() > 0) {
      temp.move.normalize().multiplyScalar(WALK_SPEED * (keys.current.ShiftLeft ? 2.0 : 1.0) * delta);
      position.current.add(temp.move);
    }
    position.current.y = EYE_HEIGHT;

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
    camera.position.copy(position.current);
  });
  return null;
}

function Scene({ url, onCollider }: { url: string; onCollider: (mesh: Mesh | null) => void }) {
  const { scene } = useGLTF(url, "/draco/");
  // Never mount drei's cached scene directly: previews and Strict Mode mounts
  // must not share mutable node visibility or transforms.
  const model = useMemo(() => scene.clone(true) as Group, [scene]);
  useEffect(() => {
    model.updateMatrixWorld(true);
    // Doors' leaves shouldn't block movement; everything else is static.
    const staticMeshes: Mesh[] = [];
    model.traverse((node) => {
      if (
        (node as Mesh).isMesh &&
        node.visible &&
        !/DoorLeaf/i.test(node.name) &&
        !/(ceiling|roof)/i.test(node.name)
      ) {
        staticMeshes.push(node as Mesh);
      }
    });
    if (staticMeshes.length === 0) {
      onCollider(null);
      return;
    }
    // Generate directly from the original meshes. Reparenting clones loses
    // transforms inherited from GLTF node groups and misaligns collisions.
    const generator = new StaticGeometryGenerator(staticMeshes);
    generator.applyWorldTransforms = true;
    generator.attributes = ["position"];
    const merged = generator.generate();
    (merged as any).boundsTree = new MeshBVH(merged);
    const collider = new Mesh(merged);
    collider.visible = false;
    onCollider(collider);
    return () => {
      onCollider(null);
      (merged as any).boundsTree?.dispose?.();
      merged.dispose();
    };
  }, [model, onCollider]);
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
  // Baked builds carry their lighting in the lightmaps; screen-space AO on
  // top double-darkens corners, so N8AO only runs for unbaked previews.
  const baked = job?.glb_source === "blender";

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
    if (firstWaypoint && isSafePlanPoint(firstWaypoint, rooms, furnitureObstacles)) {
      return planToWorld(firstWaypoint.x, firstWaypoint.y);
    }
    const roomsByArea = [...(plan?.rooms ?? [])].sort((a, b) => roomArea(b) - roomArea(a));
    for (const room of roomsByArea) {
      const point = safeRoomPoint(room, furnitureObstacles);
      if (point) return planToWorld(point.x, point.y);
    }
    if (
      plan?.entrance &&
      ((rooms.length === 0 && !furnitureObstacles.some((item) => insideFurniture(plan.entrance!, item))) ||
        isSafePlanPoint(plan.entrance, rooms, furnitureObstacles))
    ) {
      return planToWorld(plan.entrance.x, plan.entrance.y);
    }
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
            camera={{ position: start.toArray(), fov: 70, near: 0.05 }}
            gl={{ toneMapping: ACESFilmicToneMapping, outputColorSpace: SRGBColorSpace }}
          >
            <ambientLight intensity={0.35} />
            <hemisphereLight intensity={0.25} color="#cfe4ff" groundColor="#413a33" />
            <Suspense fallback={null}>
              <Scene key={sceneResetKey} url={url} onCollider={setCollider} />
            </Suspense>
            {collider && plan && (
              <Player
                collider={collider}
                start={start}
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
            {baked ? (
              <EffectComposer>
                <SMAA />
              </EffectComposer>
            ) : (
              <EffectComposer>
                <N8AO aoRadius={0.6} intensity={2.5} distanceFalloff={0.6} />
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
