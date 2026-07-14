import { PointerLockControls, useGLTF, useProgress } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { EffectComposer, N8AO, SMAA } from "@react-three/postprocessing";
import nipplejs from "nipplejs";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ACESFilmicToneMapping,
  Box3,
  CatmullRomCurve3,
  Group,
  Line3,
  Matrix4,
  Mesh,
  SRGBColorSpace,
  Vector3,
} from "three";
import { MeshBVH, StaticGeometryGenerator } from "three-mesh-bvh";
import { getEditData, getJob, versionedGlbUrl } from "../api";
import type { FloorPlanModel, JobRecord } from "../types";

const EYE_HEIGHT = 1.6;
const WALK_SPEED = 1.4; // m/s
const CAPSULE_RADIUS = 0.3;

/** The GLB is Y-up (metres); plan coords (x, y) map to world (x, -y). */
function planToWorld(x: number, y: number, height = EYE_HEIGHT): Vector3 {
  return new Vector3(x, height, -y);
}

function useKeys() {
  const keys = useRef<Record<string, boolean>>({});
  useEffect(() => {
    const down = (e: KeyboardEvent) => (keys.current[e.code] = true);
    const up = (e: KeyboardEvent) => (keys.current[e.code] = false);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);
  return keys;
}

interface PlayerProps {
  collider: Mesh | null;
  start: Vector3;
  joystick: React.MutableRefObject<{ x: number; y: number }>;
  tour: CatmullRomCurve3 | null;
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

  useFrame((_, rawDelta) => {
    const delta = Math.min(rawDelta, 0.05);
    if (tourActive && tour) {
      tourT.current = (tourT.current + delta * 0.03) % 1;
      const point = tour.getPointAt(tourT.current);
      const ahead = tour.getPointAt((tourT.current + 0.015) % 1);
      camera.position.lerp(point, 0.12);
      camera.lookAt(ahead);
      position.current.copy(camera.position);
      if (keys.current.KeyW || keys.current.KeyS || keys.current.KeyA || keys.current.KeyD) onLeaveTour();
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
              const direction = closestOnSegment.sub(closestOnTri).normalize();
              direction.y = 0; // walls only; stay on the floor plane
              temp.segment.start.addScaledVector(direction, depth);
              temp.segment.end.addScaledVector(direction, depth);
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

function Scene({ url, onCollider }: { url: string; onCollider: (mesh: Mesh) => void }) {
  const { scene } = useGLTF(url, "/draco/");
  const built = useRef(false);
  useEffect(() => {
    if (built.current) return;
    built.current = true;
    const group = scene as Group;
    // Doors' leaves shouldn't block movement; everything else is static.
    const staticMeshes: Mesh[] = [];
    group.traverse((node) => {
      if ((node as Mesh).isMesh && !node.name.includes("DoorLeaf")) staticMeshes.push(node as Mesh);
    });
    const holder = new Group();
    staticMeshes.forEach((mesh) => holder.add(mesh.clone()));
    const generator = new StaticGeometryGenerator(holder);
    generator.attributes = ["position"];
    const merged = generator.generate();
    (merged as any).boundsTree = new MeshBVH(merged);
    const collider = new Mesh(merged);
    collider.visible = false;
    onCollider(collider);
  }, [scene, onCollider]);
  return <primitive object={scene} />;
}

function LoadingOverlay() {
  const { progress, active } = useProgress();
  if (!active && progress >= 100) return null;
  return <div className="loading-overlay">loading model… {progress.toFixed(0)}%</div>;
}

export default function WalkthroughPage() {
  const { jobId = "" } = useParams();
  const [collider, setCollider] = useState<Mesh | null>(null);
  const [plan, setPlan] = useState<FloorPlanModel | null>(null);
  const [job, setJob] = useState<JobRecord | null>(null);
  const [tourActive, setTourActive] = useState(false);
  const [locked, setLocked] = useState(false);
  const joystick = useRef({ x: 0, y: 0 });
  // Versioned URL: never render a stale cached GLB after a Blender rebuild.
  const url = job ? versionedGlbUrl(jobId, job.glb_version) : null;
  // Baked builds carry their lighting in the lightmaps; screen-space AO on
  // top double-darkens corners, so N8AO only runs for unbaked previews.
  const baked = job?.glb_source === "blender";

  useEffect(() => {
    getEditData(jobId)
      .then((data) => setPlan(data.model))
      .catch(() => setPlan(null));
    getJob(jobId)
      .then(setJob)
      .catch(() => setJob(null));
  }, [jobId]);

  useEffect(() => {
    if (!("ontouchstart" in window)) return;
    const zone = document.getElementById("joystick-zone");
    if (!zone) return;
    const manager = nipplejs.create({ zone, mode: "static", position: { left: "80px", bottom: "80px" } });
    manager.on("move", (_, data) => {
      const radians = data.angle?.radian ?? 0;
      const force = Math.min(data.force ?? 0, 1);
      joystick.current = { x: Math.cos(radians) * force, y: Math.sin(radians) * force };
    });
    manager.on("end", () => (joystick.current = { x: 0, y: 0 }));
    return () => manager.destroy();
  }, []);

  const start = useMemo(() => {
    if (plan?.entrance) return planToWorld((plan.entrance as any).x, (plan.entrance as any).y);
    const room = plan?.rooms?.[0];
    if (room) {
      const cx = room.points.reduce((s, p) => s + p.x, 0) / room.points.length;
      const cy = room.points.reduce((s, p) => s + p.y, 0) / room.points.length;
      return planToWorld(cx, cy);
    }
    return new Vector3(2, EYE_HEIGHT, -2);
  }, [plan]);

  const tour = useMemo(() => {
    if (!plan || plan.rooms.length < 2) return null;
    const points = plan.rooms.map((room) => {
      const cx = room.points.reduce((s, p) => s + p.x, 0) / room.points.length;
      const cy = room.points.reduce((s, p) => s + p.y, 0) / room.points.length;
      return planToWorld(cx, cy);
    });
    return new CatmullRomCurve3(points, true, "catmullrom", 0.4);
  }, [plan]);

  return (
    <div className="walkthrough-root">
      <div className="hud">
        <Link to={`/jobs/${jobId}/preview`} style={{ color: "#fff" }}>
          ← exit walkthrough
        </Link>
        <div className="row">
          {tour && (
            <button onClick={() => setTourActive((v) => !v)}>
              {tourActive ? "take manual control" : "guided tour"}
            </button>
          )}
        </div>
      </div>
      {!locked && !tourActive && <div className="hint">click to look around — WASD to walk, Shift to hurry, Esc to release</div>}
      <Canvas
        camera={{ position: start.toArray(), fov: 70, near: 0.05 }}
        gl={{ toneMapping: ACESFilmicToneMapping, outputColorSpace: SRGBColorSpace }}
      >
        <ambientLight intensity={0.35} />
        <hemisphereLight intensity={0.25} color="#cfe4ff" groundColor="#413a33" />
        <Suspense fallback={null}>
          {url && <Scene key={url} url={url} onCollider={setCollider} />}
        </Suspense>
        <Player
          collider={collider}
          start={start}
          joystick={joystick}
          tour={tour}
          tourActive={tourActive}
          onLeaveTour={() => setTourActive(false)}
        />
        {!tourActive && <PointerLockControls onLock={() => setLocked(true)} onUnlock={() => setLocked(false)} />}
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
      <div id="joystick-zone" />
      <LoadingOverlay />
    </div>
  );
}
