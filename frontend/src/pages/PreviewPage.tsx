import { OrbitControls, useGLTF } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ACESFilmicToneMapping, Box3, PerspectiveCamera, SRGBColorSpace, Vector3 } from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { getJob, glbUrl, versionedGlbUrl } from "../api";
import type { JobRecord } from "../types";

/** Loads the GLB and frames it: orbit target = bounding-box center, camera
 *  placed from the bbox size so the whole building is visible. The pipeline
 *  uses a bottom-left origin, so models sit far from (0,0,0) — never assume
 *  the world origin is inside the building. */
function FramedBuilding({ url, resetNonce }: { url: string; resetNonce: number }) {
  const { scene } = useGLTF(url, "/draco/");
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null;

  useEffect(() => {
    if (!controls) return;
    const box = new Box3().setFromObject(scene);
    if (box.isEmpty()) return;
    const center = box.getCenter(new Vector3());
    const size = box.getSize(new Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 1);
    const distance = maxDim * 1.5;
    camera.position.set(center.x + distance * 0.75, center.y + distance * 0.65, center.z + distance * 0.75);
    if (camera instanceof PerspectiveCamera) {
      camera.near = Math.max(0.01, distance / 1000);
      camera.far = distance * 40;
      camera.updateProjectionMatrix();
    }
    controls.target.copy(center);
    controls.minDistance = maxDim * 0.15;
    controls.maxDistance = maxDim * 6;
    controls.update();
  }, [scene, controls, camera, resetNonce]);

  return <primitive object={scene} />;
}

export default function PreviewPage() {
  const { jobId = "" } = useParams();
  const [job, setJob] = useState<JobRecord | null>(null);
  const [resetNonce, setResetNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const refresh = async () => {
      const latest = await getJob(jobId);
      if (!cancelled) setJob(latest);
    };
    refresh();
    const timer = window.setInterval(refresh, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [jobId]);

  // glb_version changes whenever the server writes a new GLB, so the URL —
  // and therefore drei's cache key — changes with it.
  const url = job ? versionedGlbUrl(jobId, job.glb_version) : null;

  return (
    <div className="preview-layout">
      <div className="preview-toolbar">
        <Link to={`/jobs/${jobId}/edit`}>← Editor</Link>
        <span style={{ fontSize: 13, color: "#555" }}>
          {job ? `${job.status}${job.glb_source ? ` · model: ${job.glb_source}` : ""} — ${job.message}` : "…"}
        </span>
        <span style={{ flex: 1 }} />
        <button onClick={() => setResetNonce((n) => n + 1)}>Reset view</button>
        <a href={glbUrl(jobId)} download="building.glb">
          <button>Download GLB</button>
        </a>
        <Link to={`/jobs/${jobId}/walkthrough`}>
          <button className="primary">Enter walkthrough</button>
        </Link>
      </div>
      <div className="canvas-fill">
        <Canvas
          camera={{ position: [8, 9, 8], fov: 50 }}
          gl={{ toneMapping: ACESFilmicToneMapping, outputColorSpace: SRGBColorSpace }}
        >
          <ambientLight intensity={0.7} />
          <directionalLight position={[6, 12, 6]} intensity={1.4} />
          <Suspense fallback={null}>
            {url && <FramedBuilding key={url} url={url} resetNonce={resetNonce} />}
          </Suspense>
          <OrbitControls
            makeDefault
            enableDamping
            dampingFactor={0.08}
            maxPolarAngle={Math.PI * 0.495}
          />
        </Canvas>
      </div>
    </div>
  );
}
