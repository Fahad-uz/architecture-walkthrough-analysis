import { OrbitControls, useGLTF, useProgress } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ACESFilmicToneMapping, Box3, PerspectiveCamera, SRGBColorSpace, Vector3 } from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { getJob, versionedGlbUrl } from "../api";
import SceneErrorBoundary from "../components/SceneErrorBoundary";
import type { JobRecord } from "../types";

/** Loads the GLB and frames it: orbit target = bounding-box center, camera
 *  placed from the bbox size so the whole building is visible. The pipeline
 *  uses a bottom-left origin, so models sit far from (0,0,0) — never assume
 *  the world origin is inside the building. */
function FramedBuilding({
  url,
  resetNonce,
  showCeiling,
}: {
  url: string;
  resetNonce: number;
  showCeiling: boolean;
}) {
  const { scene } = useGLTF(url, "/draco/");
  // useGLTF caches its scene. Clone the node hierarchy so a roof visibility
  // change here cannot leak into the walkthrough's cached copy.
  const model = useMemo(() => scene.clone(true), [scene]);
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null;

  useEffect(() => {
    model.traverse((node) => {
      if (/(ceiling|roof)/i.test(node.name)) node.visible = showCeiling;
    });
  }, [model, showCeiling]);

  useEffect(() => {
    if (!controls) return;
    const box = new Box3().setFromObject(model);
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
  }, [model, controls, camera, resetNonce]);

  return <primitive object={model} />;
}

function ModelLoadingOverlay() {
  const { active, progress } = useProgress();
  if (!active) return null;
  return (
    <div className="model-overlay" role="status" aria-live="polite">
      Loading 3D model… {Math.round(progress)}%
    </div>
  );
}

export default function PreviewPage() {
  const { jobId = "" } = useParams();
  const [job, setJob] = useState<JobRecord | null>(null);
  const [resetNonce, setResetNonce] = useState(0);
  const [showCeiling, setShowCeiling] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [sceneError, setSceneError] = useState<string | null>(null);
  const [sceneRetry, setSceneRetry] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;
    const refresh = async () => {
      try {
        const latest = await getJob(jobId);
        if (!cancelled) {
          setJob(latest);
          setRefreshError(null);
        }
      } catch (error) {
        if (!cancelled) setRefreshError(`Could not refresh model status: ${String(error)}`);
      } finally {
        if (!cancelled) timer = window.setTimeout(() => void refresh(), 2500);
      }
    };
    void refresh();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [jobId]);

  // glb_version changes whenever the server writes a new GLB, so the URL —
  // and therefore drei's cache key — changes with it.
  const url = job?.glb_url ? versionedGlbUrl(jobId, job.glb_version) : null;
  const sceneResetKey = url ? `${url}:${sceneRetry}` : null;
  useEffect(() => {
    setSceneError(null);
    setSceneRetry(0);
  }, [url]);

  return (
    <div className="preview-layout">
      <div className="preview-toolbar" aria-label="Model preview controls">
        <Link to={`/jobs/${jobId}/edit`}>← Editor</Link>
        <span className="preview-status" role="status" aria-live="polite">
          {job
            ? `${job.status}${job.glb_source ? ` · model: ${job.glb_source}` : ""} — ${job.message}`
            : refreshError ?? "Loading model status…"}
        </span>
        <span style={{ flex: 1 }} />
        <button
          type="button"
          aria-pressed={showCeiling}
          disabled={!url}
          onClick={() => setShowCeiling((visible) => !visible)}
        >
          {showCeiling ? "Hide roof" : "Show roof"}
        </button>
        <button type="button" disabled={!url} onClick={() => setResetNonce((n) => n + 1)}>
          Reset view
        </button>
        {url && (
          <>
            <a className="button-link" href={url} download="building.glb">
              Download GLB
            </a>
            <Link className="button-link primary" to={`/jobs/${jobId}/walkthrough`}>
              Enter walkthrough
            </Link>
          </>
        )}
      </div>
      <div className="canvas-fill">
        {url ? (
          <SceneErrorBoundary
            resetKey={sceneResetKey}
            fallback={
              <div className="model-empty" role="alert">
                <span>The 3D model could not be displayed.</span>
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
              aria-label="Interactive 3D building preview"
              camera={{ position: [8, 9, 8], fov: 50 }}
              gl={{ toneMapping: ACESFilmicToneMapping, outputColorSpace: SRGBColorSpace }}
            >
              <ambientLight intensity={0.7} />
              <directionalLight position={[6, 12, 6]} intensity={1.4} />
              <Suspense fallback={null}>
                <FramedBuilding
                  key={sceneResetKey}
                  url={url}
                  resetNonce={resetNonce}
                  showCeiling={showCeiling}
                />
              </Suspense>
              <OrbitControls
                makeDefault
                enableDamping
                dampingFactor={0.08}
                maxPolarAngle={Math.PI * 0.495}
              />
            </Canvas>
          </SceneErrorBoundary>
        ) : (
          <div className="model-empty" role="status">
            {refreshError ?? "A preview will appear here after the plan has been analyzed and saved."}
          </div>
        )}
        {url && !sceneError && <ModelLoadingOverlay />}
        {refreshError && job && <div className="connection-notice">{refreshError}</div>}
      </div>
    </div>
  );
}
