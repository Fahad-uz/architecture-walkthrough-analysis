import { OrbitControls, useGLTF, useProgress } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { EffectComposer, N8AO, SMAA, ToneMapping } from "@react-three/postprocessing";
import { ToneMappingMode } from "postprocessing";
import { Suspense, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Box3, NoToneMapping, Object3D, PerspectiveCamera, SRGBColorSpace, Vector3 } from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { getJob, versionedGlbUrl } from "../api";
import NeutralEnvironment from "../components/NeutralEnvironment";
import SceneErrorBoundary from "../components/SceneErrorBoundary";
import ModelShadows from "../components/ModelShadows";
import type { JobRecord } from "../types";
import useCoarsePointer from "../useCoarsePointer";

/** Loads the GLB and frames it: orbit target = bounding-box center, camera
 *  placed from the bbox size so the whole building is visible. The pipeline
 *  uses a bottom-left origin, so models sit far from (0,0,0) — never assume
 *  the world origin is inside the building. */
function FramedBuilding({
  url,
  resetNonce,
  showCeiling,
  previewLighting,
  coarsePointer,
}: {
  url: string;
  resetNonce: number;
  showCeiling: boolean;
  previewLighting: boolean;
  coarsePointer: boolean;
}) {
  const { scene } = useGLTF(url, "/draco/");
  // useGLTF caches its scene. Clone the node hierarchy so a roof visibility
  // change here cannot leak into the walkthrough's cached copy.
  const model = useMemo(() => scene.clone(true), [scene]);
  const camera = useThree((s) => s.camera);
  const viewportSize = useThree((s) => s.size);
  const bounds = useMemo(() => new Box3().setFromObject(model), [model]);
  const center = useMemo(() => bounds.getCenter(new Vector3()), [bounds]);
  const maxDim = useMemo(() => Math.max(...bounds.getSize(new Vector3()).toArray(), 1), [bounds]);
  const sunTarget = useMemo(() => {
    const target = new Object3D();
    target.position.copy(center);
    return target;
  }, [center]);
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null;

  useEffect(() => {
    model.traverse((node) => {
      if (/(ceiling|roof)/i.test(node.name)) node.visible = showCeiling;
    });
  }, [model, showCeiling]);

  useEffect(() => {
    if (!controls) return;
    if (bounds.isEmpty()) return;
    const radius = bounds.getSize(new Vector3()).length() / 2;
    let distance = maxDim * 1.5;
    if (camera instanceof PerspectiveCamera) {
      const verticalFov = camera.fov * Math.PI / 180;
      const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * viewportSize.width / Math.max(viewportSize.height, 1));
      distance = radius / Math.sin(Math.min(verticalFov, horizontalFov) / 2) * 1.12;
    }
    camera.position.copy(center).add(new Vector3(0.65, 1.2, 0.75).normalize().multiplyScalar(distance));
    if (camera instanceof PerspectiveCamera) {
      camera.near = Math.max(0.01, distance / 1000);
      camera.far = distance * 40;
      camera.updateProjectionMatrix();
    }
    controls.target.copy(center);
    controls.minDistance = maxDim * 0.15;
    controls.maxDistance = Math.max(maxDim * 6, distance * 2);
    controls.update();
  }, [bounds, center, maxDim, controls, camera, resetNonce, viewportSize.width, viewportSize.height]);

  return <>
    <primitive object={model} />
    <ModelShadows model={model} coarsePointer={coarsePointer} revision={Number(showCeiling)} />
    {previewLighting && <>
      <primitive object={sunTarget} />
      <directionalLight
        position={[center.x + maxDim * 0.6, center.y + maxDim, center.z + maxDim * 0.8]}
        target={sunTarget}
        intensity={1.1}
        color="#fff3e0"
        castShadow
        shadow-mapSize-width={coarsePointer ? 512 : 1024}
        shadow-mapSize-height={coarsePointer ? 512 : 1024}
        shadow-bias={-0.00015}
        shadow-normalBias={0.025}
        shadow-camera-left={-maxDim}
        shadow-camera-right={maxDim}
        shadow-camera-top={maxDim}
        shadow-camera-bottom={-maxDim}
        shadow-camera-near={0.1}
        shadow-camera-far={maxDim * 5}
      />
    </>}
  </>;
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
  const coarsePointer = useCoarsePointer();

  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;
    setJob(null);
    setRefreshError(null);
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
  const baked =
    job?.glb_bake_mode == null
      ? job?.glb_source === "blender"
      : job.glb_bake_mode !== "none";
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
              camera={{ position: [8, 9, 8], fov: coarsePointer ? 58 : 50 }}
              dpr={coarsePointer ? [1, 1.25] : [1, 1.75]}
              shadows
              gl={{ toneMapping: NoToneMapping, outputColorSpace: SRGBColorSpace }}
            >
              <color attach="background" args={["#c9ced3"]} />
              <NeutralEnvironment />
              <ambientLight intensity={0.32} />
              <hemisphereLight intensity={0.34} color="#ffffff" groundColor="#8f9498" />
              <Suspense fallback={null}>
                <FramedBuilding
                  key={sceneResetKey}
                  url={url}
                  resetNonce={resetNonce}
                  showCeiling={showCeiling}
                  previewLighting={job?.glb_source === "preview"}
                  coarsePointer={coarsePointer}
                />
              </Suspense>
              <OrbitControls
                makeDefault
                enableDamping
                dampingFactor={0.08}
                maxPolarAngle={Math.PI * 0.495}
              />
              <EffectComposer multisampling={0}>
                  {[...(!baked ? [<N8AO key="ao"
                    screenSpaceRadius
                    aoRadius={32}
                    intensity={1.8}
                    distanceFalloff={0.2}
                    quality={coarsePointer ? "performance" : "medium"}
                    halfRes={coarsePointer}
                    depthAwareUpsampling
                  />] : []),
                  <ToneMapping key="tone" mode={ToneMappingMode.ACES_FILMIC} />,
                  <SMAA key="antialias" />]}
              </EffectComposer>
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
