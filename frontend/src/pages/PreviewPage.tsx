import { OrbitControls, useGLTF } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Suspense, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ACESFilmicToneMapping, SRGBColorSpace } from "three";
import { getJob, glbUrl } from "../api";
import type { JobRecord } from "../types";

function Building({ url }: { url: string }) {
  const { scene } = useGLTF(url, "/draco/");
  return <primitive object={scene} />;
}

export default function PreviewPage() {
  const { jobId = "" } = useParams();
  const [job, setJob] = useState<JobRecord | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(async () => {
      const latest = await getJob(jobId);
      setJob((previous) => {
        if (previous && previous.glb_source !== latest.glb_source) {
          useGLTF.clear(`${glbUrl(jobId)}?v=${reloadKey}`);
          setReloadKey((k) => k + 1);
        }
        return latest;
      });
    }, 2500);
    getJob(jobId).then(setJob);
    return () => window.clearInterval(timer);
  }, [jobId]);

  const url = `${glbUrl(jobId)}?v=${reloadKey}`;

  return (
    <div className="preview-layout">
      <div className="preview-toolbar">
        <Link to={`/jobs/${jobId}/edit`}>← Editor</Link>
        <span style={{ fontSize: 13, color: "#555" }}>
          {job ? `${job.status}${job.glb_source ? ` · model: ${job.glb_source}` : ""} — ${job.message}` : "…"}
        </span>
        <span style={{ flex: 1 }} />
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
            <Building key={reloadKey} url={url} />
          </Suspense>
          <OrbitControls makeDefault target={[4.5, 0, -3.5]} />
        </Canvas>
      </div>
    </div>
  );
}
