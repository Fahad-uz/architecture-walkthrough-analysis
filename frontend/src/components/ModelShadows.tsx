import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import { Box3, DirectionalLight, Group, Mesh, PointLight, SpotLight, Vector3 } from "three";

type ShadowLight = DirectionalLight | PointLight | SpotLight;

/** glTF does not carry shadow-map settings. Render a small number of nearby
 * light shadows and cache them until visibility or the active lights change. */
export default function ModelShadows({ model, coarsePointer, revision = 0 }: {
  model: Group;
  coarsePointer: boolean;
  revision?: number;
}) {
  const camera = useThree((state) => state.camera);
  const state = useMemo(() => ({
    lights: [] as ShadowLight[],
    position: new Vector3(),
    elapsed: 1,
  }), [model]);

  useEffect(() => {
    model.updateMatrixWorld(true);
    const bounds = new Box3().setFromObject(model);
    const extent = Math.max(bounds.getSize(new Vector3()).length(), 5);
    state.lights = [];
    model.traverse((node) => {
      if (node instanceof Mesh) {
        const materials = Array.isArray(node.material) ? node.material : [node.material];
        node.castShadow = materials.some((material) => !material.transparent && material.depthWrite);
        node.receiveShadow = true;
      }
      if (node instanceof PointLight || node instanceof SpotLight || node instanceof DirectionalLight) {
        node.castShadow = false;
        node.shadow.autoUpdate = false;
        node.shadow.mapSize.setScalar(coarsePointer ? 256 : 512);
        node.shadow.bias = -0.00015;
        node.shadow.normalBias = 0.025;
        node.shadow.camera.near = 0.08;
        node.shadow.camera.far = extent * 2;
        node.shadow.camera.updateProjectionMatrix();
        // The sun's default 10 m shadow box often misses a plan whose origin
        // is offset. Room lights have no such bounds problem.
        if (!(node instanceof DirectionalLight)) state.lights.push(node);
      }
    });
    state.elapsed = 1;
    return () => {
      for (const light of state.lights) {
        light.castShadow = false;
        light.shadow.map?.dispose();
        light.shadow.map = null;
      }
      state.lights = [];
    };
  }, [model, coarsePointer, state]);

  useEffect(() => {
    for (const light of state.lights) light.shadow.needsUpdate = true;
    state.elapsed = 1;
  }, [revision, state]);

  useFrame((_, delta) => {
    state.elapsed += delta;
    if (state.elapsed < 0.35) return;
    state.elapsed = 0;
    const active = new Set([...state.lights]
      .sort((a, b) => a.getWorldPosition(state.position).distanceToSquared(camera.position)
        - b.getWorldPosition(state.position).distanceToSquared(camera.position))
      .slice(0, coarsePointer ? 1 : 3));
    for (const light of state.lights) {
      const enabled = active.has(light);
      if (enabled && !light.castShadow) light.shadow.needsUpdate = true;
      light.castShadow = enabled;
    }
  });
  return null;
}
