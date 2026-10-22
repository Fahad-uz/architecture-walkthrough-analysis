import { useThree } from "@react-three/fiber";
import { useLayoutEffect } from "react";
import { Color, Light, Mesh, MeshBasicMaterial, PMREMGenerator } from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import type { LightColorScale } from "../lighting";

/** Offline studio reflections for PBR materials. Unlike an HDR preset this
 * never depends on a CDN, so previews remain reliable on private networks. */
export default function NeutralEnvironment({ intensity = 0.62, colorScale = [1, 1, 1] }: { intensity?: number; colorScale?: LightColorScale }) {
  const { gl, scene } = useThree();

  useLayoutEffect(() => {
    const generator = new PMREMGenerator(gl);
    const room = new RoomEnvironment();
    const tint = new Color(...colorScale);
    room.traverse((node) => {
      if (node instanceof Light) node.color.multiply(tint);
      if (node instanceof Mesh) {
        const materials = Array.isArray(node.material) ? node.material : [node.material];
        for (const material of materials) {
          if (material instanceof MeshBasicMaterial) material.color.multiply(tint);
        }
      }
    });
    const target = generator.fromScene(room, 0.04);
    room.dispose();
    generator.dispose();

    const previousEnvironment = scene.environment;
    const previousIntensity = scene.environmentIntensity;
    scene.environment = target.texture;
    scene.environmentIntensity = intensity;

    return () => {
      if (scene.environment === target.texture) {
        scene.environment = previousEnvironment;
        scene.environmentIntensity = previousIntensity;
      }
      target.dispose();
    };
  }, [gl, intensity, scene, colorScale[0], colorScale[1], colorScale[2]]);

  return null;
}
