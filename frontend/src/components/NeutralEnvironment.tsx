import { useThree } from "@react-three/fiber";
import { useLayoutEffect } from "react";
import { PMREMGenerator } from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

/** Offline studio reflections for PBR materials. Unlike an HDR preset this
 * never depends on a CDN, so previews remain reliable on private networks. */
export default function NeutralEnvironment({ intensity = 0.62 }: { intensity?: number }) {
  const { gl, scene } = useThree();

  useLayoutEffect(() => {
    const generator = new PMREMGenerator(gl);
    const room = new RoomEnvironment();
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
  }, [gl, intensity, scene]);

  return null;
}
