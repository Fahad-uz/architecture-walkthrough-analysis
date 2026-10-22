import { useThree } from "@react-three/fiber";
import { useLayoutEffect } from "react";
import { Color, type ColorRepresentation } from "three";
import { effectiveLighting, type LightColorScale } from "../lighting";
import type { LightingSettings } from "../types";
import NeutralEnvironment from "./NeutralEnvironment";

export function tintedLightColor(base: ColorRepresentation, scale: LightColorScale): Color {
  return new Color(base).multiply(new Color(...scale));
}

/** Shared saved appearance for both 3D views. GLB lights are left as exported. */
export default function SavedLighting({ settings }: { settings?: LightingSettings }) {
  const gl = useThree((state) => state.gl);
  const { powerFactor, colorScale, exposureFactor } = effectiveLighting(settings);

  useLayoutEffect(() => {
    const previous = gl.toneMappingExposure;
    gl.toneMappingExposure = exposureFactor;
    return () => { gl.toneMappingExposure = previous; };
  }, [gl, exposureFactor]);

  return <>
    <NeutralEnvironment intensity={0.62 * powerFactor} colorScale={colorScale} />
    <ambientLight intensity={0.32 * powerFactor} color={tintedLightColor("#ffffff", colorScale)} />
    <hemisphereLight intensity={0.34 * powerFactor} color={tintedLightColor("#ffffff", colorScale)} groundColor={tintedLightColor("#8f9498", colorScale)} />
  </>;
}
