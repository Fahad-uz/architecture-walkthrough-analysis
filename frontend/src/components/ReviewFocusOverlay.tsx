import { Line } from "@react-three/drei";
import type { Review3DFocus } from "../review3D";

/** Independent review wires remain visible through opaque building parts.
 * They never change the loaded model's materials or geometry. */
export default function ReviewFocusOverlay({ focus }: { focus: Review3DFocus }) {
  return <group name="Review_Marker" renderOrder={1000}>
    {focus.shapes.flatMap((shape) => shape.paths.map((points, index) =>
      <Line key={`${shape.target.kind}:${shape.target.id}:${index}`} points={points} color="#ff8a00"
        lineWidth={3} depthTest={false} depthWrite={false} toneMapped={false} transparent opacity={0.95} renderOrder={1000} raycast={() => undefined} />))}
  </group>;
}
