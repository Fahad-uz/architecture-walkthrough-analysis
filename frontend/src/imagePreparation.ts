import type { CropRect } from "./crop";

export type ImagePreparationState = { crop: CropRect | null; rotationQuarterTurns: number };
export const originalPreparation = (): ImagePreparationState => ({ crop: null, rotationQuarterTurns: 0 });

/** Continuous pixel-edge coordinates, matching the server's clockwise turns. */
export function preparedImageFrame(width: number, height: number, turns: number) {
  if (!Number.isInteger(turns) || turns < 0 || turns > 3) throw new Error("Invalid image rotation");
  const matrices = [
    [1, 0, 0, 1, 0, 0],
    [0, 1, -1, 0, height, 0],
    [-1, 0, 0, -1, width, height],
    [0, -1, 1, 0, 0, width],
  ];
  return {
    width: turns % 2 ? height : width,
    height: turns % 2 ? width : height,
    matrix: matrices[turns],
    svgTransform: `matrix(${matrices[turns].join(" ")})`,
  };
}

/** A turn clears the old crop; undo restores both its crop and orientation. */
export function turnPreparation(state: ImagePreparationState, direction: -1 | 1): ImagePreparationState {
  return { crop: null, rotationQuarterTurns: (state.rotationQuarterTurns + direction + 4) % 4 };
}
