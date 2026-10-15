import type { CropRect } from "./crop";

export type ImagePreparationState = { crop: CropRect | null; rotationQuarterTurns: number; straightenDegrees: number };
export const originalPreparation = (): ImagePreparationState => ({ crop: null, rotationQuarterTurns: 0, straightenDegrees: 0 });

function validateStraightening(angle: number) {
  if (!Number.isFinite(angle) || Math.abs(angle) > 15) throw new Error("Straightening must be between −15° and 15°.");
}

/** Continuous pixel-edge coordinates, matching the server's clockwise turns. */
export function preparedImageFrame(width: number, height: number, turns: number, angle = 0) {
  if (!Number.isInteger(turns) || turns < 0 || turns > 3) throw new Error("Invalid image rotation");
  validateStraightening(angle);
  const matrices = [
    [1, 0, 0, 1, 0, 0],
    [0, 1, -1, 0, height, 0],
    [-1, 0, 0, -1, width, height],
    [0, -1, 1, 0, 0, width],
  ];
  const quarterWidth = turns % 2 ? height : width;
  const quarterHeight = turns % 2 ? width : height;
  let outputWidth = quarterWidth, outputHeight = quarterHeight, matrix = matrices[turns];
  if (angle !== 0) {
    const radians = angle * Math.PI / 180, cosine = Math.cos(radians), sine = Math.sin(radians);
    outputWidth = Math.ceil(Math.abs(quarterWidth * cosine) + Math.abs(quarterHeight * sine) - 1e-9);
    outputHeight = Math.ceil(Math.abs(quarterWidth * sine) + Math.abs(quarterHeight * cosine) - 1e-9);
    const tx = outputWidth / 2 - cosine * quarterWidth / 2 + sine * quarterHeight / 2;
    const ty = outputHeight / 2 - sine * quarterWidth / 2 - cosine * quarterHeight / 2;
    const [a, b, c, d, e, f] = matrix;
    matrix = [cosine * a - sine * b, sine * a + cosine * b,
      cosine * c - sine * d, sine * c + cosine * d,
      cosine * e - sine * f + tx, sine * e + cosine * f + ty];
  }
  return { width: outputWidth, height: outputHeight, matrix, svgTransform: `matrix(${matrix.join(" ")})` };
}

/** A turn clears the old crop; undo restores both its crop and orientation. */
export function turnPreparation(state: ImagePreparationState, direction: -1 | 1): ImagePreparationState {
  return { ...state, crop: null, rotationQuarterTurns: (state.rotationQuarterTurns + direction + 4) % 4 };
}

export function straightenPreparation(state: ImagePreparationState, angle: number): ImagePreparationState {
  validateStraightening(angle);
  if (angle === state.straightenDegrees) return state;
  return { ...state, crop: null, straightenDegrees: angle };
}
