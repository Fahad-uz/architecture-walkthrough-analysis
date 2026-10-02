export type CropRect = { x: number; y: number; width: number; height: number };
export type ImagePoint = { x: number; y: number };

/** Integer source pixels, independent of the size of the on-screen preview. */
export function cropBetween(a: ImagePoint, b: ImagePoint, width: number, height: number): CropRect | null {
  if (![a.x, a.y, b.x, b.y, width, height].every(Number.isFinite) || width < 1 || height < 1) return null;
  const left = Math.max(0, Math.min(width, Math.floor(Math.min(a.x, b.x))));
  const top = Math.max(0, Math.min(height, Math.floor(Math.min(a.y, b.y))));
  const right = Math.max(0, Math.min(width, Math.ceil(Math.max(a.x, b.x))));
  const bottom = Math.max(0, Math.min(height, Math.ceil(Math.max(a.y, b.y))));
  return right > left && bottom > top ? { x: left, y: top, width: right - left, height: bottom - top } : null;
}
