import { useEffect, useRef, useState } from "react";
import { cropBetween, type CropRect, type ImagePoint } from "../crop";

type Props = { file: File; crop: CropRect | null; onCrop: (crop: CropRect | null) => void; disabled: boolean };

export default function ImagePreparation({ file, crop, onCrop, disabled }: Props) {
  const [source, setSource] = useState<{ url: string; width: number; height: number } | null>(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState<CropRect | null>(null);
  const [past, setPast] = useState<(CropRect | null)[]>([]);
  const [future, setFuture] = useState<(CropRect | null)[]>([]);
  const start = useRef<ImagePoint | null>(null);

  useEffect(() => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    let cancelled = false;
    setSource(null);
    setError("");
    img.onload = () => { if (!cancelled) setSource({ url, width: img.naturalWidth, height: img.naturalHeight }); };
    img.onerror = () => { if (!cancelled) setError("This image could not be previewed. Please choose a PNG, JPEG or WebP image."); };
    img.src = url;
    return () => { cancelled = true; URL.revokeObjectURL(url); };
  }, [file]);

  const commit = (next: CropRect | null) => {
    if (JSON.stringify(next) === JSON.stringify(crop)) return;
    setPast((items) => [...items.slice(-29), crop]);
    setFuture([]);
    onCrop(next);
  };
  const point = (event: React.PointerEvent<SVGSVGElement>): ImagePoint | null => {
    const matrix = event.currentTarget.getScreenCTM();
    if (!matrix) return null;
    const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    return { x: p.x, y: p.y };
  };
  if (error) return <p role="alert">{error}</p>;
  if (!source) return <p role="status">Loading image preview…</p>;
  const visible = pending ?? crop;
  return (
    <section aria-label="Image preparation" style={{ display: "grid", gap: 8 }}>
      <p style={{ margin: 0, fontSize: 13 }}>Drag over the plan to select a crop. Keep dimension labels needed for scale. Analysis keeps a small margin around the selection and preserves the original upload.</p>
      <svg viewBox={`0 0 ${source.width} ${source.height}`} aria-label="Source plan crop preview" role="img"
        style={{ width: "100%", maxHeight: 450, background: "#eee", touchAction: "none", cursor: disabled ? "default" : "crosshair" }}
        onPointerDown={(e) => {
          if (disabled || e.button !== 0) return;
          start.current = point(e);
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const p = point(e);
          if (start.current && p) setPending(cropBetween(start.current, p, source.width, source.height));
        }}
        onPointerUp={(e) => {
          const p = point(e);
          if (start.current && p) {
            const next = cropBetween(start.current, p, source.width, source.height);
            if (next && next.width >= 32 && next.height >= 32) commit(next);
          }
          start.current = null;
          setPending(null);
          if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
        }}
        onPointerCancel={() => { start.current = null; setPending(null); }}>
        <image href={source.url} width={source.width} height={source.height} />
        {visible && <rect x={visible.x} y={visible.y} width={visible.width} height={visible.height}
          fill="rgba(31,122,77,0.12)" stroke="#16814c" strokeWidth="3" vectorEffect="non-scaling-stroke" />}
      </svg>
      <div className="row">
        <button type="button" disabled={disabled || !crop} onClick={() => commit(null)}>Use full original</button>
        <button type="button" disabled={disabled || !past.length} onClick={() => {
          onCrop(past[past.length - 1]); setPast(past.slice(0, -1)); setFuture([crop, ...future]);
        }}>Undo crop</button>
        <button type="button" disabled={disabled || !future.length} onClick={() => {
          onCrop(future[0]); setFuture(future.slice(1)); setPast([...past, crop]);
        }}>Redo crop</button>
      </div>
      <p style={{ margin: 0, fontSize: 12 }} aria-live="polite">
        {source.width} × {source.height} pixels{crop ? ` · selected ${crop.width} × ${crop.height}` : " · using the full original"}.
        {Math.min(crop?.width ?? source.width, crop?.height ?? source.height) < 600
          ? " Small image: fine walls and text may be difficult to read. Use a larger original if possible." : " Check that wall lines and dimension text are readable."}
      </p>
      <details>
        <summary>Set crop precisely</summary>
        <div className="row">
          {(["x", "y", "width", "height"] as const).map((field) => {
            const current = crop ?? { x: 0, y: 0, width: source.width, height: source.height };
            return <label key={field}>{field}<input type="number" value={current[field]} min={field === "x" || field === "y" ? 0 : 32}
              max={field === "x" || field === "width" ? source.width : source.height} disabled={disabled} style={{ width: 100 }}
              onChange={(e) => {
                if (!e.target.value) return;
                const next = { ...current, [field]: Number(e.target.value) };
                if (next.width < 32 || next.height < 32) return;
                const clipped = cropBetween(next, { x: next.x + next.width, y: next.y + next.height }, source.width, source.height);
                if (clipped && clipped.width >= 32 && clipped.height >= 32) commit(clipped);
              }} /></label>;
          })}
        </div>
      </details>
    </section>
  );
}
