import { useEffect, useRef, useState } from "react";
import { cropBetween, type CropRect, type ImagePoint } from "../crop";
import { originalPreparation, preparedImageFrame, straightenPreparation, turnPreparation, type ImagePreparationState } from "../imagePreparation";

type Props = { file: File; preparation: ImagePreparationState; onChange: (state: ImagePreparationState) => void; disabled: boolean };

export default function ImagePreparation({ file, preparation, onChange, disabled }: Props) {
  const { crop, rotationQuarterTurns, straightenDegrees } = preparation;
  const [angleInput, setAngleInput] = useState(String(straightenDegrees));
  const [angleError, setAngleError] = useState("");
  const [source, setSource] = useState<{ url: string; width: number; height: number } | null>(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState<CropRect | null>(null);
  const [past, setPast] = useState<ImagePreparationState[]>([]);
  const [future, setFuture] = useState<ImagePreparationState[]>([]);
  const start = useRef<ImagePoint | null>(null);

  useEffect(() => { setAngleInput(String(straightenDegrees)); setAngleError(""); }, [straightenDegrees]);

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

  const commit = (next: ImagePreparationState) => {
    if (JSON.stringify(next) === JSON.stringify(preparation)) return;
    start.current = null;
    setPending(null);
    setPast((items) => [...items.slice(-29), preparation]);
    setFuture([]);
    onChange(next);
  };
  const applyStraightening = () => {
    try {
      if (!angleInput.trim()) throw new Error("Enter an angle between −15° and 15°.");
      commit(straightenPreparation(preparation, Number(angleInput)));
      setAngleError("");
    } catch (error) { setAngleError(error instanceof Error ? error.message : String(error)); }
  };
  const point = (event: React.PointerEvent<SVGSVGElement>): ImagePoint | null => {
    const matrix = event.currentTarget.getScreenCTM();
    if (!matrix) return null;
    const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    return { x: p.x, y: p.y };
  };
  if (error) return <p role="alert">{error}</p>;
  if (!source) return <p role="status">Loading image preview…</p>;
  const frame = preparedImageFrame(source.width, source.height, rotationQuarterTurns, straightenDegrees);
  const visible = pending ?? crop;
  return (
    <section aria-label="Image preparation" style={{ display: "grid", gap: 8 }}>
      <p style={{ margin: 0, fontSize: 13 }}>Rotate the plan upright, then drag to select a crop. Rotating clears the crop; undo restores it. Keep dimension labels needed for scale. The original upload is preserved.</p>
      <div className="row">
        <button type="button" disabled={disabled} onClick={() => commit(turnPreparation(preparation, -1))}>Rotate left 90°</button>
        <button type="button" disabled={disabled} onClick={() => commit(turnPreparation(preparation, 1))}>Rotate right 90°</button>
      </div>
      <div className="row">
        <label>Straighten (degrees, −15 to 15)
          <input type="number" min={-15} max={15} step="0.1" value={angleInput} disabled={disabled}
            style={{ width: 130 }} onChange={(e) => setAngleInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); applyStraightening(); } }} />
        </label>
        <button type="button" disabled={disabled} onClick={applyStraightening}>Apply straightening</button>
      </div>
      <p style={{ margin: 0, fontSize: 12 }}>Clockwise is positive. Apply to update the preview, then select a crop.</p>
      {angleError && <p role="alert" style={{ margin: 0 }}>{angleError}</p>}
      <svg viewBox={`0 0 ${frame.width} ${frame.height}`} aria-label="Prepared plan crop preview" role="img"
        style={{ width: "100%", maxHeight: 450, background: "#fff", touchAction: "none", cursor: disabled ? "default" : "crosshair" }}
        onPointerDown={(e) => {
          if (disabled || e.button !== 0) return;
          start.current = point(e);
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const p = point(e);
          if (start.current && p) setPending(cropBetween(start.current, p, frame.width, frame.height));
        }}
        onPointerUp={(e) => {
          const p = point(e);
          if (start.current && p) {
            const next = cropBetween(start.current, p, frame.width, frame.height);
            if (next && next.width >= 32 && next.height >= 32) commit({ ...preparation, crop: next });
          }
          start.current = null;
          setPending(null);
          if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
        }}
        onPointerCancel={() => { start.current = null; setPending(null); }}>
        <image href={source.url} width={source.width} height={source.height} transform={frame.svgTransform} />
        {visible && <rect x={visible.x} y={visible.y} width={visible.width} height={visible.height}
          fill="rgba(31,122,77,0.12)" stroke="#16814c" strokeWidth="3" vectorEffect="non-scaling-stroke" />}
      </svg>
      <div className="row">
        <button type="button" disabled={disabled || !crop} onClick={() => commit({ ...preparation, crop: null })}>Clear crop</button>
        <button type="button" disabled={disabled || (!crop && !rotationQuarterTurns && !straightenDegrees)} onClick={() => commit(originalPreparation())}>Reset to original</button>
        <button type="button" disabled={disabled || !past.length} onClick={() => {
          onChange(past[past.length - 1]); setPast(past.slice(0, -1)); setFuture([preparation, ...future]);
        }}>Undo preparation</button>
        <button type="button" disabled={disabled || !future.length} onClick={() => {
          onChange(future[0]); setFuture(future.slice(1)); setPast([...past, preparation]);
        }}>Redo preparation</button>
      </div>
      <p style={{ margin: 0, fontSize: 12 }} aria-live="polite">
        {frame.width} × {frame.height} pixels · rotation {rotationQuarterTurns * 90}° · straighten {straightenDegrees}°{crop ? ` · selected ${crop.width} × ${crop.height}` : " · using the full image"}.
        {Math.min(crop?.width ?? frame.width, crop?.height ?? frame.height) < 600
          ? " Small image: fine walls and text may be difficult to read. Use a larger original if possible." : " Check that wall lines and dimension text are readable."}
      </p>
      <details>
        <summary>Compare with original</summary>
        <img src={source.url} alt="Original plan before rotation and crop" style={{ width: "100%", maxHeight: 450, objectFit: "contain" }} />
      </details>
      <details>
        <summary>Set crop precisely</summary>
        <div className="row">
          {(["x", "y", "width", "height"] as const).map((field) => {
            const current = crop ?? { x: 0, y: 0, width: frame.width, height: frame.height };
            return <label key={field}>{field}<input type="number" value={current[field]} min={field === "x" || field === "y" ? 0 : 32}
              max={field === "x" || field === "width" ? frame.width : frame.height} disabled={disabled} style={{ width: 100 }}
              onChange={(e) => {
                if (!e.target.value) return;
                const next = { ...current, [field]: Number(e.target.value) };
                if (next.width < 32 || next.height < 32) return;
                const clipped = cropBetween(next, { x: next.x + next.width, y: next.y + next.height }, frame.width, frame.height);
                if (clipped && clipped.width >= 32 && clipped.height >= 32) commit({ ...preparation, crop: clipped });
              }} /></label>;
          })}
        </div>
      </details>
    </section>
  );
}
