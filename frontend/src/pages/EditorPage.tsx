import Konva from "konva";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Circle, Group, Image as KonvaImage, Layer, Line, Stage, Text } from "react-konva";
import { Link, useParams } from "react-router-dom";
import { generateModel, getEditData, getJob, saveCorrections, validateCorrections } from "../api";
import type { FloorPlanModel, Opening, Point2D, QualityReport, RoomPolygon, SanityWarning, WallSegment } from "../types";

type Tool = "select" | "wall" | "door" | "window" | "scale";
type Selection = { kind: "walls" | "doors" | "windows" | "rooms"; index: number } | null;

/** Editor works in image-pixel space; the model stores metres with origin at
 *  the image's bottom-left. ppm converts, imageHeight flips y. */
function makeTransforms(ppm: number, imageHeight: number) {
  const toPx = (p: Point2D) => ({ x: p.x * ppm, y: imageHeight - p.y * ppm });
  const toM = (p: Point2D) => ({ x: p.x / ppm, y: (imageHeight - p.y) / ppm });
  return { toPx, toM };
}

function wallLength(w: WallSegment): number {
  return Math.hypot(w.end.x - w.start.x, w.end.y - w.start.y);
}

function openingInterval(o: Opening): [number, number] {
  if (o.start_offset_m != null && o.end_offset_m != null) return [o.start_offset_m, o.end_offset_m];
  const width = o.width_m ?? 0.9;
  const mid = o.offset_m ?? 0;
  return [mid - width / 2, mid + width / 2];
}

function pointOnWall(w: WallSegment, offset: number): Point2D {
  const len = wallLength(w) || 1;
  const ux = (w.end.x - w.start.x) / len;
  const uy = (w.end.y - w.start.y) / len;
  return { x: w.start.x + ux * offset, y: w.start.y + uy * offset };
}

interface WallHit {
  wall: WallSegment;
  index: number;
  offset: number;
  dist: number;
  point: Point2D;
}

function nearestWall(walls: WallSegment[], p: Point2D): WallHit | null {
  let best: WallHit | null = null;
  for (let index = 0; index < walls.length; index += 1) {
    const wall = walls[index];
    const len2 = (wall.end.x - wall.start.x) ** 2 + (wall.end.y - wall.start.y) ** 2 || 1;
    const t = Math.max(
      0,
      Math.min(1, ((p.x - wall.start.x) * (wall.end.x - wall.start.x) + (p.y - wall.start.y) * (wall.end.y - wall.start.y)) / len2),
    );
    const q = { x: wall.start.x + t * (wall.end.x - wall.start.x), y: wall.start.y + t * (wall.end.y - wall.start.y) };
    const dist = Math.hypot(p.x - q.x, p.y - q.y);
    if (best === null || dist < best.dist) {
      best = { wall, index, offset: Math.sqrt(len2) * t, dist, point: q };
    }
  }
  return best;
}

function roomStats(room: RoomPolygon): string {
  const xs = room.points.map((p) => p.x);
  const ys = room.points.map((p) => p.y);
  const width = Math.max(...xs) - Math.min(...xs);
  const depth = Math.max(...ys) - Math.min(...ys);
  // Shoelace area of the polygon, in m².
  let area = 0;
  for (let i = 0; i < room.points.length; i += 1) {
    const a = room.points[i];
    const b = room.points[(i + 1) % room.points.length];
    area += a.x * b.y - b.x * a.y;
  }
  return `${width.toFixed(2)} × ${depth.toFixed(2)} m · ${Math.abs(area / 2).toFixed(1)} m²`;
}

/** Changing pixels_per_metre rescales every metre-quantity derived from pixels. */
function rescaleModel(model: FloorPlanModel, factor: number): FloorPlanModel {
  const sp = (p: Point2D) => ({ x: p.x * factor, y: p.y * factor });
  return {
    ...model,
    walls: model.walls.map((w) => ({ ...w, start: sp(w.start), end: sp(w.end) })),
    doors: model.doors.map((o) => scaleOpening(o, factor, sp)),
    windows: model.windows.map((o) => scaleOpening(o, factor, sp)),
    rooms: model.rooms.map((r) => ({ ...r, points: r.points.map(sp) })),
  };
}

function scaleOpening(o: Opening, factor: number, sp: (p: Point2D) => Point2D): Opening {
  return {
    ...o,
    center: sp(o.center),
    width_m: o.width_m * factor,
    offset_m: o.offset_m != null ? o.offset_m * factor : o.offset_m,
    start_offset_m: o.start_offset_m != null ? o.start_offset_m * factor : o.start_offset_m,
    end_offset_m: o.end_offset_m != null ? o.end_offset_m * factor : o.end_offset_m,
  };
}

export default function EditorPage() {
  const { jobId = "" } = useParams();
  const [model, setModel] = useState<FloorPlanModel | null>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [tool, setTool] = useState<Tool>("select");
  const [selection, setSelection] = useState<Selection>(null);
  const [hover, setHover] = useState<Selection>(null);
  const [pending, setPending] = useState<Point2D | null>(null); // first click of 2-point tools (metres)
  const [status, setStatus] = useState("loading…");
  const [report, setReport] = useState<QualityReport | null>(null);
  const [bakeMode, setBakeMode] = useState("draft");
  const [force, setForce] = useState(false);
  const [generating, setGenerating] = useState(false);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const data = await getEditData(jobId);
        setModel(data.model);
        const img = new window.Image();
        img.onload = () => setImage(img);
        img.src = data.image_url;
        setStatus(`loaded: ${data.model.walls.length} walls, ${data.model.rooms.length} rooms`);
      } catch (exc) {
        setStatus(String(exc));
      }
    })();
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, [jobId]);

  const ppm = model?.pixels_per_metre ?? 100;
  const imageHeight = image?.naturalHeight ?? 1000;
  const imageWidth = image?.naturalWidth ?? 1000;
  const { toPx, toM } = useMemo(() => makeTransforms(ppm, imageHeight), [ppm, imageHeight]);

  const update = useCallback((mutate: (m: FloorPlanModel) => FloorPlanModel) => {
    setModel((current) => (current ? mutate(structuredClone(current)) : current));
  }, []);

  const stagePointer = (e: Konva.KonvaEventObject<MouseEvent>): Point2D | null => {
    const pos = e.target.getStage()?.getPointerPosition();
    return pos ? toM(pos) : null;
  };

  const handleStageClick = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (!model) return;
    const p = stagePointer(e);
    if (!p) return;
    if (tool === "select") {
      if (e.target === e.target.getStage()) setSelection(null);
      return;
    }
    if (tool === "wall") {
      if (!pending) {
        setPending(p);
        return;
      }
      update((m) => {
        m.walls.push({
          id: `cw${Date.now()}`,
          start: pending,
          end: p,
          thickness_m: 0.12,
          height_m: 2.8,
          external: false,
          wall_type: "internal",
          confidence: 1,
          evidence_source: "manual_correction",
        });
        return m;
      });
      setPending(null);
      return;
    }
    if (tool === "scale") {
      if (!pending) {
        setPending(p);
        setStatus("scale: click the second reference point");
        return;
      }
      const distPx = Math.hypot(p.x - pending.x, p.y - pending.y) * ppm;
      const metres = window.prompt("Real-world distance between the two points, in metres:", "1.0");
      setPending(null);
      const value = metres ? parseFloat(metres) : NaN;
      if (!value || value <= 0) return;
      const newPpm = distPx / value;
      const factor = ppm / newPpm;
      update((m) => {
        const rescaled = rescaleModel(m, factor);
        rescaled.pixels_per_metre = newPpm;
        rescaled.metadata = { ...rescaled.metadata, scale_source: "manual", scale_confidence: 1.0 };
        return rescaled;
      });
      setStatus(`scale set: ${newPpm.toFixed(2)} px/m (manual reference)`);
      setTool("select");
      return;
    }
    if (tool === "door" || tool === "window") {
      const hit = nearestWall(model.walls, p);
      if (!hit) return;
      const width = tool === "door" ? 0.9 : 1.2;
      const item: Opening = {
        id: `${tool}_manual_${Date.now()}`,
        center: hit.point,
        width_m: width,
        height_m: tool === "door" ? 2.1 : 1.2,
        ...(tool === "window" ? { sill_height_m: 0.9 } : {}),
        wall_id: hit.wall.id ?? null,
        offset_m: hit.offset,
        start_offset_m: hit.offset - width / 2,
        end_offset_m: hit.offset + width / 2,
        opening_type: tool === "door" ? "single_leaf" : "fixed",
        confidence: 1,
        evidence_source: "manual_correction",
      };
      const key = tool === "door" ? "doors" : "windows";
      update((m) => {
        (m as any)[key].push(item);
        return m;
      });
      setSelection({ kind: key as "doors" | "windows", index: (model as any)[key].length });
      return;
    }
  };

  const moveOpening = (kind: "doors" | "windows", index: number, posPx: Point2D) => {
    if (!model) return;
    const p = toM(posPx);
    update((m) => {
      const item = (m as any)[kind][index] as Opening;
      const hit = nearestWall(m.walls, p);
      if (hit) {
        item.center = hit.point;
        item.wall_id = hit.wall.id ?? null;
        item.offset_m = hit.offset;
        item.start_offset_m = hit.offset - item.width_m / 2;
        item.end_offset_m = hit.offset + item.width_m / 2;
      }
      return m;
    });
  };

  const save = async (): Promise<boolean> => {
    if (!model) return false;
    setStatus("saving…");
    try {
      const result = await saveCorrections(jobId, model);
      setModel(result.model);
      setReport(result);
      setStatus(`saved — rooms regenerated from wall graph (${result.model.rooms.length} rooms)`);
      return true;
    } catch (exc) {
      setStatus(String(exc));
      return false;
    }
  };

  const validate = async () => {
    if (!(await save())) return;
    try {
      setReport(await validateCorrections(jobId));
    } catch (exc) {
      setStatus(String(exc));
    }
  };

  const generate = async () => {
    if (!(await save())) return;
    try {
      await generateModel(jobId, force, bakeMode);
      setGenerating(true);
      setStatus(`Blender build started (${bakeMode})…`);
      pollRef.current = window.setInterval(async () => {
        const job = await getJob(jobId);
        if (job.status === "model_generated") {
          if (pollRef.current) window.clearInterval(pollRef.current);
          setGenerating(false);
          setStatus("3D model ready — open the preview page");
        } else if (["blocked", "generation_failed"].includes(job.status)) {
          if (pollRef.current) window.clearInterval(pollRef.current);
          setGenerating(false);
          setStatus(`${job.status}: ${job.message}`);
        }
      }, 2000);
    } catch (exc) {
      setStatus(String(exc));
    }
  };

  const removeSelected = () => {
    if (!selection) return;
    update((m) => {
      (m as any)[selection.kind].splice(selection.index, 1);
      return m;
    });
    setSelection(null);
  };

  const selectedItem: any = selection && model ? (model as any)[selection.kind][selection.index] : null;
  const warnings: SanityWarning[] = (model?.metadata?.sanity_warnings as SanityWarning[]) ?? [];

  return (
    <div className="editor-layout">
      <div className="stage-wrap">
        {model && image && (
          <Stage width={imageWidth} height={imageHeight} onMouseDown={handleStageClick}>
            <Layer listening={false}>
              <KonvaImage image={image} />
            </Layer>
            <Layer>
              {model.rooms.map((room, index) => {
                const selected = selection?.kind === "rooms" && selection.index === index;
                const hovered = hover?.kind === "rooms" && hover.index === index;
                const centroid = room.points.reduce(
                  (acc, p) => ({ x: acc.x + p.x / room.points.length, y: acc.y + p.y / room.points.length }),
                  { x: 0, y: 0 },
                );
                const labelAt = toPx(centroid);
                return (
                  <Group key={`room-${index}`}>
                    <Line
                      points={room.points.flatMap((p) => {
                        const q = toPx(p);
                        return [q.x, q.y];
                      })}
                      closed
                      fill={
                        selected
                          ? "rgba(255,159,28,0.28)"
                          : hovered
                            ? "rgba(46,160,67,0.20)"
                            : "rgba(46,160,67,0.08)"
                      }
                      stroke={selected ? "#ff9f1c" : hovered ? "#1f7a3a" : "#2ea043"}
                      strokeWidth={selected ? 4 : hovered ? 3 : 2}
                      onMouseEnter={(e) => {
                        if (tool !== "select") return;
                        setHover({ kind: "rooms", index });
                        const stage = e.target.getStage();
                        if (stage) stage.container().style.cursor = "pointer";
                      }}
                      onMouseLeave={(e) => {
                        setHover(null);
                        const stage = e.target.getStage();
                        if (stage) stage.container().style.cursor = "default";
                      }}
                      onMouseDown={(e) => {
                        if (tool !== "select") return;
                        e.cancelBubble = true;
                        setSelection({ kind: "rooms", index });
                      }}
                    />
                    <Text
                      x={labelAt.x - 40}
                      y={labelAt.y - 8}
                      width={80}
                      align="center"
                      text={room.name ?? "unnamed"}
                      fontSize={13}
                      fontStyle={room.name ? "bold" : "italic"}
                      fill={room.name ? "#0a5f2c" : "#8a938c"}
                      listening={false}
                    />
                  </Group>
                );
              })}
              {model.walls.map((wall, index) => {
                const a = toPx(wall.start);
                const b = toPx(wall.end);
                const selected = selection?.kind === "walls" && selection.index === index;
                return (
                  <Group key={`wall-${index}`}>
                    <Line
                      points={[a.x, a.y, b.x, b.y]}
                      stroke={selected ? "#ff9f1c" : "#0072bc"}
                      strokeWidth={wall.external ? 7 : 4}
                      hitStrokeWidth={14}
                      lineCap="round"
                      onMouseDown={(e) => {
                        if (tool !== "select") return;
                        e.cancelBubble = true;
                        setSelection({ kind: "walls", index });
                      }}
                    />
                    {selected &&
                      (["start", "end"] as const).map((key) => {
                        const p = toPx(wall[key]);
                        return (
                          <Circle
                            key={key}
                            x={p.x}
                            y={p.y}
                            radius={7}
                            fill="#fff"
                            stroke="#222"
                            strokeWidth={2}
                            draggable
                            onDragMove={(e) => {
                              const q = toM({ x: e.target.x(), y: e.target.y() });
                              update((m) => {
                                (m.walls[index] as any)[key] = q;
                                return m;
                              });
                            }}
                          />
                        );
                      })}
                  </Group>
                );
              })}
              {(["doors", "windows"] as const).map((kind) =>
                model[kind].map((item, index) => {
                  const wall = model.walls.find((w) => w.id === item.wall_id) ?? nearestWall(model.walls, item.center)?.wall;
                  let a = toPx(item.center);
                  let b = { x: a.x + 8, y: a.y };
                  if (wall) {
                    const [s, e] = openingInterval(item);
                    a = toPx(pointOnWall(wall, s));
                    b = toPx(pointOnWall(wall, e));
                  }
                  const selected = selection?.kind === kind && selection.index === index;
                  return (
                    <Line
                      key={`${kind}-${index}`}
                      points={[a.x, a.y, b.x, b.y]}
                      stroke={selected ? "#ff9f1c" : kind === "doors" ? "#be6a2f" : "#58c4dd"}
                      strokeWidth={kind === "doors" ? 9 : 7}
                      hitStrokeWidth={16}
                      lineCap="square"
                      draggable
                      onMouseDown={(e) => {
                        e.cancelBubble = true;
                        setSelection({ kind, index });
                      }}
                      onDragMove={(e) => {
                        const stage = e.target.getStage();
                        const pos = stage?.getPointerPosition();
                        if (pos) moveOpening(kind, index, pos);
                        e.target.position({ x: 0, y: 0 });
                      }}
                    />
                  );
                }),
              )}
              {warnings.map((w, index) => (
                <Group key={`warn-${index}`} x={w.x * imageWidth} y={w.y * imageHeight}>
                  <Circle
                    radius={11}
                    fill="rgba(233,84,32,0.25)"
                    stroke="#e95420"
                    strokeWidth={2}
                    onMouseDown={(e) => {
                      e.cancelBubble = true;
                      setStatus(`Gemini: [${w.kind}] ${w.description}`);
                    }}
                  />
                  <Text text="!" x={-3} y={-7} fontSize={14} fontStyle="bold" fill="#e95420" listening={false} />
                </Group>
              ))}
              {pending && (
                <Circle x={toPx(pending).x} y={toPx(pending).y} radius={6} stroke="#ff9f1c" strokeWidth={3} dash={[6, 4]} />
              )}
            </Layer>
          </Stage>
        )}
      </div>
      <aside className="sidebar">
        <div className="panel">
          <div className="row">
            {(["select", "wall", "door", "window", "scale"] as Tool[]).map((t) => (
              <button key={t} className={tool === t ? "active" : ""} onClick={() => { setTool(t); setPending(null); }}>
                {t === "scale" ? "scale (2 pts)" : t}
              </button>
            ))}
          </div>
          <div className="row">
            <button className="danger" onClick={removeSelected} disabled={!selection || selection.kind === "rooms"}>
              Delete selected
            </button>
          </div>
          <div style={{ fontSize: 12, color: "#667078" }}>
            Rooms are derived from the wall graph and regenerate on save; label them via the
            inspector instead of drawing them.
          </div>
        </div>
        {selectedItem && (
          <div className="panel">
            <strong style={{ fontSize: 13 }}>
              {selection!.kind} {selectedItem.id ?? selection!.index}
            </strong>
            {selection!.kind === "rooms" && (
              <>
                <label>
                  Room name
                  <input
                    value={selectedItem.name ?? ""}
                    placeholder="e.g. Kitchen"
                    onChange={(e) =>
                      update((m) => {
                        m.rooms[selection!.index].name = e.target.value || null;
                        return m;
                      })
                    }
                  />
                </label>
                <div style={{ fontSize: 12, color: "#667078", display: "grid", gap: 2 }}>
                  <span>{roomStats(selectedItem)}</span>
                  {selectedItem.dimension_m && (
                    <span>
                      labeled: {Number(selectedItem.dimension_m[0]).toFixed(2)} ×{" "}
                      {Number(selectedItem.dimension_m[1]).toFixed(2)} m
                    </span>
                  )}
                  <span>face: {selectedItem.face_id ?? "—"} (name survives wall edits via face matching)</span>
                </div>
              </>
            )}
            {selection!.kind === "walls" && (
              <>
                <label>
                  Thickness (m)
                  <input
                    type="number"
                    step="0.01"
                    value={selectedItem.thickness_m}
                    onChange={(e) =>
                      update((m) => {
                        m.walls[selection!.index].thickness_m = parseFloat(e.target.value) || 0.12;
                        return m;
                      })
                    }
                  />
                </label>
                <label className="row" style={{ display: "flex" }}>
                  <input
                    type="checkbox"
                    checked={!!selectedItem.external}
                    onChange={(e) =>
                      update((m) => {
                        m.walls[selection!.index].external = e.target.checked;
                        m.walls[selection!.index].wall_type = e.target.checked ? "external" : "internal";
                        return m;
                      })
                    }
                  />
                  External wall
                </label>
              </>
            )}
            {(selection!.kind === "doors" || selection!.kind === "windows") && (
              <label>
                Width (m)
                <input
                  type="number"
                  step="0.05"
                  value={selectedItem.width_m}
                  onChange={(e) =>
                    update((m) => {
                      const item = (m as any)[selection!.kind][selection!.index] as Opening;
                      item.width_m = parseFloat(e.target.value) || item.width_m;
                      if (item.offset_m != null) {
                        item.start_offset_m = item.offset_m - item.width_m / 2;
                        item.end_offset_m = item.offset_m + item.width_m / 2;
                      }
                      return m;
                    })
                  }
                />
              </label>
            )}
          </div>
        )}
        <div className="panel">
          <div className="row">
            <button className="primary" onClick={save}>Save</button>
            <button onClick={validate}>Validate</button>
          </div>
          <label>
            Bake mode for 3D generation
            <select value={bakeMode} onChange={(e) => setBakeMode(e.target.value)}>
              <option value="final">final — full-quality bake (slow)</option>
              <option value="draft">draft — fast bake</option>
              <option value="none">none — real-time lights only</option>
            </select>
          </label>
          <label className="row" style={{ display: "flex" }}>
            <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
            Override quality gate (export anyway)
          </label>
          <button className="primary" onClick={generate} disabled={generating}>
            {generating ? "Generating…" : "Generate 3D"}
          </button>
          <Link to={`/jobs/${jobId}/preview`}>Model preview →</Link>
        </div>
        {report && (
          <div className="panel">
            <span className={`quality-chip quality-${report.quality_state}`}>
              {report.quality_state} · {(report.quality_score * 100).toFixed(0)}%
            </span>
            <div style={{ fontSize: 12 }}>
              {Object.entries(report.components).map(([key, value]) => (
                <div key={key}>
                  {key}: {(value * 100).toFixed(0)}%
                </div>
              ))}
            </div>
            <div style={{ fontSize: 12, color: "#7a5a10", maxHeight: 140, overflow: "auto" }}>
              {report.issues.map((issue, index) => (
                <div key={index}>
                  [{issue.severity}] {issue.message}
                </div>
              ))}
            </div>
          </div>
        )}
        <div className="status-box">{status}</div>
      </aside>
    </div>
  );
}
