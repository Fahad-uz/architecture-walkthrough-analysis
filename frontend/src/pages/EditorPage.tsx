import Konva from "konva";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Circle, Group, Image as KonvaImage, Layer, Line, Stage, Text } from "react-konva";
import { Link, useParams } from "react-router-dom";
import { generateModel, getEditData, getJob, saveCorrections } from "../api";
import type { FloorPlanModel, Opening, Point2D, QualityReport, RoomPolygon, SanityWarning, WallSegment } from "../types";

type Tool = "select" | "wall" | "door" | "window" | "scale";
type BakeMode = "final" | "draft" | "none";
type Selection = { kind: "walls" | "doors" | "windows" | "rooms"; index: number } | null;
type SavedCorrection = QualityReport & { model: FloorPlanModel };

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

const OPENING_EDGE_CLEARANCE_M = 0.05;
const MIN_OPENING_WIDTH_M = 0.15;

interface OpeningPlacement {
  center: Point2D;
  width: number;
  offset: number;
  start: number;
  end: number;
}

/** Keep an opening wholly inside its host wall, including a small reveal at
 * both ends. Short walls shrink the opening instead of producing invalid
 * negative/out-of-range intervals. */
function openingPlacement(wall: WallSegment, desiredOffset: number, desiredWidth: number): OpeningPlacement | null {
  const length = wallLength(wall);
  if (!Number.isFinite(length) || length < 0.05) return null;
  const edge = Math.min(OPENING_EDGE_CLEARANCE_M, length * 0.1);
  const maxWidth = length - edge * 2;
  if (maxWidth <= 0) return null;
  const minWidth = Math.min(MIN_OPENING_WIDTH_M, maxWidth);
  const safeDesiredWidth = Number.isFinite(desiredWidth) ? desiredWidth : minWidth;
  const width = Math.min(maxWidth, Math.max(minWidth, safeDesiredWidth));
  const minOffset = edge + width / 2;
  const maxOffset = length - edge - width / 2;
  const safeDesiredOffset = Number.isFinite(desiredOffset) ? desiredOffset : length / 2;
  const offset = Math.min(maxOffset, Math.max(minOffset, safeDesiredOffset));
  return {
    center: pointOnWall(wall, offset),
    width,
    offset,
    start: offset - width / 2,
    end: offset + width / 2,
  };
}

function applyOpeningPlacement(item: Opening, wall: WallSegment, placement: OpeningPlacement): void {
  item.center = placement.center;
  item.width_m = placement.width;
  item.wall_id = wall.id ?? null;
  item.offset_m = placement.offset;
  item.start_offset_m = placement.start;
  item.end_offset_m = placement.end;
}

function openingWall(walls: WallSegment[], item: Opening): WallSegment | null {
  return walls.find((wall) => wall.id === item.wall_id) ?? nearestWall(walls, item.center)?.wall ?? null;
}

function reclampWallOpenings(model: FloorPlanModel, wall: WallSegment): void {
  if (!wall.id) return;
  for (const kind of ["doors", "windows"] as const) {
    model[kind] = model[kind].filter((item) => {
      if (item.wall_id !== wall.id) return true;
      const [start, end] = openingInterval(item);
      const placement = openingPlacement(wall, item.offset_m ?? (start + end) / 2, item.width_m);
      if (!placement) return false;
      applyOpeningPlacement(item, wall, placement);
      return true;
    });
  }
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
    walls: model.walls.map((w) => ({
      ...w,
      start: sp(w.start),
      end: sp(w.end),
      thickness_m: w.thickness_m * factor,
    })),
    doors: model.doors.map((o) => scaleOpening(o, factor, sp)),
    windows: model.windows.map((o) => scaleOpening(o, factor, sp)),
    rooms: model.rooms.map((r) => ({ ...r, points: r.points.map(sp) })),
    balconies: model.balconies?.map((r) => ({ ...r, points: r.points.map(sp) })),
    slabs: model.slabs?.map((r) => ({ ...r, points: r.points.map(sp) })),
    special_elements: model.special_elements?.map((element) => ({
      ...element,
      polygon: element.polygon?.map(sp),
      center: element.center ? sp(element.center) : element.center,
      width_m: element.width_m != null ? element.width_m * factor : element.width_m,
      depth_m: element.depth_m != null ? element.depth_m * factor : element.depth_m,
    })),
    furniture: model.furniture?.map((item) => ({
      ...item,
      center: sp(item.center),
      width_m: item.width_m * factor,
      depth_m: item.depth_m * factor,
    })),
    asset_placements: model.asset_placements?.map((item) => ({
      ...item,
      center: sp(item.center),
      width_m: item.width_m * factor,
      depth_m: item.depth_m * factor,
    })),
    entrance: model.entrance ? sp(model.entrance) : model.entrance,
    camera_waypoints: model.camera_waypoints?.map((waypoint) => ({
      ...waypoint,
      position: sp(waypoint.position),
      look_at: waypoint.look_at ? sp(waypoint.look_at) : waypoint.look_at,
    })),
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

function qualityReportFromModel(model: FloorPlanModel): QualityReport | null {
  const state = model.reconstruction?.quality_state;
  const score = model.reconstruction?.quality_score;
  if (state == null || score == null) return null;
  const rawComponents = model.metadata.quality_components;
  const components =
    rawComponents && typeof rawComponents === "object" && !Array.isArray(rawComponents)
      ? Object.fromEntries(
          Object.entries(rawComponents).filter((entry): entry is [string, number] => typeof entry[1] === "number"),
        )
      : {};
  return {
    quality_state: state,
    quality_score: score,
    components,
    issues: model.validation_issues ?? [],
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
  // Iterate quickly while geometry is still under review. A final bake is an
  // explicit last step because it can take tens of minutes.
  const [bakeMode, setBakeMode] = useState<BakeMode>("draft");
  const [force, setForce] = useState(false);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [generating, setGenerating] = useState(false);
  const pollRef = useRef<number | null>(null);
  const revisionRef = useRef(0);
  const savePromiseRef = useRef<{
    jobId: string;
    requestId: symbol;
    promise: Promise<SavedCorrection | null>;
  } | null>(null);
  const currentJobIdRef = useRef(jobId);
  currentJobIdRef.current = jobId;
  const stageWrapRef = useRef<HTMLDivElement | null>(null);
  const [stageSize, setStageSize] = useState({ width: 800, height: 600 });
  // Viewport transform: the Stage itself is panned/zoomed; all shapes stay in
  // plan-pixel coordinates underneath it.
  const [view, setView] = useState({ x: 0, y: 0, scale: 1 });

  useEffect(() => {
    let cancelled = false;
    let img: HTMLImageElement | null = null;
    setModel(null);
    setImage(null);
    setSelection(null);
    setHover(null);
    setPending(null);
    setReport(null);
    setDirty(false);
    setSaving(false);
    setGenerating(false);
    revisionRef.current = 0;
    setStatus("loading…");
    void (async () => {
      try {
        const data = await getEditData(jobId);
        if (cancelled) return;
        setModel(data.model);
        setReport(qualityReportFromModel(data.model));
        img = new window.Image();
        img.onload = () => {
          if (!cancelled) setImage(img);
        };
        img.onerror = () => {
          if (!cancelled) setStatus("The plan data loaded, but its source image could not be displayed.");
        };
        img.src = data.image_url;
        const issueCount = data.model.validation_issues?.length ?? 0;
        setStatus(
          `loaded: ${data.model.walls.length} walls, ${data.model.rooms.length} rooms${
            issueCount ? ` · ${issueCount} validation issue${issueCount === 1 ? "" : "s"}` : ""
          }`,
        );
      } catch (exc) {
        if (!cancelled) setStatus(String(exc));
      }
    })();
    return () => {
      cancelled = true;
      if (img) {
        img.onload = null;
        img.onerror = null;
      }
      if (pollRef.current !== null) {
        window.clearTimeout(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [jobId]);

  useEffect(() => {
    const element = stageWrapRef.current;
    if (!element) return;
    const observer = new ResizeObserver(() => {
      setStageSize({ width: element.clientWidth, height: element.clientHeight });
    });
    observer.observe(element);
    setStageSize({ width: element.clientWidth, height: element.clientHeight });
    return () => observer.disconnect();
  }, []);

  const ppm = model?.pixels_per_metre ?? 100;
  const imageHeight = image?.naturalHeight ?? 1000;
  const imageWidth = image?.naturalWidth ?? 1000;
  const { toPx, toM } = useMemo(() => makeTransforms(ppm, imageHeight), [ppm, imageHeight]);

  const fitToPlan = useCallback(() => {
    const element = stageWrapRef.current;
    if (!element || !image) return;
    const cw = element.clientWidth;
    const ch = element.clientHeight;
    const scale = Math.min(cw / image.naturalWidth, ch / image.naturalHeight) * 0.97;
    setView({
      x: (cw - image.naturalWidth * scale) / 2,
      y: (ch - image.naturalHeight * scale) / 2,
      scale,
    });
  }, [image]);

  useEffect(() => {
    fitToPlan(); // frame the whole plan once the image is known
  }, [fitToPlan]);

  const update = useCallback((mutate: (m: FloorPlanModel) => FloorPlanModel) => {
    revisionRef.current += 1;
    setDirty(true);
    setReport(null);
    setModel((current) => (current ? mutate(structuredClone(current)) : current));
  }, []);

  /** Pointer position in plan-pixel space, accounting for the pan/zoom transform. */
  const pointerPlanPx = (stage: Konva.Stage | null): { x: number; y: number } | null => {
    const pos = stage?.getPointerPosition();
    if (!pos || !stage) return null;
    return { x: (pos.x - stage.x()) / stage.scaleX(), y: (pos.y - stage.y()) / stage.scaleY() };
  };

  const stagePointer = (e: Konva.KonvaEventObject<MouseEvent>): Point2D | null => {
    const pos = pointerPlanPx(e.target.getStage());
    return pos ? toM(pos) : null;
  };

  const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const stage = e.target.getStage();
    const pointer = stage?.getPointerPosition();
    if (!stage || !pointer) return;
    const factor = e.evt.deltaY > 0 ? 1 / 1.12 : 1.12;
    const nextScale = Math.min(8, Math.max(0.1, view.scale * factor));
    // Keep the plan point under the cursor fixed while zooming.
    const planPoint = { x: (pointer.x - view.x) / view.scale, y: (pointer.y - view.y) / view.scale };
    setView({
      scale: nextScale,
      x: pointer.x - planPoint.x * nextScale,
      y: pointer.y - planPoint.y * nextScale,
    });
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
      if (Math.hypot(p.x - pending.x, p.y - pending.y) < 0.05) {
        setPending(null);
        setStatus("wall: the two endpoints must be at least 0.05 m apart");
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
      if (!Number.isFinite(value) || value <= 0 || distPx <= 0) {
        setStatus("scale: enter a positive real-world distance");
        return;
      }
      const newPpm = distPx / value;
      if (!Number.isFinite(newPpm) || newPpm <= 0) {
        setStatus("scale: that reference did not produce a valid scale");
        return;
      }
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
      const snapDistancePx = hit.dist * ppm * view.scale;
      if (snapDistancePx > 24) {
        setStatus(`${tool}: click closer to the wall where the opening belongs`);
        return;
      }
      const width = tool === "door" ? 0.9 : 1.2;
      const placement = openingPlacement(hit.wall, hit.offset, width);
      if (!placement) {
        setStatus(`${tool}: selected wall is too short for an opening`);
        return;
      }
      const item: Opening = {
        id: `${tool}_manual_${Date.now()}`,
        center: placement.center,
        width_m: placement.width,
        height_m: tool === "door" ? 2.1 : 1.2,
        ...(tool === "window" ? { sill_height_m: 0.9 } : {}),
        wall_id: hit.wall.id ?? null,
        offset_m: placement.offset,
        start_offset_m: placement.start,
        end_offset_m: placement.end,
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
      if (placement.width < width) {
        setStatus(`${tool} width reduced to ${placement.width.toFixed(2)} m to fit the selected wall`);
      }
      return;
    }
  };

  const moveOpening = (kind: "doors" | "windows", index: number, posPx: Point2D) => {
    if (!model) return;
    const p = toM(posPx);
    update((m) => {
      const item = (m as any)[kind][index] as Opening;
      const hit = nearestWall(m.walls, p);
      if (hit && hit.dist * ppm * view.scale <= 24) {
        const placement = openingPlacement(hit.wall, hit.offset, item.width_m);
        if (placement) applyOpeningPlacement(item, hit.wall, placement);
      }
      return m;
    });
  };

  const save = async (): Promise<SavedCorrection | null> => {
    if (!model) return null;
    if (savePromiseRef.current?.jobId === jobId) return savePromiseRef.current.promise;

    const savedJobId = jobId;
    const savedRevision = revisionRef.current;
    const correction = structuredClone(model);
    correction.walls.forEach((wall) => reclampWallOpenings(correction, wall));
    const requestId = Symbol(savedJobId);
    setSaving(true);
    setStatus("saving…");

    const task: Promise<SavedCorrection | null> = (async () => {
      try {
        const result = await saveCorrections(savedJobId, correction);
        if (currentJobIdRef.current !== savedJobId) return null;
        if (revisionRef.current !== savedRevision) {
          setReport(null);
          setStatus("A snapshot was saved, but newer local edits remain. Save again before generating 3D.");
          return null;
        }
        setModel(result.model);
        setSelection(null);
        setHover(null);
        setPending(null);
        setReport(result);
        setDirty(false);
        setStatus(`saved — rooms regenerated from wall graph (${result.model.rooms.length} rooms)`);
        return result;
      } catch (exc) {
        if (currentJobIdRef.current === savedJobId) setStatus(String(exc));
        return null;
      } finally {
        if (savePromiseRef.current?.requestId === requestId) {
          savePromiseRef.current = null;
          if (currentJobIdRef.current === savedJobId) setSaving(false);
        }
      }
    })();
    savePromiseRef.current = { jobId: savedJobId, requestId, promise: task };
    return task;
  };

  const validate = async () => {
    const result = await save();
    if (!result) return;
    const blocking = result.issues.filter((issue) => ["error", "severe"].includes(issue.severity));
    setStatus(
      blocking.length
        ? `validation found ${blocking.length} blocking issue${blocking.length === 1 ? "" : "s"}`
        : `validation complete — ${(result.quality_score * 100).toFixed(0)}% (${result.quality_state})`,
    );
  };

  const generate = async () => {
    if (generating) return;
    const result = await save();
    if (!result) return;
    const blocking = result.issues.filter((issue) => ["error", "severe"].includes(issue.severity));
    if (blocking.length > 0 && !force) {
      setStatus(
        `model build blocked: fix ${blocking.length} validation ${blocking.length === 1 ? "issue" : "issues"}, or explicitly enable the quality-gate override`,
      );
      return;
    }
    if (
      blocking.length > 0 &&
      force &&
      !window.confirm(
        `This model has ${blocking.length} blocking validation ${blocking.length === 1 ? "issue" : "issues"}. Generate it anyway?`,
      )
    ) {
      setStatus("model build cancelled — the lightweight preview remains available");
      return;
    }
    if (
      bakeMode === "final" &&
      result.quality_state !== "high" &&
      !window.confirm(
        `This plan is still ${result.quality_state} at ${(result.quality_score * 100).toFixed(0)}%. A final bake can take tens of minutes. Continue anyway?`,
      )
    ) {
      setStatus("final bake cancelled — continue correcting, or use draft for a fast check");
      return;
    }
    try {
      const generationJobId = jobId;
      setGenerating(true);
      await generateModel(generationJobId, blocking.length > 0 && force, bakeMode);
      if (currentJobIdRef.current !== generationJobId) return;
      setStatus(`Blender build started (${bakeMode})…`);
      if (pollRef.current !== null) window.clearTimeout(pollRef.current);
      const poll = async () => {
        try {
          const job = await getJob(generationJobId);
          if (currentJobIdRef.current !== generationJobId) return;
          if (job.status === "model_generated") {
            pollRef.current = null;
            setGenerating(false);
            setStatus("3D model ready — open the preview page");
            return;
          } else if (["blocked", "generation_failed"].includes(job.status)) {
            pollRef.current = null;
            setGenerating(false);
            setStatus(`${job.status}: ${job.message}`);
            return;
          }
        } catch (exc) {
          if (currentJobIdRef.current !== generationJobId) return;
          setStatus(`waiting for model status: ${String(exc)}`);
        }
        if (currentJobIdRef.current === generationJobId) {
          pollRef.current = window.setTimeout(() => void poll(), 2000);
        }
      };
      pollRef.current = window.setTimeout(() => void poll(), 500);
    } catch (exc) {
      if (currentJobIdRef.current === jobId) {
        setGenerating(false);
        setStatus(String(exc));
      }
    }
  };

  const removeSelected = () => {
    if (!selection) return;
    update((m) => {
      if (selection.kind === "walls") {
        const wall = m.walls[selection.index];
        if (wall?.id) {
          m.doors = m.doors.filter((item) => item.wall_id !== wall.id);
          m.windows = m.windows.filter((item) => item.wall_id !== wall.id);
        }
      }
      (m as any)[selection.kind].splice(selection.index, 1);
      return m;
    });
    setSelection(null);
  };

  const selectedItem: any = selection && model ? (model as any)[selection.kind][selection.index] : null;
  const warnings: SanityWarning[] = (model?.metadata?.sanity_warnings as SanityWarning[]) ?? [];
  const blockingIssueCount = report?.issues.filter((issue) => ["error", "severe"].includes(issue.severity)).length ?? 0;

  return (
    <div className="editor-layout">
      <div
        className="stage-wrap"
        ref={stageWrapRef}
        role="region"
        aria-label="Interactive floor-plan correction canvas"
      >
        <div className="stage-hud">
          <button type="button" onClick={fitToPlan} disabled={!image}>
            Fit to plan
          </button>
          <span>{Math.round(view.scale * 100)}%</span>
        </div>
        {model && image && (
          <Stage
            width={stageSize.width}
            height={stageSize.height}
            x={view.x}
            y={view.y}
            scaleX={view.scale}
            scaleY={view.scale}
            draggable={tool === "select"}
            onDragEnd={(e) => {
              // Child shapes bubble drag events; only sync the pan when the
              // stage itself moved.
              if (e.target === e.target.getStage()) {
                setView((v) => ({ ...v, x: e.target.x(), y: e.target.y() }));
              }
            }}
            onWheel={handleWheel}
            onMouseDown={handleStageClick}
          >
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
                                const editedWall = m.walls[index];
                                (editedWall as any)[key] = q;
                                reclampWallOpenings(m, editedWall);
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
                        const pos = pointerPlanPx(e.target.getStage());
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
                      setStatus(`Review note: [${w.kind}] ${w.description}`);
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
              <button
                key={t}
                type="button"
                aria-pressed={tool === t}
                className={tool === t ? "active" : ""}
                onClick={() => {
                  setTool(t);
                  setPending(null);
                }}
              >
                {t === "scale" ? "scale (2 pts)" : t}
              </button>
            ))}
          </div>
          <div className="row">
            <button
              type="button"
              className="danger"
              onClick={removeSelected}
              disabled={!selection || selection.kind === "rooms"}
            >
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
                  min={MIN_OPENING_WIDTH_M}
                  step="0.05"
                  value={selectedItem.width_m}
                  onChange={(e) =>
                    update((m) => {
                      const item = (m as any)[selection!.kind][selection!.index] as Opening;
                      const requestedWidth = parseFloat(e.target.value);
                      if (!Number.isFinite(requestedWidth) || requestedWidth <= 0) return m;
                      const wall = openingWall(m.walls, item);
                      if (wall) {
                        const [start, end] = openingInterval(item);
                        const placement = openingPlacement(
                          wall,
                          item.offset_m ?? (start + end) / 2,
                          requestedWidth,
                        );
                        if (placement) applyOpeningPlacement(item, wall, placement);
                      } else {
                        item.width_m = requestedWidth;
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
            <button
              type="button"
              className="primary"
              onClick={() => void save()}
              disabled={!model || saving || generating}
            >
              {saving ? "Saving…" : dirty ? "Save changes" : "Save"}
            </button>
            <button
              type="button"
              onClick={() => void validate()}
              disabled={!model || saving || generating}
            >
              Validate
            </button>
            {dirty && <span className="unsaved-indicator">Unsaved changes</span>}
          </div>
          <label>
            Bake mode for 3D generation
            <select
              value={bakeMode}
              disabled={saving || generating}
              onChange={(e) => setBakeMode(e.target.value as BakeMode)}
            >
              <option value="final">final — full-quality bake (slow)</option>
              <option value="draft">draft — fast geometry/material check</option>
              <option value="none">none — real-time lights only</option>
            </select>
          </label>
          <div style={{ fontSize: 12, color: bakeMode === "final" ? "#8a4d12" : "#667078" }}>
            {bakeMode === "final"
              ? blockingIssueCount > 0
                ? `Final bake is blocked by ${blockingIssueCount} validation issue${blockingIssueCount === 1 ? "" : "s"}.`
                : "Final can take tens of minutes; use it only after the draft geometry looks correct."
              : "Draft is recommended after structural validation; the lightweight preview updates on every save."}
          </div>
          <label className="row" style={{ display: "flex" }}>
            <input
              type="checkbox"
              checked={force}
              disabled={saving || generating}
              onChange={(e) => setForce(e.target.checked)}
            />
            Override quality gate (export anyway)
          </label>
          <button
            type="button"
            className="primary"
            onClick={() => void generate()}
            disabled={!model || saving || generating}
          >
            {generating ? "Generating…" : "Generate 3D"}
          </button>
          <Link to={`/jobs/${jobId}/preview`}>
            Model preview{dirty ? " (last saved)" : ""} →
          </Link>
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
            <div
              role="list"
              aria-label="Validation issues"
              style={{ fontSize: 12, color: "#7a5a10", maxHeight: 140, overflow: "auto" }}
            >
              {report.issues.map((issue, index) => (
                <div
                  role="listitem"
                  key={`${issue.code}-${index}`}
                  style={{ color: ["error", "severe"].includes(issue.severity) ? "#a12622" : undefined }}
                >
                  [{issue.severity}] {issue.message}
                </div>
              ))}
            </div>
          </div>
        )}
        <div className="status-box" role="status" aria-live="polite" aria-atomic="true">
          {status}
        </div>
      </aside>
    </div>
  );
}
