export interface Point2D {
  x: number;
  y: number;
}

export interface WallSegment {
  id?: string | null;
  start: Point2D;
  end: Point2D;
  thickness_m: number;
  height_m: number;
  external: boolean;
  wall_type?: string;
  confidence?: number;
  evidence_source?: string;
  [key: string]: unknown;
}

export interface Opening {
  id?: string | null;
  center: Point2D;
  width_m: number;
  height_m: number;
  sill_height_m?: number;
  wall_id?: string | null;
  offset_m?: number | null;
  start_offset_m?: number | null;
  end_offset_m?: number | null;
  hinge_side?: "start" | "end" | null;
  swing_side?: "left" | "right" | null;
  opening_type?: string;
  confidence?: number;
  evidence_source?: string;
  [key: string]: unknown;
}

export interface RoomPolygon {
  id?: string | null;
  face_id?: string | null;
  name?: string | null;
  points: Point2D[];
  confidence?: number;
  [key: string]: unknown;
}

export interface SanityWarning {
  kind: string;
  description: string;
  x: number;
  y: number;
  confidence: number;
}

export interface FloorPlanModel {
  coordinate_system: string;
  schema_version: string;
  pixels_per_metre: number | null;
  walls: WallSegment[];
  doors: Opening[];
  windows: Opening[];
  rooms: RoomPolygon[];
  camera_waypoints?: { position: Point2D; look_at?: Point2D | null; pause_seconds?: number }[];
  metadata: Record<string, unknown> & { sanity_warnings?: SanityWarning[] };
  reconstruction?: { quality_state?: string; quality_score?: number };
  validation_issues?: { code: string; severity: string; message: string }[];
  [key: string]: unknown;
}

export interface JobRecord {
  job_id: string;
  status: string;
  message: string;
  glb_url?: string | null;
  glb_source?: string | null;
  quality_state?: string | null;
  quality_score?: number | null;
  ai_assist_error?: string | null;
}

export interface QualityReport {
  quality_state: string;
  quality_score: number;
  components: Record<string, number>;
  issues: { code: string; severity: string; message: string }[];
}
