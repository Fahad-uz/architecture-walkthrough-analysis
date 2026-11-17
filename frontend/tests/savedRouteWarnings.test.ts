import assert from "node:assert/strict";
import test from "node:test";
import { savedRouteWarnings } from "../src/savedRouteWarnings.ts";
import type { FloorPlanModel, FurniturePlacement, Point2D } from "../src/types.ts";

function model(points: Point2D[] = [{ x: -100, y: 0 }, { x: 100, y: 0 }]): FloorPlanModel {
  return {
    coordinate_system: "metres", schema_version: "3.0", pixels_per_metre: 100,
    walls: [{ id: "wall-a", start: { x: -5, y: -5 }, end: { x: 5, y: -5 }, thickness_m: 0.2, height_m: 2.8, external: true }],
    doors: [{ id: "door-a", wall_id: "wall-a", center: { x: 0, y: -5 }, width_m: 1, height_m: 2.1,
      start_offset_m: 4.5, end_offset_m: 5.5 }],
    windows: [{ id: "window-a", wall_id: "wall-a", center: { x: 3, y: -5 }, width_m: 1, height_m: 1 }],
    rooms: [{ id: "room-a", name: "Living", points: [{ x: -5, y: -5 }, { x: 5, y: -5 }, { x: 5, y: 5 }, { x: -5, y: 5 }] }],
    furniture: [], camera_waypoints: points.map((position) => ({ position })), metadata: {},
    reconstruction: { quality_score: 0.7, quality_state: "review_required" },
    validation_issues: [{ code: "review", severity: "warning", message: "Review source evidence." }],
  };
}

function furniture(overrides: Partial<FurniturePlacement> = {}): FurniturePlacement {
  return { id: "table-a", category: "table", center: { x: 0, y: 0 }, width_m: 1, depth_m: 1, rotation_deg: 0, ...overrides };
}

function kinds(current: FloorPlanModel, item = furniture(), saved: FloorPlanModel | null = current): string[] {
  return savedRouteWarnings(current, item, saved).map((warning) => warning.kind);
}

test("a continuous saved segment detects furniture between distant waypoints and warns once", () => {
  const saved = model();
  saved.camera_waypoints!.push({ position: { x: -100, y: 0 } });
  const warnings = savedRouteWarnings(saved, furniture({ center: { x: 23.456, y: 0 } }), saved);
  assert.deepEqual(warnings, [{ kind: "saved_route", message: "This furniture overlaps the saved tour path. Move it or save to regenerate and review the route." }]);
});

test("saved tour clearance uses the capsule radius along the segment and at its endpoints", () => {
  assert.deepEqual(kinds(model(), furniture({ center: { x: 0, y: 0.79 } })), ["saved_route"]);
  assert.deepEqual(kinds(model(), furniture({ center: { x: 0, y: 0.81 } })), []);
  assert.deepEqual(kinds(model([{ x: -5, y: 0 }, { x: -0.7, y: 0 }])), ["saved_route"]);
  assert.deepEqual(kinds(model([{ x: -5, y: 0 }, { x: -0.81, y: 0 }])), []);
  assert.deepEqual(kinds(model([{ x: -5, y: 0 }, { x: -0.8, y: 0 }])), ["saved_route"], "boundary contact survives floating-point rounding");
});

test("rounded capsule corners reject an expanded-rectangle near miss", () => {
  const square = furniture({ width_m: 2, depth_m: 2 });
  assert.deepEqual(kinds(model([{ x: 1.25, y: 1.25 }, { x: 2, y: 2 }]), square), []);
  assert.deepEqual(kinds(model([{ x: 1.2, y: 1.2 }, { x: 2, y: 2 }]), square), ["saved_route"]);
  assert.deepEqual(kinds(model([{ x: 1.2, y: 1.25 }, { x: 1.2, y: 2 }]), square), []);
});

test("rotation uses the actual oriented footprint, including diagonal near misses", () => {
  const narrowTable = furniture({ width_m: 4, depth_m: 0.2 });
  const horizontalRoute = model([{ x: -3, y: 1 }, { x: 3, y: 1 }]);
  assert.deepEqual(kinds(horizontalRoute, narrowTable), []);
  assert.deepEqual(kinds(horizontalRoute, { ...narrowTable, rotation_deg: 90 }), ["saved_route"]);
  assert.deepEqual(kinds(horizontalRoute, { ...narrowTable, rotation_deg: -270 }), ["saved_route"]);
  const cornerRoute = model([{ x: 1.3, y: -1.3 }, { x: 2, y: -2 }]);
  assert.deepEqual(kinds(cornerRoute, { ...narrowTable, rotation_deg: 45 }), []);
  assert.deepEqual(kinds(model([{ x: -2, y: -2 }, { x: 2, y: 2 }]), { ...narrowTable, rotation_deg: 45 }), ["saved_route"]);
});

test("walkable rugs, carpets, floor patches and door categories use navigation semantics", () => {
  for (const category of ["rug", "living_room_rug", "carpet", "floor_patch", "wood-floor-patch", "door_panel"]) {
    assert.deepEqual(kinds(model(), furniture({ category })), [], category);
  }
  assert.deepEqual(kinds(model(), furniture({ category: "wardrobe" })), ["saved_route"]);
});

test("missing, non-metre, single-point and stationary saved tours never invent a route", () => {
  const saved = model();
  assert.deepEqual(kinds(saved, furniture(), null), []);
  for (const camera_waypoints of [undefined, null, {}, [], [{ position: { x: 0, y: 0 } }],
    [{ position: { x: 0, y: 0 } }, { position: { x: 0, y: 0 } }]]) {
    const changed = { ...saved, camera_waypoints } as FloorPlanModel;
    assert.deepEqual(kinds(changed), []);
  }
  for (const coordinate_system of ["pixels", "unknown", ""]) {
    const changed = { ...saved, coordinate_system };
    assert.deepEqual(kinds(changed), []);
  }
  assert.deepEqual(kinds({ ...saved, coordinate_system: "cartesian_metres" }), ["saved_route"]);
});

test("malformed waypoints break adjacency while valid segments after a gap remain usable", () => {
  for (const malformed of [null, {}, { position: null }, { position: { x: NaN, y: 0 } },
    { position: { x: Infinity, y: 0 } }, { position: { x: "0", y: 0 } }, { position: { x: 0 } }]) {
    const saved = model();
    saved.camera_waypoints = [saved.camera_waypoints![0], malformed, saved.camera_waypoints![1]] as any;
    assert.deepEqual(kinds(saved), [], JSON.stringify(malformed));
    saved.camera_waypoints!.push({ position: { x: 101, y: 0 } });
    assert.deepEqual(kinds(saved), [], "the valid far-away segment must not bridge the gap");
    saved.camera_waypoints!.push({ position: { x: -100, y: 0 } });
    assert.deepEqual(kinds(saved), ["saved_route"]);
  }
});

test("tour segments do not close into a loop or extend to entrances or room centres", () => {
  const saved = model([{ x: -2, y: 2 }, { x: 2, y: 2 }, { x: 2, y: -2 }]);
  saved.entrance = { x: 0, y: 0 };
  assert.deepEqual(kinds(saved), []);
});

test("the saved source supplies the route even when live waypoints change or disappear", () => {
  const saved = model();
  const current = structuredClone(saved);
  current.camera_waypoints = [{ position: { x: 0, y: 100 } }, { position: { x: 10, y: 100 } }];
  assert.deepEqual(kinds(current, furniture(), saved), ["saved_route"]);
  current.camera_waypoints = [];
  assert.deepEqual(kinds(current, furniture(), saved), ["saved_route"]);
  assert.deepEqual(kinds(saved, furniture(), current), [], "a live route cannot replace an absent saved path");
});

test("scale, coordinate frame and structural edits require a new save before comparison", () => {
  const saved = model();
  const changes: ((plan: FloorPlanModel) => void)[] = [
    (plan) => { plan.pixels_per_metre = 101; },
    (plan) => { plan.coordinate_system = "pixels"; },
    (plan) => { plan.walls[0].end.x += 1e-12; },
    (plan) => { plan.walls[0].thickness_m += 0.01; },
    (plan) => { plan.walls.push({ ...plan.walls[0], id: "wall-b" }); },
    (plan) => { plan.doors[0].start_offset_m = 4; },
    (plan) => { plan.windows[0].width_m = 2; },
    (plan) => { plan.rooms[0].points[0].x -= 1; },
    (plan) => { plan.balconies = [structuredClone(plan.rooms[0])]; },
    (plan) => { plan.slabs = [structuredClone(plan.rooms[0])]; },
    (plan) => { plan.special_elements = [{ id: "shaft", kind: "shaft", center: { x: 2, y: 2 } }]; },
    (plan) => { plan.entrance = { x: 3, y: 3 }; },
    (plan) => { plan.plan_roi = { x: 0, y: 0, width: 300, height: 200 }; },
    (plan) => { plan.metadata.ceiling_void_room_ids = ["room-a"]; },
    (plan) => { plan.metadata.floor_boundary_inferences = [{ room_id: "room-a" }]; },
  ];
  for (const change of changes) {
    const current = structuredClone(saved);
    change(current);
    const warnings = savedRouteWarnings(current, furniture(), saved);
    assert.equal(warnings.length, 1);
    assert.equal(warnings[0].kind, "saved_route_stale");
    assert.match(warnings[0].message, /Save changes/);
  }
  const rescaled = structuredClone(saved);
  rescaled.pixels_per_metre = 50;
  rescaled.camera_waypoints = rescaled.camera_waypoints!.map((waypoint) => ({ position: { x: waypoint.position.x * 2, y: waypoint.position.y * 2 } }));
  assert.deepEqual(kinds(rescaled, furniture(), saved), ["saved_route_stale"], "live rescaled waypoints must not make an old saved tour comparable");
});

test("furniture, labels, finishes, lighting and quality edits preserve the structural comparison", () => {
  const saved = model();
  const current = structuredClone(saved);
  current.rooms[0].name = "Study";
  current.rooms[0].floor_material_preset = "wood_floor";
  current.rooms[0].wall_material_preset = "painted_wall";
  current.furniture = [furniture()];
  current.metadata.material_preset = "oak";
  current.lighting = { preset: "warm", color_temperature: 3000, brightness: 1.2, intensity: 450, exposure: 0 };
  current.reconstruction!.quality_score = 0.2;
  current.validation_issues = [];
  current.walls[0].confidence = 0.3;
  assert.deepEqual(kinds(current, furniture(), saved), ["saved_route"]);
  current.walls.push({ ...current.walls[0], id: "wall-b" });
  saved.walls.push({ ...saved.walls[0], id: "wall-b" });
  current.walls.reverse();
  assert.deepEqual(kinds(current, furniture(), saved), ["saved_route"], "element ordering has no geometric meaning");
});

test("invalid furniture and non-finite structural comparisons produce no overlap conclusion", () => {
  const saved = model();
  for (const override of [{ width_m: 0 }, { width_m: -1 }, { width_m: Infinity }, { depth_m: NaN },
    { rotation_deg: NaN }, { center: { x: NaN, y: 0 } }, { category: null }]) {
    assert.deepEqual(kinds(saved, furniture(override as Partial<FurniturePlacement>)), []);
  }
  for (const value of [NaN, Infinity, -10]) {
    const current = structuredClone(saved);
    current.pixels_per_metre = value;
    assert.deepEqual(kinds(current, furniture(), saved), ["saved_route_stale"]);
  }
  const invalid = structuredClone(saved);
  invalid.walls[0].start.x = NaN;
  assert.deepEqual(kinds(invalid), ["saved_route_stale"]);
});

test("placement hints never mutate either model, route, furniture, validation or quality", () => {
  const saved = model();
  const current = structuredClone(saved);
  current.furniture = [furniture()];
  const before = structuredClone({ saved, current });
  const freeze = (value: any) => {
    if (value && typeof value === "object") { Object.values(value).forEach(freeze); Object.freeze(value); }
  };
  freeze(saved);
  freeze(current);
  assert.deepEqual(kinds(current, current.furniture![0], saved), ["saved_route"]);
  assert.deepEqual({ saved, current }, before);
});
