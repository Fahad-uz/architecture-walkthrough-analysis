import assert from "node:assert/strict";
import test from "node:test";
import {
  addPosition, createFurniture, FURNITURE_PRESETS, furnitureFootprint, furnitureLabel, overlapWarnings,
} from "../src/editorFurniture.ts";
import { appendDraft, createHistory, parseDraft, recordEdit, redoEdit, undoEdit } from "../src/editorHistory.ts";
import { pointInPolygon } from "../src/navigation.ts";
import type { FloorPlanModel, FurniturePlacement, Point2D, RoomPolygon } from "../src/types.ts";

function model(): FloorPlanModel {
  return {
    coordinate_system: "metres", schema_version: "3.0", pixels_per_metre: 100,
    walls: [{ id: "wall-a", start: { x: 0, y: 0 }, end: { x: 5, y: 0 }, thickness_m: 0.2, height_m: 2.8, external: false }],
    rooms: [], doors: [], windows: [], furniture: [], metadata: {},
  };
}

function item(overrides: Partial<FurniturePlacement> = {}): FurniturePlacement {
  return { category: "table", center: { x: 2, y: 0 }, width_m: 0.4, depth_m: 0.4, ...overrides };
}

function closePoint(actual: Point2D, expected: Point2D) {
  assert.ok(Math.abs(actual.x - expected.x) < 1e-9, `${actual.x} != ${expected.x}`);
  assert.ok(Math.abs(actual.y - expected.y) < 1e-9, `${actual.y} != ${expected.y}`);
}

test("supported presets create distinct user items with metre dimensions and independent centres", () => {
  const center = { x: 2, y: 3 };
  for (const preset of FURNITURE_PRESETS) {
    const first = createFurniture(preset.category, center);
    const second = createFurniture(preset.category, center);
    assert.equal(first.provenance, "user");
    assert.match(first.id!, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    assert.notEqual(first.id, second.id);
    assert.equal(first.width_m, preset.width_m);
    assert.equal(first.depth_m, preset.depth_m);
    assert.equal(first.height_m, preset.height_m);
    assert.equal(first.rotation_deg, 0);
    assert.equal(furnitureLabel(first.category), preset.label);
    assert.notEqual(first.center, center);
  }
  assert.equal(furnitureLabel("dining_table"), "Dining Table");
  assert.throws(() => createFurniture("unsupported", center), /supported furniture/);
});

test("footprints retain metre sizes and rotate counterclockwise around their centre", () => {
  const rectangle = item({ center: { x: 4, y: 3 }, width_m: 2, depth_m: 1 });
  assert.deepEqual(furnitureFootprint(rectangle), [
    { x: 3, y: 2.5 }, { x: 5, y: 2.5 }, { x: 5, y: 3.5 }, { x: 3, y: 3.5 },
  ]);
  const rotated = furnitureFootprint({ ...rectangle, rotation_deg: 90 });
  [
    { x: 4.5, y: 2 }, { x: 4.5, y: 4 }, { x: 3.5, y: 4 }, { x: 3.5, y: 2 },
  ].forEach((point, index) => closePoint(rotated[index], point));
  let twiceArea = 0;
  rotated.forEach((point, index) => {
    const next = rotated[(index + 1) % rotated.length];
    twiceArea += point.x * next.y - next.x * point.y;
  });
  assert.ok(twiceArea > 0);
  furnitureFootprint({ ...rectangle, rotation_deg: -270 }).forEach((point, index) => closePoint(point, rotated[index]));
});

test("add placement uses the selected room's area centroid and preserves a visible fallback", () => {
  const selected: RoomPolygon = { points: [{ x: 1, y: 1 }, { x: 5, y: 1 }, { x: 5, y: 3 }, { x: 1, y: 3 }] };
  const visible = { x: 20, y: 30 };
  assert.deepEqual(addPosition(model(), selected, visible), { x: 3, y: 2 });
  assert.deepEqual(addPosition(model(), { points: [...selected.points].reverse() }, visible), { x: 3, y: 2 });
  assert.deepEqual(addPosition(model(), null, visible), visible);
  assert.notEqual(addPosition(model(), null, visible), visible);
  assert.deepEqual(addPosition(model(), { points: [] }, visible), visible);
});

test("concave and narrow rooms receive an interior point even when the centroid lies outside", () => {
  const room: RoomPolygon = {
    points: [
      { x: 0, y: 0 }, { x: 6, y: 0 }, { x: 6, y: 6 }, { x: 5.999, y: 6 },
      { x: 5.999, y: 0.001 }, { x: 0.001, y: 0.001 }, { x: 0.001, y: 6 }, { x: 0, y: 6 },
    ],
  };
  assert.equal(pointInPolygon({ x: 3, y: 2 }, room.points), false);
  const position = addPosition(model(), room, { x: 3, y: 3 });
  assert.equal(pointInPolygon(position, room.points), true);
  assert.equal(pointInPolygon(addPosition(model(), { points: [...room.points].reverse() }, { x: 3, y: 3 }), room.points), true);
});

test("wall hints use the rotated footprint and ignore flush edge contact", () => {
  assert.deepEqual(overlapWarnings(model(), item()).map((warning) => warning.kind), ["wall"]);
  assert.deepEqual(overlapWarnings(model(), item({ center: { x: 2, y: 0.3 } })), []);
  const longTable = item({ center: { x: 2, y: 0.7 }, width_m: 2, depth_m: 0.2 });
  assert.deepEqual(overlapWarnings(model(), longTable), []);
  assert.deepEqual(overlapWarnings(model(), { ...longTable, rotation_deg: 90 }).map((warning) => warning.kind), ["wall"]);

  const separated = model();
  separated.walls[0].start = { x: 0.3, y: 1.6 };
  separated.walls[0].end = { x: 0.5, y: 1.6 };
  assert.deepEqual(overlapWarnings(separated, item({ center: { x: 1, y: 1 }, width_m: 2, depth_m: 0.2, rotation_deg: 45 })), []);
});

test("door spans are removed from wall hints, including legacy midpoint and projected-centre forms", () => {
  const plan = model();
  plan.doors = [{ id: "door-a", wall_id: "wall-a", center: { x: 2, y: 0 }, width_m: 1, height_m: 2.1,
    start_offset_m: 1.5, end_offset_m: 2.5 }];
  assert.deepEqual(overlapWarnings(plan, item()), [{ kind: "door", message: "Overlaps a door opening.", targetId: "door-a" }]);
  assert.deepEqual(overlapWarnings(plan, item({ center: { x: 1.5, y: 0 } })).map((warning) => warning.kind).sort(), ["door", "wall"]);
  assert.deepEqual(overlapWarnings(plan, item({ center: { x: 3, y: 0 } })).map((warning) => warning.kind), ["wall"]);
  assert.deepEqual(overlapWarnings(plan, item({ center: { x: 2, y: 0.5 } })), []);

  delete plan.doors[0].start_offset_m;
  delete plan.doors[0].end_offset_m;
  plan.doors[0].offset_m = 2;
  assert.deepEqual(overlapWarnings(plan, item()).map((warning) => warning.kind), ["door"]);
  delete plan.doors[0].offset_m;
  assert.deepEqual(overlapWarnings(plan, item()).map((warning) => warning.kind), ["door"]);
  plan.doors[0].wall_id = "another-wall";
  assert.deepEqual(overlapWarnings(plan, item()).map((warning) => warning.kind), ["wall"]);
});

test("diagonal walls and overlapping door intervals keep the correct remaining solids", () => {
  const plan = model();
  plan.walls[0].end = { x: 4, y: 4 };
  plan.doors = [
    { id: "door-a", wall_id: "wall-a", center: { x: 1, y: 1 }, width_m: 1, height_m: 2.1, start_offset_m: 1, end_offset_m: 2 },
    { id: "door-b", wall_id: "wall-a", center: { x: 1, y: 1 }, width_m: 1, height_m: 2.1, start_offset_m: 1.5, end_offset_m: 2.5 },
  ];
  const middle = 1.75 / Math.sqrt(2);
  const warnings = overlapWarnings(plan, item({ center: { x: middle, y: middle }, width_m: 0.1, depth_m: 0.1 }));
  assert.deepEqual(warnings.map((warning) => warning.kind), ["door", "door"]);
  assert.deepEqual(overlapWarnings(plan, item({ center: { x: 3, y: 3 } })).map((warning) => warning.kind), ["wall"]);
});

test("furniture IDs, provenance and source metadata survive edits, history and draft recovery", () => {
  const original = model();
  const detected = item({ id: "detected-a", provenance: "detected", evidence_source: "local_symbols", detector: { version: 2 } });
  const added = createFurniture("bed", { x: 2, y: 3 });
  original.furniture = [detected, added];
  const changed = structuredClone(original);
  changed.furniture![0] = { ...changed.furniture![0], center: { x: 3, y: 4 }, rotation_deg: 37 };
  changed.furniture![1] = { ...changed.furniture![1], width_m: 1.8 };
  const history = recordEdit(createHistory(original), changed);
  assert.deepEqual(undoEdit(history).present, original);
  const redone = redoEdit(undoEdit(history)).present;
  assert.deepEqual(redone, changed);
  assert.equal(redone.furniture![0].id, "detected-a");
  assert.equal(redone.furniture![0].provenance, "detected");
  assert.equal(redone.furniture![0].evidence_source, "local_symbols");
  assert.deepEqual(redone.furniture![0].detector, { version: 2 });
  assert.equal(redone.furniture![1].id, added.id);
  assert.equal(redone.furniture![1].provenance, "user");
  const draft = appendDraft(null, "job-a", redone, "manual");
  assert.deepEqual(parseDraft(JSON.stringify(draft), "job-a", "3.0")!.versions[0].model.furniture, changed.furniture);
  const removed = { ...redone, furniture: redone.furniture!.filter((entry) => entry.id !== added.id) };
  assert.deepEqual(undoEdit(recordEdit(history, removed)).present.furniture, changed.furniture);
});
