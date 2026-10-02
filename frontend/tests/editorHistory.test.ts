import assert from "node:assert/strict";
import test from "node:test";
import {
  appendDraft, createHistory, draftStorageKey, HISTORY_LIMIT, parseDraft, recordEdit, redoEdit, undoEdit, writeDraft,
} from "../src/editorHistory.ts";
import type { FloorPlanModel } from "../src/types.ts";

function model(): FloorPlanModel {
  return {
    coordinate_system: "cartesian_metres", schema_version: "1.0", pixels_per_metre: 100,
    walls: [{ id: "wall-a", start: { x: 0, y: 0 }, end: { x: 5, y: 0 }, thickness_m: 0.12, height_m: 2.8, external: true }],
    doors: [], windows: [],
    rooms: [{ id: "room-a", name: "Kitchen", points: [{ x: 0, y: 0 }, { x: 5, y: 0 }, { x: 5, y: 4 }, { x: 0, y: 4 }] }],
    metadata: { scale_source: "manual" },
  };
}

test("undo restores deleted walls together with their openings; redo restores the complete edit", () => {
  const original = model();
  original.doors.push({ id: "door-a", wall_id: "wall-a", center: { x: 2, y: 0 }, width_m: 0.9, height_m: 2.1 });
  const edited = structuredClone(original);
  edited.walls = [];
  edited.doors = [];
  const history = recordEdit(createHistory(original), edited);
  const restored = undoEdit(history);
  assert.deepEqual(restored.present, original);
  assert.deepEqual(redoEdit(restored).present, edited);
  assert.deepEqual(history.present, edited);
});

test("a drag is one undo step, while the next independent edit is separate", () => {
  const original = model();
  let history = createHistory(original);
  for (let x = 6; x <= 20; x += 1) {
    const next = structuredClone(history.present);
    next.walls[0].end.x = x;
    history = recordEdit(history, next, x > 6);
  }
  assert.equal(history.past.length, 1);
  assert.deepEqual(undoEdit(history).present, original);
  const rename = structuredClone(history.present);
  rename.rooms[0].name = "Living room";
  const next = recordEdit(history, rename);
  assert.equal(next.past.length, 2);
  assert.equal(undoEdit(next).present.walls[0].end.x, 20);
  assert.equal(undoEdit(next).present.rooms[0].name, "Kitchen");
});

test("branching after undo clears redo without mutating snapshots; no-op updates keep redo", () => {
  const first = model();
  const second = structuredClone(first);
  second.rooms[0].name = "Bedroom";
  const undone = undoEdit(recordEdit(createHistory(first), second));
  assert.equal(recordEdit(undone, structuredClone(first)), undone);
  const third = structuredClone(first);
  third.rooms[0].name = "Study";
  const branched = recordEdit(undone, third);
  third.rooms[0].name = "Unexpected mutation";
  assert.equal(branched.present.rooms[0].name, "Study");
  assert.equal(branched.future.length, 0);
  assert.equal(undone.future[0].rooms[0].name, "Bedroom");
});

test("history remains bounded after hundreds of changes and boundary undo/redo are safe", () => {
  let history = createHistory(0);
  assert.equal(undoEdit(history), history);
  assert.equal(redoEdit(history), history);
  for (let i = 1; i <= 500; i += 1) history = recordEdit(history, i);
  assert.equal(history.past.length, HISTORY_LIMIT);
  for (let i = 0; i < HISTORY_LIMIT; i += 1) history = undoEdit(history);
  assert.equal(history.present, 500 - HISTORY_LIMIT);
  assert.equal(undoEdit(history), history);
  for (let i = 0; i < HISTORY_LIMIT; i += 1) history = redoEdit(history);
  assert.equal(history.present, 500);
});

test("versioned drafts survive reload, isolate jobs, and retain only the newest five layouts", () => {
  let draft = appendDraft(null, "job/a", model(), "manual", new Date("2026-10-03T09:00:00Z"));
  for (let i = 1; i <= 8; i += 1) {
    const edited = model();
    edited.rooms[0].name = `Room ${i}`;
    draft = appendDraft(draft, "job/a", edited, "automatic", new Date(`2026-10-03T09:0${i}:00Z`));
  }
  const recovered = parseDraft(JSON.stringify(draft), "job/a", "1.0")!;
  assert.equal(recovered.versions.length, 5);
  assert.equal(recovered.versions[0].model.rooms[0].name, "Room 8");
  assert.equal(recovered.versions[4].model.rooms[0].name, "Room 4");
  assert.notEqual(draftStorageKey("job/a"), draftStorageKey("job%2Fa"));
  assert.throws(() => parseDraft(JSON.stringify(draft), "job-b", "1.0"), /does not match/);
  assert.throws(() => appendDraft(draft, "job-b", model(), "automatic"), /different plan/);
  assert.throws(() => parseDraft(JSON.stringify(draft), "job/a", "2.0"), /does not match/);
});

test("automatic saves deduplicate unchanged layouts and draft snapshots are independent", () => {
  const original = model();
  const draft = appendDraft(null, "job-a", original, "automatic");
  assert.equal(appendDraft(draft, "job-a", original, "automatic"), draft);
  original.walls[0].end.x = 90;
  assert.equal(draft.versions[0].model.walls[0].end.x, 5);
  assert.equal(appendDraft(draft, "job-a", draft.versions[0].model, "manual").versions.length, 2);
});

test("damaged or mismatched local data cannot enter the editor", () => {
  assert.equal(parseDraft(null, "job-a", "1.0"), null);
  assert.throws(() => parseDraft("broken{", "job-a", "1.0"), /damaged/);
  const raw = appendDraft(null, "job-a", model(), "automatic");
  const malformedModels = [
    { ...model(), walls: [{ start: null }] },
    { ...model(), pixels_per_metre: 0 },
    { ...model(), rooms: [{ points: [null, null, null] }] },
    { ...model(), furniture: [{ center: null }] },
    { ...model(), camera_waypoints: [{ position: "missing" }] },
    { ...model(), metadata: { sanity_warnings: [null] } },
    { ...model(), validation_issues: [null] },
  ];
  for (const malformed of malformedModels) {
    const damaged = structuredClone(raw);
    damaged.versions[0].model = malformed as FloorPlanModel;
    assert.throws(() => parseDraft(JSON.stringify(damaged), "job-a", "1.0"), /does not match/);
  }
});

test("storage quota failure preserves the last recoverable version", () => {
  const draft = appendDraft(null, "job-a", model(), "automatic");
  const previousRaw = JSON.stringify(draft);
  const storage = {
    getItem: () => previousRaw,
    setItem: () => { throw new Error("QuotaExceededError"); },
  };
  const edited = model();
  edited.rooms[0].name = "New room";
  assert.throws(() => writeDraft(storage, appendDraft(draft, "job-a", edited, "automatic"), previousRaw), /QuotaExceededError/);
  assert.equal(parseDraft(storage.getItem(), "job-a", "1.0")!.versions[0].model.rooms[0].name, "Kitchen");
});

test("an older tab cannot overwrite a newer local draft", () => {
  const values = new Map<string, string>();
  const storage = { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => { values.set(key, value); } };
  const initial = appendDraft(null, "job-a", model(), "automatic");
  const initialRaw = writeDraft(storage, initial, null);
  const changed = model();
  changed.rooms[0].name = "Newer tab";
  const newer = appendDraft(initial, "job-a", changed, "automatic");
  const newerRaw = writeDraft(storage, newer, initialRaw);
  assert.throws(() => writeDraft(storage, initial, initialRaw), /Another tab/);
  assert.equal(storage.getItem(draftStorageKey("job-a")), newerRaw);
});
