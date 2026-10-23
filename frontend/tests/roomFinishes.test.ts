import assert from "node:assert/strict";
import test from "node:test";
import { appendDraft, createHistory, isDraftModel, parseDraft, recordEdit, redoEdit, undoEdit } from "../src/editorHistory.ts";
import { changeRoomFinish, roomFinishValue } from "../src/roomFinishes.ts";
import { acknowledgeIssue, acknowledgementState, collectReviewIssues } from "../src/editorWarnings.ts";
import type { FloorPlanModel } from "../src/types.ts";

function model(): FloorPlanModel {
  return {
    coordinate_system: "metres", schema_version: "3.0", pixels_per_metre: 100,
    walls: [{ id: "wall-a", start: { x: 0, y: 0 }, end: { x: 5, y: 0 }, thickness_m: 0.2, height_m: 2.8, external: false }],
    rooms: [
      { id: "room-a", face_id: "face-a", name: "Study", points: [{ x: 0, y: 0 }, { x: 2, y: 0 }, { x: 0, y: 3 }], floor_material_preset: "custom_oak", confidence: 0.8 },
      { id: "room-b", name: "Hall", points: [{ x: 2, y: 0 }, { x: 5, y: 0 }, { x: 5, y: 3 }] },
    ],
    doors: [], windows: [], metadata: {},
    material_assignments: [{ target: "room-a", preset: "marble" }],
    validation_issues: [{ code: "room_review", severity: "warning", message: "Review room boundary" }],
  };
}

test("room finishes affect only the chosen room and preserve geometry and legacy assignments", () => {
  const original = model();
  const snapshot = structuredClone(original);
  const changed = changeRoomFinish(original, 0, "wall_material_preset", "painted_wall");
  assert.equal(changed.rooms[0].wall_material_preset, "painted_wall");
  assert.equal(changed.rooms[0].floor_material_preset, "custom_oak");
  assert.deepEqual(original, snapshot);
  assert.deepEqual(changed.rooms[1], original.rooms[1]);
  assert.deepEqual(changed.rooms[0].points, original.rooms[0].points);
  assert.deepEqual(changed.material_assignments, original.material_assignments);
  assert.deepEqual(changed.walls, original.walls);
  assert.equal(changed.rooms[0].face_id, "face-a");
  assert.equal(changed.rooms[0].confidence, 0.8);
  assert.equal(changeRoomFinish(original, 5, "floor_material_preset", "wood"), original);
  assert.equal(changeRoomFinish(original, 0, "floor_material_preset", "unlisted"), original);
  assert.equal(roomFinishValue(original.rooms[0], "floor_material_preset"), "custom_oak");
  assert.equal(roomFinishValue(original.rooms[1], "wall_material_preset"), "");
});

test("finish reset, undo/redo and recovered drafts retain separate room choices", () => {
  const original = model();
  const changed = changeRoomFinish(changeRoomFinish(original, 0, "floor_material_preset", "ceramic_tile"), 1, "wall_material_preset", "wood");
  const history = recordEdit(createHistory(original), changed);
  assert.deepEqual(undoEdit(history).present, original);
  assert.deepEqual(redoEdit(undoEdit(history)).present, changed);
  const draft = appendDraft(null, "job-a", changed, "manual");
  assert.deepEqual(parseDraft(JSON.stringify(draft), "job-a", "3.0")!.versions[0].model, changed);
  const reset = changeRoomFinish(changed, 0, "floor_material_preset", "");
  assert.equal(reset.rooms[0].floor_material_preset, null);
  assert.equal(reset.rooms[1].wall_material_preset, "wood");
  assert.deepEqual(reset.material_assignments, original.material_assignments);
});

test("finish-only edits retain structural review confirmations and quality", () => {
  const original = model();
  const issue = collectReviewIssues(original)[0];
  const acknowledged = acknowledgeIssue(original, issue, true);
  const changed = changeRoomFinish(acknowledged, 0, "wall_material_preset", "wood");
  assert.equal(acknowledgementState(changed, issue), "confirmed");
  assert.deepEqual(changed.validation_issues, original.validation_issues);
  assert.deepEqual(changed.reconstruction, original.reconstruction);
});

test("draft recovery accepts missing/null/custom finishes and rejects damaged values", () => {
  assert.equal(isDraftModel(model()), true);
  for (const value of [null, undefined, "custom_oak"]) {
    const candidate = model();
    candidate.rooms[0].floor_material_preset = value;
    assert.equal(isDraftModel(candidate), true);
  }
  for (const value of ["", " ", "x".repeat(129), 12, {}, ["wood"], false]) {
    const candidate = model();
    (candidate.rooms[0] as Record<string, unknown>).wall_material_preset = value;
    assert.equal(isDraftModel(candidate), false);
  }
});
