import assert from "node:assert/strict";
import test from "node:test";
import {
  acknowledgeIssue, acknowledgementState, collectReviewIssues, issueFingerprint, reviewTargetPoints,
} from "../src/editorWarnings.ts";
import type { FloorPlanModel } from "../src/types.ts";

function model(): FloorPlanModel {
  return {
    coordinate_system: "cartesian_metres", schema_version: "1.0", pixels_per_metre: 100,
    walls: [
      { id: "wall-a", start: { x: 0, y: 0 }, end: { x: 5, y: 0 }, thickness_m: 0.12, height_m: 2.8, external: true },
      { id: "wall-b", start: { x: 5, y: 0 }, end: { x: 5, y: 4 }, thickness_m: 0.12, height_m: 2.8, external: true },
    ],
    doors: [{ id: "door-a", wall_id: "wall-a", center: { x: 2, y: 0 }, width_m: 0.9, height_m: 2.1 }], windows: [],
    rooms: [{ id: "room-a", name: "Kitchen", points: [{ x: 0, y: 0 }, { x: 5, y: 0 }, { x: 5, y: 4 }, { x: 0, y: 4 }] }],
    metadata: { quality_components: { topology: 0.3 } },
    reconstruction: { quality_score: 0.3, quality_state: "review_required" },
    validation_issues: [{ code: "opening_outside_wall_span", severity: "error", message: "Opening exceeds wall.", element_id: "door-a" } as any],
  };
}

test("explicit issue targets resolve by identity and survive reordered validation lists", () => {
  const original = model();
  original.validation_issues!.push({ code: "missing_scale", severity: "severe", message: "Scale unknown" });
  const issues = collectReviewIssues(original);
  assert.deepEqual(issues[0].targets, [{ kind: "doors", index: 0, id: "door-a" }]);
  assert.equal(issues[1].targets.length, 0);
  assert.equal(issues[1].location, undefined);
  const reordered = collectReviewIssues(original, [...original.validation_issues!].reverse());
  assert.equal(issues[0].id, reordered[1].id);
  assert.equal(issues[1].id, reordered[0].id);
});

test("legacy gap and ambiguous-opening messages focus only verified referenced walls", () => {
  const original = model();
  const issues = collectReviewIssues(original, [
    { code: "unclosed_wall_gap", severity: "warning", message: "unclosed gap of 0.90 m between walls wall-a and wall-b" },
    { code: "ambiguous_opening", severity: "warning", message: "ambiguous on wall-a at 1.20-2.10 m (width heuristic only)" },
    { code: "unclosed_wall_gap", severity: "warning", message: "unclosed gap of 0.90 m between walls missing-a and missing-b" },
    { code: "unknown", severity: "warning", message: "Maybe somewhere around wall-a" },
  ]);
  assert.deepEqual(issues[0].targets.map((target) => target.id), ["wall-a", "wall-b"]);
  assert.equal(issues[1].targets[0].id, "wall-a");
  assert.deepEqual(issues[2].targets, []);
  assert.deepEqual(issues[3].targets, []);
});

test("semantic source locations are deduplicated and invalid locations never fabricate focus", () => {
  const original = model();
  original.metadata.sanity_warnings = [
    { kind: "door", description: "Check this opening", x: 0.2, y: 0.4, confidence: 0.6 },
    { kind: "layout", description: "Review overall layout", x: -1, y: 8, confidence: 0.2 },
  ];
  const issues = collectReviewIssues(original, [
    { code: "gemini_door", severity: "warning", message: "Check this opening (at 0.20, 0.40 normalized)" },
  ]);
  assert.equal(issues.length, 2);
  assert.deepEqual(issues[0].location, { x: 0.2, y: 0.4 });
  assert.equal(issues[1].location, undefined);
  assert.equal(issues[1].targets.length, 0);
});

test("acknowledgement round trips without altering warnings, quality, or export severity", () => {
  const original = model();
  const issue = collectReviewIssues(original)[0];
  const confirmed = acknowledgeIssue(original, issue, true, new Date("2026-10-03T12:00:00Z"));
  assert.equal(acknowledgementState(original, issue), "unreviewed");
  const reloaded = JSON.parse(JSON.stringify(confirmed));
  assert.equal(acknowledgementState(reloaded, collectReviewIssues(reloaded)[0]), "confirmed");
  assert.deepEqual(confirmed.validation_issues, original.validation_issues);
  assert.deepEqual(confirmed.reconstruction, original.reconstruction);
  assert.deepEqual(confirmed.metadata.quality_components, original.metadata.quality_components);
  assert.equal(confirmed.validation_issues![0].severity, "error");
  assert.equal(acknowledgementState(acknowledgeIssue(confirmed, issue, false), issue), "unreviewed");
});

test("opening confirmation invalidates for host geometry, width, sibling openings, and scale", () => {
  const original = model();
  const issue = collectReviewIssues(original)[0];
  const confirmed = acknowledgeIssue(original, issue, true);
  const edits: ((value: FloorPlanModel) => void)[] = [
    (value) => { value.walls[0].end.x += 1; },
    (value) => { value.doors[0].width_m += 0.1; },
    (value) => { value.windows.push({ id: "window-a", wall_id: "wall-a", center: { x: 3, y: 0 }, width_m: 1, height_m: 1 }); },
    (value) => { value.pixels_per_metre = 80; },
    (value) => { value.doors = []; },
  ];
  for (const edit of edits) {
    const changed = structuredClone(confirmed);
    edit(changed);
    assert.equal(acknowledgementState(changed, issue), "stale");
  }
  const irrelevant = structuredClone(confirmed);
  irrelevant.walls[1].end.y += 1;
  irrelevant.furniture = [{ category: "chair", center: { x: 3, y: 2 }, width_m: 0.5, depth_m: 0.5 }];
  assert.equal(acknowledgementState(irrelevant, issue), "confirmed");
});

test("room confirmations invalidate for enclosing geometry but ignore unrelated rendering metadata", () => {
  const original = model();
  const issue = collectReviewIssues(original, [{ code: "invalid_room_polygon", severity: "error", message: "Room invalid", element_id: "room-a" }])[0];
  const confirmed = acknowledgeIssue(original, issue, true);
  const changed = structuredClone(confirmed);
  changed.walls[1].end.y += 1;
  assert.equal(acknowledgementState(changed, issue), "stale");
  const rendered = structuredClone(confirmed);
  rendered.camera_waypoints = [{ position: { x: 2, y: 2 } }];
  rendered.metadata.material_preset = "oak";
  assert.equal(acknowledgementState(rendered, issue), "confirmed");
});

test("whole-plan findings cover all layout geometry and remain unlocated", () => {
  const original = model();
  const issue = collectReviewIssues(original, [{ code: "no_openings_detected", severity: "warning", message: "Check plan openings" }])[0];
  const confirmed = acknowledgeIssue(original, issue, true);
  const changed = structuredClone(confirmed);
  changed.walls.push({ ...changed.walls[0], id: "new-wall" });
  assert.equal(acknowledgementState(changed, issue), "stale");
  assert.deepEqual(issue.targets, []);
});

test("target geometry follows IDs after array reordering and missing objects produce no substitute", () => {
  const original = model();
  const issue = collectReviewIssues(original, [{ code: "wall_crossing_without_junction", severity: "warning", message: "walls wall-a and wall-b cross without a shared junction", element_id: "wall-a" }])[0];
  const fingerprint = issueFingerprint(original, issue);
  original.walls.reverse();
  assert.deepEqual(reviewTargetPoints(original, issue.targets[0]), [{ x: 0, y: 0 }, { x: 5, y: 0 }]);
  assert.equal(issueFingerprint(original, issue), fingerprint);
  original.walls = original.walls.filter((wall) => wall.id !== "wall-a");
  assert.deepEqual(reviewTargetPoints(original, issue.targets[0]), []);
});

test("source review follows authoritative opening spans on diagonal hosts despite stale centres", () => {
  const original = model();
  original.walls[0].start = { x: 1, y: 2 };
  original.walls[0].end = { x: 4, y: 6 };
  Object.assign(original.doors[0], { center: { x: 999, y: -999 }, width_m: 99, offset_m: 300, start_offset_m: 1, end_offset_m: 3 });
  const target = collectReviewIssues(original)[0].targets[0];
  const expected = [{ x: 1.6, y: 2.8 }, { x: 2.8, y: 4.4 }];
  assert.deepEqual(reviewTargetPoints(original, target), expected);
  original.walls.reverse();
  assert.deepEqual(reviewTargetPoints(original, target), expected);
  delete original.doors[0].start_offset_m;
  delete original.doors[0].end_offset_m;
  original.doors[0].offset_m = 2;
  original.doors[0].width_m = 2;
  assert.deepEqual(reviewTargetPoints(original, target), expected);
});

test("invalid anchored review locations never substitute a cached centre or another host", () => {
  const original = model();
  Object.assign(original.doors[0], { start_offset_m: 1, end_offset_m: 2 });
  const target = collectReviewIssues(original)[0].targets[0];
  for (const mutate of [
    (plan: FloorPlanModel) => { plan.doors[0].wall_id = "missing"; },
    (plan: FloorPlanModel) => { plan.walls.push({ ...plan.walls[0] }); },
    (plan: FloorPlanModel) => { plan.doors[0].end_offset_m = 6; },
    (plan: FloorPlanModel) => { plan.doors[0].start_offset_m = -1; },
    (plan: FloorPlanModel) => { delete plan.doors[0].end_offset_m; },
    (plan: FloorPlanModel) => { plan.walls[0].end = { ...plan.walls[0].start }; },
  ]) {
    const changed = structuredClone(original);
    mutate(changed);
    assert.deepEqual(reviewTargetPoints(changed, target), []);
  }
});

test("collected review references retain a missing half of a two-wall finding", () => {
  const original = model();
  original.walls = original.walls.slice(0, 1);
  const finding = collectReviewIssues(original, [{ code: "unclosed_wall_gap", severity: "warning", message: "unclosed gap of 0.90 m between walls wall-a and wall-b" }])[0];
  assert.deepEqual(finding.referencedIds, ["wall-a", "wall-b"]);
  assert.deepEqual(finding.targets.map((target) => target.id), ["wall-a"]);
});
