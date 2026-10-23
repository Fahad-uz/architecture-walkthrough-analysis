import assert from "node:assert/strict";
import test from "node:test";
import { collectReviewIssues, type ReviewIssue } from "../src/editorWarnings.ts";
import { findingPath, resolveReview3DFocus, review3DLinkState, reviewIssue3DFocus, reviewWorldPoint, type Review3DFocus, type ReviewWorldPoint } from "../src/review3D.ts";
import type { FloorPlanModel } from "../src/types.ts";

function model(): FloorPlanModel {
  return {
    coordinate_system: "metres", schema_version: "3.0", pixels_per_metre: 100,
    walls: [
      { id: "wall-a", start: { x: 1, y: 2 }, end: { x: 5, y: 2 }, thickness_m: 0.2, height_m: 3, external: true },
      { id: "wall-b", start: { x: 5, y: 2 }, end: { x: 5, y: 6 }, thickness_m: 0.15, height_m: 2.8, external: false },
    ],
    doors: [{ id: "door-a", wall_id: "wall-a", center: { x: 999, y: -999 }, width_m: 99, height_m: 2.1, start_offset_m: 1, end_offset_m: 2 }],
    windows: [{ id: "window-a", wall_id: "wall-a", center: { x: 999, y: -999 }, width_m: 1, height_m: 1.2, sill_height_m: 0.9, offset_m: 3 }],
    rooms: [{ id: "room-a", points: [{ x: 1, y: 2 }, { x: 5, y: 2 }, { x: 5, y: 6 }, { x: 1, y: 6 }] }],
    metadata: {}, validation_issues: [],
  };
}

function issue(plan: FloorPlanModel, id: string): ReviewIssue {
  return collectReviewIssues(plan, [{ code: "object_review", severity: "warning", message: "Review saved object", element_id: id }])[0];
}

function addIssue(plan: FloorPlanModel, id: string): ReviewIssue {
  plan.validation_issues = [{ code: "object_review", severity: "warning", message: "Review saved object", element_id: id } as any];
  return collectReviewIssues(plan)[0];
}

function closePoint(actual: ReviewWorldPoint, expected: ReviewWorldPoint) {
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < 1e-10, `${actual} != ${expected}`));
}

function noMarker(focus: Review3DFocus) {
  assert.deepEqual(focus.shapes, []);
  assert.equal(focus.bounds, undefined);
  assert.ok(focus.reason);
}

test("plan coordinates map to glTF Y-up with negative plan Y on world Z", () => {
  assert.deepEqual(reviewWorldPoint({ x: 4, y: 7 }, 2.1), [4, 2.1, -7]);
  assert.deepEqual(reviewWorldPoint({ x: -3, y: -2 }), [-3, 0, 2]);
});

test("wall markers follow saved thickness and height, resolving IDs after reordering", () => {
  const plan = model();
  const finding = addIssue(plan, "wall-a");
  const original = structuredClone(plan);
  const focus = resolveReview3DFocus(plan, finding.id);
  closePoint(focus.bounds!.min, [1, 0, -2.1]);
  closePoint(focus.bounds!.max, [5, 3, -1.9]);
  assert.equal(focus.shapes[0].paths.length, 6);
  plan.walls.reverse();
  plan.validation_issues!.unshift({ code: "missing_scale", severity: "warning", message: "Check scale" });
  assert.deepEqual(resolveReview3DFocus(plan, finding.id).bounds, focus.bounds);
  assert.deepEqual(reviewIssue3DFocus(plan, finding).bounds, focus.bounds);
  plan.walls.reverse();
  plan.validation_issues!.shift();
  assert.deepEqual(plan, original, "creating markers must not mutate the saved plan");
});

test("opening bounds use the authoritative interval even when center, width and midpoint disagree", () => {
  const plan = model();
  plan.doors[0].offset_m = 300;
  const finding = issue(plan, "door-a");
  const focus = reviewIssue3DFocus(plan, finding);
  closePoint(focus.bounds!.min, [2, 0, -2.1]);
  closePoint(focus.bounds!.max, [3, 2.1, -1.9]);
  plan.doors[0].center = { x: NaN, y: Infinity };
  assert.deepEqual(reviewIssue3DFocus(plan, finding).bounds, focus.bounds, "the cached center is not the aperture's position");
});

test("legacy midpoint and width locate a window and preserve its sill elevation", () => {
  const plan = model();
  const focus = reviewIssue3DFocus(plan, issue(plan, "window-a"));
  closePoint(focus.bounds!.min, [3.5, 0.9, -2.1]);
  closePoint(focus.bounds!.max, [4.5, 2.1, -1.9]);
});

test("diagonal host walls preserve aperture orientation and metre offsets", () => {
  const plan = model();
  plan.walls[0].end = { x: 4, y: 6 };
  plan.doors[0].start_offset_m = 1;
  plan.doors[0].end_offset_m = 3;
  const wallFocus = reviewIssue3DFocus(plan, issue(plan, "wall-a"));
  closePoint(wallFocus.bounds!.min, [0.92, 0, -6.06]);
  closePoint(wallFocus.bounds!.max, [4.08, 3, -1.94]);
  const doorFocus = reviewIssue3DFocus(plan, issue(plan, "door-a"));
  closePoint(doorFocus.bounds!.min, [1.52, 0, -4.46]);
  closePoint(doorFocus.bounds!.max, [2.88, 2.1, -2.74]);
});

test("rooms and balconies mark floor outlines while slab thickness extends below ground", () => {
  const plan = model();
  plan.balconies = [{ ...plan.rooms[0], id: "balcony-a" }];
  plan.slabs = [{ ...plan.rooms[0], id: "slab-a", thickness_m: 0.22 }];
  for (const id of ["room-a", "balcony-a"]) {
    const focus = reviewIssue3DFocus(plan, issue(plan, id));
    closePoint(focus.bounds!.min, [1, 0, -6]);
    closePoint(focus.bounds!.max, [5, 0, -2]);
    assert.equal(focus.shapes[0].paths.length, 1);
    assert.deepEqual(focus.shapes[0].paths[0][0], focus.shapes[0].paths[0].at(-1));
  }
  const slab = reviewIssue3DFocus(plan, issue(plan, "slab-a"));
  closePoint(slab.bounds!.min, [1, -0.22, -6]);
  closePoint(slab.bounds!.max, [5, 0, -2]);
});

test("legacy two-wall findings frame only fully resolved references and never substitute another wall", () => {
  const plan = model();
  plan.validation_issues = [{ code: "unclosed_wall_gap", severity: "warning", message: "unclosed gap of 0.90 m between walls wall-a and wall-b" }];
  const finding = collectReviewIssues(plan)[0];
  const focus = resolveReview3DFocus(plan, finding.id);
  assert.equal(focus.shapes.length, 2);
  assert.deepEqual(new Set(focus.shapes.map((shape) => shape.target.id)), new Set(["wall-a", "wall-b"]));
  plan.walls = plan.walls.filter((wall) => wall.id !== "wall-b");
  noMarker(reviewIssue3DFocus(plan, finding));
  noMarker(resolveReview3DFocus(plan, finding.id));
});

test("stale, whole-plan and image-only findings never manufacture a 3D location", () => {
  const plan = model();
  const finding = addIssue(plan, "wall-a");
  plan.validation_issues = [];
  noMarker(resolveReview3DFocus(plan, finding.id));
  noMarker(resolveReview3DFocus(plan, "untrusted-or-obsolete-id"));
  noMarker(reviewIssue3DFocus(plan, collectReviewIssues(plan, [{ code: "missing_scale", severity: "warning", message: "Scale unknown" }])[0]));
  plan.metadata.sanity_warnings = [{ kind: "door", description: "Check the source opening", x: 0.25, y: 0.6, confidence: 0.4 }];
  const sourceOnly = collectReviewIssues(plan)[0];
  assert.deepEqual(sourceOnly.location, { x: 0.25, y: 0.6 });
  const focus = resolveReview3DFocus(plan, sourceOnly.id);
  noMarker(focus);
  assert.match(focus.reason!, /source image only/);
});

test("an opening needs a unique host and a complete, finite interval inside that wall", () => {
  const invalidEdits: ((plan: FloorPlanModel) => void)[] = [
    (plan) => { plan.doors[0].wall_id = null; },
    (plan) => { plan.doors[0].wall_id = "missing-wall"; },
    (plan) => { plan.walls.push({ ...plan.walls[0] }); },
    (plan) => { delete plan.doors[0].start_offset_m; },
    (plan) => { plan.doors[0].start_offset_m = -0.1; },
    (plan) => { plan.doors[0].end_offset_m = 4.1; },
    (plan) => { plan.doors[0].end_offset_m = 1; },
    (plan) => { plan.doors[0].end_offset_m = NaN; },
    (plan) => { delete plan.doors[0].start_offset_m; delete plan.doors[0].end_offset_m; },
    (plan) => { plan.walls[0].end = { ...plan.walls[0].start }; },
    (plan) => { plan.walls[0].thickness_m = 0; },
  ];
  for (const edit of invalidEdits) {
    const plan = model();
    const finding = issue(plan, "door-a");
    edit(plan);
    noMarker(reviewIssue3DFocus(plan, finding));
  }
  const plan = model();
  plan.windows[0].sill_height_m = 4;
  noMarker(reviewIssue3DFocus(plan, issue(plan, "window-a")));
});

test("malformed, ambiguous, unsupported and pixel-space geometry produces no marker", () => {
  const plan = model();
  const finding = issue(plan, "wall-a");
  plan.coordinate_system = "pixels";
  noMarker(reviewIssue3DFocus(plan, finding));
  plan.coordinate_system = "metres";
  plan.walls[0].start.x = Infinity;
  noMarker(reviewIssue3DFocus(plan, finding));
  const duplicate = model();
  duplicate.rooms[0].id = "wall-a";
  noMarker(reviewIssue3DFocus(duplicate, issue(duplicate, "wall-a")));
  const invalidRoom = model();
  invalidRoom.rooms[0].points[1].y = NaN;
  noMarker(reviewIssue3DFocus(invalidRoom, issue(invalidRoom, "room-a")));
  const unsupported = model();
  unsupported.special_elements = [{ id: "stairs-a", kind: "stairs", center: { x: 2, y: 3 } }];
  noMarker(reviewIssue3DFocus(unsupported, issue(unsupported, "stairs-a")));
});

test("reference heights are finite and bounded while horizontal geometry stays intact", () => {
  const plan = model();
  plan.walls[0].height_m = 300;
  const focus = reviewIssue3DFocus(plan, issue(plan, "wall-a"));
  assert.equal(focus.bounds!.max[1], 20);
  plan.walls[0].height_m = NaN;
  assert.equal(reviewIssue3DFocus(plan, issue(plan, "wall-a")).bounds!.max[1], 2.8);
  plan.walls[1].height_m = NaN;
  assert.equal(reviewIssue3DFocus(plan, issue(plan, "wall-a")).bounds!.max[1], 3);
});

test("3D links require verified geometry, a completed save and an available saved model", () => {
  const plan = model();
  const focus = reviewIssue3DFocus(plan, issue(plan, "wall-a"));
  const ready = { dirty: false, saving: false, modelAvailable: true };
  assert.deepEqual(review3DLinkState(focus, ready), { enabled: true });
  for (const state of [{ ...ready, dirty: true }, { ...ready, saving: true }, { ...ready, dirty: true, saving: true }]) {
    const gate = review3DLinkState(focus, state);
    assert.equal(gate.enabled, false);
    assert.match(gate.reason!, /Save changes first/);
  }
  const noModel = review3DLinkState(focus, { ...ready, modelAvailable: false });
  assert.equal(noModel.enabled, false);
  assert.match(noModel.reason!, /unavailable/);
  assert.equal(review3DLinkState(resolveReview3DFocus(plan, "gone"), ready).enabled, false);
  assert.equal(review3DLinkState(null, ready).enabled, false);
});

test("finding links preserve complete stable IDs and encode route and query text safely", () => {
  const plan = model();
  const finding = addIssue(plan, "wall-a");
  const url = new URL(findingPath("job / one", "preview", finding.id), "https://example.test");
  assert.equal(url.pathname, "/jobs/job%20%2F%20one/preview");
  assert.equal(url.searchParams.get("finding"), finding.id);
  assert.ok(resolveReview3DFocus(plan, url.searchParams.get("finding")!).bounds);
  assert.equal(new URL(findingPath("job-a", "edit", finding.id), url).searchParams.get("finding"), finding.id);
});
