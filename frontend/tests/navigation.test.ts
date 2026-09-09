import assert from "node:assert/strict";
import test from "node:test";
import {
  insideFurniture, isSafePlanPoint, isStructuralColliderName, safeRoomPoint,
  pointInPolygon, interestingViewTarget, openViewTarget, resolveFurniture,
} from "../src/navigation.ts";

const room = { points: [{ x: 0, y: 0 }, { x: 6, y: 0 }, { x: 6, y: 6 }, { x: 0, y: 6 }] };
const walls = room.points.map((start, i) => ({ start, end: room.points[(i + 1) % 4], thickness_m: 0.2, height_m: 3, external: true }));
const rug = { category: "living_room_rug", center: { x: 3, y: 3 }, width_m: 6, depth_m: 6 };

test("a full-room rug does not prevent a safe spawn or camera movement", () => {
  const spawn = safeRoomPoint(room, [rug], walls);
  assert.ok(spawn);
  assert.equal(isSafePlanPoint(spawn, [room], [rug], walls), true);
  assert.equal(insideFurniture(spawn, rug), false);
  assert.ok(openViewTarget(spawn, [room], [rug], walls));
});

test("solid asset placements remain obstacles even on a floor covering", () => {
  const cabinet = { category: "cabinet", center: { x: 3, y: 3 }, width_m: 1, depth_m: 2 };
  assert.equal(isSafePlanPoint(cabinet.center, [room], [rug, cabinet], walls), false);
  const spawn = safeRoomPoint(room, [rug, cabinet], walls);
  assert.ok(spawn);
  assert.equal(insideFurniture(spawn, cabinet), false);
});

test("rotated furniture clearance follows the rendered footprint", () => {
  const sofa = { category: "sofa", center: { x: 3, y: 3 }, width_m: 3, depth_m: 0.8, rotation_deg: 90 };
  assert.equal(insideFurniture({ x: 3, y: 4.2 }, sofa, 0), true);
  assert.equal(insideFurniture({ x: 4.2, y: 3 }, sofa, 0), false);
});

test("concave rooms spawn inside the floor polygon, away from the missing corner", () => {
  const concave = { points: [{ x: 0, y: 0 }, { x: 5, y: 0 }, { x: 5, y: 1.5 }, { x: 1.5, y: 1.5 }, { x: 1.5, y: 5 }, { x: 0, y: 5 }] };
  const spawn = safeRoomPoint(concave, [], []);
  assert.ok(spawn);
  assert.equal(pointInPolygon(spawn, concave.points), true);
  assert.equal(isSafePlanPoint(spawn, [concave], [], []), true);
});

test("an internal wall still blocks a point inside a valid room polygon", () => {
  const partition = { start: { x: 3, y: 1 }, end: { x: 3, y: 5 }, thickness_m: 0.2, height_m: 3, external: false };
  assert.equal(isSafePlanPoint({ x: 3.2, y: 3 }, [room], [], [...walls, partition]), false);
  const spawn = safeRoomPoint(room, [], [...walls, partition]);
  assert.ok(spawn);
  assert.ok(Math.abs(spawn.x - 3) > 0.4);
});

test("a blocked or sub-capsule room is never reported as safe", () => {
  const tiny = { points: [{ x: 0, y: 0 }, { x: 0.5, y: 0 }, { x: 0.5, y: 0.5 }, { x: 0, y: 0.5 }] };
  assert.equal(safeRoomPoint(tiny, [], []), null);
  assert.equal(safeRoomPoint(room, [{ ...rug, category: "wardrobe" }], walls), null);
});

test("initial camera interest ignores the rug and aims at furniture", () => {
  const sofa = { category: "sofa", center: { x: 3, y: 4 }, width_m: 1.5, depth_m: 0.8 };
  assert.deepEqual(interestingViewTarget({ x: 3, y: 1 }, [room], [rug, sofa], walls), sofa.center);
});

test("detailed sofa metadata preserves asset height and the rug without hiding the initial camera target", () => {
  const sofa = { category: "sofa", center: { x: 3, y: 4 }, width_m: 1.5, depth_m: 0.8, height_m: 0.7 };
  const detailedSofa = { ...sofa, category: "sofa_3_seat", height_m: 1.05 };
  const overlappingRug = { ...sofa, category: "rug", height_m: 0.02 };
  const origin = { x: 3, y: 1 };

  // Reproduce the failure: each copy of the sofa occludes the other copy.
  assert.equal(interestingViewTarget(origin, [room], [sofa, detailedSofa, overlappingRug], walls), null);

  const resolved = resolveFurniture([sofa], [detailedSofa, overlappingRug]);
  assert.equal(resolved.length, 2);
  assert.deepEqual(resolved[0], { ...sofa, height_m: 1.05 });
  assert.deepEqual(resolved[1], overlappingRug);
  assert.equal(insideFurniture(overlappingRug.center, resolved[1]), false);
  assert.deepEqual(interestingViewTarget(origin, [room], resolved, walls), sofa.center);
  assert.equal(sofa.height_m, 0.7, "resolving asset metadata must not mutate the editable source");
});

test("both baked and preview walls collide, but decorative wall caps do not", () => {
  for (const name of ["Wall_003", "Walls_Joined", "Walls_Joined_1", "DoorLeaf_001", "WindowGlass_002", "Special_001_lift_Rear"]) {
    assert.equal(isStructuralColliderName(name), true, name);
  }
  for (const name of ["Wall_Cap_003", "Ceilings_Joined", "Floor_001", "Furniture_001_sofa_Cushion"]) {
    assert.equal(isStructuralColliderName(name), false, name);
  }
});
