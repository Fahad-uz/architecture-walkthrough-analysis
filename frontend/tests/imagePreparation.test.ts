import assert from "node:assert/strict";
import test from "node:test";
import { originalPreparation, preparedImageFrame, straightenPreparation, turnPreparation } from "../src/imagePreparation.ts";
import { cropBetween } from "../src/crop.ts";
import { uploadPlan } from "../src/api.ts";

test("quarter turns keep asymmetric pixel centres inside the correct prepared frame", () => {
  const expected = [[10.5, 20.5], [69.5, 10.5], [109.5, 69.5], [20.5, 109.5]];
  for (let turn = 0; turn < 4; turn++) {
    const frame = preparedImageFrame(120, 90, turn);
    const [a, b, c, d, e, f] = frame.matrix;
    assert.deepEqual([a * 10.5 + c * 20.5 + e, b * 10.5 + d * 20.5 + f], expected[turn]);
    assert.deepEqual([frame.width, frame.height], turn % 2 ? [90, 120] : [120, 90]);
    assert.deepEqual(cropBetween({ x: 0, y: 0 }, { x: 999, y: 999 }, frame.width, frame.height),
      { x: 0, y: 0, width: frame.width, height: frame.height });
  }
  assert.throws(() => preparedImageFrame(120, 90, 4));
  assert.throws(() => preparedImageFrame(120, 90, 0.5));
});

test("rotation clears a previous crop without mutating the snapshot needed by undo", () => {
  const before = { crop: { x: 20, y: 10, width: 80, height: 40 }, rotationQuarterTurns: 0, straightenDegrees: 0 };
  assert.deepEqual(turnPreparation(before, -1), { crop: null, rotationQuarterTurns: 3, straightenDegrees: 0 });
  assert.deepEqual(before.crop, { x: 20, y: 10, width: 80, height: 40 });
  let state = originalPreparation();
  for (let i = 0; i < 4; i++) state = turnPreparation(state, 1);
  assert.deepEqual(state, originalPreparation());
});

test("upload sends rotation with prepared-frame crop and preserves original file bytes", async (t) => {
  const file = new File([new Uint8Array([1, 2, 3])], "original.png", { type: "image/png" });
  let received: FormData | undefined;
  t.mock.method(globalThis, "fetch", async (_url: string, init: RequestInit) => {
    received = init.body as FormData;
    return new Response(JSON.stringify({ job_id: "test" }), { headers: { "Content-Type": "application/json" } });
  });
  await uploadPlan(file, false, "0.01", { x: 0, y: 30, width: 90, height: 90 }, 1, 2.5);
  assert.equal(received?.get("rotation_quarter_turns"), "1");
  assert.equal(received?.get("straighten_degrees"), "2.5");
  assert.equal(received?.get("crop_width"), "90");
  assert.equal(received?.get("crop_y"), "30");
  assert.equal(received?.get("manual_scale"), "0.01");
  assert.deepEqual(new Uint8Array(await (received?.get("file") as File).arrayBuffer()), new Uint8Array([1, 2, 3]));
  await uploadPlan(file, false, "");
  assert.equal(received?.has("rotation_quarter_turns"), false);
  assert.equal(received?.has("straighten_degrees"), false);
  assert.equal(received?.has("crop_x"), false);
});

test("fine rotation preserves the source centre and keeps every edge inside the expanded canvas", () => {
  for (const angle of [-15, -2.5, 0.1, 5, 15]) {
    for (let turn = 0; turn < 4; turn++) {
      const frame = preparedImageFrame(123, 91, turn, angle);
      const [a, b, c, d, e, f] = frame.matrix;
      const transform = (x: number, y: number) => [a * x + c * y + e, b * x + d * y + f];
      const centre = transform(61.5, 45.5);
      assert.ok(Math.abs(centre[0] - frame.width / 2) < 1e-10);
      assert.ok(Math.abs(centre[1] - frame.height / 2) < 1e-10);
      assert.ok(Math.abs(a * d - b * c - 1) < 1e-10);
      for (const [x, y] of [[0, 0], [123, 0], [0, 91], [123, 91]]) {
        const [px, py] = transform(x, y);
        assert.ok(px >= -1e-9 && py >= -1e-9 && px <= frame.width + 1e-9 && py <= frame.height + 1e-9);
      }
    }
  }
  assert.deepEqual(preparedImageFrame(123, 91, 1, 0), preparedImageFrame(123, 91, 1));
});

test("straightening snapshots preserve quarter turns and clear a crop; invalid angles are rejected", () => {
  const before = { crop: { x: 10, y: 20, width: 50, height: 60 }, rotationQuarterTurns: 3, straightenDegrees: 0 };
  assert.equal(straightenPreparation(before, 0), before);
  const after = straightenPreparation(before, -2.5);
  assert.deepEqual(after, { crop: null, rotationQuarterTurns: 3, straightenDegrees: -2.5 });
  assert.equal(turnPreparation(after, 1).straightenDegrees, -2.5);
  assert.ok(before.crop);
  for (const angle of [NaN, Infinity, -Infinity, 15.01, -15.01]) {
    assert.throws(() => straightenPreparation(before, angle));
    assert.throws(() => preparedImageFrame(123, 91, 0, angle));
  }
});
