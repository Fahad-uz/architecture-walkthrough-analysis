import assert from "node:assert/strict";
import test from "node:test";
import { originalPreparation, preparedImageFrame, turnPreparation } from "../src/imagePreparation.ts";
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
  const before = { crop: { x: 20, y: 10, width: 80, height: 40 }, rotationQuarterTurns: 0 };
  assert.deepEqual(turnPreparation(before, -1), { crop: null, rotationQuarterTurns: 3 });
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
  await uploadPlan(file, false, "0.01", { x: 0, y: 30, width: 90, height: 90 }, 1);
  assert.equal(received?.get("rotation_quarter_turns"), "1");
  assert.equal(received?.get("crop_width"), "90");
  assert.equal(received?.get("crop_y"), "30");
  assert.equal(received?.get("manual_scale"), "0.01");
  assert.deepEqual(new Uint8Array(await (received?.get("file") as File).arrayBuffer()), new Uint8Array([1, 2, 3]));
  await uploadPlan(file, false, "");
  assert.equal(received?.has("rotation_quarter_turns"), false);
  assert.equal(received?.has("crop_x"), false);
});
