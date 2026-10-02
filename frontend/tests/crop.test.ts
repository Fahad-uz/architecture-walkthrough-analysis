import assert from "node:assert/strict";
import test from "node:test";
import { cropBetween } from "../src/crop.ts";

test("reverse drag retains identical crop and encloses fractional pixels", () => {
  const a = { x: 510.2, y: 601.6 }, b = { x: 20.8, y: 30.9 };
  const expected = { x: 20, y: 30, width: 491, height: 572 };
  assert.deepEqual(cropBetween(a, b, 1000, 1200), expected);
  assert.deepEqual(cropBetween(b, a, 1000, 1200), expected);
});

test("pointer capture outside image clips crop to original pixels", () => {
  assert.deepEqual(cropBetween({ x: -20, y: -30 }, { x: 2000, y: 2000 }, 1000, 1200),
    { x: 0, y: 0, width: 1000, height: 1200 });
  assert.equal(cropBetween({ x: 1200, y: 20 }, { x: 1500, y: 90 }, 1000, 1200), null);
});

test("empty or non-finite crop cannot be submitted", () => {
  assert.equal(cropBetween({ x: 30, y: 40 }, { x: 30, y: 40 }, 100, 100), null);
  assert.equal(cropBetween({ x: NaN, y: 40 }, { x: 70, y: 90 }, 100, 100), null);
});
