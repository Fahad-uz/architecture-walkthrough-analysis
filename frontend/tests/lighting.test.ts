import assert from "node:assert/strict";
import test from "node:test";
import { changeLighting, DEFAULT_LIGHTING, effectiveLighting, isLightingSettings, LIGHTING_TONES, lightingSettings } from "../src/lighting.ts";
import { appendDraft, createHistory, isDraftModel, parseDraft, recordEdit, redoEdit, undoEdit } from "../src/editorHistory.ts";
import type { FloorPlanModel } from "../src/types.ts";

function model(): FloorPlanModel {
  return {
    coordinate_system: "cartesian_metres", schema_version: "3.0", pixels_per_metre: 100,
    walls: [{ id: "wall-a", start: { x: 0, y: 0 }, end: { x: 5, y: 0 }, thickness_m: 0.2, height_m: 2.8, external: false }],
    rooms: [], doors: [], windows: [], metadata: {},
  };
}

test("absent and legacy lighting retain the existing neutral appearance", () => {
  assert.deepEqual(effectiveLighting(), { powerFactor: 1, colorScale: [1, 1, 1], exposureFactor: 1 });
  assert.deepEqual(effectiveLighting({ preset: "neutral_studio", intensity: 450, exposure: 0, color_temperature: 4500 }), effectiveLighting());
  const resolved = lightingSettings();
  resolved.brightness = 2;
  assert.equal(DEFAULT_LIGHTING.brightness, 1);
});

test("browser tones match the exporter color math across supported Kelvin values", () => {
  // Expected values from the shared Python exporter helper, including limits.
  const samples = [
    [1000, [1, 0.3122315571559754, 0]],
    [3000, [1, 0.8145932472432359, 0.586431873954059]],
    [4500, [1, 1, 1]],
    [6500, [0.7496101581433231, 0.8756565506122055, 1]],
    [12000, [0.5508419015602293, 0.7146351581287834, 1]],
  ] as const;
  for (const [temperature, expected] of samples) {
    const actual = effectiveLighting({ color_temperature: temperature }).colorScale;
    actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < 1e-12));
  }
  assert.deepEqual(LIGHTING_TONES.map((tone) => tone.temperature), [3000, 4500, 6500]);
});

test("power applies brightness and legacy intensity once, with exposure separate", () => {
  const lighting = effectiveLighting({ brightness: 1.5, intensity: 900, exposure: 2 });
  assert.equal(lighting.powerFactor, 3);
  assert.equal(lighting.exposureFactor, 4);
  assert.equal(effectiveLighting({ brightness: 2, intensity: 0, exposure: 5 }).powerFactor, 0);
  assert.equal(effectiveLighting({ exposure: -5 }).exposureFactor, 1 / 32);
});

test("ordinary lighting edits preserve saved custom settings until reset", () => {
  const original = { preset: "custom_studio", color_temperature: 5100, intensity: 765, exposure: -0.5 };
  const brighter = changeLighting(original, { brightness: 1.4 });
  assert.deepEqual(brighter, { ...original, brightness: 1.4 });
  const warm = changeLighting(brighter, { color_temperature: 3000 });
  assert.deepEqual(warm, { ...original, brightness: 1.4, color_temperature: 3000 });
  assert.equal(original.color_temperature, 5100);
  assert.deepEqual(lightingSettings(), DEFAULT_LIGHTING);
});

test("lighting edits and reset survive history and local draft recovery without changing geometry", () => {
  const original = model();
  const changed = { ...original, lighting: changeLighting(undefined, { brightness: 1.5, color_temperature: 3000 }) };
  const history = recordEdit(createHistory(original), changed);
  assert.deepEqual(undoEdit(history).present, original);
  assert.deepEqual(redoEdit(undoEdit(history)).present, changed);
  const draft = appendDraft(null, "job-a", changed, "manual");
  assert.deepEqual(parseDraft(JSON.stringify(draft), "job-a", "3.0")!.versions[0].model, changed);
  const reset = recordEdit(history, { ...changed, lighting: { ...DEFAULT_LIGHTING } });
  assert.deepEqual(undoEdit(reset).present, changed);
  assert.deepEqual(reset.present.walls, original.walls);
  assert.deepEqual(reset.present.rooms, original.rooms);
  assert.deepEqual(reset.present.metadata, original.metadata);
});

test("draft validation permits absent and partial legacy settings and rejects corrupt lighting", () => {
  assert.equal(isDraftModel(model()), true);
  assert.equal(isLightingSettings({ intensity: 900, exposure: -1, color_temperature: 5100 }), true);
  assert.equal(isLightingSettings({ brightness: 0.25, intensity: 0, exposure: -5, color_temperature: 1000 }), true);
  assert.equal(isLightingSettings({ brightness: 2, intensity: 4500, exposure: 5, color_temperature: 12000 }), true);
  for (const invalid of [
    null, [], "warm", { preset: "" }, { preset: "x".repeat(65) }, { preset: 1 }, { brightness: 0 },
    { brightness: 2.01 }, { brightness: "1" }, { brightness: true }, { brightness: NaN },
    { brightness: Infinity }, { color_temperature: 4500.5 }, { color_temperature: 999 },
    { color_temperature: 12001 }, { intensity: -1 }, { intensity: 4501 }, { exposure: -6 },
    { exposure: 6 }, { tone: "warm" }, { toString: "unexpected" },
  ]) {
    assert.equal(isLightingSettings(invalid), false, JSON.stringify(invalid));
    assert.equal(isDraftModel({ ...model(), lighting: invalid }), false, JSON.stringify(invalid));
  }
  const draft = appendDraft(null, "job-a", model(), "manual");
  const corrupt = { ...draft, versions: [{ ...draft.versions[0], model: { ...model(), lighting: { brightness: 8 } } }] };
  assert.throws(() => parseDraft(JSON.stringify(corrupt), "job-a", "3.0"), /does not match/);
});
