import type { LightingSettings } from "./types";

export const DEFAULT_LIGHTING: Readonly<LightingSettings> = Object.freeze({
  preset: "neutral_studio", color_temperature: 4500, brightness: 1, intensity: 450, exposure: 0,
});

export const LIGHTING_TONES = [
  { value: "warm", label: "Warm", temperature: 3000 },
  { value: "neutral", label: "Neutral", temperature: 4500 },
  { value: "cool", label: "Cool", temperature: 6500 },
] as const;

/** Older saved plans may omit the new settings or individual default fields. */
export function lightingSettings(settings?: Partial<LightingSettings>): LightingSettings {
  return { ...DEFAULT_LIGHTING, ...settings };
}

/** Editing the ordinary controls retains any saved fine adjustments. */
export function changeLighting(settings: Partial<LightingSettings> | undefined, change: Partial<Pick<LightingSettings, "brightness" | "color_temperature">>): LightingSettings {
  return { ...lightingSettings(settings), ...change };
}

export function isLightingSettings(value: unknown): boolean {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const settings = value as Record<string, unknown>;
  if (Object.keys(settings).some((key) => !Object.prototype.hasOwnProperty.call(DEFAULT_LIGHTING, key))) return false;
  if (settings.preset !== undefined && (typeof settings.preset !== "string" || settings.preset.length < 1 || settings.preset.length > 64)) return false;
  const inRange = (key: string, min: number, max: number) => settings[key] === undefined ||
    (typeof settings[key] === "number" && Number.isFinite(settings[key]) && settings[key] >= min && settings[key] <= max);
  return inRange("brightness", 0.25, 2) && inRange("intensity", 0, 4500) && inRange("exposure", -5, 5) &&
    inRange("color_temperature", 1000, 12000) &&
    (settings.color_temperature === undefined || Number.isInteger(settings.color_temperature));
}

export type LightColorScale = [number, number, number];

/** Keep this approximation in sync with scene/lighting_values.py. */
function kelvinRgb(temperature: number): LightColorScale {
  const scaled = temperature / 100;
  const values = scaled <= 66 ? [
    255,
    99.4708025861 * Math.log(scaled) - 161.1195681661,
    scaled <= 19 ? 0 : 138.5177312231 * Math.log(scaled - 10) - 305.0447927307,
  ] : [
    329.698727446 * (scaled - 60) ** -0.1332047592,
    288.1221695283 * (scaled - 60) ** -0.0755148492,
    255,
  ];
  return values.map((value) => Math.max(0, Math.min(255, value)) / 255) as LightColorScale;
}

/** Imported punctual lights already contain the exported power and color.
 * These values control only browser fill lights, reflections, and exposure. */
export function effectiveLighting(saved?: Partial<LightingSettings>) {
  const settings = lightingSettings(saved);
  const neutral = kelvinRgb(DEFAULT_LIGHTING.color_temperature);
  const ratios = kelvinRgb(settings.color_temperature).map((value, index) => value / neutral[index]);
  const peak = Math.max(...ratios);
  return {
    powerFactor: settings.brightness * (settings.intensity / DEFAULT_LIGHTING.intensity),
    colorScale: ratios.map((value) => value / peak) as LightColorScale,
    exposureFactor: 2 ** settings.exposure,
  };
}
