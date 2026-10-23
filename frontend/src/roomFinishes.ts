import type { FloorPlanModel, RoomPolygon } from "./types";

export type RoomFinishField = "floor_material_preset" | "wall_material_preset";

/** These names are existing local material registry keys, not downloaded assets. */
export const FLOOR_FINISHES = [
  { value: "wood", label: "Wood" },
  { value: "marble", label: "Light stone" },
  { value: "ceramic_tile", label: "Grey tile" },
] as const;
export const WALL_FINISHES = [
  { value: "painted_wall", label: "Warm plaster" },
  { value: "wood", label: "Wood" },
  { value: "ceramic_tile", label: "Grey tile" },
] as const;

export function roomFinishValue(room: RoomPolygon, field: RoomFinishField): string {
  return typeof room[field] === "string" ? room[field] : "";
}

/** Unknown existing registry keys may be displayed and preserved, but the UI
 * only creates one of the supported choices or clears an explicit override. */
export function changeRoomFinish(model: FloorPlanModel, index: number, field: RoomFinishField, value: string): FloorPlanModel {
  const choices = field === "floor_material_preset" ? FLOOR_FINISHES : WALL_FINISHES;
  if (!Number.isInteger(index) || !model.rooms[index] || (value !== "" && !choices.some((item) => item.value === value))) return model;
  return { ...model, rooms: model.rooms.map((room, roomIndex) => roomIndex === index
    ? { ...room, [field]: value || null } : room) };
}

/** New optional fields also occur on legacy-derived slab/balcony polygons. */
export function validRoomFinishes(value: Record<string, unknown>): boolean {
  return (["floor_material_preset", "wall_material_preset"] as const).every((field) =>
    value[field] == null || (typeof value[field] === "string" && value[field].trim().length > 0 && value[field].length <= 128));
}
