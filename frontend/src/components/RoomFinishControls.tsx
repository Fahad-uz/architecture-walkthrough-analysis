import { FLOOR_FINISHES, WALL_FINISHES, roomFinishValue, type RoomFinishField } from "../roomFinishes";
import type { RoomPolygon } from "../types";

export default function RoomFinishControls({ room, disabled, onChange }: {
  room: RoomPolygon;
  disabled: boolean;
  onChange: (field: RoomFinishField, value: string) => void;
}) {
  return <>
    <strong style={{ fontSize: 13 }}>Room finishes</strong>
    {([
      ["floor_material_preset", "Floor finish", FLOOR_FINISHES],
      ["wall_material_preset", "Wall finish", WALL_FINISHES],
    ] as const).map(([field, label, choices]) => {
      const value = roomFinishValue(room, field);
      const custom = value && !choices.some((choice) => choice.value === value);
      return <label key={field}>
        {label}
        <select value={value} disabled={disabled} onChange={(event) => onChange(field, event.target.value)}>
          <option value="">Automatic (saved defaults)</option>
          {custom && <option value={value}>Custom ({value})</option>}
          {choices.map((choice) => <option key={choice.value} value={choice.value}>{choice.label}</option>)}
        </select>
      </label>;
    })}
    <p style={{ fontSize: 12, color: "#667078" }}>
      Finishes are your design choices. Save changes to update 3D. Wall finishes apply to the room's inside faces;
      automatic restores the saved material rules.
    </p>
  </>;
}
