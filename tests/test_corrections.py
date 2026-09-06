from __future__ import annotations

from pathlib import Path

import pytest

from architecture_walkthrough.geometry.floorplan import load_corrected_floorplan


def test_load_corrected_floorplan_accepts_fixture() -> None:
    model = load_corrected_floorplan(Path("tests/fixtures/sample_floorplan.json"))
    assert len(model.walls) == 4
    assert model.camera_waypoints


def test_load_corrected_floorplan_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"coordinate_system": "metres", "unknown": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid correction"):
        load_corrected_floorplan(path)
