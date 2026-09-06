from __future__ import annotations

import numpy as np

from architecture_walkthrough.geometry.models import Point2D, WallSegment
from architecture_walkthrough.vision.semantic_filter import (
    SemanticExclusion,
    filter_walls_crossing_repetitive_details,
    filter_walls_inside_semantic_objects,
    filter_walls_near_colored_objects,
    filter_walls_near_text_regions,
    semantic_exclusions_from_local_footprints,
)
from architecture_walkthrough.vision.furniture_detection import LocalObjectFootprint
from architecture_walkthrough.vision.ocr import OCRText
from architecture_walkthrough.vision.wall_detection import RepetitiveDetailRegion


def test_furniture_outline_is_removed_but_crossing_structural_wall_survives() -> None:
    exclusion = SemanticExclusion(
        category="kitchen counter",
        center=Point2D(x=50, y=50),
        width_px=30,
        depth_px=20,
        rotation_deg=0,
    )
    furniture_edge = WallSegment(
        start=Point2D(x=36, y=42),
        end=Point2D(x=64, y=42),
        evidence_source="wall_band",
    )
    building_wall = WallSegment(
        start=Point2D(x=0, y=50),
        end=Point2D(x=100, y=50),
        evidence_source="wall_band",
    )

    result = filter_walls_inside_semantic_objects(
        [furniture_edge, building_wall],
        [exclusion],
    )

    assert result.walls == [building_wall]
    assert result.rejected == [furniture_edge]


def test_external_grounded_wall_is_preserved_at_balcony_edge() -> None:
    exclusion = SemanticExclusion(
        category="balcony",
        center=Point2D(x=50, y=10),
        width_px=100,
        depth_px=20,
        rotation_deg=0,
    )
    wall = WallSegment(
        start=Point2D(x=0, y=10),
        end=Point2D(x=100, y=10),
        external=True,
        evidence_source="ai_proposal+local_structural_ink",
    )

    result = filter_walls_inside_semantic_objects(
        [wall],
        [exclusion],
        preserve_grounded_external=True,
    )

    assert result.walls == [wall]


def test_local_color_footprint_rejects_furniture_edge_but_keeps_envelope() -> None:
    mask = np.zeros((120, 180), dtype=np.uint8)
    mask[45:76, 50:131] = 255
    furniture_edge = WallSegment(
        start=Point2D(x=45, y=42),
        end=Point2D(x=135, y=42),
        evidence_source="wall_band",
    )
    crossing_wall = WallSegment(
        start=Point2D(x=10, y=60),
        end=Point2D(x=170, y=60),
        evidence_source="wall_band",
    )
    envelope = WallSegment(
        start=Point2D(x=175, y=0),
        end=Point2D(x=175, y=119),
        external=True,
        evidence_source="wall_band",
    )

    result = filter_walls_near_colored_objects(
        [furniture_edge, crossing_wall, envelope],
        mask,
        dilation_ratio=0.025,
        coverage_threshold=0.70,
    )

    assert result.rejected == [furniture_edge]
    assert result.walls == [crossing_wall, envelope]


def test_grounded_wall_through_stair_treads_is_rejected() -> None:
    region = RepetitiveDetailRegion(
        orientation="h",
        rect=(40.0, 60.0, 140.0, 180.0),
        spacing_px=20.0,
        line_count=8,
    )
    stair_stringer = WallSegment(
        start=Point2D(x=110, y=20),
        end=Point2D(x=110, y=300),
        evidence_source="ai_proposal+local_structural_ink",
    )
    perimeter = WallSegment(
        start=Point2D(x=40, y=20),
        end=Point2D(x=40, y=300),
        evidence_source="ai_proposal+local_structural_ink",
    )

    result = filter_walls_crossing_repetitive_details([stair_stringer, perimeter], [region])

    assert result.rejected == [stair_stringer]
    assert result.walls == [perimeter]


def test_local_oriented_footprint_rejects_counter_outline_and_preserves_external() -> None:
    footprint = LocalObjectFootprint(
        category="kitchen_counter",
        center=Point2D(x=80, y=80),
        width_px=90,
        depth_px=50,
        rotation_deg=0,
    )
    counter_edge = WallSegment(
        start=Point2D(x=38, y=58),
        end=Point2D(x=122, y=58),
        evidence_source="wall_band",
    )
    external = counter_edge.model_copy(update={"external": True})

    result = filter_walls_inside_semantic_objects(
        [counter_edge, external],
        semantic_exclusions_from_local_footprints([footprint]),
        coverage_threshold=0.70,
        preserve_external=True,
    )

    assert result.rejected == [counter_edge]
    assert result.walls == [external]


def test_ocr_box_rejects_glyph_stroke_but_not_crossing_room_wall() -> None:
    text = OCRText(
        text="BALCONY",
        normalized_text="BALCONY",
        semantic_type="balcony_label",
        confidence=0.95,
        polygon=[(40, 40), (120, 40), (120, 60), (40, 60)],
    )
    glyph = WallSegment(
        start=Point2D(x=78, y=38),
        end=Point2D(x=78, y=63),
        evidence_source="wall_band",
    )
    crossing = WallSegment(
        start=Point2D(x=0, y=50),
        end=Point2D(x=180, y=50),
        evidence_source="wall_band",
    )

    result = filter_walls_near_text_regions([glyph, crossing], [text])

    assert result.rejected == [glyph]
    assert result.walls == [crossing]
