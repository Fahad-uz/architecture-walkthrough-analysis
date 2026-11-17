from __future__ import annotations

import numpy as np
import pytest
from shapely.affinity import rotate, translate
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from architecture_walkthrough.geometry.models import (
    FloorPlanModel, Point2D, RoomPolygon, SlabPolygon, WallSegment,
)
from architecture_walkthrough.scene import floor_builder
from architecture_walkthrough.scene.floor_builder import polygon_floor_mesh
from architecture_walkthrough.scene.simple_glb import (
    COLORS, _floor_meshes_for_model, _merged_room_floors, _room_floor_coverage,
)


def _points(coordinates):
    return [Point2D(x=float(x), y=float(y)) for x, y in coordinates]


def _assert_solid_matches_polygon(mesh, polygon, thickness=0.1):
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.is_volume
    assert mesh.volume == pytest.approx(polygon.area * thickness, abs=1e-8)
    assert mesh.area == pytest.approx(2 * polygon.area + polygon.length * thickness, abs=1e-8)
    top = mesh.face_normals[:, 2] > 0.999
    bottom = mesh.face_normals[:, 2] < -0.999
    assert mesh.area_faces[top].sum() == pytest.approx(polygon.area, abs=1e-8)
    assert mesh.area_faces[bottom].sum() == pytest.approx(polygon.area, abs=1e-8)
    assert np.allclose(mesh.triangles[top, :, 2], 0)
    assert np.allclose(mesh.triangles[bottom, :, 2], -thickness)
    projected = [Polygon(triangle[:, :2]) for triangle in mesh.triangles[top]]
    # Check complete triangle footprints, not just their centroids: no
    # triangle may bridge a courtyard even if its center lies outside it.
    for triangle in projected:
        assert triangle.difference(polygon).area < 1e-8
    assert unary_union(projected).symmetric_difference(polygon).area < 1e-8
    for hole in polygon.interiors:
        courtyard = Polygon(hole)
        assert sum(triangle.intersection(courtyard).area for triangle in projected) < 1e-8
        assert not any(triangle.covers(courtyard.representative_point()) for triangle in projected)
    # The side normals point out of the solid, including *into* courtyards.
    for center, normal in zip(mesh.triangles_center[~(top | bottom)], mesh.face_normals[~(top | bottom)]):
        outside = Point(*(center[:2] + normal[:2] * 1e-5))
        inside = Point(*(center[:2] - normal[:2] * 1e-5))
        assert not polygon.contains(outside)
        assert polygon.contains(inside)


def _courtyard_model(angle=0):
    xs, ys = [0, 4, 6, 10], [0, 3, 5, 8]
    room_shapes = [box(xs[x], ys[y], xs[x + 1], ys[y + 1])
                   for x in range(3) for y in range(3) if (x, y) != (1, 1)]
    room_shapes = [rotate(polygon, angle, origin=(0, 0)) for polygon in room_shapes]
    envelope = rotate(box(0, 0, 10, 8), angle, origin=(0, 0))
    coordinates = list(envelope.exterior.coords)
    walls = [WallSegment(start=Point2D(x=a[0], y=a[1]), end=Point2D(x=b[0], y=b[1]))
             for a, b in zip(coordinates, coordinates[1:])]
    return FloorPlanModel(walls=walls, rooms=[
        RoomPolygon(points=_points(polygon.exterior.coords[:-1])) for polygon in room_shapes
    ]), unary_union(room_shapes)


@pytest.mark.parametrize("angle", [0, 17, 47, 90])
def test_merged_room_floor_preserves_courtyard_at_any_orientation(angle):
    model, polygon = _courtyard_model(angle)
    assert _room_floor_coverage(model) >= 0.45
    merged = _merged_room_floors(model)
    assert len(merged) == 1
    assert len(merged[0].interiors) == 1
    floors = _floor_meshes_for_model(model)
    assert len(floors) == 1
    _assert_solid_matches_polygon(floors[0], polygon)
    assert floors[0].euler_number == 0


def test_concave_exterior_supports_multiple_rotated_holes_and_narrow_void():
    exterior = [(0, 0), (12, 0), (12, 10), (7, 10), (7, 5), (0, 5)]
    holes = [translate(rotate(box(-0.8, -0.8, 0.8, 0.8), 17), 3, 2.5),
             translate(rotate(box(-1.3, -0.6, 1.3, 0.6), 41), 8, 2.4),
             translate(rotate(box(-0.1, -1, 0.1, 1), 33), 9.5, 7.5)]
    polygon = Polygon(exterior, [hole.exterior.coords for hole in holes])
    assert polygon.is_valid
    mesh = polygon_floor_mesh(_points(exterior), 0.1, COLORS["floor"],
                              holes=[_points(hole.exterior.coords[:-1]) for hole in holes])
    _assert_solid_matches_polygon(mesh, polygon)
    assert mesh.euler_number == 2 - 2 * len(holes)


def test_disconnected_room_floor_union_keeps_hole_and_separate_island():
    model, ring = _courtyard_model()
    island = box(11, 1, 12, 2)
    model.rooms.append(RoomPolygon(points=_points(island.exterior.coords[:-1])))
    floors = _floor_meshes_for_model(model)
    assert len(floors) == 2
    _assert_solid_matches_polygon(floors[0], ring)
    _assert_solid_matches_polygon(floors[1], island)


@pytest.mark.parametrize("clockwise", [False, True])
def test_no_hole_floor_uses_existing_triangulation_without_partition(monkeypatch, clockwise):
    def unexpected_partition(*_args, **_kwargs):
        raise AssertionError("ordinary no-hole floor must keep its existing triangulation path")

    monkeypatch.setattr(floor_builder, "partition_finish_mesh", unexpected_partition)
    coordinates = [(0, 0), (4, 0), (4, 3), (0, 3)]
    if clockwise:
        coordinates.reverse()
    plain = polygon_floor_mesh(_points(coordinates), 0.1, COLORS["floor"])
    empty_holes = polygon_floor_mesh(_points(coordinates), 0.1, COLORS["floor"], holes=[])
    assert len(plain.vertices) == 8
    assert len(plain.faces) == 12
    np.testing.assert_array_equal(plain.vertices, empty_holes.vertices)
    np.testing.assert_array_equal(plain.faces, empty_holes.faces)
    _assert_solid_matches_polygon(plain, Polygon(coordinates))
    model = FloorPlanModel(walls=[
        WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=4, y=0)),
        WallSegment(start=Point2D(x=4, y=0), end=Point2D(x=4, y=3)),
    ], rooms=[RoomPolygon(points=_points(coordinates))])
    merged = _floor_meshes_for_model(model)[0]
    assert len(merged.vertices) == 8
    assert len(merged.faces) == 12
    _assert_solid_matches_polygon(merged, Polygon(coordinates))


def test_explicit_slab_remains_authoritative_over_room_courtyard():
    model, _ = _courtyard_model()
    slab = box(0, 0, 10, 8)
    model.slabs = [SlabPolygon(points=_points(slab.exterior.coords[:-1]), thickness_m=0.3)]
    floors = _floor_meshes_for_model(model)
    assert len(floors) == 1
    _assert_solid_matches_polygon(floors[0], slab, thickness=0.3)


def test_low_room_coverage_keeps_existing_padded_fallback_policy():
    model, _ = _courtyard_model()
    model.walls.append(WallSegment(start=Point2D(x=0, y=0), end=Point2D(x=30, y=30)))
    assert _room_floor_coverage(model) < 0.45
    floors = _floor_meshes_for_model(model)
    assert len(floors) == 1
    _assert_solid_matches_polygon(floors[0], box(-0.3, -0.3, 30.3, 30.3))


def test_invalid_hole_configuration_is_rejected():
    with pytest.raises(ValueError, match="floor polygon is invalid"):
        polygon_floor_mesh(_points([(0, 0), (4, 0), (4, 4), (0, 4)]), 0.1, COLORS["floor"],
                           holes=[_points([(3, 3), (5, 3), (5, 5), (3, 5)])])


@pytest.mark.parametrize("holes", [
    [[(0, 0), (3, 1), (1, 3)]],
    [[(2, 2), (4, 2), (4, 4), (2, 4)], [(4, 4), (6, 4), (6, 6), (4, 6)]],
])
def test_point_touching_hole_boundaries_are_rejected_before_extrusion(monkeypatch, holes):
    exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
    # These are valid 2D polygons, but touching rings produce a pinched edge
    # in the extruded solid. No mesh should be created for either input.
    assert Polygon(exterior, holes).is_valid

    def unexpected_extrusion(*_args, **_kwargs):
        raise AssertionError("touching boundaries must be rejected before extrusion")

    monkeypatch.setattr(floor_builder.trimesh.creation, "extrude_triangulation", unexpected_extrusion)
    with pytest.raises(ValueError, match="floor hole boundaries must be disjoint"):
        polygon_floor_mesh(_points(exterior), 0.1, COLORS["floor"],
                           holes=[_points(hole) for hole in holes])
