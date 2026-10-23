"""Partition existing surfaces at finish boundaries, without adding any skins.

This module deliberately uses only Python's standard library: Blender loads the
same implementation with runpy, independently of the server's dependencies.
All faces are cut by the same planes (including unpainted bottom/edge faces),
so the partition retains shared edges, closed solids and the original volume.
"""

from __future__ import annotations

import math

EPS = 1e-9


def _inside(x, y, points):
    inside = False
    for a, b in zip(points, points[1:] + points[:1]):
        if (a[1] > y) != (b[1] > y):
            crossing = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < crossing:
                inside = not inside
    return inside


def _split(polygon, plane):
    nx, ny, distance = plane
    values = [nx * p[0] + ny * p[1] - distance for p in polygon]
    if min(values) >= -EPS or max(values) <= EPS:
        return [polygon]
    halves = [[], []]
    for index, a in enumerate(polygon):
        b = polygon[(index + 1) % len(polygon)]
        da, db = values[index], values[(index + 1) % len(polygon)]
        for half, sign in zip(halves, (1, -1)):
            if sign * da >= -EPS:
                half.append(a)
        if (da > EPS and db < -EPS) or (da < -EPS and db > EPS):
            t = da / (da - db)
            intersection = tuple(a[k] + t * (b[k] - a[k]) for k in range(3))
            for half in halves:
                half.append(intersection)
    return [half for half in halves if len(half) >= 3]


def _normal(triangle):
    a, b, c = triangle
    u = [b[k] - a[k] for k in range(3)]
    v = [c[k] - a[k] for k in range(3)]
    cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
             u[0] * v[1] - u[1] * v[0])
    length = math.sqrt(sum(value * value for value in cross))
    return tuple(value / length for value in cross) if length > EPS * EPS else (0, 0, 0)


def partition_finish_mesh(vertices, faces, regions, *, wall=None):
    """Return vertices, triangles, preset-or-None labels and source face indices.

    Floor regions contain ``points`` and ``preset``. Wall regions are already
    resolved, nonoverlapping ``side``, ``start_m``, ``end_m`` intervals. A None
    label always means retain that source face's original material, including
    ambiguous floor overlaps. Input faces must be triangles in world metres.
    """
    if not regions:
        return vertices, faces, [None] * len(faces), list(range(len(faces)))
    if wall is None:
        min_x, max_x = min(point[0] for point in vertices), max(point[0] for point in vertices)
        min_y, max_y = min(point[1] for point in vertices), max(point[1] for point in vertices)
        regions = [region for region in regions
                   if min(point[0] for point in region["points"]) <= max_x + EPS
                   and max(point[0] for point in region["points"]) >= min_x - EPS
                   and min(point[1] for point in region["points"]) <= max_y + EPS
                   and max(point[1] for point in region["points"]) >= min_y - EPS]
        if not regions:
            return vertices, faces, [None] * len(faces), list(range(len(faces)))
    planes = []
    if wall is None:
        for region in regions:
            points = region["points"]
            for a, b in zip(points, points[1:] + points[:1]):
                dx, dy = b[0] - a[0], b[1] - a[1]
                length = math.hypot(dx, dy)
                if length > EPS:
                    planes.append((-dy / length, dx / length, (-dy * a[0] + dx * a[1]) / length))
    else:
        sx, sy = wall["start"]["x"], wall["start"]["y"]
        dx, dy = wall["end"]["x"] - sx, wall["end"]["y"] - sy
        length = math.hypot(dx, dy)
        if length <= EPS:
            return vertices, faces, [None] * len(faces), list(range(len(faces)))
        ux, uy = dx / length, dy / length
        for region in regions:
            for offset in (region["start_m"], region["end_m"]):
                planes.append((ux, uy, ux * sx + uy * sy + offset))
    # A sign-normalized plane key avoids repeated subdivision at shared edges.
    unique = {}
    for plane in planes:
        sign = -1 if plane[0] < -EPS or (abs(plane[0]) < EPS and plane[1] < 0) else 1
        normalized = tuple(sign * value for value in plane)
        unique[tuple(round(value, 9) for value in normalized)] = normalized
    planes = list(unique.values())
    output_vertices, output_faces, labels, sources = [], [], [], []
    vertex_indices = {}

    def vertex_index(point):
        key = tuple(round(float(value), 10) for value in point)
        if key not in vertex_indices:
            vertex_indices[key] = len(output_vertices)
            output_vertices.append(key)
        return vertex_indices[key]

    for source, face in enumerate(faces):
        triangle = [tuple(vertices[int(index)]) for index in face]
        normal = _normal(triangle)
        cells = [triangle]
        for plane in planes:
            cells = [half for cell in cells for half in _split(cell, plane)]
        for cell in cells:
            center = tuple(sum(point[k] for point in cell) / len(cell) for k in range(3))
            preset = None
            if wall is None:
                if normal[2] > 0.999:
                    containing = [region for region in regions
                                  if _inside(center[0], center[1], region["points"])]
                    matches = {region["preset"] for region in containing}
                    if len(matches) == 1 and any(region.get("apply", True) for region in containing):
                        preset = next(iter(matches))
            else:
                side_dot = normal[0] * -uy + normal[1] * ux
                side = 1 if side_dot > 0.999 else -1 if side_dot < -0.999 else 0
                position = (center[0] - sx) * ux + (center[1] - sy) * uy
                side_distance = (center[0] - sx) * -uy + (center[1] - sy) * ux
                # Exclude opening reveals, end caps and any unrelated surface.
                if side and abs(abs(side_distance) - wall.get("thickness_m", 0.12) / 2) < 1e-6:
                    matches = {region["preset"] for region in regions if region["side"] == side
                               and region["start_m"] - EPS < position < region["end_m"] + EPS}
                    if len(matches) == 1:
                        preset = next(iter(matches))
            # Center fans retain every split boundary vertex, including
            # collinear points. Dropping those points creates T junctions.
            triangles = [cell] if len(cell) == 3 else [
                [center, a, b] for a, b in zip(cell, cell[1:] + cell[:1])
            ]
            for tri in triangles:
                if _normal(tri) == (0, 0, 0):
                    continue
                output_faces.append([vertex_index(point) for point in tri])
                labels.append(preset)
                sources.append(source)
    return output_vertices, output_faces, labels, sources
