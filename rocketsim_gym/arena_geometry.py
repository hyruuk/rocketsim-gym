"""The homemade soccar arena as one triangle mesh, and RocketSim's collision-mesh files.

RocketSim adds a soccar arena's floor, a ceiling plane and the two side walls itself. This mesh
is the rest, built from Rocket League's published field dimensions: the curves from the floor
up the walls and from the walls into the ceiling, the corner and back walls with their goal
mouths, the ceiling, and the two goals. The curve radii and the rounding where two walls meet
are estimates. Lengths here are Unreal units (UU); a collision-mesh file holds Bullet units.

``python -m rocketsim_gym.arena_geometry FOLDER`` writes the homemade arena into ``FOLDER``:
its mesh file under ``soccar/``, and a manifest of the file's SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

#: Unreal units per Bullet unit: a collision-mesh file's lengths are UU divided by this.
UNREAL_UNITS_PER_BULLET_UNIT = 50.0
#: RocketSim refuses a file with more triangles or vertices than this.
MAXIMUM_TRIANGLES_OR_VERTICES = 1_000_000
#: The subfolder of a collision-mesh folder that RocketSim reads the soccar meshes from.
SOCCAR_FOLDER_NAME = "soccar"
#: The file in a collision-mesh folder that lists the SHA-256 of each of its mesh files.
MANIFEST_FILE_NAME = "manifest.json"
#: The homemade arena's one mesh file, under its soccar folder.
HOMEMADE_ARENA_FILE_NAME = "arena.cmf"
#: The collision-mesh folder of the homemade arena, shipped with this package.
HOMEMADE_ARENA_FOLDER = Path(__file__).resolve().parent / "collision_meshes" / "homemade"
#: Coordinates are rounded to this many decimals of a UU, so that equal points weld.
_WELD_DECIMALS = 6
#: How close (UU) two lengths must be to count as the same.
_TOLERANCE = 1e-6


@dataclass(frozen=True)
class SoccarArenaDimensions:
    """The shape of a soccar arena, in UU. The defaults are the homemade arena.

    :ivar corner_wall_line: ``|x| + |y|`` along the four corner walls.
    :ivar wall_joint_radius: the radius, seen from above, that rounds off where two walls meet.
    :ivar longest_edge: no piece of a flat wall, the goals or the ceiling is longer than this.
    """

    side_wall_x: float = 4096.0
    back_wall_y: float = 5120.0
    corner_wall_line: float = 8064.0
    ceiling_z: float = 2044.0
    floor_curve_radius: float = 256.0
    ceiling_curve_radius: float = 256.0
    wall_joint_radius: float = 512.0
    goal_half_width: float = 892.755
    goal_height: float = 642.775
    goal_back_y: float = 6000.0
    segments_per_quarter_circle: int = 16
    longest_edge: float = 256.0


@dataclass(frozen=True)
class TriangleMesh:
    """Vertices in UU, ``(n, 3)`` float64, and triangles as vertex indices, ``(m, 3)`` int32.

    A triangle is wound counter-clockwise seen from the side it faces.
    """

    vertices: np.ndarray
    triangles: np.ndarray


class _Span(Enum):
    """What lies between two neighbouring points of the footprint."""

    SIDE_WALL = "side wall"
    CORNER_WALL = "corner wall"
    BACK_WALL = "back wall"
    JOINT = "rounded joint of two walls"


class _Band(Enum):
    """What lies between two neighbouring points of a wall's cross-section."""

    FLOOR_CURVE = "floor curve"
    WALL = "wall"
    CEILING_CURVE = "ceiling curve"


@dataclass(frozen=True)
class _Footprint:
    """Points around the foot of the walls, counter-clockwise seen from above.

    :ivar points: ``(n, 2)`` positions on the floor.
    :ivar inward: ``(n, 2)`` unit normals pointing into the field.
    :ivar spans: what lies between point ``i`` and point ``i + 1``.
    """

    points: np.ndarray
    inward: np.ndarray
    spans: tuple[_Span, ...]


@dataclass(frozen=True)
class _ProfilePoint:
    """A point of a wall's cross-section: how far in from the foot of the wall, how high, and
    the band from it to the next point (``None`` for the last point)."""

    inset: float
    height: float
    band: _Band | None


class _MeshBuilder:
    """Triangles added one at a time, their vertices welded, each wound to face a given way."""

    def __init__(self) -> None:
        self._index_of_vertex: dict[tuple[float, float, float], int] = {}
        self._vertices: list[tuple[float, float, float]] = []
        self._triangles: list[tuple[int, int, int]] = []

    def _vertex(self, point) -> int:
        key = tuple(float(c) for c in np.round(np.asarray(point, dtype=np.float64),
                                                _WELD_DECIMALS))
        index = self._index_of_vertex.get(key)
        if index is None:
            index = len(self._vertices)
            self._index_of_vertex[key] = index
            self._vertices.append(key)
        return index

    def add_triangle(self, a, b, c, facing) -> None:
        """Add the triangle ``a b c``, wound so that its normal points along ``facing``."""
        a, b, c = (np.asarray(p, dtype=np.float64) for p in (a, b, c))
        if np.dot(np.cross(b - a, c - a), facing) < 0:
            b, c = c, b
        self._triangles.append((self._vertex(a), self._vertex(b), self._vertex(c)))

    def add_quad(self, a, b, c, d, facing) -> None:
        """Add the quad ``a b c d`` (corners in order around it) as two triangles."""
        self.add_triangle(a, b, c, facing)
        self.add_triangle(a, c, d, facing)

    def mesh(self) -> TriangleMesh:
        return TriangleMesh(np.array(self._vertices, dtype=np.float64),
                            np.array(self._triangles, dtype=np.int32))


def build_homemade_soccar_arena(
        dimensions: SoccarArenaDimensions = SoccarArenaDimensions()) -> TriangleMesh:
    """The homemade soccar arena: everything RocketSim's own planes leave out.

    :param dimensions: the arena's shape.
    :return: one mesh, every triangle facing the playable volume.
    """
    footprint = _footprint(dimensions)
    profile = _profile(dimensions)
    builder = _MeshBuilder()
    _add_walls_and_curves(builder, footprint, profile, dimensions)
    _add_ceiling(builder, footprint, dimensions)
    for side in (1.0, -1.0):
        _add_goal(builder, footprint, profile, dimensions, side)
    return builder.mesh()


def _subdivided(breakpoints: list[float], longest: float) -> list[float]:
    """``breakpoints`` (in order, either way) with points added so no piece exceeds ``longest``."""
    values = [breakpoints[0]]
    for a, b in zip(breakpoints, breakpoints[1:]):
        pieces = max(1, math.ceil(abs(b - a) / longest - _TOLERANCE))
        values += [a + (b - a) * i / pieces for i in range(1, pieces)] + [b]
    return values


def _left_normal(direction: np.ndarray) -> np.ndarray:
    """The unit normal on the left of ``direction``: inward, going counter-clockwise."""
    return np.array([-direction[1], direction[0]])


def _turn(incoming: np.ndarray, outgoing: np.ndarray) -> float:
    """The angle (radians) the walls turn through at a corner."""
    return math.acos(float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0)))


def _footprint(dimensions: SoccarArenaDimensions) -> _Footprint:
    """The foot of the walls: an octagon with its eight joints rounded off."""
    x_side, y_back = dimensions.side_wall_x, dimensions.back_wall_y
    y_side_end = dimensions.corner_wall_line - x_side
    x_back_end = dimensions.corner_wall_line - y_back
    corners = np.array([(x_side, -y_side_end), (x_side, y_side_end), (x_back_end, y_back),
                        (-x_back_end, y_back), (-x_side, y_side_end), (-x_side, -y_side_end),
                        (-x_back_end, -y_back), (x_back_end, -y_back)])
    walls = (_Span.SIDE_WALL, _Span.CORNER_WALL, _Span.BACK_WALL, _Span.CORNER_WALL) * 2
    directions = np.roll(corners, -1, axis=0) - corners
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radius = dimensions.wall_joint_radius
    tangents = [radius * math.tan(_turn(directions[k - 1], directions[k]) / 2) for k in range(8)]
    points, inward, spans = [], [], []
    for k, wall in enumerate(walls):
        joint_points, joint_inward = _rounded_joint(
            corners[k], directions[k - 1], directions[k], tangents[k], dimensions)
        points += joint_points
        inward += joint_inward
        spans += [_Span.JOINT] * len(joint_points)
        run_points = _straight_run(corners[k] + tangents[k] * directions[k],
                                   corners[(k + 1) % 8] - tangents[(k + 1) % 8] * directions[k],
                                   wall, dimensions)
        points += run_points
        inward += [_left_normal(directions[k])] * len(run_points)
        spans += [wall] * len(run_points)
    return _Footprint(np.array(points), np.array(inward), tuple(spans))


def _rounded_joint(corner: np.ndarray, incoming: np.ndarray, outgoing: np.ndarray,
                   tangent: float, dimensions: SoccarArenaDimensions) -> tuple[list, list]:
    """The arc that rounds ``corner`` off, without its last point (the next wall's first)."""
    radius = dimensions.wall_joint_radius
    start = corner - tangent * incoming
    center = start + radius * _left_normal(incoming)
    turn = _turn(incoming, outgoing)
    first_angle = math.atan2(start[1] - center[1], start[0] - center[0])
    segments = max(2, math.ceil(dimensions.segments_per_quarter_circle * turn / (math.pi / 2)))
    points, inward = [start], [_left_normal(incoming)]
    for j in range(1, segments):
        radial = np.array([math.cos(first_angle + turn * j / segments),
                           math.sin(first_angle + turn * j / segments)])
        points.append(center + radius * radial)
        inward.append(-radial)
    return points, inward


def _straight_run(start: np.ndarray, end: np.ndarray, wall: _Span,
                  dimensions: SoccarArenaDimensions) -> list:
    """Points along a wall from ``start`` towards ``end``, without ``end`` itself.

    A back wall gets a point at each goal post, so that the goal mouth starts and ends on one.
    """
    length = float(np.linalg.norm(end - start))
    direction = (end - start) / length
    breaks = {0.0, length}
    if wall is _Span.BACK_WALL:
        for post_x in (dimensions.goal_half_width, -dimensions.goal_half_width):
            breaks.add(float((post_x - start[0]) / direction[0]))
    distances = _subdivided(sorted(breaks), dimensions.longest_edge)[:-1]
    return [start + s * direction for s in distances]


def _profile(dimensions: SoccarArenaDimensions) -> list[_ProfilePoint]:
    """A wall's cross-section from the floor to the ceiling: floor curve, wall, ceiling curve."""
    floor_radius, ceiling_radius = dimensions.floor_curve_radius, dimensions.ceiling_curve_radius
    wall_top = dimensions.ceiling_z - ceiling_radius
    if not floor_radius < dimensions.goal_height < wall_top:
        raise ValueError("the goal's crossbar must be on the upright part of the back wall")
    segments = dimensions.segments_per_quarter_circle
    points = []
    for j in range(segments):
        angle = (math.pi / 2) * j / segments
        points.append(_ProfilePoint(floor_radius * (1 - math.sin(angle)),
                                    floor_radius * (1 - math.cos(angle)), _Band.FLOOR_CURVE))
    heights = _subdivided([floor_radius, dimensions.goal_height, wall_top],
                          dimensions.longest_edge)
    points += [_ProfilePoint(0.0, z, _Band.WALL) for z in heights[:-1]]
    for j in range(segments):
        angle = (math.pi / 2) * j / segments
        points.append(_ProfilePoint(ceiling_radius * (1 - math.cos(angle)),
                                    wall_top + ceiling_radius * math.sin(angle),
                                    _Band.CEILING_CURVE))
    points.append(_ProfilePoint(ceiling_radius, dimensions.ceiling_z, None))
    return points


def _wall_point(footprint: _Footprint, station: int, point: _ProfilePoint) -> np.ndarray:
    """Where the profile ``point`` is above footprint point ``station``."""
    x, y = footprint.points[station] + point.inset * footprint.inward[station]
    return np.array([x, y, point.height])


def _band_facing(lower: _ProfilePoint, upper: _ProfilePoint, inward: np.ndarray,
                 dimensions: SoccarArenaDimensions) -> np.ndarray:
    """Which way a band of the profile faces: into the field, towards its curve's centre."""
    inset, height = (lower.inset + upper.inset) / 2, (lower.height + upper.height) / 2
    if lower.band is _Band.FLOOR_CURVE:
        radius = dimensions.floor_curve_radius
        return np.array([*(inward * (radius - inset)), radius - height])
    if lower.band is _Band.CEILING_CURVE:
        radius = dimensions.ceiling_curve_radius
        return np.array([*(inward * (radius - inset)),
                         dimensions.ceiling_z - radius - height])
    return np.array([*inward, 0.0])


def _is_goal_mouth(footprint: _Footprint, station: int, dimensions: SoccarArenaDimensions) -> bool:
    """Whether the span from footprint point ``station`` to the next is a goal's mouth."""
    following = (station + 1) % len(footprint.points)
    return (footprint.spans[station] is _Span.BACK_WALL
            and abs(footprint.points[station][0]) <= dimensions.goal_half_width + _TOLERANCE
            and abs(footprint.points[following][0]) <= dimensions.goal_half_width + _TOLERANCE)


def _add_walls_and_curves(builder: _MeshBuilder, footprint: _Footprint,
                          profile: list[_ProfilePoint], dimensions: SoccarArenaDimensions) -> None:
    """The profile swept around the footprint, less the side walls' planes and the goal mouths."""
    count = len(footprint.points)
    for station in range(count):
        following = (station + 1) % count
        inward = footprint.inward[station] + footprint.inward[following]
        inward /= np.linalg.norm(inward)
        is_goal_mouth = _is_goal_mouth(footprint, station, dimensions)
        for lower, upper in zip(profile, profile[1:]):
            if footprint.spans[station] is _Span.SIDE_WALL and lower.band is _Band.WALL:
                continue
            if is_goal_mouth and upper.height <= dimensions.goal_height + _TOLERANCE:
                continue
            builder.add_quad(_wall_point(footprint, station, lower),
                             _wall_point(footprint, following, lower),
                             _wall_point(footprint, following, upper),
                             _wall_point(footprint, station, upper),
                             _band_facing(lower, upper, inward, dimensions))


def _add_ceiling(builder: _MeshBuilder, footprint: _Footprint,
                 dimensions: SoccarArenaDimensions) -> None:
    """The flat ceiling inside the top of the ceiling curve: rings that shrink to its centre."""
    height, down = dimensions.ceiling_z, np.array([0.0, 0.0, -1.0])
    ring = footprint.points + dimensions.ceiling_curve_radius * footprint.inward
    rings = math.ceil(float(np.linalg.norm(ring, axis=1).max()) / dimensions.longest_edge)
    count = len(ring)
    outer = ring
    for k in range(1, rings):
        inner = ring * (1 - k / rings)
        for i in range(count):
            following = (i + 1) % count
            builder.add_quad([*outer[i], height], [*outer[following], height],
                             [*inner[following], height], [*inner[i], height], down)
        outer = inner
    for i in range(count):
        builder.add_triangle([0.0, 0.0, height], [*outer[i], height],
                             [*outer[(i + 1) % count], height], down)


def _add_goal(builder: _MeshBuilder, footprint: _Footprint, profile: list[_ProfilePoint],
              dimensions: SoccarArenaDimensions, side: float) -> None:
    """The goal behind the back wall at ``side * back_wall_y``: roof, sides, back, and the caps
    that close the floor curve where the goal mouth cuts it."""
    post, crossbar = dimensions.goal_half_width, dimensions.goal_height
    mouth_y, back_y = side * dimensions.back_wall_y, side * dimensions.goal_back_y
    mouth_xs = sorted({round(float(x), _WELD_DECIMALS) for x, y in footprint.points
                       if abs(y - mouth_y) < _TOLERANCE and abs(x) <= post + _TOLERANCE})
    heights = [0.0] + [p.height for p in profile
                       if p.band is _Band.WALL and p.height <= crossbar + _TOLERANCE]
    depths = _subdivided([mouth_y, back_y], dimensions.longest_edge)
    down, toward_field = np.array([0.0, 0.0, -1.0]), np.array([0.0, -side, 0.0])
    for x0, x1 in zip(mouth_xs, mouth_xs[1:]):
        for y0, y1 in zip(depths, depths[1:]):
            builder.add_quad([x0, y0, crossbar], [x1, y0, crossbar], [x1, y1, crossbar],
                             [x0, y1, crossbar], down)
        for z0, z1 in zip(heights, heights[1:]):
            builder.add_quad([x0, back_y, z0], [x1, back_y, z0], [x1, back_y, z1],
                             [x0, back_y, z1], toward_field)
    floor_curve = [p for p in profile if p.band is _Band.FLOOR_CURVE]
    floor_curve.append(next(p for p in profile if p.band is _Band.WALL))
    for post_x in (post, -post):
        toward_middle = np.array([-math.copysign(1.0, post_x), 0.0, 0.0])
        for y0, y1 in zip(depths, depths[1:]):
            for z0, z1 in zip(heights, heights[1:]):
                builder.add_quad([post_x, y0, z0], [post_x, y1, z0], [post_x, y1, z1],
                                 [post_x, y0, z1], toward_middle)
        foot = [post_x, mouth_y, 0.0]
        for p, q in zip(floor_curve, floor_curve[1:]):
            builder.add_triangle(foot, [post_x, mouth_y - side * p.inset, p.height],
                                 [post_x, mouth_y - side * q.inset, q.height], toward_middle)


def write_collision_mesh_file(mesh: TriangleMesh, path: Path) -> None:
    """Write ``mesh`` as RocketSim reads a collision-mesh file.

    The file is the triangle count and the vertex count (int32), every triangle's three vertex
    indices (int32), then every vertex (float32, Bullet units), all little-endian.
    """
    vertices = (mesh.vertices / UNREAL_UNITS_PER_BULLET_UNIT).astype("<f4")
    triangles = mesh.triangles.astype("<i4")
    Path(path).write_bytes(struct.pack("<ii", len(triangles), len(vertices))
                           + triangles.tobytes() + vertices.tobytes())


def read_collision_mesh_file(path: Path) -> TriangleMesh:
    """Read a collision-mesh file, refusing anything RocketSim would refuse or misread.

    :param path: the ``.cmf`` file.
    :return: its mesh, vertices in UU.
    :raises ValueError: the file's counts, length, indices or vertices are not a mesh.
    """
    raw = Path(path).read_bytes()
    if len(raw) < 8:
        raise ValueError(f"{path}: {len(raw)} bytes is too short for a collision-mesh file")
    triangle_count, vertex_count = struct.unpack_from("<ii", raw)
    if not (0 < triangle_count <= MAXIMUM_TRIANGLES_OR_VERTICES
            and 0 < vertex_count <= MAXIMUM_TRIANGLES_OR_VERTICES):
        raise ValueError(f"{path}: {triangle_count} triangles and {vertex_count} vertices "
                         "is not a collision mesh RocketSim reads")
    expected_length = 8 + 12 * (triangle_count + vertex_count)
    if len(raw) != expected_length:
        raise ValueError(f"{path}: {len(raw)} bytes, but its counts make {expected_length}")
    triangles = np.frombuffer(raw, "<i4", 3 * triangle_count, 8).reshape(-1, 3)
    vertices = np.frombuffer(raw, "<f4", 3 * vertex_count, 8 + 12 * triangle_count).reshape(-1, 3)
    if triangles.min() < 0 or triangles.max() >= vertex_count:
        raise ValueError(f"{path}: a triangle names a vertex the file does not have")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{path}: a vertex is not a finite number")
    return TriangleMesh(vertices.astype(np.float64) * UNREAL_UNITS_PER_BULLET_UNIT,
                        triangles.astype(np.int32))


def rocketsim_mesh_hash(mesh: TriangleMesh) -> int:
    """The hash RocketSim gives a collision mesh, as it computes it on x86-64.

    RocketSim converts each float coordinate to uint32, which wraps negative values on
    x86-64 and clamps them on ARM, so only the x86-64 value matches its list of the meshes
    dumped from Rocket League.
    """
    vertices = (mesh.vertices / UNREAL_UNITS_PER_BULLET_UNIT).astype(np.float32)
    mask = 0xFFFFFFFF
    result = (len(vertices) + len(mesh.triangles) * len(vertices)) & mask
    values = np.trunc(vertices[mesh.triangles].reshape(-1).astype(np.float64))
    values = (values.astype(np.int64) & mask).astype(np.uint64)
    for _ in range(2):
        values = (((values >> np.uint64(16)) ^ values) * np.uint64(0x45D9F3B)) & np.uint64(mask)
    values = ((values >> np.uint64(16)) ^ values) & np.uint64(mask)
    for value in values.tolist():
        result ^= (value + 0x9E3779B9 + ((result << 6) & mask) + (result >> 2)) & mask
    return result


def file_sha256(path: Path) -> str:
    """The SHA-256 of a file's bytes, in hex."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collision_mesh_files(folder: Path) -> list[Path]:
    """The soccar mesh files RocketSim would load from ``folder``, in name order."""
    soccar = Path(folder) / SOCCAR_FOLDER_NAME
    return sorted(p for p in soccar.iterdir() if p.suffix == ".cmf") if soccar.is_dir() else []


def collision_mesh_manifest(folder: Path) -> dict[str, str]:
    """The SHA-256 of each soccar mesh file in ``folder``, by its path relative to ``folder``."""
    return {path.relative_to(folder).as_posix(): file_sha256(path)
            for path in collision_mesh_files(folder)}


def write_manifest(folder: Path, manifest: dict[str, str]) -> Path:
    """Write ``manifest`` as ``folder``'s manifest file.

    :return: the manifest file.
    """
    path = Path(folder) / MANIFEST_FILE_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def read_manifest(path: Path) -> dict[str, str]:
    """A manifest file: each mesh file's path, relative to its folder, and its SHA-256.

    :raises ValueError: the file is not an object of strings.
    """
    manifest = json.loads(Path(path).read_text())
    if not (isinstance(manifest, dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in manifest.items())):
        raise ValueError(f"{path}: a manifest maps each mesh file's path to its SHA-256")
    return manifest


def write_homemade_arena(folder: Path,
                         dimensions: SoccarArenaDimensions = SoccarArenaDimensions()) -> Path:
    """Write the homemade arena's mesh file into ``folder/soccar/``, and the folder's manifest.

    :return: the mesh file.
    """
    path = Path(folder) / SOCCAR_FOLDER_NAME / HOMEMADE_ARENA_FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    write_collision_mesh_file(build_homemade_soccar_arena(dimensions), path)
    write_manifest(folder, collision_mesh_manifest(folder))
    return path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m rocketsim_gym.arena_geometry FOLDER")
    print(write_homemade_arena(Path(sys.argv[1])))
