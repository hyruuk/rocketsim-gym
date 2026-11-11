"""Frames of a soccar match, drawn offscreen with OpenGL through moderngl.

The arena drawn is the one the physics uses: its collision meshes, plus the floor and the side
walls, which RocketSim adds as planes. One light shades every face, in team colors: blue's half
and goal at negative y, orange's at positive y. The meshes face the field, so from outside the
arena its walls are not drawn, as in Rocket League. A frame depends on its :class:`Scene` alone,
and is read back whole before :meth:`Renderer.draw` returns.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import moderngl
import numpy as np

from .arena_geometry import TriangleMesh
from .camera import Camera
from .match import Team

#: Samples per pixel for anti-aliasing.
SAMPLES = 4
NEAR, FAR = 10.0, 30000.0
BACKGROUND = (0.03, 0.04, 0.08)
LIGHT_DIRECTION = (0.25, -0.35, 0.9)

GRASS = ((0.17, 0.40, 0.19), (0.15, 0.36, 0.17))
FIELD_LINE = (0.85, 0.88, 0.85)
WALL = {Team.BLUE: (0.17, 0.27, 0.52), Team.ORANGE: (0.55, 0.31, 0.15)}
GOAL = {Team.BLUE: (0.20, 0.42, 0.95), Team.ORANGE: (1.00, 0.52, 0.14)}
CEILING = (0.13, 0.14, 0.18)
CAR = {Team.BLUE: (0.10, 0.38, 0.95), Team.ORANGE: (1.00, 0.48, 0.08)}
CAR_WINDOWS = (0.05, 0.06, 0.08)
TIRES = (0.04, 0.04, 0.04)
BALL = (0.86, 0.86, 0.83)
PAD_BASE = (0.22, 0.22, 0.24)
BOOST = {True: (1.00, 0.74, 0.12), False: (1.00, 0.86, 0.36)}

_SIDE_WALL_X, _BACK_WALL_Y, _GOAL_BACK_Y, _CEILING_Z = 4096.0, 5120.0, 6000.0, 2044.0
_STRIP = 512.0
_BALL_RADIUS = 91.25

_VERTEX_SHADER = """
#version 330
uniform mat4 view_projection;
uniform mat4 model;
in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;
out vec3 v_normal;
out vec3 v_color;
void main() {
    v_normal = mat3(model) * in_normal;
    v_color = in_color;
    gl_Position = view_projection * model * vec4(in_position, 1.0);
}
"""
_FRAGMENT_SHADER = """
#version 330
uniform vec3 light_direction;
in vec3 v_normal;
in vec3 v_color;
out vec4 f_color;
void main() {
    vec3 normal = normalize(v_normal);
    float light = 0.42 + 0.38 * max(dot(normal, light_direction), 0.0) + 0.2 * (0.5 + 0.5 * normal.z);
    f_color = vec4(v_color * light, 1.0);
}
"""


@dataclass(frozen=True)
class CarPose:
    """A car where the frame shows it.

    :ivar forward, right, up: its directions, as RocketSim gives them.
    :ivar visible: False while it is demolished.
    """

    team: Team
    position: tuple[float, float, float]
    forward: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    visible: bool = True


@dataclass(frozen=True)
class Scene:
    """Everything a frame shows.

    :ivar ball_position: the ball, or None while it is out of play.
    :ivar pads_active: each boost pad, in the renderer's order: whether it can be picked up.
    """

    camera: Camera
    ball_position: tuple[float, float, float] | None
    cars: tuple[CarPose, ...]
    pads_active: tuple[bool, ...]


class Renderer:
    """Draws frames of one arena, at one size, offscreen.

    :param width: the frame's width, in pixels.
    :param height: the frame's height, in pixels.
    :param arena_meshes: the arena's collision meshes.
    :param pads: each boost pad's position (uu) and whether it is big, in RocketSim's order.
    :raises ValueError: a size is not a positive whole number.
    :raises RuntimeError: no OpenGL 3.3 context opens on this machine.
    """

    def __init__(self, width: int, height: int, arena_meshes: Sequence[TriangleMesh],
                 pads: Sequence[tuple[tuple[float, float, float], bool]]) -> None:
        for name, size in (("width", width), ("height", height)):
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(f"{name} {size!r}: a frame's size is a positive whole number")
        self.width, self.height = width, height
        self._pads = tuple((np.array(position, float), big) for position, big in pads)
        try:
            self._context = _standalone_context()
        except Exception as error:
            raise RuntimeError(f"no OpenGL 3.3 context opens to draw frames: {error}") from error
        context = self._context
        with context:
            self._program = context.program(vertex_shader=_VERTEX_SHADER,
                                            fragment_shader=_FRAGMENT_SHADER)
            self._arena = self._vertex_array(_arena_vertices(arena_meshes))
            self._ball = self._vertex_array(_sphere(_BALL_RADIUS, BALL, subdivisions=3))
            self._cars = {team: self._vertex_array(_car(CAR[team])) for team in Team}
            self._pad_bases = {big: self._vertex_array(_cylinder(radius, 3.0, PAD_BASE))
                               for big, radius in ((True, 160.0), (False, 100.0))}
            self._pad_boosts = {True: self._vertex_array(_sphere(45.0, BOOST[True], 2)),
                                False: self._vertex_array(_cylinder(42.0, 10.0, BOOST[False]))}
            self._framebuffer = context.framebuffer(
                color_attachments=[context.renderbuffer((width, height), 4, samples=SAMPLES)],
                depth_attachment=context.depth_renderbuffer((width, height), samples=SAMPLES))
            self._resolved = context.framebuffer(
                color_attachments=[context.renderbuffer((width, height), 4)])
        self.gl_renderer = context.info["GL_RENDERER"]

    def draw(self, scene: Scene) -> np.ndarray:
        """The frame of ``scene``: ``(height, width, 3)`` RGB bytes, top row first.

        :raises ValueError: the scene's boost pads are not the renderer's.
        """
        if len(scene.pads_active) != len(self._pads):
            raise ValueError(f"the scene has {len(scene.pads_active)} boost pads, the arena "
                             f"{len(self._pads)}")
        context = self._context
        with context:
            self._framebuffer.use()
            context.clear(*BACKGROUND, 1.0, depth=1.0)
            context.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
            # The view keeps the car's right on the screen's right, which mirrors Rocket
            # League's left-handed world: faces seen from the front wind clockwise.
            context.front_face = "cw"
            view_projection = (_projection(scene.camera, self.width / self.height)
                               @ _view(scene.camera))
            self._program["view_projection"].write(_matrix_bytes(view_projection))
            light = np.array(LIGHT_DIRECTION)
            self._program["light_direction"].value = tuple(light / np.linalg.norm(light))
            self._render(self._arena, np.eye(4))
            for (position, big), active in zip(self._pads, scene.pads_active, strict=True):
                self._render(self._pad_bases[big], _translation((position[0], position[1], 0.0)))
                if active:
                    self._render(self._pad_boosts[big], _translation(
                        position if big else (position[0], position[1], 0.0)))
            if scene.ball_position is not None:
                self._render(self._ball, _translation(scene.ball_position))
            for car in scene.cars:
                if car.visible:
                    self._render(self._cars[car.team], _car_matrix(car))
            context.copy_framebuffer(self._resolved, self._framebuffer)
            pixels = self._resolved.read(components=3, alignment=1)
        return np.frombuffer(pixels, np.uint8).reshape(self.height, self.width, 3)[::-1].copy()

    def close(self) -> None:
        """Free the OpenGL context and everything in it."""
        self._context.release()

    def _vertex_array(self, vertices: np.ndarray) -> moderngl.VertexArray:
        buffer = self._context.buffer(vertices.astype("f4").tobytes())
        return self._context.vertex_array(
            self._program, [(buffer, "3f 3f 3f", "in_position", "in_normal", "in_color")])

    def _render(self, vertex_array: moderngl.VertexArray, model: np.ndarray) -> None:
        self._program["model"].write(_matrix_bytes(model))
        vertex_array.render(moderngl.TRIANGLES)


def _standalone_context() -> moderngl.Context:
    """An OpenGL context without a window: EGL on Linux, which needs no display."""
    if sys.platform.startswith("linux"):
        return moderngl.create_standalone_context(require=330, backend="egl")
    return moderngl.create_standalone_context(require=330)


def _matrix_bytes(matrix: np.ndarray) -> bytes:
    """``matrix`` as OpenGL reads a ``mat4``: column by column."""
    return np.asarray(matrix, "f4").T.tobytes()


def _translation(position) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = position
    return matrix


def _car_matrix(car: CarPose) -> np.ndarray:
    """From the car's own axes (x forward, y right, z up) to the world."""
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2] = car.forward, car.right, car.up
    matrix[:3, 3] = car.position
    return matrix


def _view(camera: Camera) -> np.ndarray:
    forward, up = np.array(camera.forward), np.array(camera.up)
    right = np.cross(up, forward)
    position = np.array(camera.position)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = right, up, -forward
    view[:3, 3] = -view[:3, :3] @ position
    return view


def _projection(camera: Camera, aspect: float) -> np.ndarray:
    """A perspective of ``camera``'s horizontal field of view."""
    half_width = math.tan(math.radians(camera.field_of_view) / 2)
    projection = np.zeros((4, 4))
    projection[0, 0] = 1 / half_width
    projection[1, 1] = aspect / half_width
    projection[2, 2] = (FAR + NEAR) / (NEAR - FAR)
    projection[2, 3] = 2 * FAR * NEAR / (NEAR - FAR)
    projection[3, 2] = -1.0
    return projection


def _flat(corners: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Triangles ``corners`` (n, 3, 3) as vertices of position, face normal and color."""
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    colors = np.broadcast_to(np.asarray(colors, float), (len(corners), 3))
    return np.concatenate([corners, np.repeat(normals[:, None], 3, axis=1),
                           np.repeat(colors[:, None], 3, axis=1)], axis=2).reshape(-1, 9)


def _quad(corner, edge_a, edge_b, color) -> np.ndarray:
    """A parallelogram, facing ``edge_a`` x ``edge_b``."""
    corner, edge_a, edge_b = (np.asarray(v, float) for v in (corner, edge_a, edge_b))
    corners = np.array([[corner, corner + edge_a, corner + edge_a + edge_b],
                        [corner, corner + edge_a + edge_b, corner + edge_b]])
    return _flat(corners, color)


def _side_of(y: np.ndarray) -> np.ndarray:
    """How far toward orange's half: 0 at blue's end, 1 at orange's, smooth across the middle."""
    t = np.clip((np.asarray(y) + 1500.0) / 3000.0, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _wall_color(points: np.ndarray) -> np.ndarray:
    side = _side_of(points[:, 1])[:, None]
    color = np.array(WALL[Team.BLUE]) * (1 - side) + np.array(WALL[Team.ORANGE]) * side
    in_goal = np.abs(points[:, 1]) > _BACK_WALL_Y + 1.0
    color[in_goal] = np.where(points[in_goal, 1:2] > 0, GOAL[Team.ORANGE], GOAL[Team.BLUE])
    color[(points[:, 2] > _CEILING_Z - 100.0) & ~in_goal] = CEILING
    return color


def _arena_vertices(meshes: Sequence[TriangleMesh]) -> np.ndarray:
    parts = []
    for mesh in meshes:
        corners = mesh.vertices[mesh.triangles]
        parts.append(_flat(corners, _wall_color(corners.mean(axis=1))))
    strips = np.arange(-12 * _STRIP, 12 * _STRIP, _STRIP)
    for index, y in enumerate(strips):
        parts.append(_quad((-_SIDE_WALL_X, y, 0.0), (2 * _SIDE_WALL_X, 0.0, 0.0),
                           (0.0, _STRIP, 0.0), GRASS[index % 2]))
        for x, edges in ((_SIDE_WALL_X, ((0.0, 0.0, _CEILING_Z), (0.0, _STRIP, 0.0))),
                         (-_SIDE_WALL_X, ((0.0, _STRIP, 0.0), (0.0, 0.0, _CEILING_Z)))):
            if abs(y + _STRIP / 2) < _BACK_WALL_Y:
                color = _wall_color(np.array([[x, y + _STRIP / 2, 0.0]]))[0]
                parts.append(_quad((x, y, 0.0), *edges, color))
    parts.append(_quad((-_SIDE_WALL_X, -20.0, 1.0), (2 * _SIDE_WALL_X, 0.0, 0.0),
                       (0.0, 40.0, 0.0), FIELD_LINE))
    return np.concatenate(parts)


def _box(center, size, color) -> np.ndarray:
    """A box, its faces outward."""
    center, half = np.asarray(center, float), np.asarray(size, float) / 2
    parts = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            normal = np.zeros(3)
            normal[axis] = sign
            a, b = np.zeros(3), np.zeros(3)
            a[(axis + 1) % 3], b[(axis + 2) % 3] = 2 * half[(axis + 1) % 3], 2 * half[(axis + 2) % 3]
            if sign < 0:
                a, b = b, a
            corner = center + normal * half - a / 2 - b / 2
            parts.append(_quad(corner, a, b, color))
    return np.concatenate(parts)


def _car(color) -> np.ndarray:
    """An Octane-sized car in its own axes: x forward, y right, z up, the origin where RocketSim
    puts the car's. The body is the hitbox, reaching down over the wheels."""
    wheels = [_box((x, side * y, z), (2 * radius, 16.0, 2 * radius), TIRES)
              for x, y, z, radius in ((51.25, 36.0, -4.5, 12.5), (-33.75, 36.0, -2.0, 15.0))
              for side in (-1.0, 1.0)]
    return np.concatenate([_box((13.88, 0.0, 16.0), (120.5, 86.7, 48.0), color),
                           _box((-8.0, 0.0, 46.0), (56.0, 66.0, 12.0), CAR_WINDOWS), *wheels])


def _cylinder(radius: float, height: float, color, segments: int = 24) -> np.ndarray:
    """An upright cylinder standing on the origin, with its top."""
    angles = np.linspace(0.0, 2 * math.pi, segments + 1)
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)], 1)
    top = ring + (0.0, 0.0, height)
    sides = [_quad(ring[i], ring[i + 1] - ring[i], top[i] - ring[i], color)
             for i in range(segments)]
    centre = np.array([0.0, 0.0, height])
    caps = _flat(np.stack([np.repeat(centre[None], segments, 0), top[:-1], top[1:]], 1), color)
    return np.concatenate([*sides, caps])


def _sphere(radius: float, color, subdivisions: int) -> np.ndarray:
    """A sphere centred on the origin, an icosahedron subdivided, its normals smooth."""
    t = (1 + 5 ** 0.5) / 2
    points = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t),
              (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    triangles = np.array([[points[i] for i in face] for face in faces], float)
    triangles /= np.linalg.norm(triangles, axis=2, keepdims=True)
    for _ in range(subdivisions):
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        ab, bc, ca = (_unit(a + b), _unit(b + c), _unit(c + a))
        triangles = np.concatenate([np.stack(face, 1) for face in
                                    ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    normals = triangles.reshape(-1, 3)
    colors = np.broadcast_to(np.asarray(color, float), normals.shape)
    return np.concatenate([normals * radius, normals, colors], axis=1)


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
