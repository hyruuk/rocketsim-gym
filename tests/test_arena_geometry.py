"""The homemade arena's mesh, and RocketSim's collision-mesh files: no RocketSim needed."""

from __future__ import annotations

import collections
import struct

import numpy as np
import pytest

from rocketsim_gym.arena_geometry import (HOMEMADE_ARENA_FILE_NAME, HOMEMADE_ARENA_FOLDER,
                                          MANIFEST_FILE_NAME, SOCCAR_FOLDER_NAME,
                                          SoccarArenaDimensions, TriangleMesh,
                                          build_homemade_soccar_arena, collision_mesh_manifest,
                                          read_collision_mesh_file, read_manifest,
                                          write_collision_mesh_file)

DIMENSIONS = SoccarArenaDimensions()


@pytest.fixture(scope="module")
def arena() -> TriangleMesh:
    return build_homemade_soccar_arena()


def _edge_uses(mesh: TriangleMesh) -> tuple[collections.Counter, collections.Counter]:
    undirected, directed = collections.Counter(), collections.Counter()
    for triangle in mesh.triangles.tolist():
        for a, b in zip(triangle, triangle[1:] + triangle[:1]):
            undirected[tuple(sorted((a, b)))] += 1
            directed[(a, b)] += 1
    return undirected, directed


def test_the_mesh_meets_rocketsims_planes_without_gaps(arena):
    """Every open edge lies on the floor or a side wall, which RocketSim adds as planes."""
    undirected, directed = _edge_uses(arena)
    assert max(undirected.values()) == 2
    vertices = arena.vertices
    for a, b in (edge for edge, uses in undirected.items() if uses == 1):
        on_floor = vertices[a][2] == 0 and vertices[b][2] == 0
        on_side_wall = abs(vertices[a][0]) == abs(vertices[b][0]) == DIMENSIONS.side_wall_x
        assert on_floor or on_side_wall, (vertices[a], vertices[b])
    assert max(directed.values()) == 1, "two neighbouring triangles are wound opposite ways"


def test_every_triangle_faces_the_playable_volume(arena):
    corners = arena.vertices[arena.triangles]
    centroids = corners.mean(axis=1)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    in_goal = np.abs(centroids[:, 1]) > DIMENSIONS.back_wall_y
    inside = np.where(in_goal[:, None],
                      np.stack([np.zeros(len(centroids)),
                                np.sign(centroids[:, 1]) * 5560.0,
                                np.full(len(centroids), 320.0)], axis=1),
                      np.array([0.0, 0.0, 1000.0]))
    assert (np.einsum("ij,ij->i", normals, inside - centroids) > 0).all()


def test_the_arena_stays_inside_the_field(arena):
    low, high = arena.vertices.min(axis=0), arena.vertices.max(axis=0)
    assert low.tolist() == [-DIMENSIONS.side_wall_x, -DIMENSIONS.goal_back_y, 0.0]
    assert high.tolist() == [DIMENSIONS.side_wall_x, DIMENSIONS.goal_back_y, DIMENSIONS.ceiling_z]


def test_the_shipped_arena_is_what_the_builder_makes(arena):
    folder = HOMEMADE_ARENA_FOLDER
    shipped = read_collision_mesh_file(folder / SOCCAR_FOLDER_NAME / HOMEMADE_ARENA_FILE_NAME)
    assert np.array_equal(shipped.triangles, arena.triangles)
    assert np.abs(shipped.vertices - arena.vertices).max() < 1e-3
    assert read_manifest(folder / MANIFEST_FILE_NAME) == collision_mesh_manifest(folder)


def test_a_mesh_file_reads_back_as_written(arena, tmp_path):
    path = tmp_path / "arena.cmf"
    write_collision_mesh_file(arena, path)
    assert np.array_equal(read_collision_mesh_file(path).triangles, arena.triangles)
    assert np.abs(read_collision_mesh_file(path).vertices - arena.vertices).max() < 1e-3


@pytest.mark.parametrize("damage, problem", [
    (lambda raw: raw[:-4], "bytes, but its counts make"),
    (lambda raw: struct.pack("<ii", 0, 3) + raw[8:], "is not a collision mesh"),
    (lambda raw: raw[:8] + struct.pack("<i", 10**6) + raw[12:], "a vertex the file does not"),
    (lambda raw: raw[:3], "too short"),
])
def test_reading_refuses_a_file_rocketsim_would_refuse(arena, tmp_path, damage, problem):
    path = tmp_path / "arena.cmf"
    write_collision_mesh_file(arena, path)
    path.write_bytes(damage(path.read_bytes()))
    with pytest.raises(ValueError, match=problem):
        read_collision_mesh_file(path)
