"""Check a folder of soccar collision meshes, and write its manifest.

    python -m rocketsim_gym.check_collision_meshes FOLDER

For each ``FOLDER/soccar/*.cmf``: its triangle and vertex counts, its bounding box in UU, its
SHA-256, and whether its RocketSim hash is one of the meshes dumped from Rocket League that
RocketSim knows. Then RocketSim loads the folder, and a ball, an Octane and a shot at each goal
show how the arena plays. The report is printed and written to ``FOLDER/check_report.json``,
the manifest (each file's SHA-256) to ``FOLDER/manifest.json``.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .arena_geometry import (collision_mesh_files, collision_mesh_manifest, file_sha256,
                             read_collision_mesh_file, rocketsim_mesh_hash, write_manifest)
from .engine import load_rocketsim, new_soccar_arena

#: The hashes of the 16 soccar meshes dumped from Rocket League, as RocketSim lists them
#: (``src/RocketSim.cpp``, ``MeshHashSet``).
ROCKET_LEAGUE_SOCCAR_MESH_HASHES = frozenset({
    0xA160BAF9, 0x2811EEE8, 0xB81AC8B9, 0x760358D3, 0x73AE4940, 0x918F4A4E, 0x1F8EE550,
    0x255BA8C1, 0x14B84668, 0xEC759EBF, 0x94FB0D5C, 0xDEA07102, 0xBD4FBEA8, 0x39A47F63,
    0x3D79D25D, 0xD84C7A68,
})
#: The name of the report written into the checked folder.
REPORT_FILE_NAME = "check_report.json"


def describe_mesh_file(path: Path, folder: Path) -> dict:
    """One mesh file's counts, bounding box (UU), SHA-256 and RocketSim hash."""
    mesh = read_collision_mesh_file(path)
    mesh_hash = rocketsim_mesh_hash(mesh)
    return {
        "path": path.relative_to(folder).as_posix(),
        "triangles": int(len(mesh.triangles)),
        "vertices": int(len(mesh.vertices)),
        "bounds_uu": [np.round(mesh.vertices.min(axis=0), 2).tolist(),
                      np.round(mesh.vertices.max(axis=0), 2).tolist()],
        "sha256": file_sha256(path),
        "rocketsim_hash": f"0x{mesh_hash:08X}",
        "is_known_rocket_league_mesh": mesh_hash in ROCKET_LEAGUE_SOCCAR_MESH_HASHES,
    }


def measure_arena() -> dict:
    """How the loaded soccar arena plays: where a ball and an Octane come to rest, which team a
    shot at each goal scores for and on which tick, and how high a car drives up a side wall."""
    import RocketSim as rs

    arena = new_soccar_arena()
    arena.ball.set_state(rs.BallState(pos=rs.Vec(0, 0, 300), vel=rs.Vec(0, 0, -1)))
    arena.step(120 * 20)
    ball_rest_z = arena.ball.get_state().pos.z
    car = arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
    x, y, yaw = rs.RLConst.CAR_SPAWN_LOCATIONS_SOCCAR[0]
    car.set_state(rs.CarState(pos=rs.Vec(x, y, 17), rot_mat=rs.Angle(yaw, 0, 0).as_rot_mat()))
    arena.step(120 * 2)
    car_state = car.get_state()
    shots = {}
    for side, name in ((1, "shot_at_positive_y_goal"), (-1, "shot_at_negative_y_goal")):
        arena = new_soccar_arena()
        arena.ball.set_state(rs.BallState(pos=rs.Vec(0, side * 4000, 93.15),
                                          vel=rs.Vec(0, side * 2000, 0)))
        for tick in range(1, 361):
            arena.step(1)
            if arena.blue_score or arena.orange_score:
                break
        shots[name] = {"blue_score": arena.blue_score, "orange_score": arena.orange_score,
                       "tick": tick}
    arena = new_soccar_arena()
    car = arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
    car.set_state(rs.CarState(pos=rs.Vec(2500, 0, 17), boost=100))
    car.set_controls(rs.CarControls(throttle=1, boost=True))
    highest_z = 0.0
    for _ in range(120 * 3):
        arena.step(1)
        highest_z = max(highest_z, car.get_state().pos.z)
    return {"ball_rest_z": round(ball_rest_z, 3),
            "car_rest_z": round(car_state.pos.z, 3),
            "car_rest_wheels_in_contact": int(sum(car_state.wheels_with_contact)),
            **shots,
            "side_wall_drive_highest_z": round(highest_z, 1)}


def check_collision_meshes(folder: Path) -> dict:
    """Check ``folder``'s soccar meshes, load them into RocketSim and measure the arena.

    :return: the report.
    :raises ValueError: no soccar mesh, or a file RocketSim would refuse.
    """
    folder = Path(folder).resolve()
    files = [describe_mesh_file(path, folder) for path in collision_mesh_files(folder)]
    if not files:
        raise ValueError(f"{folder}: no soccar collision mesh (soccar/*.cmf)")
    manifest = collision_mesh_manifest(folder)
    load_rocketsim(folder, manifest)
    return {
        "folder": str(folder),
        "rocketsim_version": importlib.metadata.version("rocketsim"),
        "platform": f"{platform.system()} {platform.machine()}",
        "files": files,
        "known_rocket_league_meshes": sum(f["is_known_rocket_league_mesh"] for f in files),
        "arena": measure_arena(),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m rocketsim_gym.check_collision_meshes FOLDER")
    checked_folder = Path(sys.argv[1])
    checked_report = check_collision_meshes(checked_folder)
    write_manifest(checked_folder, collision_mesh_manifest(checked_folder))
    (checked_folder / REPORT_FILE_NAME).write_text(json.dumps(checked_report, indent=2) + "\n")
    print(json.dumps(checked_report, indent=2))
