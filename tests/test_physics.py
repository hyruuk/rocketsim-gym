"""The homemade arena as RocketSim plays it: rest heights, walls, goals, posts, and no leaks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rocketsim_gym.arena_geometry import HOMEMADE_ARENA_FOLDER, MANIFEST_FILE_NAME, read_manifest
from rocketsim_gym.engine import check_collision_meshes_folder, load_rocketsim, new_soccar_arena

rs = pytest.importorskip("RocketSim")


@pytest.fixture(scope="module", autouse=True)
def homemade_arena_loaded():
    load_rocketsim(HOMEMADE_ARENA_FOLDER, read_manifest(HOMEMADE_ARENA_FOLDER / MANIFEST_FILE_NAME))


def _car(arena, position, yaw=0.0, boost=100.0):
    car = arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
    car.set_state(rs.CarState(pos=rs.Vec(*position), rot_mat=rs.Angle(yaw, 0, 0).as_rot_mat(),
                              boost=boost))
    return car


def _ball_after(position, velocity, ticks):
    arena = new_soccar_arena()
    arena.ball.set_state(rs.BallState(pos=rs.Vec(*position), vel=rs.Vec(*velocity)))
    arena.step(ticks)
    return arena, arena.ball.get_state()


def test_the_ball_comes_to_rest_where_rocket_league_rests_it():
    _, ball = _ball_after((1000, 1000, 300), (0, 0, -1), 120 * 20)
    assert ball.pos.z == pytest.approx(rs.RLConst.BALL_REST_Z, abs=0.05)


def test_an_octane_rests_on_four_wheels():
    arena = new_soccar_arena()
    x, y, yaw = rs.RLConst.CAR_SPAWN_LOCATIONS_SOCCAR[0]
    car = _car(arena, (x, y, 17), yaw)
    arena.step(240)
    assert car.get_state().pos.z == pytest.approx(rs.RLConst.CAR_SPAWN_REST_Z, abs=0.5)
    assert all(car.get_state().wheels_with_contact)


@pytest.mark.parametrize("start, yaw", [
    ((2500, 0, 17), 0.0),                              # a side wall
    ((-2500, 0, 17), math.pi),
    ((2000, 3000, 17), math.pi / 4),                   # a corner wall
    ((-2000, -3000, 17), -3 * math.pi / 4),
    ((2000, 3500, 17), math.pi / 2),                   # a back wall, beside the goal
    ((-2000, -3500, 17), -math.pi / 2),
], ids=["side+", "side-", "corner++", "corner--", "back+", "back-"])
def test_a_car_drives_up_a_wall_onto_the_ceiling(start, yaw):
    arena = new_soccar_arena()
    car = _car(arena, start, yaw)
    car.set_controls(rs.CarControls(throttle=1, boost=True))
    on_wall_with_four_wheels, highest = 0, 0.0
    for _ in range(120 * 3):
        arena.step(1)
        state = car.get_state()
        highest = max(highest, state.pos.z)
        on_wall_with_four_wheels += state.pos.z > 300 and all(state.wheels_with_contact)
    assert highest > 2000
    assert on_wall_with_four_wheels > 100


@pytest.mark.parametrize("side", [1, -1])
def test_a_shot_scores_once_for_the_team_attacking_that_goal(side):
    arena, ball = _ball_after((0, side * 4000, 93.15), (0, side * 2000, 0), 120 * 3)
    assert (arena.blue_score, arena.orange_score) == ((1, 0) if side > 0 else (0, 1))
    assert abs(ball.pos.y) > 5215 and abs(ball.pos.x) < 892.755


@pytest.mark.parametrize("position, expected_direction", [
    ((892.755 + 40, 4900, 300), (1, -1, 0)),     # past the post's outer edge: out and back
    ((892.755 - 40, 4900, 300), (-1, -1, 0)),    # past its inner edge: in and back
    ((0, 4900, 642.775 + 40), (0, -1, 1)),       # over the crossbar's top edge: up and back
    ((0, 4900, 642.775 - 40), (0, -1, -1)),      # under it: down and back
], ids=["post outside", "post inside", "crossbar above", "crossbar below"])
def test_the_goal_frame_deflects_a_ball_that_clips_it(position, expected_direction):
    arena, ball = _ball_after(position, (0, 3000, 0), 15)
    for velocity, direction in zip(ball.vel.as_tuple(), expected_direction):
        assert direction == 0 or math.copysign(1, velocity) == direction
    assert arena.blue_score == 0


def test_nothing_leaves_the_arena():
    rng = np.random.default_rng(0)

    def inside(margin):
        while True:
            point = rng.uniform([-3800, -4800, 150], [3800, 4800, 1800])
            if abs(point[0]) + abs(point[1]) < 8064 - 512 - margin:
                return point

    def has_left(position):
        x, y, z = position
        in_goal = abs(x) <= 892.755 + 30 and z <= 642.775 + 30
        return (abs(x) > 4101 or not -5 < z < 2053 or abs(x) + abs(y) > 8074
                or abs(y) > (6005 if in_goal else 5125))

    for _ in range(1000):
        arena = new_soccar_arena()
        speed = rng.uniform(1000, 6000)
        direction = rng.normal(size=3)
        arena.ball.set_state(rs.BallState(pos=rs.Vec(*inside(150)),
                                          vel=rs.Vec(*(speed * direction / np.linalg.norm(direction)))))
        car = _car(arena, inside(250), float(rng.uniform(-math.pi, math.pi)))
        car.set_controls(rs.CarControls(throttle=float(rng.uniform(-1, 1)),
                                        steer=float(rng.uniform(-1, 1)), boost=True,
                                        jump=bool(rng.integers(2))))
        for _ in range(60):
            arena.step(4)
            assert not has_left(arena.ball.get_state().pos.as_tuple())
            assert not has_left(car.get_state().pos.as_tuple())


def test_loading_refuses_a_folder_whose_files_differ_from_its_manifest(tmp_path):
    manifest = read_manifest(HOMEMADE_ARENA_FOLDER / MANIFEST_FILE_NAME)
    (tmp_path / "soccar").mkdir()
    for relative_path in manifest:
        (tmp_path / relative_path).write_bytes((HOMEMADE_ARENA_FOLDER / relative_path).read_bytes())
    check_collision_meshes_folder(tmp_path, manifest)
    (tmp_path / "soccar" / "extra.cmf").write_bytes(b"")
    with pytest.raises(ValueError, match="not listed"):
        check_collision_meshes_folder(tmp_path, manifest)


def test_one_process_loads_one_folder(tmp_path):
    with pytest.raises(ValueError, match="one process takes one folder"):
        load_rocketsim(tmp_path, {})
