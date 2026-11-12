"""Snapshots: a match at the end of a step, and its bytes. Unpacked, a snapshot gives the
observation, the sample and the frame the snapshot it was packed from gives, and none of it
needs RocketSim."""

import dataclasses
import subprocess
import sys
from types import MappingProxyType

import numpy as np
import pytest

from rocketsim_gym.camera import CameraSettings
from rocketsim_gym.controls import ANALOG_HIGH, ANALOG_LOW, BUTTON_COUNT, Button, Controls
from rocketsim_gym.env import PLAYER, SoccarEnv
from rocketsim_gym.match import TICKS_PER_SECOND, Phase, Team
from rocketsim_gym.simulation import pad_spots
from rocketsim_gym.snapshot import FORMAT_VERSION, Snapshot
from rocketsim_gym.view import SoccarView

WIDTH, HEIGHT = 160, 90
#: The most a server frame may take, to travel in one datagram with room for the room's state.
DATAGRAM_BUDGET = 1200


def _env(team_size=1, input_mode="digital", render=False):
    controller = {"deadzone": 0.1, "steering_sensitivity": 1.3} if input_mode == "analog" else None
    try:
        return SoccarEnv(match_seconds=20, team_size=team_size, input_mode=input_mode,
                         controller=controller, render_mode="rgb_array" if render else None,
                         render_width=WIDTH, render_height=HEIGHT)
    except RuntimeError as error:
        pytest.skip(str(error))


def _actions(input_mode, seed):
    rng = np.random.default_rng(seed)
    while True:
        if input_mode == "digital":
            yield rng.integers(0, 2, BUTTON_COUNT, dtype=np.int8)
        else:
            yield rng.uniform(ANALOG_LOW, ANALOG_HIGH).astype(np.float32)


@pytest.mark.parametrize("team_size", [1, 3])
@pytest.mark.parametrize("input_mode", ["digital", "analog"])
def test_an_unpacked_snapshot_gives_the_same_observation_sample_and_info(team_size, input_mode):
    env = _env(team_size, input_mode)
    env.reset(seed=4)
    actions, phases = _actions(input_mode, 4), set()
    for _ in range(2500):
        terminated = env.step(next(actions))[2]
        live = env._snapshot
        phases.add(live.match.phase)
        packed = live.pack()
        unpacked = Snapshot.unpack(packed)
        assert unpacked.pack() == packed
        observation, sample, info = env._observation(), env.sample(), env._info()
        env._snapshot = unpacked
        assert env._observation().tobytes() == observation.tobytes()
        assert repr(env.sample()) == repr(sample)
        assert env._info() == info
        env._snapshot = live
        if terminated:
            break
    assert {Phase.COUNTDOWN, Phase.KICKOFF, Phase.PLAY} <= phases


def test_every_field_survives_its_bytes():
    env = _env(team_size=2)
    env.reset(seed=1)
    actions = _actions("digital", 1)
    for _ in range(300):
        env.step(next(actions))
    snapshot = env._snapshot
    demolished = snapshot.cars[3].car
    match = dataclasses.replace(
        snapshot.match, phase=Phase.GOAL, phase_ticks_left=200, overtime=True,
        overtime_ticks=1234, score=MappingProxyType({Team.BLUE: 3, Team.ORANGE: 7}),
        last_goal=Team.ORANGE, ticks_since_go=4321,
        respawn_ticks_left=MappingProxyType({demolished: 250}))
    controls = Controls(-0.37, 0.123456789012345, -1.0, 0.0, -0.0, True, True, True)
    cars = tuple(dataclasses.replace(car, controls=controls, touched_ball=True,
                                     was_demolished=True, is_demolished=True)
                 if car.car == demolished else car for car in snapshot.cars)
    edited = dataclasses.replace(snapshot, step=987654, match=match, goal=Team.ORANGE, cars=cars,
                                 pads_active=(False, True) * 17)
    unpacked = Snapshot.unpack(edited.pack())
    assert unpacked.match == edited.match and unpacked.match.winner is None
    assert (unpacked.step, unpacked.goal, unpacked.pads_active) == (987654, Team.ORANGE,
                                                                  (False, True) * 17)
    assert unpacked.ball_in_play is None
    car = unpacked.car(demolished)
    assert car.controls == controls and repr(car.controls.roll) == "-0.0"
    assert car.touched_ball and car.was_demolished and car.is_demolished
    for name in ("position", "velocity", "angular_velocity", "forward", "right", "up"):
        assert getattr(car, name).dtype == np.float32
        assert np.array_equal(getattr(car, name), getattr(edited.car(demolished), name))


def test_a_3v3_snapshot_fits_in_one_datagram():
    env = _env(team_size=3)
    env.reset(seed=2)
    assert len(env._snapshot.pack()) < DATAGRAM_BUDGET


def test_unpacking_refuses_bytes_of_another_format_or_length():
    env = _env()
    env.reset(seed=2)
    packed = env._snapshot.pack()
    other_format = bytes([FORMAT_VERSION + 1]) + packed[1:]
    with pytest.raises(ValueError, match="format"):
        Snapshot.unpack(other_format)
    with pytest.raises(ValueError, match="cut short"):
        Snapshot.unpack(packed[:-1])
    with pytest.raises(ValueError, match="too many"):
        Snapshot.unpack(packed + b"\0")


def test_a_view_drawing_unpacked_snapshots_shows_the_env_s_frames():
    env = _env(render=True)
    view = SoccarView(PLAYER, CameraSettings(), (WIDTH, HEIGHT), pad_spots())
    env.reset(seed=8)
    view.start(Snapshot.unpack(env._snapshot.pack()))
    assert np.array_equal(view.frame(), env.render())
    actions = _actions("digital", 8)
    for _ in range(400):
        action = next(actions)
        env.step(action)
        view.press_ball_cam(bool(action[Button.BALL_CAM]))
        view.advance(Snapshot.unpack(env._snapshot.pack()), 2 / TICKS_PER_SECOND)
        assert np.array_equal(view.frame(), env.render())
    view.close()
    env.close()


def test_snapshots_and_views_need_no_rocketsim():
    code = ("import sys; sys.modules['RocketSim'] = None; "
            "import rocketsim_gym.snapshot, rocketsim_gym.view")
    subprocess.run([sys.executable, "-c", code], check=True)
