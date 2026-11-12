"""A player's side of a LAN match: the frames it shows and when, the env's LAN mode with two
players on one server, and that each player's play redraws from its records alone, even where
RocketSim is missing."""

import hashlib
import pickle
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from rocketsim_gym.boost_pads import BOOST_PADS
from rocketsim_gym.controls import BUTTON_COUNT
from rocketsim_gym.env import LAN_SAMPLE_FIELDS, SoccarEnv
from rocketsim_gym.lan.client import (BURST, FIRST_TARGET, REPEATS_TO_RAISE, SKIP_AFTER_STEPS,
                                      FrameBuffer)
from rocketsim_gym.lan.protocol import RoomStatus, ServerFrame
from rocketsim_gym.lan.server import MatchServer
from rocketsim_gym.simulation import pad_spots

WIDTH, HEIGHT = 160, 90
SEATS = ["blue1", "orange1"]


def _frames(numbers, status=RoomStatus.PLAYING):
    return [ServerFrame(number, status, 1, b"") for number in numbers]


def _shown(buffer):
    frame = buffer.next()
    return None if frame is None else frame.number


def test_the_boost_pads_are_rocketsims():
    assert BOOST_PADS == pad_spots()


def test_frames_show_in_order_and_the_last_repeats_while_the_next_has_not_come():
    buffer = FrameBuffer()
    assert _shown(buffer) is None
    buffer.add(_frames([1, 2, 3]))
    assert [_shown(buffer), _shown(buffer), _shown(buffer)] == [3 - FIRST_TARGET, 3, 3]
    buffer.add(_frames([5]))
    assert _shown(buffer) == 5  # 4 was lost on the way


def test_a_burst_of_frames_jumps_to_the_target():
    buffer = FrameBuffer()
    buffer.add(_frames([1]))
    _shown(buffer)
    buffer.add(_frames(range(2, BURST + 5)))
    assert _shown(buffer) == BURST + 4 - FIRST_TARGET


def test_frames_ahead_for_long_skip_one():
    buffer = FrameBuffer()
    buffer.add(_frames([1, 2]))
    shown = [_shown(buffer)]
    buffer.add(_frames([3, 4]))  # two at once: two ahead of the target from now on
    newest = 4
    for _ in range(SKIP_AFTER_STEPS + 5):
        shown.append(_shown(buffer))
        newest += 1
        buffer.add(_frames([newest]))
    steps = np.diff(shown).tolist()
    assert set(steps) == {1, 2} and steps.count(2) == 1


def test_repeats_that_recur_raise_the_target():
    buffer, number = FrameBuffer(), 1
    buffer.add(_frames([number]))
    _shown(buffer)
    for _ in range(REPEATS_TO_RAISE):
        _shown(buffer)  # nothing came: a repeat
        number += 1
        buffer.add(_frames([number]))
        _shown(buffer)
    assert buffer.target == FIRST_TARGET + 1


def test_a_new_buffer_ignores_the_frames_of_a_finished_match():
    buffer = FrameBuffer()
    buffer.add(_frames([1, 2], RoomStatus.ENDED))
    assert _shown(buffer) is None
    buffer.add(_frames([3], RoomStatus.LOBBY) + _frames([4], RoomStatus.ENDED))
    assert [_shown(buffer), _shown(buffer)] == [3, 4]


@pytest.mark.parametrize("arguments, problem", [
    ({"seat": "blue1"}, "give its server"),
    ({"server": "auto", "seat": "blue2", "seats": ["blue1"]}, "seat 'blue2'"),
    ({"server": "auto", "seat": "blue1", "seats": "blue1"}, "seats"),
    ({"server": "auto", "seat": "blue1", "seats": ["blue1", "blue9"]}, "seats"),
    ({"server": "auto", "seat": "blue1", "seats": ["blue1"], "ticks_per_step": 1}, "LAN match"),
])
def test_lan_arguments_that_make_no_match_are_refused(arguments, problem):
    with pytest.raises(ValueError, match=problem):
        SoccarEnv(**arguments)


def test_a_match_on_this_computer_does_not_replay_from_records():
    with pytest.raises(ValueError, match="seed and actions"):
        SoccarEnv().replay_from([])


@pytest.fixture
def server(tmp_path):
    server = MatchServer(port=0, log_folder=tmp_path, seed=9, discovery_port=None,
                         report=lambda text: None)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    yield server
    server.submit("quit")
    thread.join(timeout=5)


def _lan_env(port, seat, render=True):
    try:
        return SoccarEnv(server=f"127.0.0.1:{port}", seat=seat, seats=SEATS, match_seconds=20,
                         render_mode="rgb_array" if render else None, render_width=WIDTH,
                         render_height=HEIGHT)
    except RuntimeError as error:
        pytest.skip(str(error))


_REDRAW = """
import hashlib, pickle, sys
from rocketsim_gym.env import SoccarEnv
with open(sys.argv[1], "rb") as file:
    played = pickle.load(file)
try:
    env = SoccarEnv(server="127.0.0.1:9", seat="blue1", seats=["blue1", "orange1"],
                    match_seconds=20, render_mode="rgb_array", render_width=160,
                    render_height=90)
except RuntimeError:
    print("no-opengl")
    sys.exit()
env.replay_from(played["records"])
env.reset()
states, pixels = hashlib.sha256(), hashlib.sha256()
for action in played["actions"]:
    observation = env.step(action)[0]
    states.update(observation.tobytes() + repr(sorted(env.sample().items())).encode())
    pixels.update(env.render().tobytes())
print(states.hexdigest(), pixels.hexdigest())
"""


def test_two_players_see_one_match_and_each_redraws_from_its_records(server, tmp_path):
    # Frames are compared between two fresh processes, as the renderer's own test does: a
    # virtual machine's GPU (GitHub's macOS runners) can draw differently in a process that
    # has drawn other frames before.
    blue = _lan_env(server.port, "blue1", render=False)
    orange = _lan_env(server.port, "orange1", render=False)
    blue.reset()
    orange.reset()
    rng = np.random.default_rng(4)
    played, states = {"records": [], "actions": []}, hashlib.sha256()
    # Each player's observation and room state, by the number of the server frame it showed.
    seen = {"blue1": {}, "orange1": {}}

    def step(name, env, action):
        observation, _, terminated, truncated, info = env.step(action)
        record = info["server_frame"]
        if len(record) > 1:
            seen[name][ServerFrame.unpack(record[1:]).number] = (observation.tobytes(),
                                                                 env.sample()["room"])
        if name == "blue1":
            played["records"].append(record)
            played["actions"].append(action)
            states.update(observation.tobytes() + repr(sorted(env.sample().items())).encode())
        return terminated, truncated

    deadline = time.monotonic() + 30
    while sum(room == "playing" for _, room in seen["blue1"].values()) < 300:
        assert time.monotonic() < deadline, "the match did not get going"
        for name, env in (("blue1", blue), ("orange1", orange)):
            step(name, env, rng.integers(0, 2, BUTTON_COUNT, dtype=np.int8))
        time.sleep(1 / 60)
    shared = set(seen["blue1"]) & set(seen["orange1"])
    assert len(shared) > 100
    assert all(seen["blue1"][number][0] == seen["orange1"][number][0] for number in shared)
    server.submit("end")
    for _ in range(600):
        terminated, truncated = step("blue1", blue, np.zeros(BUTTON_COUNT, np.int8))
        if truncated:
            break
        time.sleep(1 / 60)
    assert truncated and not terminated
    assert blue.sample()["room"] == "stopped"
    for env in (blue, orange):
        env.close()
    with open(tmp_path / "played.pickle", "wb") as file:
        pickle.dump(played, file)
    redraws = [subprocess.run([sys.executable, "-c", _REDRAW, str(tmp_path / "played.pickle")],
                              capture_output=True, text=True) for _ in range(2)]
    for redraw in redraws:
        assert redraw.returncode == 0, redraw.stderr
    if redraws[0].stdout.strip() == "no-opengl":
        pytest.skip("no OpenGL 3.3 context to draw the frames")
    first, second = (redraw.stdout.split() for redraw in redraws)
    assert first[0] == states.hexdigest()
    assert first == second


_WITHOUT_ROCKETSIM = """
import pickle, sys
sys.modules["RocketSim"] = None
from rocketsim_gym.env import LAN_SAMPLE_FIELDS, SoccarEnv
with open(sys.argv[1], "rb") as file:
    records = pickle.load(file)
env = SoccarEnv(server="127.0.0.1:9", seat="orange1", seats=["blue1", "orange1"])
env.replay_from(records)
env.reset()
observation = env.step([0] * 12)[0]
sample = env.sample()
print(int(observation.any()), sample["client"], sample["room"])
"""


def test_a_lan_player_needs_no_rocketsim(server, tmp_path):
    orange = _lan_env(server.port, "orange1", render=False)
    orange.reset()
    deadline = time.monotonic() + 10
    while True:
        record = orange.step(np.zeros(BUTTON_COUNT, np.int8))[4]["server_frame"]
        if len(record) > 1:
            break
        assert time.monotonic() < deadline, "no frame came"
        time.sleep(1 / 60)
    assert set(LAN_SAMPLE_FIELDS) <= set(orange.sample())
    orange.close()
    with open(tmp_path / "records.pickle", "wb") as file:
        pickle.dump([record], file)
    result = subprocess.run([sys.executable, "-c", _WITHOUT_ROCKETSIM,
                             str(tmp_path / "records.pickle")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["1", "live", "lobby"]
