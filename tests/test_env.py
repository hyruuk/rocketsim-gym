"""The env: Gymnasium's contract, the match it plays, what it logs, and that a match replays."""

import os
import subprocess
import sys
import textwrap

import gymnasium
import numpy as np
import pytest
import RocketSim as rs
from gymnasium.utils.env_checker import check_env

import rocketsim_gym  # noqa: F401  (registers RocketSimSoccar-v0)
from rocketsim_gym.arena_geometry import HOMEMADE_ARENA_FOLDER, MANIFEST_FILE_NAME, read_manifest
from rocketsim_gym.controls import ANALOG_HIGH, ANALOG_LOW, BUTTON_COUNT, Button, analog_from_buttons
from rocketsim_gym.env import OBSERVATION_SIZE, SAMPLE_FIELDS, SoccarEnv
from rocketsim_gym.match import COUNTDOWN_TICKS

IDLE = np.zeros(BUTTON_COUNT, np.int8)


def press(*buttons):
    action = IDLE.copy()
    action[list(buttons)] = 1
    return action


# A physics state has no natural bound; Gymnasium's own MuJoCo envs leave theirs infinite too.
@pytest.mark.filterwarnings("ignore:.*Box observation space m")
def test_it_passes_gymnasiums_env_checker():
    check_env(SoccarEnv(match_seconds=5), skip_render_check=True)


def test_gymnasium_makes_it_by_its_id():
    env = gymnasium.make("RocketSimSoccar-v0", match_seconds=5)
    observation, _ = env.reset(seed=0)
    assert observation.shape == (OBSERVATION_SIZE,) and observation.dtype == np.float32


def test_a_match_starts_with_a_kickoff_that_the_countdown_holds_still():
    env = SoccarEnv()
    first, info = env.reset(seed=0)
    assert info == {"phase": "countdown", "blue_score": 0, "orange_score": 0}
    assert first[0:3] == pytest.approx((0, 0, rs.RLConst.BALL_REST_Z))
    assert first[29:31] == pytest.approx(-first[9:11]) and first[10] < 0
    assert first[83:87].tolist() == [300, 0, 0, 0]
    full_speed = press(Button.THROTTLE, Button.BOOST)
    for _ in range(COUNTDOWN_TICKS // 2):
        observation, _, _, _, info = env.step(full_speed)
    assert np.array_equal(observation, first) and info["phase"] == "kickoff"
    for _ in range(30):
        observation, _, _, _, info = env.step(full_speed)
    assert np.linalg.norm(observation[12:14]) > 500 and env.sample()["blue_input_throttle"] == 1


@pytest.mark.parametrize("ticks_per_step", [1, 2, 8])
def test_a_match_against_the_bot_ends_and_its_rewards_add_up_to_the_score(ticks_per_step):
    env = SoccarEnv(match_seconds=20, ticks_per_step=ticks_per_step)
    env.reset(seed=1)
    total, terminated = 0.0, False
    while not terminated:
        _, reward, terminated, truncated, info = env.step(IDLE)
        total += reward
        assert not truncated
    assert info["phase"] == "ended" and info["blue_score"] != info["orange_score"]
    assert total == info["blue_score"] - info["orange_score"]
    assert env.sample()["seconds_left"] == 0
    with pytest.raises(ValueError, match="ended"):
        env.step(IDLE)


def test_ball_cam_switches_on_each_press_not_while_held():
    env = SoccarEnv()
    env.reset(seed=0)
    switches = []
    for action in (press(Button.BALL_CAM), press(Button.BALL_CAM), IDLE, press(Button.BALL_CAM)):
        env.step(action)
        switches.append(env.sample()["ball_cam"])
    assert switches == [False, False, False, True]


def test_sample_logs_the_same_fields_every_step_and_the_controls_each_car_obeyed():
    env = SoccarEnv()
    env.reset(seed=0)
    throttle = press(Button.THROTTLE)
    for step in range(COUNTDOWN_TICKS // 2 + 5):
        env.step(throttle)
        sample = env.sample()
        assert tuple(sample) == SAMPLE_FIELDS
        assert all(isinstance(value, (str, bool, int, float)) for value in sample.values())
        in_countdown = step < COUNTDOWN_TICKS // 2
        assert sample["blue_input_throttle"] == (0 if in_countdown else 1)
        assert sample["orange_input_throttle"] == (0 if in_countdown else 1)


def test_provenance_names_the_versions_and_the_meshes():
    provenance = SoccarEnv().provenance()
    assert provenance["collision_meshes"] == read_manifest(HOMEMADE_ARENA_FOLDER / MANIFEST_FILE_NAME)
    assert provenance["rocketsim_version"] == "2.2.1"
    assert (provenance["ticks_per_second"], provenance["ticks_per_step"]) == (120, 2)
    assert SoccarEnv(camera={"distance": 300}).provenance()["camera"]["distance"] == 300


@pytest.mark.parametrize("arguments", [
    {"match_seconds": 0}, {"match_seconds": -5}, {"match_seconds": True}, {"match_seconds": "300"},
    {"ticks_per_step": 0}, {"ticks_per_step": 1.5}, {"ticks_per_step": True},
    {"render_mode": "human"}, {"render_width": 0}, {"render_height": 72.5},
    {"camera": {"field_of_view": 130}}, {"camera": {"swivel_speed": 5}},
])
def test_arguments_out_of_range_are_refused(arguments):
    with pytest.raises(ValueError):
        SoccarEnv(**arguments)


def test_stepping_before_reset_is_refused():
    with pytest.raises(ValueError, match="reset"):
        SoccarEnv().step(IDLE)


def _play(seed, steps):
    env, actions = SoccarEnv(), np.random.default_rng(seed)
    observations = [env.reset(seed=seed)[0]]
    for _ in range(steps):
        observations.append(env.step(actions.integers(0, 2, BUTTON_COUNT, dtype=np.int8))[0])
    return np.array(observations)


def test_the_same_seed_and_actions_play_the_same_match_and_seeds_vary_the_kickoffs():
    assert np.array_equal(_play(4, 2000), _play(4, 2000))
    assert len({tuple(_play(seed, 0)[0][9:11]) for seed in range(8)}) > 1


_MATCH_DIGEST = textwrap.dedent("""
    import hashlib
    import numpy as np
    from rocketsim_gym.env import SoccarEnv

    env, actions = SoccarEnv(match_seconds=30), np.random.default_rng(7)
    digest = hashlib.sha256(env.reset(seed=3)[0].tobytes())
    terminated = False
    while not terminated:
        observation, reward, terminated, _, info = env.step(actions.integers(0, 2, 12, dtype=np.int8))
        digest.update(observation.tobytes() + repr((reward, sorted(env.sample().items()))).encode())
    print(digest.hexdigest(), info["blue_score"], info["orange_score"])
""")


def test_a_match_replays_identically_in_another_process():
    runs = [subprocess.run([sys.executable, "-c", _MATCH_DIGEST], capture_output=True, text=True,
                           check=True, env={**os.environ, "PYTHONHASHSEED": hash_seed}).stdout
            for hash_seed in ("1", "2")]
    assert runs[0] == runs[1]
    _, blue_score, orange_score = runs[0].split()
    assert int(blue_score) + int(orange_score) > 0


@pytest.mark.filterwarnings("ignore:.*Box observation space m")
def test_analog_mode_passes_gymnasiums_env_checker():
    check_env(SoccarEnv(match_seconds=5, input_mode="analog"), skip_render_check=True)


def test_analog_pushed_all_the_way_plays_the_same_match_as_digital():
    digital, analog = SoccarEnv(), SoccarEnv(input_mode="analog")
    observations = [digital.reset(seed=6)[0]], [analog.reset(seed=6)[0]]
    buttons = np.random.default_rng(6).integers(0, 2, (1500, BUTTON_COUNT), dtype=np.int8)
    for pressed in buttons:
        observations[0].append(digital.step(pressed)[0])
        observations[1].append(analog.step(analog_from_buttons(pressed.astype(bool)))[0])
    assert np.array_equal(np.array(observations[0]), np.array(observations[1]))


def test_an_analog_match_replays_identically():
    def play():
        env = SoccarEnv(input_mode="analog", controller={"deadzone": 0.1})
        actions = np.random.default_rng(8)
        observations = [env.reset(seed=8)[0]]
        for _ in range(1500):
            action = actions.uniform(ANALOG_LOW, ANALOG_HIGH).astype(np.float32)
            observations.append(env.step(action)[0])
        return np.array(observations)

    assert np.array_equal(play(), play())


@pytest.mark.parametrize("arguments", [
    {"input_mode": "keyboard"}, {"controller": {"deadzone": 1.5}}, {"controller": {"sensitivity": 1}},
])
def test_input_arguments_out_of_range_are_refused(arguments):
    with pytest.raises(ValueError):
        SoccarEnv(**arguments)


def test_provenance_names_the_input_mode_and_the_controller_settings():
    provenance = SoccarEnv(input_mode="analog", controller={"deadzone": 0.15}).provenance()
    assert provenance["input_mode"] == "analog" and provenance["controller"]["deadzone"] == 0.15
