"""The frames: what lands where on the screen, that drawing changes nothing, and that the same
match draws the same pixels in another process."""

import hashlib
import os
import subprocess
import sys
import textwrap
import threading

import numpy as np
import pytest

from rocketsim_gym.arena_geometry import (HOMEMADE_ARENA_FOLDER, collision_mesh_files,
                                          read_collision_mesh_file)
from rocketsim_gym.camera import Camera
from rocketsim_gym.controls import BUTTON_COUNT
from rocketsim_gym.engine import SoccarArena
from rocketsim_gym.env import SoccarEnv
from rocketsim_gym.match import Team
from rocketsim_gym.renderer import CarPose, Renderer, Scene

WIDTH, HEIGHT = 320, 180
NO_PADS = (False,) * 34  # soccar's boost pads, none up
# A camera at the blue end of the field, looking down it toward +y.
CAMERA = Camera((0.0, -1500.0, 150.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 110.0)


@pytest.fixture(scope="module")
def renderer():
    meshes = [read_collision_mesh_file(path) for path in collision_mesh_files(HOMEMADE_ARENA_FOLDER)]
    try:
        renderer = Renderer(WIDTH, HEIGHT, meshes, SoccarArena().pad_spots())
    except RuntimeError as error:
        pytest.skip(str(error))
    yield renderer
    renderer.close()


def _drawn(renderer, ball=None, cars=()):
    """The pixels the ball and cars change on an empty field, and the frame they are in."""
    empty = renderer.draw(Scene(CAMERA, None, (), NO_PADS))
    frame = renderer.draw(Scene(CAMERA, ball, cars, NO_PADS))
    return (frame != empty).any(axis=2), frame


def _around(changed, row, column, reach=30):
    """``changed`` within ``reach`` pixels of ``(row, column)``. GitHub's macOS runner, a
    virtual machine, also changes a few pixels of the ceiling when it draws the ball."""
    around = np.zeros_like(changed)
    around[max(0, row - reach):row + reach + 1, max(0, column - reach):column + reach + 1] = True
    return changed & around


def test_a_frame_has_the_size_asked_for(renderer):
    frame = renderer.draw(Scene(CAMERA, None, (), NO_PADS))
    assert frame.shape == (HEIGHT, WIDTH, 3) and frame.dtype == np.uint8


def test_the_ball_ahead_is_at_the_centre_of_the_screen(renderer):
    changed = _drawn(renderer, ball=(0.0, 0.0, 150.0))[0]
    rows, columns = np.nonzero(_around(changed, HEIGHT // 2, WIDTH // 2))
    where = f"{len(columns)} pixels changed, rows {rows.min()}-{rows.max()}, columns {columns.min()}-{columns.max()}"
    assert len(columns) > 50, where
    assert abs(columns.mean() - WIDTH / 2) < 2 and abs(rows.mean() - HEIGHT / 2) < 2, where


def test_the_screens_right_is_the_cars_right(renderer):
    """Facing +y, a car's right is -x in Rocket League's world: a ball there is on the right
    of the screen."""
    _, columns = np.nonzero(_drawn(renderer, ball=(-600.0, 0.0, 150.0))[0])
    assert columns.mean() > WIDTH * 0.6


def test_cars_are_drawn_in_their_team_color_unless_demolished(renderer):
    def drawn(team, visible=True):
        car = CarPose(team, (0.0, -1100.0, 17.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0),
                      (0.0, 0.0, 1.0), visible)
        return _drawn(renderer, cars=(car,))

    (blue_pixels, blue_frame), (orange_pixels, orange_frame) = drawn(Team.BLUE), drawn(Team.ORANGE)
    blue = blue_frame[blue_pixels].astype(int).mean(axis=0)
    orange = orange_frame[orange_pixels].astype(int).mean(axis=0)
    assert blue_pixels.sum() > 300
    assert blue[2] > blue[0] + 40 and orange[0] > orange[2] + 40
    assert not drawn(Team.BLUE, visible=False)[0].any()


def test_a_scene_with_other_boost_pads_is_refused(renderer):
    with pytest.raises(ValueError, match="boost pads"):
        renderer.draw(Scene(CAMERA, None, (), NO_PADS[:-1]))


def test_the_renderer_names_the_opengl_renderer(renderer):
    assert renderer.gl_renderer


def test_opengl_runs_off_the_callers_thread(renderer, monkeypatch):
    # A window drawn with OpenGL on the caller's thread (rocketsim-play's, through SDL) stops
    # showing frames on Linux when another context is made current there.
    threads = []
    draw = renderer._draw
    monkeypatch.setattr(renderer, "_draw",
                        lambda scene: threads.append(threading.get_ident()) or draw(scene))
    renderer.draw(Scene(CAMERA, None, (), NO_PADS))
    assert threads and threads[0] != threading.get_ident()


@pytest.mark.parametrize("size", [(0, 180), (320, -1), (320.5, 180), (True, 180)])
def test_a_frame_size_is_a_positive_whole_number(size):
    with pytest.raises(ValueError):
        Renderer(*size, [], [])


def _play(render, steps=400):
    try:
        env = SoccarEnv(render_mode="rgb_array" if render else None, render_width=WIDTH,
                        render_height=HEIGHT)
    except RuntimeError as error:
        pytest.skip(str(error))
    actions = np.random.default_rng(5)
    observations = [env.reset(seed=5)[0]]
    for _ in range(steps):
        observations.append(env.step(actions.integers(0, 2, BUTTON_COUNT, dtype=np.int8))[0])
        if render:
            frame = env.render()
            assert env.render() is frame and not frame.flags.writeable
    env.close()
    return np.array(observations)


def test_drawing_changes_nothing_in_the_match():
    assert np.array_equal(_play(render=True), _play(render=False))


_FRAMES_DIGEST = textwrap.dedent("""
    import hashlib
    import numpy as np
    from rocketsim_gym.env import SoccarEnv

    env = SoccarEnv(render_mode="rgb_array", render_width=320, render_height=180)
    actions, digest = np.random.default_rng(9), hashlib.sha256()
    env.reset(seed=9)
    for _ in range(400):
        env.step(actions.integers(0, 2, 12, dtype=np.int8))
        digest.update(env.render().tobytes())
    print(digest.hexdigest())
""")


def test_a_match_draws_the_same_pixels_in_another_process():
    _play(render=True, steps=1)  # skips when this machine has no OpenGL 3.3
    digests = [subprocess.run([sys.executable, "-c", _FRAMES_DIGEST], capture_output=True,
                              text=True, check=True,
                              env={**os.environ, "PYTHONHASHSEED": seed}).stdout
               for seed in ("1", "2")]
    assert digests[0] == digests[1] and len(digests[0].strip()) == len(hashlib.sha256().hexdigest())


def test_a_boosting_car_trails_a_flame(renderer):
    def car(boosting):
        return CarPose(Team.BLUE, (0.0, -1100.0, 17.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0),
                       (0.0, 0.0, 1.0), boosting=boosting)

    still = renderer.draw(Scene(CAMERA, None, (car(False),), NO_PADS))
    boosting = renderer.draw(Scene(CAMERA, None, (car(True),), NO_PADS))
    assert (still != boosting).any(axis=2).sum() > 50


def test_a_ball_in_the_air_casts_a_shadow_on_the_floor(renderer):
    changed, frame = _drawn(renderer, ball=(0.0, -900.0, 600.0))
    empty = renderer.draw(Scene(CAMERA, None, (), NO_PADS))
    darker = changed & (frame.astype(int).sum(axis=2) < empty.astype(int).sum(axis=2) - 30)
    rows = np.nonzero(darker)[0]
    assert len(rows) > 100 and rows.min() > HEIGHT / 2


def test_the_ball_turns_with_its_axes(renderer):
    def ball(axes):
        return renderer.draw(Scene(CAMERA, (0.0, -900.0, 150.0), (), NO_PADS, axes))

    turned = ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    assert not np.array_equal(ball(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))), ball(turned))
