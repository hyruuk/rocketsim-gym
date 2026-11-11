"""The ball-chaser: which way it steers, and that it reaches the ball from every kickoff."""

import numpy as np
import pytest

from rocketsim_gym.bot import GOAL_LINE_Y, ball_chaser
from rocketsim_gym.engine import KICKOFF_SPOT_COUNT, SoccarArena
from rocketsim_gym.match import Team

# A car at the centre facing +y; in Rocket League's coordinates its right is -x.
POSITION, FORWARD, RIGHT = np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0])


def _controls(ball_position):
    return ball_chaser(POSITION, FORWARD, RIGHT, np.array(ball_position, float), -GOAL_LINE_Y)


def test_it_drives_straight_at_a_ball_ahead_boosting():
    controls = _controls([0, 1500, 93])
    assert controls.throttle == 1 and controls.steer == pytest.approx(0)
    assert controls.boost and not controls.handbrake


@pytest.mark.parametrize("ball_x, side", [(-800, 1), (800, -1)], ids=["right", "left"])
def test_it_steers_toward_a_ball_to_one_side(ball_x, side):
    controls = _controls([ball_x, 800, 93])
    assert np.sign(controls.steer) == side and not controls.boost


def test_it_powerslides_toward_a_ball_behind_it():
    controls = _controls([300, -1500, 93])
    assert controls.handbrake and not controls.boost and abs(controls.steer) == 1


@pytest.mark.parametrize("spot", range(KICKOFF_SPOT_COUNT))
def test_it_reaches_the_ball_from_every_kickoff_spot(spot):
    arena = SoccarArena()
    arena.place_for_kickoff(spot)
    for _ in range(4 * 120):
        car = arena.car(Team.ORANGE)
        arena.set_controls(Team.ORANGE, ball_chaser(
            car.pos.as_numpy(), car.rot_mat.forward.as_numpy(), car.rot_mat.right.as_numpy(),
            arena.ball.pos.as_numpy(), -GOAL_LINE_Y))
        if Team.ORANGE in arena.tick().touched:
            return
    pytest.fail("the bot did not touch the ball within 4 s")
