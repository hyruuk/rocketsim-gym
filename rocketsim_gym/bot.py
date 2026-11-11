"""The opponent: a scripted ball-chaser.

It drives at the ball, aiming a little behind it so that the touch sends the ball toward the goal
it attacks, boosts while it faces its target and powerslides through sharp turns. It never jumps.
Its controls follow from the state alone, so the same state always gives the same controls.
"""

from __future__ import annotations

import math

import numpy as np

from .controls import Controls

#: The back walls' y, where the goal mouths open.
GOAL_LINE_Y = 5120.0
#: Full steering lock for a heading error of 1 / STEERING_GAIN radians or more.
STEERING_GAIN = 5.0
#: Boost while the target is within this many radians of straight ahead.
BOOST_MAXIMUM_HEADING = 0.3
#: Powerslide while the target is more than this many radians off.
POWERSLIDE_MINIMUM_HEADING = 1.5
#: How far behind the ball, from the goal it attacks, the bot aims (uu).
AIM_BEHIND_BALL = 150.0


def ball_chaser(position: np.ndarray, forward: np.ndarray, right: np.ndarray,
                ball_position: np.ndarray, attacked_goal_y: float) -> Controls:
    """The controls that chase the ball.

    :param position: the car's position (uu).
    :param forward: the car's forward direction.
    :param right: the car's right direction (RocketSim's ``rot_mat.right``).
    :param ball_position: the ball's position (uu).
    :param attacked_goal_y: the y of the goal line the car attacks.
    """
    to_goal = np.array([0.0, attacked_goal_y]) - ball_position[:2]
    distance_to_goal = float(np.hypot(*to_goal))
    target = ball_position[:2]
    if distance_to_goal > 1.0:
        target = target - AIM_BEHIND_BALL * to_goal / distance_to_goal
    to_target = target - position[:2]
    heading = math.atan2(float(to_target @ right[:2]), float(to_target @ forward[:2]))
    return Controls(throttle=1.0,
                    steer=float(np.clip(STEERING_GAIN * heading, -1.0, 1.0)),
                    boost=abs(heading) < BOOST_MAXIMUM_HEADING,
                    handbrake=abs(heading) > POWERSLIDE_MINIMUM_HEADING)
