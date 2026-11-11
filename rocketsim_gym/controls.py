"""The env's twelve buttons, and the car controls they make.

An action holds each button pressed or not, the way a controller's digital inputs are; the
sticks and triggers are buttons too, at full deflection. The buttons map to Rocket League's
default controls:

=====  ===============  =========================================================
index  button           car control
=====  ===============  =========================================================
0      throttle         throttle +1
1      reverse          throttle -1 (with throttle: none)
2      stick_left       steer -1, and in the air yaw -1 (roll -1 with powerslide)
3      stick_right      steer +1, and in the air yaw +1 (roll +1 with powerslide)
4      stick_up         pitch -1: nose down
5      stick_down       pitch +1: nose up
6      jump             jump
7      boost            boost
8      powerslide       handbrake on the ground; in the air, the stick rolls
9      ball_cam         no control: switches the camera between car and ball
10     air_roll_left    roll -1
11     air_roll_right   roll +1
=====  ===============  =========================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Button(IntEnum):
    """A button's index in an action."""

    THROTTLE = 0
    REVERSE = 1
    STICK_LEFT = 2
    STICK_RIGHT = 3
    STICK_UP = 4
    STICK_DOWN = 5
    JUMP = 6
    BOOST = 7
    POWERSLIDE = 8
    BALL_CAM = 9
    AIR_ROLL_LEFT = 10
    AIR_ROLL_RIGHT = 11


#: The number of buttons in an action.
BUTTON_COUNT = len(Button)


@dataclass(frozen=True)
class Controls:
    """A car's controls for a tick, as RocketSim takes them; each axis is in [-1, 1]."""

    throttle: float = 0.0
    steer: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    jump: bool = False
    boost: bool = False
    handbrake: bool = False


#: A car left alone.
NO_CONTROLS = Controls()


def buttons_from_action(action: Sequence[bool | int] | np.ndarray) -> tuple[bool, ...]:
    """The buttons an action holds pressed.

    :param action: one value per button, each ``0``, ``1``, ``False`` or ``True``: a list, or a
        boolean or integer array.
    :raises ValueError: the action is not that.
    """
    values = np.asarray(action)
    if values.shape != (BUTTON_COUNT,):
        raise ValueError(f"an action holds {BUTTON_COUNT} buttons, not shape {values.shape}")
    if values.dtype != np.bool_ and not (np.issubdtype(values.dtype, np.integer)
                                          and np.isin(values, (0, 1)).all()):
        raise ValueError(f"{action!r}: each button is 0 or 1, or False or True")
    return tuple(bool(value) for value in values)


def controls_from_buttons(pressed: Sequence[bool]) -> Controls:
    """The car controls the pressed buttons make (the table in this module's docstring)."""

    def axis(negative: Button, positive: Button) -> float:
        return float(pressed[positive]) - float(pressed[negative])

    stick_x = axis(Button.STICK_LEFT, Button.STICK_RIGHT)
    free_air_roll = pressed[Button.POWERSLIDE]
    roll = (stick_x if free_air_roll else 0.0) + axis(Button.AIR_ROLL_LEFT, Button.AIR_ROLL_RIGHT)
    return Controls(
        throttle=axis(Button.REVERSE, Button.THROTTLE),
        steer=stick_x,
        pitch=axis(Button.STICK_UP, Button.STICK_DOWN),
        yaw=0.0 if free_air_roll else stick_x,
        roll=max(-1.0, min(1.0, roll)),
        jump=pressed[Button.JUMP],
        boost=pressed[Button.BOOST],
        handbrake=free_air_roll,
    )
