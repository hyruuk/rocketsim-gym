"""The env's inputs, digital or analog, and the car controls they make.

Both input modes describe a game controller, the way Rocket League's default controls use one,
and both go through one :class:`Pad` to the car. **Digital** inputs are twelve buttons, each
held or not; the sticks and triggers are buttons too, at full deflection, as keys are:

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

**Analog** inputs are ten values: the triggers and the stick as far as they are pushed, and the
buttons, each pressed when above 0.5:

=====  ===============  =====  ==============================================
index  input            range  car control
=====  ===============  =====  ==============================================
0      throttle         0, 1   throttle, less the reverse trigger
1      reverse          0, 1   throttle backward
2      stick_x          -1, 1  steer, and in the air yaw (roll with powerslide)
3      stick_y          -1, 1  pitch: -1 pushed up (nose down), +1 pulled down
4      jump             0, 1   jump
5      boost            0, 1   boost
6      powerslide       0, 1   handbrake; in the air, the stick rolls
7      ball_cam         0, 1   no control: switches the camera
8      air_roll_left    0, 1   roll -1
9      air_roll_right   0, 1   roll +1
=====  ===============  =====  ==============================================

A player's :class:`ControllerSettings` shape the stick as Rocket League's do: a dead zone, and a
sensitivity on the ground and in the air.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np


class InputMode(Enum):
    """How an action holds the controller: buttons only, or the sticks and triggers as far as
    they are pushed."""

    DIGITAL = "digital"
    ANALOG = "analog"


class Button(IntEnum):
    """A button's index in a digital action."""

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


class AnalogInput(IntEnum):
    """An input's index in an analog action."""

    THROTTLE = 0
    REVERSE = 1
    STICK_X = 2
    STICK_Y = 3
    JUMP = 4
    BOOST = 5
    POWERSLIDE = 6
    BALL_CAM = 7
    AIR_ROLL_LEFT = 8
    AIR_ROLL_RIGHT = 9


#: The number of values in a digital action and in an analog one.
BUTTON_COUNT = len(Button)
ANALOG_INPUT_COUNT = len(AnalogInput)
#: Each analog input's lowest and highest value.
ANALOG_LOW = np.array([0, 0, -1, -1, 0, 0, 0, 0, 0, 0], np.float32)
ANALOG_HIGH = np.ones(ANALOG_INPUT_COUNT, np.float32)
#: An analog button is pressed above this value.
PRESSED = 0.5


@dataclass(frozen=True)
class Pad:
    """A game controller's state for a step.

    :ivar throttle, reverse: the right and left triggers, 0 to 1.
    :ivar stick_x: the stick, -1 left to 1 right.
    :ivar stick_y: the stick, -1 pushed up to 1 pulled down, as controllers report it.
    """

    throttle: float = 0.0
    reverse: float = 0.0
    stick_x: float = 0.0
    stick_y: float = 0.0
    jump: bool = False
    boost: bool = False
    powerslide: bool = False
    ball_cam: bool = False
    air_roll_left: bool = False
    air_roll_right: bool = False


@dataclass(frozen=True)
class ControllerSettings:
    """A player's controller settings, as Rocket League's settings menu has them.

    :ivar deadzone: how far the stick moves before it counts, 0 to 0.95; past it, the stick's
        travel is spread over the full range again.
    :ivar steering_sensitivity: steer is the stick times this, 0.1 to 10, at most full lock.
    :ivar aerial_sensitivity: pitch, yaw and roll from the stick are it times this, 0.1 to 10.
    :raises ValueError: a setting is not a number in its range.
    """

    deadzone: float = 0.0
    steering_sensitivity: float = 1.0
    aerial_sensitivity: float = 1.0

    def __post_init__(self) -> None:
        for name, (lowest, highest) in (("deadzone", (0.0, 0.95)),
                                        ("steering_sensitivity", (0.1, 10.0)),
                                        ("aerial_sensitivity", (0.1, 10.0))):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not lowest <= value <= highest:
                raise ValueError(f"controller {name} {value!r}: a number from {lowest} to "
                                 f"{highest}")


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
    """The buttons a digital action holds pressed.

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


def pad_from_buttons(pressed: Sequence[bool]) -> Pad:
    """The pad that buttons make: each trigger and stick direction at full deflection."""

    def axis(negative: Button, positive: Button) -> float:
        return float(pressed[positive]) - float(pressed[negative])

    return Pad(throttle=float(pressed[Button.THROTTLE]), reverse=float(pressed[Button.REVERSE]),
               stick_x=axis(Button.STICK_LEFT, Button.STICK_RIGHT),
               stick_y=axis(Button.STICK_UP, Button.STICK_DOWN),
               jump=pressed[Button.JUMP], boost=pressed[Button.BOOST],
               powerslide=pressed[Button.POWERSLIDE], ball_cam=pressed[Button.BALL_CAM],
               air_roll_left=pressed[Button.AIR_ROLL_LEFT],
               air_roll_right=pressed[Button.AIR_ROLL_RIGHT])


def analog_from_buttons(pressed: Sequence[bool]) -> np.ndarray:
    """The analog action that holds what ``pressed`` does: its triggers and stick directions
    pushed all the way."""
    action = np.zeros(ANALOG_INPUT_COUNT, np.float32)
    action[AnalogInput.THROTTLE] = pressed[Button.THROTTLE]
    action[AnalogInput.REVERSE] = pressed[Button.REVERSE]
    action[AnalogInput.STICK_X] = float(pressed[Button.STICK_RIGHT]) - float(pressed[Button.STICK_LEFT])
    action[AnalogInput.STICK_Y] = float(pressed[Button.STICK_DOWN]) - float(pressed[Button.STICK_UP])
    for analog, button in ((AnalogInput.JUMP, Button.JUMP), (AnalogInput.BOOST, Button.BOOST),
                           (AnalogInput.POWERSLIDE, Button.POWERSLIDE),
                           (AnalogInput.BALL_CAM, Button.BALL_CAM),
                           (AnalogInput.AIR_ROLL_LEFT, Button.AIR_ROLL_LEFT),
                           (AnalogInput.AIR_ROLL_RIGHT, Button.AIR_ROLL_RIGHT)):
        action[analog] = pressed[button]
    return action


def pad_from_analog(action: Sequence[float] | np.ndarray) -> Pad:
    """The pad an analog action holds.

    :param action: the ten analog inputs, each within its range (the table above).
    :raises ValueError: the action is not that.
    """
    values = np.asarray(action, dtype=float)
    if values.shape != (ANALOG_INPUT_COUNT,):
        raise ValueError(f"an analog action holds {ANALOG_INPUT_COUNT} values, not shape "
                         f"{values.shape}")
    if not (np.isfinite(values).all() and (values >= ANALOG_LOW).all()
            and (values <= ANALOG_HIGH).all()):
        raise ValueError(f"{action!r}: the triggers and buttons go from 0 to 1, the stick from "
                         f"-1 to 1")
    pressed = values > PRESSED
    return Pad(throttle=float(values[AnalogInput.THROTTLE]),
               reverse=float(values[AnalogInput.REVERSE]),
               stick_x=float(values[AnalogInput.STICK_X]),
               stick_y=float(values[AnalogInput.STICK_Y]),
               jump=bool(pressed[AnalogInput.JUMP]), boost=bool(pressed[AnalogInput.BOOST]),
               powerslide=bool(pressed[AnalogInput.POWERSLIDE]),
               ball_cam=bool(pressed[AnalogInput.BALL_CAM]),
               air_roll_left=bool(pressed[AnalogInput.AIR_ROLL_LEFT]),
               air_roll_right=bool(pressed[AnalogInput.AIR_ROLL_RIGHT]))


def controls_from_pad(pad: Pad, settings: ControllerSettings = ControllerSettings()) -> Controls:
    """The car controls a pad makes, through a player's controller settings."""
    stick_x = _past_deadzone(pad.stick_x, settings.deadzone)
    stick_y = _past_deadzone(pad.stick_y, settings.deadzone)
    air_x = _clamped(stick_x * settings.aerial_sensitivity)
    roll = (air_x if pad.powerslide else 0.0) + float(pad.air_roll_right) - float(pad.air_roll_left)
    return Controls(
        throttle=_clamped(pad.throttle - pad.reverse),
        steer=_clamped(stick_x * settings.steering_sensitivity),
        pitch=_clamped(stick_y * settings.aerial_sensitivity),
        yaw=0.0 if pad.powerslide else air_x,
        roll=_clamped(roll),
        jump=pad.jump,
        boost=pad.boost,
        handbrake=pad.powerslide,
    )


def controls_from_buttons(pressed: Sequence[bool]) -> Controls:
    """The car controls the pressed buttons make (the digital table above)."""
    return controls_from_pad(pad_from_buttons(pressed))


def _past_deadzone(value: float, deadzone: float) -> float:
    """``value`` once the dead zone is taken off its travel, the rest spread back to 1."""
    if abs(value) <= deadzone:
        return 0.0
    return float(np.sign(value)) * (abs(value) - deadzone) / (1.0 - deadzone)


def _clamped(value: float) -> float:
    return max(-1.0, min(1.0, value))
