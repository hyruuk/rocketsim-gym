"""The buttons: the controls they make, the actions accepted, and which way RocketSim turns."""

import math

import numpy as np
import pytest
import RocketSim as rs

from rocketsim_gym.controls import (BUTTON_COUNT, Button, Controls, buttons_from_action,
                                    controls_from_buttons)
from rocketsim_gym.engine import SoccarArena
from rocketsim_gym.match import Team


def pressed(*buttons):
    return tuple(index in buttons for index in range(BUTTON_COUNT))


@pytest.mark.parametrize("buttons, controls", [
    ((), Controls()),
    ((Button.THROTTLE,), Controls(throttle=1)),
    ((Button.REVERSE,), Controls(throttle=-1)),
    ((Button.THROTTLE, Button.REVERSE), Controls()),
    ((Button.STICK_LEFT,), Controls(steer=-1, yaw=-1)),
    ((Button.STICK_RIGHT,), Controls(steer=1, yaw=1)),
    ((Button.STICK_LEFT, Button.STICK_RIGHT), Controls()),
    ((Button.STICK_UP,), Controls(pitch=-1)),
    ((Button.STICK_DOWN,), Controls(pitch=1)),
    ((Button.JUMP,), Controls(jump=True)),
    ((Button.BOOST,), Controls(boost=True)),
    ((Button.POWERSLIDE,), Controls(handbrake=True)),
    ((Button.POWERSLIDE, Button.STICK_RIGHT), Controls(steer=1, roll=1, handbrake=True)),
    ((Button.AIR_ROLL_LEFT,), Controls(roll=-1)),
    ((Button.AIR_ROLL_RIGHT, Button.STICK_LEFT), Controls(steer=-1, yaw=-1, roll=1)),
    ((Button.POWERSLIDE, Button.STICK_RIGHT, Button.AIR_ROLL_RIGHT),
     Controls(steer=1, roll=1, handbrake=True)),
    ((Button.BALL_CAM,), Controls()),
])
def test_each_button_makes_its_controls(buttons, controls):
    assert controls_from_buttons(pressed(*buttons)) == controls


@pytest.mark.parametrize("action", [
    [0] * 12, [True] * 12, np.zeros(12, np.int8), np.ones(12, np.int64), np.zeros(12, bool),
])
def test_twelve_buttons_given_as_bits_are_an_action(action):
    assert buttons_from_action(action) == tuple(bool(value) for value in action)


@pytest.mark.parametrize("action", [
    [0] * 11, np.zeros((2, 12), np.int8), [2] + [0] * 11, [-1] + [0] * 11, [0.0] * 12, ["1"] * 12,
])
def test_anything_else_is_refused(action):
    with pytest.raises(ValueError):
        buttons_from_action(action)


def _car_after(buttons, ticks, airborne):
    """A blue car facing +y, after ``ticks`` ticks holding ``buttons``: its state before and after."""
    arena = SoccarArena()
    arena.place_for_kickoff(4)
    car = arena._cars[Team.BLUE]
    car.set_state(rs.CarState(pos=rs.Vec(0, -2000, 1000 if airborne else 17),
                              vel=rs.Vec(0, 0 if airborne else 1000, 0),
                              rot_mat=rs.Angle(math.pi / 2, 0, 0).as_rot_mat()))
    before = car.get_state()
    arena.set_controls(Team.BLUE, controls_from_buttons(pressed(*buttons)))
    for _ in range(ticks):
        arena.tick()
    return before, car.get_state()


def _toward(vector, direction):
    return np.dot(vector.as_numpy(), direction.as_numpy())


def test_stick_right_steers_right():
    before, after = _car_after((Button.THROTTLE, Button.STICK_RIGHT), 60, airborne=False)
    assert _toward(after.vel, before.rot_mat.right) > 300


# In the air, 30 ticks turn the car at most 5.5 rad/s * 0.25 s, under a quarter turn, so the
# sign of each change shows the direction of the turn.

def test_stick_right_yaws_right_in_the_air():
    before, after = _car_after((Button.STICK_RIGHT,), 30, airborne=True)
    assert _toward(after.rot_mat.forward, before.rot_mat.right) > 0.05


def test_stick_down_lifts_the_nose_and_stick_up_drops_it():
    _, after = _car_after((Button.STICK_DOWN,), 30, airborne=True)
    assert after.rot_mat.forward.z > 0.05
    _, after = _car_after((Button.STICK_UP,), 30, airborne=True)
    assert after.rot_mat.forward.z < -0.05


@pytest.mark.parametrize("buttons", [(Button.AIR_ROLL_RIGHT,), (Button.POWERSLIDE, Button.STICK_RIGHT)],
                         ids=["air roll right", "powerslide and stick right"])
def test_rolling_right_tips_the_roof_to_the_right(buttons):
    before, after = _car_after(buttons, 30, airborne=True)
    assert _toward(after.rot_mat.up, before.rot_mat.right) > 0.05
    assert _toward(after.rot_mat.forward, before.rot_mat.forward) > 0.9
