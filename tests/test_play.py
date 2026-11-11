"""The play window's inputs and title: no window here."""

from collections import defaultdict

import pygame
import pytest

from rocketsim_gym.controls import BUTTON_COUNT, Button
from rocketsim_gym.play import caption, pressed_buttons

NO_MOUSE = (False, False, False)


class FakeController:
    def __init__(self, buttons=(), axes=None):
        self.buttons, self.axes = set(buttons), dict(axes or {})

    def get_button(self, button):
        return button in self.buttons

    def get_axis(self, axis):
        return self.axes.get(axis, 0)


def held(action):
    return {Button(index) for index in range(BUTTON_COUNT) if action[index]}


def keys(*held_keys):
    return defaultdict(bool, dict.fromkeys(held_keys, True))


@pytest.mark.parametrize("key, buttons", [
    (pygame.K_w, {Button.THROTTLE, Button.STICK_UP}),
    (pygame.K_s, {Button.REVERSE, Button.STICK_DOWN}),
    (pygame.K_a, {Button.STICK_LEFT}),
    (pygame.K_d, {Button.STICK_RIGHT}),
    (pygame.K_LSHIFT, {Button.POWERSLIDE}),
    (pygame.K_SPACE, {Button.BALL_CAM}),
    (pygame.K_q, {Button.AIR_ROLL_LEFT}),
    (pygame.K_e, {Button.AIR_ROLL_RIGHT}),
])
def test_each_key_holds_rocket_leagues_pc_default(key, buttons):
    assert held(pressed_buttons(keys(key), NO_MOUSE, None)) == buttons


def test_the_left_mouse_button_boosts_and_the_right_one_jumps():
    assert held(pressed_buttons(keys(), (True, False, False), None)) == {Button.BOOST}
    assert held(pressed_buttons(keys(), (False, False, True), None)) == {Button.JUMP}


def test_a_controllers_triggers_and_stick_hold_buttons_past_their_thresholds():
    controller = FakeController(
        buttons=(pygame.CONTROLLER_BUTTON_A, pygame.CONTROLLER_BUTTON_Y),
        axes={pygame.CONTROLLER_AXIS_TRIGGERRIGHT: 20000, pygame.CONTROLLER_AXIS_TRIGGERLEFT: 5000,
              pygame.CONTROLLER_AXIS_LEFTX: -30000, pygame.CONTROLLER_AXIS_LEFTY: 10000})
    assert held(pressed_buttons(keys(), NO_MOUSE, controller)) == {
        Button.THROTTLE, Button.STICK_LEFT, Button.JUMP, Button.BALL_CAM}


def test_the_title_shows_the_score_the_clock_and_the_boost():
    sample = {"overtime": False, "overtime_seconds": 0.0, "seconds_left": 61.2, "phase": "play",
              "blue_score": 2, "orange_score": 1, "blue_boost": 33.4}
    assert caption(sample).split() == ["Blue", "2", "-", "1", "Orange", "1:02", "boost", "33"]
    sample.update(overtime=True, overtime_seconds=75.9, phase="goal")
    assert "+1:15" in caption(sample) and "goal!" in caption(sample)
