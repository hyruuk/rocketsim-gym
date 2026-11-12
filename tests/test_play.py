"""The play window's inputs and title: no window here."""

from collections import defaultdict

import pygame
import pytest

from rocketsim_gym.controls import BUTTON_COUNT, AnalogInput, Button
from rocketsim_gym.play import (_window_size, caption, lan_arguments, main, pad_inputs,
                                pressed_buttons)

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


def test_the_title_says_what_the_keys_do():
    assert caption({"phase": "play"}) == "rocketsim-gym: Esc quits"
    assert "Enter plays another match" in caption({"phase": "ended"})


@pytest.mark.parametrize("size, expected", [("1920x1080", (1920, 1080)), ("800X450", (800, 450))])
def test_the_window_size_reads_width_by_height(size, expected):
    assert _window_size(size) == expected


def test_camera_settings_out_of_range_stop_the_window_before_it_opens(capsys):
    with pytest.raises(SystemExit):
        main(["--field-of-view", "130"])
    assert "field_of_view" in capsys.readouterr().err


def test_a_team_size_rocket_league_has_no_match_for_stops_the_window_before_it_opens(capsys):
    with pytest.raises(SystemExit):
        main(["--team-size", "4"])
    assert "--team-size" in capsys.readouterr().err


def test_analog_inputs_read_the_controller_as_far_as_it_is_pushed():
    controller = FakeController(
        buttons=(pygame.CONTROLLER_BUTTON_A,),
        axes={pygame.CONTROLLER_AXIS_TRIGGERRIGHT: 16384, pygame.CONTROLLER_AXIS_LEFTX: -8192,
              pygame.CONTROLLER_AXIS_LEFTY: -32768})
    action = pad_inputs(keys(), NO_MOUSE, controller)
    assert action[AnalogInput.THROTTLE] == pytest.approx(0.5, abs=1e-3)
    assert action[AnalogInput.STICK_X] == pytest.approx(-0.25, abs=1e-3)
    assert action[AnalogInput.STICK_Y] == -1.0 and action[AnalogInput.JUMP] == 1.0
    assert action[AnalogInput.REVERSE] == 0.0


def test_a_held_key_is_a_full_push_over_the_stick_on_its_axis():
    controller = FakeController(axes={pygame.CONTROLLER_AXIS_LEFTX: -30000,
                                      pygame.CONTROLLER_AXIS_TRIGGERRIGHT: 8000})
    action = pad_inputs(keys(pygame.K_d, pygame.K_w), NO_MOUSE, controller)
    assert action[AnalogInput.STICK_X] == 1.0 and action[AnalogInput.THROTTLE] == 1.0
    assert action[AnalogInput.STICK_Y] == -1.0


def test_lan_flags_without_join_or_with_a_seed_stop_the_window_before_it_opens(capsys):
    with pytest.raises(SystemExit):
        main(["--seat", "blue1"])
    assert "--join" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--join", "127.0.0.1:47999", "--seats", "blue1", "--seed", "3"])
    assert "--seed" in capsys.readouterr().err


def test_the_first_to_join_sets_the_match_and_takes_its_first_seat_by_default():
    assert lan_arguments("127.0.0.1:47999", None, "orange1, blue1", 1, 60.0) == {
        "server": "127.0.0.1:47999", "seats": ["orange1", "blue1"], "seat": "orange1",
        "team_size": 1, "match_seconds": 60.0}


def test_the_title_of_a_lan_match_says_the_room_and_the_input_delay():
    sample = {"phase": "play", "client": "live", "room": "playing", "server_frame": 3,
              "input_delay": 2}
    assert caption(sample) == "rocketsim-gym LAN (playing, input delay 2 frames): Esc quits"
    assert "Enter plays another match" in caption({**sample, "room": "stopped"})
