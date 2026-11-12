"""Play a match against bots in a window on this computer: 1v1, 2v2 or 3v3.

    rocketsim-play [--team-size 1] [--seconds 300] [--seed N] [--window 1280x720] [--input analog]
                   [--deadzone 0.1] [--steering-sensitivity 1] [--aerial-sensitivity 1]
                   [--field-of-view 110] [--distance 270] [--height 100] [--angle -3]
                   [--stiffness 1] [--transition-speed 1]

The keyboard and mouse follow Rocket League's PC defaults, and an Xbox-style controller works
too. With ``--input analog``, the default, its triggers and left stick count as far as they are
pushed, as in Rocket League; with ``--input digital`` they count as buttons, all or nothing, as
a scanner's controller may give them. A key is always a full push. The controller and camera
options are Rocket League's settings, in its units, so a player can bring their own.

================  ====================  ==============
button            keyboard and mouse    controller
================  ====================  ==============
throttle          W                     right trigger
reverse           S                     left trigger
stick left/right  A / D                 left stick
stick up/down     W / S                 left stick
jump              right mouse button    A
boost             left mouse button     B
powerslide        left Shift            X
ball cam          Space                 Y
air roll left     Q
air roll right    E
================  ====================  ==============

Esc quits; when the match is over, Enter plays another.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402
from pygame._sdl2 import controller as sdl_controller  # noqa: E402
from pygame._sdl2.video import Renderer, Texture, Window  # noqa: E402

from .camera import CameraSettings  # noqa: E402
from .controls import (BUTTON_COUNT, AnalogInput, Button, InputMode,  # noqa: E402
                       analog_from_buttons)
from .env import TEAM_SIZES, SoccarEnv  # noqa: E402

#: Each key, and the buttons it holds.
KEYS = {
    pygame.K_w: (Button.THROTTLE, Button.STICK_UP),
    pygame.K_s: (Button.REVERSE, Button.STICK_DOWN),
    pygame.K_a: (Button.STICK_LEFT,),
    pygame.K_d: (Button.STICK_RIGHT,),
    pygame.K_LSHIFT: (Button.POWERSLIDE,),
    pygame.K_SPACE: (Button.BALL_CAM,),
    pygame.K_q: (Button.AIR_ROLL_LEFT,),
    pygame.K_e: (Button.AIR_ROLL_RIGHT,),
}
#: Each mouse button, by its index in ``pygame.mouse.get_pressed()``, and the button it holds.
MOUSE = {0: Button.BOOST, 2: Button.JUMP}
#: Each controller button, and the button it holds.
CONTROLLER_BUTTONS = {
    pygame.CONTROLLER_BUTTON_A: Button.JUMP,
    pygame.CONTROLLER_BUTTON_B: Button.BOOST,
    pygame.CONTROLLER_BUTTON_X: Button.POWERSLIDE,
    pygame.CONTROLLER_BUTTON_Y: Button.BALL_CAM,
}
#: A trigger holds its button past this share of its travel, the stick past this deflection.
TRIGGER_PRESSED = 0.3
STICK_PRESSED = 0.5
_AXIS_MAXIMUM = 32767


class Controller(Protocol):
    """What is read from a game controller: SDL's ``pygame._sdl2.controller.Controller``."""

    def get_button(self, button: int) -> bool: ...

    def get_axis(self, axis: int) -> int: ...


def pressed_buttons(keys_held: Mapping[int, bool] | Sequence[bool], mouse_held: Sequence[bool],
                    controller: Controller | None) -> np.ndarray:
    """The action the held keys, mouse buttons and controller make.

    :param keys_held: whether each key is held, by pygame key (``pygame.key.get_pressed()``).
    :param mouse_held: whether each mouse button is held (``pygame.mouse.get_pressed()``).
    :param controller: the controller, if one is connected.
    """
    action = np.zeros(BUTTON_COUNT, np.int8)
    for key, buttons in KEYS.items():
        if keys_held[key]:
            action[list(buttons)] = 1
    for index, button in MOUSE.items():
        if mouse_held[index]:
            action[button] = 1
    if controller is not None:
        for sdl_button, button in CONTROLLER_BUTTONS.items():
            if controller.get_button(sdl_button):
                action[button] = 1

        def axis(sdl_axis: int) -> float:
            return controller.get_axis(sdl_axis) / _AXIS_MAXIMUM

        stick_x, stick_y = axis(pygame.CONTROLLER_AXIS_LEFTX), axis(pygame.CONTROLLER_AXIS_LEFTY)
        for button, held in ((Button.THROTTLE, axis(pygame.CONTROLLER_AXIS_TRIGGERRIGHT)
                              > TRIGGER_PRESSED),
                             (Button.REVERSE, axis(pygame.CONTROLLER_AXIS_TRIGGERLEFT)
                              > TRIGGER_PRESSED),
                             (Button.STICK_LEFT, stick_x < -STICK_PRESSED),
                             (Button.STICK_RIGHT, stick_x > STICK_PRESSED),
                             (Button.STICK_UP, stick_y < -STICK_PRESSED),
                             (Button.STICK_DOWN, stick_y > STICK_PRESSED)):
            if held:
                action[button] = 1
    return action


def pad_inputs(keys_held: Mapping[int, bool] | Sequence[bool], mouse_held: Sequence[bool],
               controller: Controller | None) -> np.ndarray:
    """The analog action the held keys, mouse buttons and controller make: the controller's
    triggers and stick as far as they are pushed, a key as a full push.

    :param keys_held: whether each key is held, by pygame key (``pygame.key.get_pressed()``).
    :param mouse_held: whether each mouse button is held (``pygame.mouse.get_pressed()``).
    :param controller: the controller, if one is connected.
    """
    action = analog_from_buttons(pressed_buttons(keys_held, mouse_held, None))
    if controller is None:
        return action

    def axis(sdl_axis: int) -> float:
        return max(-1.0, min(1.0, controller.get_axis(sdl_axis) / _AXIS_MAXIMUM))

    for analog, sdl_axis in ((AnalogInput.THROTTLE, pygame.CONTROLLER_AXIS_TRIGGERRIGHT),
                             (AnalogInput.REVERSE, pygame.CONTROLLER_AXIS_TRIGGERLEFT)):
        action[analog] = max(action[analog], axis(sdl_axis))
    for analog, sdl_axis in ((AnalogInput.STICK_X, pygame.CONTROLLER_AXIS_LEFTX),
                             (AnalogInput.STICK_Y, pygame.CONTROLLER_AXIS_LEFTY)):
        if action[analog] == 0:
            action[analog] = axis(sdl_axis)
    for sdl_button, button in CONTROLLER_BUTTONS.items():
        if controller.get_button(sdl_button):
            action[AnalogInput[button.name]] = 1.0
    return action


def caption(sample: Mapping[str, Any]) -> str:
    """The window's title: what the keys do now."""
    if sample["phase"] == "ended":
        return "rocketsim-gym: Enter plays another match, Esc quits"
    return "rocketsim-gym: Esc quits"


def _open_window(width: int, height: int) -> tuple[Window, Renderer]:
    """The game's window, showing each frame in step with the screen's refresh (vsync).

    A frame shown while the screen is still drawing the one before tears the picture: its top
    and its bottom come from two frames, out of line wherever the view turns fast.
    """
    window = Window("rocketsim-gym", size=(width, height))
    try:
        return window, Renderer(window, vsync=True)
    except pygame.error as error:
        print(f"rocketsim-play: no vsync on this machine ({error}); frames may tear",
              file=sys.stderr)
        return window, Renderer(window)


def _window_size(text: str) -> tuple[int, int]:
    """``WIDTHxHEIGHT``, in pixels."""
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: a size such as 1280x720") from None
    return width, height


def _wait_for(moment: float) -> float:
    """Wait until ``moment`` on ``time.perf_counter``'s clock, sleeping most of the way and
    spinning the last 2 ms, since a sleep can overshoot by milliseconds; return when the frame
    that follows starts, which is now if the frames have fallen a tenth of a second behind."""
    delay = moment - time.perf_counter()
    if delay < -0.1:
        return time.perf_counter()
    if delay > 0.002:
        time.sleep(delay - 0.002)
    while time.perf_counter() < moment:
        pass
    return moment


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play a match against bots.")
    parser.add_argument("--team-size", type=int, choices=TEAM_SIZES, default=1,
                        help="cars a team: 1 (1v1, the default), 2 (2v2) or 3 (3v3); you drive "
                             "one of blue's, bots the others")
    parser.add_argument("--seconds", type=float, default=300.0, help="regulation time")
    parser.add_argument("--seed", type=int, default=None, help="seed of the kickoff spots")
    parser.add_argument("--window", type=_window_size, default=(1280, 720),
                        help="window size, in pixels (default 1280x720)")
    parser.add_argument("--input", choices=[mode.value for mode in InputMode],
                        default=InputMode.ANALOG.value,
                        help="analog: the controller's triggers and stick as far as they are "
                             "pushed (default); digital: all or nothing")
    controller = parser.add_argument_group("controller", "Rocket League's controller settings")
    for name, default, help_text in (("deadzone", 0.1, "0 to 0.95"),
                                     ("steering_sensitivity", 1.0, "0.1 to 10"),
                                     ("aerial_sensitivity", 1.0, "0.1 to 10")):
        controller.add_argument("--" + name.replace("_", "-"), type=float, dest=name,
                                default=default, help=f"{help_text}; default {default:g}")
    camera = parser.add_argument_group("camera", "Rocket League's camera settings")
    defaults = CameraSettings()
    for name, help_text in (("field_of_view", "horizontal, in degrees (60 to 110)"),
                            ("distance", "behind the car (100 to 400)"),
                            ("height", "above the car (40 to 200)"),
                            ("angle", "pitch, in degrees (-15 to 0)"),
                            ("stiffness", "0 to 1"), ("transition_speed", "1 to 2")):
        camera.add_argument("--" + name.replace("_", "-"), type=float, dest=name,
                            default=getattr(defaults, name),
                            help=f"{help_text}; default {getattr(defaults, name):g}")
    arguments = parser.parse_args(argv)
    width, height = arguments.window
    settings = {name: getattr(arguments, name) for name in
                ("field_of_view", "distance", "height", "angle", "stiffness", "transition_speed")}
    controller_settings = {name: getattr(arguments, name)
                           for name in ("deadzone", "steering_sensitivity", "aerial_sensitivity")}
    try:
        env = SoccarEnv(match_seconds=arguments.seconds, team_size=arguments.team_size,
                        render_mode="rgb_array", render_width=width, render_height=height,
                        camera=settings, input_mode=arguments.input,
                        controller=controller_settings)
    except ValueError as error:
        parser.error(str(error))
    read_inputs = pad_inputs if arguments.input == InputMode.ANALOG.value else pressed_buttons
    pygame.init()
    sdl_controller.init()
    window, renderer = _open_window(width, height)
    controller = None
    if sdl_controller.get_count() and sdl_controller.is_controller(0):
        controller = sdl_controller.Controller(0)
    frame_seconds = 1 / env.metadata["render_fps"]
    env.reset(seed=arguments.seed)
    playing = True
    frame_due = time.perf_counter()
    try:
        while True:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.WINDOWCLOSE) or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN and not playing:
                    env.reset()
                    playing = True
                if (event.type == pygame.CONTROLLERDEVICEADDED and controller is None
                        and sdl_controller.is_controller(event.device_index)):
                    controller = sdl_controller.Controller(event.device_index)
            if playing:
                action = read_inputs(pygame.key.get_pressed(), pygame.mouse.get_pressed(),
                                     controller)
                playing = not env.step(action)[2]
            frame = env.render()
            image = pygame.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
            Texture.from_surface(renderer, image).draw()
            renderer.present()
            title = caption(env.sample())
            if window.title != title:
                window.title = title
            frame_due = _wait_for(frame_due + frame_seconds)
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
