"""Play a match against the bot in a window on this computer.

    rocketsim-play [--seconds 300] [--seed N] [--width 1280] [--height 720]

The keyboard and mouse follow Rocket League's PC defaults, and an Xbox-style controller works
too, its triggers and left stick read as buttons:

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

The window's title shows the score, the clock and your boost. Esc quits; when the match is
over, Enter plays another.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402
from pygame._sdl2 import controller as sdl_controller  # noqa: E402

from .controls import BUTTON_COUNT, Button  # noqa: E402
from .env import SoccarEnv  # noqa: E402

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


def caption(sample: Mapping[str, Any]) -> str:
    """The window's title: the score, the clock and the blue car's boost."""
    if sample["overtime"]:
        clock = "+" + _minutes(sample["overtime_seconds"], math.floor)
    else:
        clock = _minutes(sample["seconds_left"], math.ceil)
    status = {"countdown": "get ready", "goal": "goal!",
              "ended": "full time: Enter plays another match, Esc quits"}.get(sample["phase"], "")
    return (f"Blue {sample['blue_score']} - {sample['orange_score']} Orange    {clock}    "
            f"boost {sample['blue_boost']:.0f}    {status}")


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


def _minutes(seconds: float, rounding) -> str:
    whole = int(rounding(seconds))
    return f"{whole // 60}:{whole % 60:02d}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play a match against the bot.")
    parser.add_argument("--seconds", type=float, default=300.0, help="regulation time")
    parser.add_argument("--seed", type=int, default=None, help="seed of the kickoff spots")
    parser.add_argument("--width", type=int, default=1280, help="window width, in pixels")
    parser.add_argument("--height", type=int, default=720, help="window height, in pixels")
    arguments = parser.parse_args(argv)
    env = SoccarEnv(match_seconds=arguments.seconds, render_mode="rgb_array",
                    render_width=arguments.width, render_height=arguments.height)
    pygame.init()
    sdl_controller.init()
    screen = pygame.display.set_mode((arguments.width, arguments.height))
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
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                                 and event.key == pygame.K_ESCAPE):
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN and not playing:
                    env.reset()
                    playing = True
                if (event.type == pygame.CONTROLLERDEVICEADDED and controller is None
                        and sdl_controller.is_controller(event.device_index)):
                    controller = sdl_controller.Controller(event.device_index)
            if playing:
                action = pressed_buttons(pygame.key.get_pressed(), pygame.mouse.get_pressed(),
                                         controller)
                playing = not env.step(action)[2]
            frame = env.render()
            screen.blit(pygame.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]),
                                                "RGB"), (0, 0))
            pygame.display.flip()
            pygame.display.set_caption(caption(env.sample()))
            frame_due = _wait_for(frame_due + frame_seconds)
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
