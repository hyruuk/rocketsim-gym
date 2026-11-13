"""Play a match in a window: against bots on this computer, or with other players on this
network, 1v1, 2v2 or 3v3.

    rocketsim-play [--team-size 1] [--seconds 300] [--seed N] [--window 1280x720] [--input analog]
                   [--deadzone 0.1] [--steering-sensitivity 1] [--aerial-sensitivity 1]
                   [--field-of-view 110] [--distance 270] [--height 100] [--angle -3]
                   [--stiffness 1] [--transition-speed 1]
                   [--join [ADDRESS] [--seat SEAT] [--seats SEAT,SEAT...]] [--stats]

With ``--join``, the window joins the match a ``rocketsim-server`` plays on this network (the one
at ADDRESS, HOST or HOST:PORT, if given): in the seat ``--seat`` names, or the first free one, of
the room's match, or, for the first player to join, of the match that ``--seats`` (the cars
players drive), ``--team-size`` and ``--seconds`` set.

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

Esc quits; when the match is over, Enter plays another (in a LAN match, once every seat's
player is ready). F3 (fn+F3 on most Mac keyboards) or the controller's Back button shows and hides
the stats in the window's corner: the frames shown a second and the slowest, and in a LAN match the
input delay, the server's frames shown twice or skipped, and the room's and the connection's state.
``--stats`` shows them from the start. They are drawn over the window, never into the game's
frames.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
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
from .lan.connection import JoinRefused, find_servers, server_address  # noqa: E402
from .pacing import wait_for  # noqa: E402

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
#: The key and the controller button that show and hide the stats.
STATS_KEY, STATS_BUTTON = pygame.K_F3, pygame.CONTROLLER_BUTTON_BACK
#: How often the stats' text changes, so that it can be read (s).
STATS_REFRESH_SECONDS = 0.5
#: The span the stats count the server's frames shown twice or skipped over (s).
SMOOTHNESS_SECONDS = 10.0
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


class PlayStats:
    """What the play window measures of the frames it shows: how many it showed over the last
    second, and the longest gap between two; in a LAN match, the input delay over the last
    second, and the server's frames shown twice or skipped over the last
    :data:`SMOOTHNESS_SECONDS`.

    :param frame_seconds: a frame's time, which turns frames of delay into milliseconds.
    """

    def __init__(self, frame_seconds: float) -> None:
        self._frame_seconds = frame_seconds
        self._shown_at: deque[float] = deque()
        self._input_delays: deque[tuple[float, int]] = deque()
        self._server_frames: deque[tuple[float, int]] = deque()

    def add(self, moment: float, sample: Mapping[str, Any], stepped: bool) -> None:
        """A frame shown at ``moment`` (``time.perf_counter``), with the env's sample after it.

        :param stepped: whether the env stepped for this frame; one that did not (the match
            over) shows its server frame again on purpose.
        """
        self._shown_at.append(moment)
        if stepped and sample.get("input_delay") is not None:
            self._input_delays.append((moment, sample["input_delay"]))
        if stepped and sample.get("server_frame") is not None:
            self._server_frames.append((moment, sample["server_frame"]))
        while self._shown_at[0] <= moment - 1.0:
            self._shown_at.popleft()
        for kept, seconds in ((self._input_delays, 1.0),
                              (self._server_frames, SMOOTHNESS_SECONDS)):
            while kept and kept[0][0] <= moment - seconds:
                kept.popleft()

    def lines(self, sample: Mapping[str, Any]) -> list[str]:
        """The stats, a line each: the frame rate, and in a LAN match (``sample`` has the
        client's state) the input delay, the frames shown twice or skipped, and the states of
        the room and of the connection."""
        gaps = np.diff(self._shown_at) if len(self._shown_at) > 1 else np.zeros(1)
        lines = [f"{len(self._shown_at)} frames a second, the slowest {1000 * gaps.max():.0f} ms"]
        if "client" not in sample:
            return lines
        if self._input_delays:
            delay = float(np.mean([frames for _, frames in self._input_delays]))
            lines.append(f"input delay {delay:.1f} frames "
                         f"({1000 * delay * self._frame_seconds:.0f} ms)")
        else:
            lines.append("input delay: none measured yet")
        steps = np.diff([number for _, number in self._server_frames]).astype(int)
        lines.append(f"last {SMOOTHNESS_SECONDS:g} s, server frames shown twice: "
                     f"{int((steps == 0).sum())}, skipped: {int((steps[steps > 1] - 1).sum())}")
        lines.append(f"room {sample['room'] or '-'}, connection {sample['client']}")
        return lines


def stats_image(lines: Sequence[str], font: pygame.font.Font) -> pygame.Surface:
    """``lines`` in white on a dark, see-through box, for the window's corner."""
    texts = [font.render(line, True, (255, 255, 255)) for line in lines]
    margin = max(1, font.get_height() // 3)
    box = pygame.Surface((max(text.get_width() for text in texts) + 2 * margin,
                          sum(text.get_height() for text in texts) + 2 * margin), pygame.SRCALPHA)
    box.fill((0, 0, 0, 170))
    top = margin
    for text in texts:
        box.blit(text, (margin, top))
        top += text.get_height()
    return box


def caption(sample: Mapping[str, Any]) -> str:
    """The window's title: what the keys do now; in a LAN match, also the room's state and the
    delay of the player's controls."""
    is_over = sample["phase"] == "ended" or sample.get("room") in ("ended", "stopped", "closing")
    keys = "Enter plays another match, Esc quits" if is_over else "Esc quits"
    if "client" not in sample:
        return f"rocketsim-gym: {keys}"
    delay = ("" if sample["input_delay"] is None
             else f", input delay {sample['input_delay']} frames")
    return f"rocketsim-gym LAN ({sample['room'] or sample['client']}{delay}): {keys}"


def lan_arguments(join: str, seat: str | None, seats: str | None, team_size: int,
                  seconds: float) -> dict[str, Any]:
    """The env's arguments for joining a LAN match: the server's address, and the match the
    flags set (``seats``, ``team_size``, ``seconds``) or else the room's, with the seat asked
    for or else the room's first free one.

    :raises ConnectionError: no server answers, with what to do.
    :raises ValueError: the room has no match and the flags set none, or no seat is free.
    """
    host, port, _ = server_address(join)
    server = f"{host}:{port}"
    if seats is not None:
        names = [name.strip() for name in seats.split(",") if name.strip()]
        return {"server": server, "seats": names, "seat": seat or (names[0] if names else None),
                "team_size": team_size, "match_seconds": seconds}
    rooms = [room for room in find_servers(hosts=(host,)) if room.port == port]
    if not rooms or rooms[0].setup is None:
        raise ValueError(f"the server at {server} has no match yet, or does not say: set one "
                         f"with --seats (and --team-size, --seconds)")
    room = rooms[0]
    free = [name for name in room.setup.seats if name not in room.seats]
    if seat is None and not free:
        raise ValueError(f"every seat of the match at {server} is taken")
    return {"server": server, "seats": list(room.setup.seats), "seat": seat or free[0],
            "team_size": room.setup.team_size, "match_seconds": room.setup.match_seconds}


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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play a match against bots.")
    parser.add_argument("--team-size", type=int, choices=TEAM_SIZES, default=1,
                        help="cars a team: 1 (1v1, the default), 2 (2v2) or 3 (3v3); on this "
                             "computer, you drive one of blue's, bots the others")
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
    lan = parser.add_argument_group("LAN play", "join a match that rocketsim-server plays")
    lan.add_argument("--join", nargs="?", const="auto", metavar="ADDRESS",
                     help="join the match of the server on this network, or of the one at "
                          "ADDRESS (HOST or HOST:PORT)")
    lan.add_argument("--seat", help="the car you drive, such as orange1 (default: the first "
                                    "free seat)")
    lan.add_argument("--seats", help="for the first to join: the cars players drive, such as "
                                     "blue1,orange1; with --team-size and --seconds, it sets the "
                                     "match (default: the room's)")
    parser.add_argument("--stats", action="store_true",
                        help="show the stats from the start: the frame rate, and in a LAN match "
                             "the input delay and the frames shown twice or skipped (F3, or the "
                             "controller's Back button, shows and hides them)")
    arguments = parser.parse_args(argv)
    width, height = arguments.window
    lan_settings: dict[str, Any] = {}
    if arguments.join is None and (arguments.seat or arguments.seats):
        parser.error("--seat and --seats: only with --join")
    if arguments.join is not None:
        if arguments.seed is not None:
            parser.error("--seed: a LAN match takes the server's seeds (rocketsim-server --seed)")
        try:
            lan_settings = lan_arguments(arguments.join, arguments.seat, arguments.seats,
                                         arguments.team_size, arguments.seconds)
        except (ConnectionError, ValueError) as error:
            parser.error(str(error))
    settings = {name: getattr(arguments, name) for name in
                ("field_of_view", "distance", "height", "angle", "stiffness", "transition_speed")}
    controller_settings = {name: getattr(arguments, name)
                           for name in ("deadzone", "steering_sensitivity", "aerial_sensitivity")}
    match = {"match_seconds": arguments.seconds, "team_size": arguments.team_size, **lan_settings}
    try:
        env = SoccarEnv(render_mode="rgb_array", render_width=width, render_height=height,
                        camera=settings, input_mode=arguments.input,
                        controller=controller_settings, **match)
    except ValueError as error:
        parser.error(str(error))
    try:
        env.reset(seed=arguments.seed)
    except (ConnectionError, JoinRefused) as error:
        env.close()
        parser.exit(1, f"rocketsim-play: {error}\n")
    read_inputs = pad_inputs if arguments.input == InputMode.ANALOG.value else pressed_buttons
    pygame.init()
    sdl_controller.init()
    window, renderer = _open_window(width, height)
    controller = None
    if sdl_controller.get_count() and sdl_controller.is_controller(0):
        controller = sdl_controller.Controller(0)
    frame_seconds = 1 / env.metadata["render_fps"]
    stats, is_showing_stats = PlayStats(frame_seconds), arguments.stats
    stats_font = pygame.font.Font(None, max(16, height // 36))
    stats_texture, stats_drawn_at = None, -math.inf
    playing = True
    frame_due = time.perf_counter()
    try:
        while True:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.WINDOWCLOSE) or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN and not playing:
                    try:
                        env.reset()
                    except (ConnectionError, JoinRefused) as error:
                        print(f"rocketsim-play: {error}", file=sys.stderr)
                        return
                    playing = True
                if (event.type == pygame.CONTROLLERDEVICEADDED and controller is None
                        and sdl_controller.is_controller(event.device_index)):
                    controller = sdl_controller.Controller(event.device_index)
                if (event.type == pygame.KEYDOWN and event.key == STATS_KEY) or (
                        event.type == pygame.CONTROLLERBUTTONDOWN
                        and event.button == STATS_BUTTON):
                    is_showing_stats = not is_showing_stats
            stepped = playing
            if playing:
                action = read_inputs(pygame.key.get_pressed(), pygame.mouse.get_pressed(),
                                     controller)
                _, _, terminated, truncated, _ = env.step(action)
                playing = not (terminated or truncated)
            frame = env.render()
            image = pygame.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
            Texture.from_surface(renderer, image).draw()
            sample = env.sample()
            if is_showing_stats:
                if time.perf_counter() - stats_drawn_at >= STATS_REFRESH_SECONDS:
                    stats_texture = Texture.from_surface(
                        renderer, stats_image(stats.lines(sample), stats_font))
                    stats_drawn_at = time.perf_counter()
                stats_texture.draw(dstrect=(8, 8, stats_texture.width, stats_texture.height))
            renderer.present()
            stats.add(time.perf_counter(), sample, stepped)
            title = caption(sample)
            if window.title != title:
                window.title = title
            frame_due = wait_for(frame_due + frame_seconds)
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
