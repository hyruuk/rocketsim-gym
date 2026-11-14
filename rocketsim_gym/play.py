"""Play a match in a window: against bots on this computer, or with other players on this
network, 1v1, 2v2 or 3v3.

    rocketsim-play                     the start menu: play, host or join, settings
    rocketsim-play --no-menu           straight into a match against bots
    rocketsim-play --join [ADDRESS]    straight into a LAN match
    rocketsim-play --experiment ...    the flags alone: no menu, no saved settings

The start menu plays a match against bots, hosts a LAN match (it starts ``rocketsim-server``
on this computer and joins it), joins one a server on this network plays, and changes the
settings: the display, Rocket League's camera and controller settings, and what holds each
button. The settings are kept in a file (:mod:`rocketsim_gym.settings`), and the flags given
override them for one run. The window opens fullscreen; F11 switches between fullscreen and a
window. During a match, Esc (or a controller's Start) opens the menu: against bots it pauses the
match; in a LAN match the match goes on.

With ``--join``, the window joins the match a ``rocketsim-server`` plays on this network (the one
at ADDRESS, HOST or HOST:PORT, if given): in the seat ``--seat`` names, or the first free one, of
the room's match, or, for the first player to join, of the match that ``--seats`` (the cars
players drive), ``--team-size`` and ``--seconds`` set. ``--team-size``, ``--seconds`` or
``--seed`` alone start a match against bots at once.

``--experiment`` is for sessions that must run the same on every computer: no menu, no settings
file (unless ``--settings`` names one, which is then only read), a window rather than
fullscreen unless ``--fullscreen``, frames of ``--window``'s size unless ``--resolution``, and
Esc quits, as the window always did.

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

The menu's Controls screen binds them anew, and the settings file keeps the bindings. When a
match is over, Enter plays another (in a LAN match, once every seat's player is ready).

F3 (fn+F3 on most Mac keyboards) or the controller's Back button shows and hides the stats in the
window's corner: the frames shown a second and the slowest, and in a LAN match the input delay,
the server's frames shown twice or skipped, and the room's and the connection's state.
``--stats`` shows them from the start. They are drawn over the window, never into the game's
frames.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402
from pygame._sdl2 import controller as sdl_controller  # noqa: E402
from pygame._sdl2.video import Renderer, Texture, Window  # noqa: E402

from .camera import CameraSettings  # noqa: E402
from .controls import BUTTON_COUNT, AnalogInput, Button, InputMode, analog_from_buttons  # noqa: E402
from .env import TEAM_SIZES, SoccarEnv  # noqa: E402
from .lan.connection import JoinRefused, find_servers, server_address  # noqa: E402
from .lan.protocol import DEFAULT_PORT, TICKS_PER_STEP, ProtocolError, Setup  # noqa: E402
from .match import TICKS_PER_SECOND  # noqa: E402
from .menu import (Command, Menu, MenuPainter, Request, Row, ServerFinder,  # noqa: E402
                   pause_screen, start_menu)
from .pacing import wait_for  # noqa: E402
from .settings import (MOUSE_BUTTONS, PAD_BUTTONS, Bindings, PlaySettings,  # noqa: E402
                       ResolvedBindings, SettingsError, key_name, load_settings,
                       save_settings, settings_path)

#: A trigger holds its button past this share of its travel, the stick past this deflection.
TRIGGER_PRESSED = 0.3
STICK_PRESSED = 0.5
_AXIS_MAXIMUM = 32767
#: A stick moves the menu's selection past this share of its travel, and again once back
#: within :data:`_STICK_REST`.
_STICK_MENU, _STICK_REST = 0.6, 0.3
#: The key and the controller button that show and hide the stats.
STATS_KEY, STATS_BUTTON = pygame.K_F3, pygame.CONTROLLER_BUTTON_BACK
#: How often the stats' text changes, so that it can be read (s).
STATS_REFRESH_SECONDS = 0.5
#: The span the stats count the server's frames shown twice or skipped over (s).
SMOOTHNESS_SECONDS = 10.0
#: How long a server's news stays over the match, in seconds.
NEWS_SECONDS = 5.0
#: How long the window waits for a server it starts to be ready, in seconds.
HOST_START_SECONDS = 20.0
_MOUSE_NAMES = {index + 1: name for name, index in MOUSE_BUTTONS.items()}
_PAD_NAMES = {number: name for name, number in PAD_BUTTONS.items()}


class Controller(Protocol):
    """What is read from a game controller: SDL's ``pygame._sdl2.controller.Controller``."""

    def get_button(self, button: int) -> bool: ...

    def get_axis(self, axis: int) -> int: ...


@cache
def default_bindings() -> ResolvedBindings:
    """Rocket League's PC and controller defaults, by pygame's numbers."""
    return Bindings().resolve()


def held_buttons(keys_held: Mapping[int, bool] | Sequence[bool], mouse_held: Sequence[bool],
                 controller: Controller | None,
                 bindings: ResolvedBindings | None = None) -> np.ndarray:
    """The buttons the held keys, mouse buttons and controller buttons hold, by ``bindings``
    (Rocket League's defaults if None); not the controller's triggers and stick."""
    bindings = bindings or default_bindings()
    action = np.zeros(BUTTON_COUNT, np.int8)
    for key, buttons in bindings.keys.items():
        if keys_held[key]:
            action[list(buttons)] = 1
    for index, buttons in bindings.mouse.items():
        if index < len(mouse_held) and mouse_held[index]:
            action[list(buttons)] = 1
    if controller is not None:
        for sdl_button, buttons in bindings.pad.items():
            if controller.get_button(sdl_button):
                action[list(buttons)] = 1
    return action


def pressed_buttons(keys_held: Mapping[int, bool] | Sequence[bool], mouse_held: Sequence[bool],
                    controller: Controller | None,
                    bindings: ResolvedBindings | None = None) -> np.ndarray:
    """The action the held keys, mouse buttons and controller make.

    :param keys_held: whether each key is held, by pygame key (``pygame.key.get_pressed()``).
    :param mouse_held: whether each mouse button is held (``pygame.mouse.get_pressed()``).
    :param controller: the controller, if one is connected.
    :param bindings: what holds each button; Rocket League's defaults if None.
    """
    action = held_buttons(keys_held, mouse_held, controller, bindings)
    if controller is not None:
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
               controller: Controller | None,
               bindings: ResolvedBindings | None = None) -> np.ndarray:
    """The analog action the held keys, mouse buttons and controller make: the controller's
    triggers and stick as far as they are pushed, a key or button as a full push.

    :param keys_held: whether each key is held, by pygame key (``pygame.key.get_pressed()``).
    :param mouse_held: whether each mouse button is held (``pygame.mouse.get_pressed()``).
    :param controller: the controller, if one is connected.
    :param bindings: what holds each button; Rocket League's defaults if None.
    """
    action = analog_from_buttons(held_buttons(keys_held, mouse_held, controller, bindings))
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


def caption(sample: Mapping[str, Any], escape: str = "quits") -> str:
    """The window's title: what the keys do now; in a LAN match, also the room's state and the
    delay of the player's controls."""
    is_over = sample["phase"] == "ended" or sample.get("room") in ("ended", "stopped", "closing")
    keys = f"Enter plays another match, Esc {escape}" if is_over else f"Esc {escape}"
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


def apply_flags(settings: PlaySettings, arguments: argparse.Namespace) -> PlaySettings:
    """``settings`` with the flags given this run over them.

    :raises ValueError: a flag's value out of its range.
    """
    changes: dict[str, Any] = {}
    if arguments.fullscreen is not None:
        changes["fullscreen"] = arguments.fullscreen
    if arguments.window is not None:
        changes["window"] = arguments.window
        if arguments.fullscreen is None:
            changes["fullscreen"] = False
    if arguments.resolution is not None:
        changes["resolution"] = None if arguments.resolution == "auto" else arguments.resolution
    if arguments.vsync is not None:
        changes["vsync"] = arguments.vsync
    if arguments.input is not None:
        changes["input_mode"] = InputMode(arguments.input)
    for group, names in (("camera", _CAMERA_FLAGS), ("controller", _CONTROLLER_FLAGS)):
        values = {name: getattr(arguments, name) for name in names
                  if getattr(arguments, name) is not None}
        if values:
            changes[group] = values
    return settings.replace(**changes)


_CAMERA_FLAGS = ("field_of_view", "distance", "height", "angle", "stiffness", "transition_speed")
_CONTROLLER_FLAGS = ("deadzone", "steering_sensitivity", "aerial_sensitivity")


class HostedServer:
    """A ``rocketsim-server`` this window started, in a process of its own, and what it says.

    :param port: its port.
    :param log_folder: where its match logs go.
    :raises ConnectionError: it did not start, with why.
    """

    def __init__(self, port: int = DEFAULT_PORT, log_folder: Path = Path("matches")) -> None:
        self.port = port
        self.addresses: list[str] = []
        self._lines: collections.deque[tuple[float, str]] = collections.deque(maxlen=50)
        self._ready = threading.Event()
        self._process = subprocess.Popen(
            [sys.executable, "-m", "rocketsim_gym.lan.server", "--port", str(port),
             "--log-dir", str(log_folder)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        threading.Thread(target=self._read, daemon=True, name="rocketsim-gym-host").start()
        deadline = time.monotonic() + HOST_START_SECONDS
        while not self._ready.wait(0.05):
            if self._process.poll() is not None or time.monotonic() > deadline:
                self.stop()
                said = " ".join(line for _, line in self._lines) or "no reason given"
                if "already in use" in said or "Another server" in said:
                    said = ("a server is already running on this computer: join it instead "
                            "(Join a LAN match)")
                raise ConnectionError(f"the server did not start: {said}")

    def _read(self) -> None:
        for line in self._process.stdout:
            line = line.strip()
            if not line:
                continue
            found = re.search(r"--join (\S+)", line)
            if found:
                self.addresses.append(found[1])
            if line.startswith("commands:"):
                self._ready.set()
                continue
            # The server's news starts with the time; the rest is its address and the like.
            self._lines.append((time.monotonic(), re.sub(r"^\d\d:\d\d:\d\d ", "", line)))

    def news(self) -> str:
        """What the server said last, if it said it within :data:`NEWS_SECONDS`."""
        if self._lines and time.monotonic() - self._lines[-1][0] < NEWS_SECONDS \
                and self._ready.is_set():
            return self._lines[-1][1]
        return ""

    def command(self, text: str) -> None:
        """One of the experimenter's commands (resume, end, bot SEAT...)."""
        try:
            self._process.stdin.write(text + "\n")
            self._process.stdin.flush()
        except (OSError, ValueError):
            pass

    def stop(self) -> None:
        """Close the server, its players told; kill it if it does not close."""
        if self._process.poll() is None:
            self.command("quit")
            try:
                self._process.wait(5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()


class PlayWindow:
    """The window: its menus, the match in it, and the server it hosts, if any.

    :param settings: the settings the menu shows, changes and saves (the settings file's).
    :param flags: the flags of this run, laid over the settings for each match.
    :param settings_file: where changed settings are saved; None saves none.
    :param experiment: whether this is a session of an experiment: no menu, Esc quits.
    """

    def __init__(self, settings: PlaySettings, flags: argparse.Namespace,
                 settings_file: Path | None, experiment: bool) -> None:
        self._flags = flags
        self._settings_file = settings_file
        self._experiment = experiment
        self._finder = ServerFinder()
        self.menu = Menu(settings, self._settings_changed, self._finder, settings_file)
        self.menu.push(start_menu(self.menu))
        self._pause: Menu | None = None
        self._env: SoccarEnv | None = None
        self._request: Request | None = None
        self._host: HostedServer | None = None
        self._playing = False
        self._bindings = self._session().bindings.resolve()
        self._controller: sdl_controller.Controller | None = None
        self._stick_held: dict[int, bool] = {}
        pygame.init()
        sdl_controller.init()
        if sdl_controller.get_count() and sdl_controller.is_controller(0):
            self._controller = sdl_controller.Controller(0)
        sizes = pygame.display.get_desktop_sizes()
        self._screen = sizes[0] if sizes else None
        session = self._session()
        self._window = Window("rocketsim-gym", size=self._window_size(session),
                              fullscreen_desktop=session.fullscreen)
        try:
            self._renderer = Renderer(self._window, vsync=session.vsync)
        except pygame.error as error:
            print(f"rocketsim-play: no vsync on this machine ({error}); frames may tear",
                  file=sys.stderr)
            self._renderer = Renderer(self._window)
        self._fullscreen = session.fullscreen
        self._input_mode = session.input_mode
        self._mouse_visible: bool | None = None
        self._stats = PlayStats(TICKS_PER_STEP / TICKS_PER_SECOND)
        self._showing_stats = bool(flags.stats)
        self._stats_texture: Texture | None = None
        self._stats_drawn_at = -math.inf
        self._stepped = False
        self._text_input = False
        self._painter: MenuPainter | None = None
        self._set_frame_size(session.frame_size(self._screen))

    # Settings

    def _session(self) -> PlaySettings:
        """The settings a match plays with now: the menu's, with this run's flags over them."""
        session = apply_flags(self.menu.settings, self._flags)
        if self._experiment and session.resolution is None:
            session = session.replace(resolution=session.window)
        return session

    def _settings_changed(self, settings: PlaySettings) -> None:
        if self._settings_file is not None:
            try:
                save_settings(settings, self._settings_file)
            except OSError as error:
                self.menu.message = f"Could not save the settings: {error}"
        session = apply_flags(settings, self._flags)
        self._bindings = session.bindings.resolve()
        if hasattr(self, "_window"):
            self._show_fullscreen(session.fullscreen, session)
            if self._env is None:
                self._set_frame_size(session.frame_size(self._screen))

    def _window_size(self, session: PlaySettings) -> tuple[int, int]:
        return (self._screen or session.window) if session.fullscreen else session.window

    def _show_fullscreen(self, fullscreen: bool, session: PlaySettings) -> None:
        if fullscreen == self._fullscreen and (fullscreen or self._window.size == session.window):
            return
        self._fullscreen = fullscreen
        if fullscreen:
            self._window.set_fullscreen(desktop=True)
        else:
            self._window.set_windowed()
            self._window.size = session.window

    def _toggle_fullscreen(self) -> None:
        """F11: fullscreen or a window, from now on (saved, but in experiment mode)."""
        fullscreen = not self._fullscreen
        # F11 is the player's say over this run's flags.
        self._flags.fullscreen = fullscreen
        if self._experiment:
            self._show_fullscreen(fullscreen, self._session())
        else:
            self.menu.change(fullscreen=fullscreen)

    def _set_frame_size(self, size: tuple[int, int]) -> None:
        self._renderer.logical_size = size
        self._stats_font = pygame.font.Font(None, max(16, size[1] // 36))
        self._stats_drawn_at = -math.inf
        if self._painter is None or (self._painter.width, self._painter.height) != size:
            self._painter = MenuPainter(*size)

    # Matches

    def start(self, request: Request) -> None:
        """Start the match ``request`` asks for.

        :raises ConnectionError, JoinRefused: a LAN match could not be joined or hosted.
        """
        session = self._session()
        size = session.frame_size(self._screen)
        lan: dict[str, Any] = {}
        if request.kind == "host":
            self._host = HostedServer()
            lan = {"server": f"127.0.0.1:{self._host.port}", "seats": list(request.seats),
                   "seat": request.seat}
        elif request.kind == "join":
            lan = {"server": request.server, "seats": list(request.seats),
                   "seat": request.seat}
        try:
            env = SoccarEnv(render_mode="rgb_array", render_width=size[0],
                            render_height=size[1], camera=dataclasses.asdict(session.camera),
                            input_mode=session.input_mode.value,
                            controller=dataclasses.asdict(session.controller),
                            team_size=request.team_size,
                            match_seconds=request.seconds, **lan)
            try:
                env.reset(seed=request.seed)
            except BaseException:
                env.close()
                raise
        except BaseException:
            self._stop_host()
            raise
        self._env, self._request, self._playing = env, request, True
        self._input_mode = session.input_mode
        self._set_frame_size(size)

    def leave(self) -> None:
        """Leave the match for the start menu."""
        if self._env is not None:
            self._env.close()
        self._env, self._request, self._pause = None, None, None
        self._stop_host()
        self._set_frame_size(self._session().frame_size(self._screen))

    def _stop_host(self) -> None:
        if self._host is not None:
            self._host.stop()
            self._host = None

    def close(self) -> None:
        self.leave()
        self._finder.close()
        pygame.quit()

    # The loop

    def run(self, request: Request | None) -> None:
        """Show the menu, or play ``request`` at once; return when the player quits."""
        if request is not None:
            self.start(request)
        # The env steps two ticks a step: 60 frames a second, a LAN match's rate too.
        frame_seconds = TICKS_PER_STEP / TICKS_PER_SECOND
        frame_due = time.perf_counter()
        while True:
            for event in pygame.event.get():
                if self._handle(event):
                    return
            surface = self._frame()
            Texture.from_surface(self._renderer, surface).draw()
            sample = self._env.sample() if self._env is not None else None
            if sample is not None and self._showing_stats:
                self._draw_stats(sample)
            self._renderer.present()
            if sample is not None:
                self._stats.add(time.perf_counter(), sample, self._stepped)
            title = self._title()
            if self._window.title != title:
                self._window.title = title
            frame_due = wait_for(frame_due + frame_seconds)

    def _draw_stats(self, sample: Mapping[str, Any]) -> None:
        """The stats in the window's corner, over the frame, never in it."""
        if time.perf_counter() - self._stats_drawn_at >= STATS_REFRESH_SECONDS:
            self._stats_texture = Texture.from_surface(
                self._renderer, stats_image(self._stats.lines(sample), self._stats_font))
            self._stats_drawn_at = time.perf_counter()
        self._stats_texture.draw(dstrect=(8, 8, self._stats_texture.width,
                                          self._stats_texture.height))

    def _handle(self, event: pygame.event.Event) -> bool:
        """Take one event; True when the player quits."""
        if event.type in (pygame.QUIT, pygame.WINDOWCLOSE):
            return True
        if (event.type == pygame.CONTROLLERDEVICEADDED and self._controller is None
                and sdl_controller.is_controller(event.device_index)):
            self._controller = sdl_controller.Controller(event.device_index)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
            self._toggle_fullscreen()
            return False
        capturing = self._pause is not None and self._pause.capturing or self.menu.capturing
        if not capturing and ((event.type == pygame.KEYDOWN and event.key == STATS_KEY) or (
                event.type == pygame.CONTROLLERBUTTONDOWN and event.button == STATS_BUTTON)):
            self._showing_stats = not self._showing_stats
            return False
        menu = self._pause if self._env is not None else self.menu
        if self._env is not None and self._pause is None:
            return self._handle_in_match(event)
        request = self._menu_event(menu, event)
        if request is None:
            return False
        if request.kind == "quit":
            return True
        try:
            self.start(request)
        except (ConnectionError, JoinRefused, ValueError) as error:
            self.menu.message = str(error)
        return False

    def _handle_in_match(self, event: pygame.event.Event) -> bool:
        opens_menu = ((event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
                      or (event.type == pygame.CONTROLLERBUTTONDOWN
                          and event.button == pygame.CONTROLLER_BUTTON_START))
        replays = ((event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN,
                                                                   pygame.K_KP_ENTER))
                   or (event.type == pygame.CONTROLLERBUTTONDOWN
                       and event.button == pygame.CONTROLLER_BUTTON_A))
        if opens_menu:
            if self._experiment:
                return True
            if not self._playing:
                self.leave()
            else:
                self._open_pause()
        elif replays and not self._playing:
            try:
                self._env.reset()
                self._playing = True
            except (ConnectionError, JoinRefused) as error:
                print(f"rocketsim-play: {error}", file=sys.stderr)
                if self._experiment:
                    return True
                self.leave()
                self.menu.message = str(error)
        return False

    def _open_pause(self) -> None:
        solo = self._request is not None and self._request.kind == "solo"
        pause = Menu(self.menu.settings, self._settings_changed, self._finder,
                     self._settings_file)

        def resume() -> None:
            self._pause = None

        def leave() -> None:
            self.leave()

        def host_rows() -> list[Row]:
            if self._host is None:
                return []
            rows = []
            if self._env is not None and self._env.sample().get("room") == "paused":
                rows.append(Row("Resume the match for everyone",
                                accept=lambda: self._host_command("resume")))
            rows.append(Row("End the match for everyone",
                            accept=lambda: self._host_command("end")))
            return rows

        pause.push(pause_screen(pause, resume, leave, host_rows, paused_game=solo))
        pause.bottom_back = resume
        self._pause = pause

    def _host_command(self, command: str) -> None:
        if self._host is not None:
            self._host.command(command)
        self._pause = None

    def _menu_event(self, menu: Menu, event: pygame.event.Event) -> Request | None:
        """A menu's answer to ``event``, if any."""
        if menu.capturing is not None:
            binding = None
            if event.type == pygame.KEYDOWN and event.key != pygame.K_ESCAPE:
                binding = "key:" + key_name(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button in _MOUSE_NAMES:
                binding = "mouse:" + _MOUSE_NAMES[event.button]
            elif event.type == pygame.CONTROLLERBUTTONDOWN and event.button in _PAD_NAMES:
                binding = "pad:" + _PAD_NAMES[event.button]
            if binding is not None:
                menu.bind(binding)
                return None
        if menu.typing is not None and event.type == pygame.TEXTINPUT:
            menu.type(event.text)
            return None
        command = self._command(menu, event)
        if command is None:
            return None
        return menu.command(command)

    def _command(self, menu: Menu, event: pygame.event.Event) -> Command | None:
        if event.type == pygame.KEYDOWN:
            keys = {pygame.K_UP: Command.UP, pygame.K_DOWN: Command.DOWN,
                    pygame.K_LEFT: Command.LEFT, pygame.K_RIGHT: Command.RIGHT,
                    pygame.K_RETURN: Command.ACCEPT, pygame.K_KP_ENTER: Command.ACCEPT,
                    pygame.K_ESCAPE: Command.BACK, pygame.K_BACKSPACE: Command.CLEAR,
                    pygame.K_DELETE: Command.CLEAR}
            if menu.typing is None:
                keys[pygame.K_SPACE] = Command.ACCEPT
            return keys.get(event.key)
        if event.type == pygame.CONTROLLERBUTTONDOWN:
            return {pygame.CONTROLLER_BUTTON_DPAD_UP: Command.UP,
                    pygame.CONTROLLER_BUTTON_DPAD_DOWN: Command.DOWN,
                    pygame.CONTROLLER_BUTTON_DPAD_LEFT: Command.LEFT,
                    pygame.CONTROLLER_BUTTON_DPAD_RIGHT: Command.RIGHT,
                    pygame.CONTROLLER_BUTTON_A: Command.ACCEPT,
                    pygame.CONTROLLER_BUTTON_B: Command.BACK,
                    pygame.CONTROLLER_BUTTON_START: Command.BACK,
                    pygame.CONTROLLER_BUTTON_X: Command.CLEAR}.get(event.button)
        if event.type == pygame.CONTROLLERAXISMOTION and event.axis in (
                pygame.CONTROLLER_AXIS_LEFTX, pygame.CONTROLLER_AXIS_LEFTY):
            value = event.value / _AXIS_MAXIMUM
            if abs(value) < _STICK_REST:
                self._stick_held[event.axis] = False
            elif abs(value) > _STICK_MENU and not self._stick_held.get(event.axis):
                self._stick_held[event.axis] = True
                if event.axis == pygame.CONTROLLER_AXIS_LEFTY:
                    return Command.UP if value < 0 else Command.DOWN
                return Command.LEFT if value < 0 else Command.RIGHT
            return None
        if event.type == pygame.MOUSEMOTION:
            index = self._painter.row_at(event.pos)
            if index is not None:
                menu.select(index)
            return None
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 3:
                return Command.BACK
            index = self._painter.row_at(event.pos) if event.button == 1 else None
            if index is None:
                return None
            menu.select(index)
            row = menu.view().rows[index]
            if row.adjust is not None:
                value = self._painter.value_rects[index]
                left = value is not None and event.pos[0] < value.left + value.width // 3
                return Command.LEFT if left else Command.RIGHT
            return Command.ACCEPT
        return None

    def _frame(self) -> pygame.Surface:
        """This frame of the window: the menu, or the match (under the pause menu, if open)."""
        in_menu = self._env is None or self._pause is not None
        if in_menu != self._mouse_visible:
            self._mouse_visible = in_menu
            pygame.mouse.set_visible(in_menu)
            if in_menu:
                pygame.key.set_repeat(350, 45)
            else:
                pygame.key.set_repeat()
        active = self._pause if self._env is not None else self.menu
        typing = active is not None and active.typing is not None
        if typing != self._text_input:
            self._text_input = typing
            (pygame.key.start_text_input if typing else pygame.key.stop_text_input)()
        if active is not None:
            active.tick()
        if self._env is None:
            return self._painter.paint(self.menu.view())
        solo = self._request is not None and self._request.kind == "solo"
        self._stepped = self._playing and not (solo and self._pause is not None)
        if self._stepped:
            if self._pause is None:
                read = pad_inputs if self._input_mode is InputMode.ANALOG else pressed_buttons
                action = read(pygame.key.get_pressed(), pygame.mouse.get_pressed(),
                              self._controller, self._bindings)
            else:
                action = self._env.action_space.sample() * 0
            _, _, terminated, truncated, _ = self._env.step(action)
            self._playing = not (terminated or truncated)
        frame = self._env.render()
        if self._pause is not None:
            return self._painter.paint(self._pause.view(), background=frame)
        surface = pygame.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]),
                                          "RGB")
        footer = self._footer()
        if footer:
            surface = surface.copy()
            self._painter.footer(surface, footer)
        return surface

    def _footer(self) -> str:
        """News over the match: where others join a match hosted here, and what its server
        says."""
        if self._host is None:
            return ""
        parts = []
        room = self._env.sample().get("room")
        if room in (None, "lobby", "paused") and self._host.addresses:
            parts.append(f"Others join at {self._host.addresses[0]}")
        news = self._host.news()
        if news:
            parts.append(news)
        return "   ·   ".join(parts)

    def _title(self) -> str:
        if self._env is None:
            return "rocketsim-gym"
        return caption(self._env.sample(),
                       "quits" if self._experiment else
                       ("opens the menu" if self._playing else "goes back to the menu"))


def _window_size(text: str) -> tuple[int, int]:
    """``WIDTHxHEIGHT``, in pixels."""
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: a size such as 1280x720") from None
    return width, height


def _resolution(text: str) -> tuple[int, int] | str:
    """``auto`` or ``WIDTHxHEIGHT``."""
    return "auto" if text.lower() == "auto" else _window_size(text)


def parser() -> argparse.ArgumentParser:
    """rocketsim-play's flags."""
    parser = argparse.ArgumentParser(
        description="Play Rocket League-style soccar: against bots, or with others on this "
                    "network. Without flags, a start menu opens.",
        epilog=f"Settings are kept in {settings_path()}; flags override them for one run.")
    parser.add_argument("--team-size", type=int, choices=TEAM_SIZES,
                        help="cars a team: 1 (1v1, the default), 2 (2v2) or 3 (3v3); on this "
                             "computer, you drive one of blue's, bots the others. Starts a "
                             "match at once")
    parser.add_argument("--seconds", type=float, help="regulation time (default 300). Starts a "
                                                      "match at once")
    parser.add_argument("--seed", type=int, default=None, help="seed of the kickoff spots. "
                                                               "Starts a match at once")
    parser.add_argument("--no-menu", action="store_true",
                        help="start a match against bots at once")
    session = parser.add_argument_group("settings", "the settings file, and experiment mode")
    session.add_argument("--settings", type=Path, metavar="FILE",
                         help=f"the settings file to use (default {settings_path()})")
    session.add_argument("--experiment", action="store_true",
                         help="for an experiment's sessions: no menu, no settings file "
                              "unless --settings (then only read), a window unless "
                              "--fullscreen, frames of --window's size unless --resolution, "
                              "and Esc quits")
    display = parser.add_argument_group("display")
    shown = display.add_mutually_exclusive_group()
    shown.add_argument("--fullscreen", action="store_true", default=None,
                       help="fill the screen (the default; F11 switches)")
    shown.add_argument("--windowed", action="store_false", dest="fullscreen",
                       help="a window of --window's size")
    display.add_argument("--window", type=_window_size, metavar="WxH",
                         help="the window's size, in pixels (default 1280x720); alone, it "
                              "means a window, not fullscreen")
    display.add_argument("--resolution", type=_resolution, metavar="WxH",
                         help="the frames' size: WIDTHxHEIGHT, or auto (the default): the "
                              "window's, or in fullscreen the screen's shape at most 1080 tall")
    display.add_argument("--vsync", action=argparse.BooleanOptionalAction, default=None,
                         help="show each frame in step with the screen's refresh (default on)")
    display.add_argument("--stats", action="store_true",
                         help="show the stats from the start: the frame rate, and in a LAN "
                              "match the input delay and the frames shown twice or skipped (F3, "
                              "or the controller's Back button, shows and hides them)")
    display.add_argument("--list-controllers", action="store_true",
                         help="list the controllers connected, and quit")
    parser.add_argument("--input", choices=[mode.value for mode in InputMode],
                        help="analog: the controller's triggers and stick as far as they are "
                             "pushed (default); digital: all or nothing")
    controller = parser.add_argument_group("controller", "Rocket League's controller settings")
    for name, default, help_text in (("deadzone", 0.1, "0 to 0.95"),
                                     ("steering_sensitivity", 1.0, "0.1 to 10"),
                                     ("aerial_sensitivity", 1.0, "0.1 to 10")):
        controller.add_argument("--" + name.replace("_", "-"), type=float, dest=name,
                                help=f"{help_text}; default {default:g}")
    camera = parser.add_argument_group("camera", "Rocket League's camera settings")
    defaults = CameraSettings()
    for name, help_text in (("field_of_view", "horizontal, in degrees (60 to 110)"),
                            ("distance", "behind the car (100 to 400)"),
                            ("height", "above the car (40 to 200)"),
                            ("angle", "pitch, in degrees (-15 to 0)"),
                            ("stiffness", "0 to 1"), ("transition_speed", "1 to 2")):
        camera.add_argument("--" + name.replace("_", "-"), type=float, dest=name,
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
    return parser


def _list_controllers() -> None:
    pygame.init()
    sdl_controller.init()
    count = sdl_controller.get_count()
    if not count:
        print("no controller connected")
    for index in range(count):
        kind = ("a game controller" if sdl_controller.is_controller(index)
                else "a joystick SDL does not know as a game controller")
        name = sdl_controller.name_forindex(index) if sdl_controller.is_controller(index) \
            else pygame.joystick.Joystick(index).get_name()
        print(f"{index}: {name} ({kind})")
    pygame.quit()


def main(argv: Sequence[str] | None = None) -> None:
    flags_parser = parser()
    arguments = flags_parser.parse_args(argv)
    if arguments.list_controllers:
        _list_controllers()
        return
    if arguments.join is None and (arguments.seat or arguments.seats):
        flags_parser.error("--seat and --seats: only with --join")
    if arguments.join is not None and arguments.seed is not None:
        flags_parser.error("--seed: a LAN match takes the server's seeds (rocketsim-server "
                           "--seed)")
    experiment = arguments.experiment
    settings_file = arguments.settings if arguments.settings is not None else (
        None if experiment else settings_path())
    try:
        # In experiment mode without a settings file: a window, as the flags alone say.
        settings = (load_settings(settings_file) if settings_file is not None
                    else PlaySettings(fullscreen=False))
        apply_flags(settings, arguments)
    except (SettingsError, ValueError) as error:
        flags_parser.error(str(error))
    team_size = arguments.team_size or 1
    seconds = arguments.seconds if arguments.seconds is not None else 300.0
    request = None
    if arguments.join is not None:
        try:
            lan = lan_arguments(arguments.join, arguments.seat, arguments.seats, team_size,
                                seconds)
        except (ConnectionError, ValueError) as error:
            flags_parser.error(str(error))
        request = Request("join", lan["team_size"], lan["match_seconds"], lan["server"],
                          tuple(lan["seats"]), lan["seat"])
        try:
            setup = Setup(request.team_size, request.seats, request.seconds)
        except ProtocolError as error:
            flags_parser.error(str(error))
        if request.seat not in setup.seats:
            flags_parser.error(f"--seat {request.seat}: one of the seats players drive, "
                               f"{list(setup.seats)}")
    elif (experiment or arguments.no_menu or arguments.team_size is not None
          or arguments.seconds is not None or arguments.seed is not None):
        request = Request("solo", team_size, seconds, seed=arguments.seed)
    window = PlayWindow(settings, arguments, None if experiment else settings_file, experiment)
    try:
        try:
            window.run(request)
        except (ConnectionError, JoinRefused, ValueError) as error:
            flags_parser.exit(1, f"rocketsim-play: {error}\n")
    finally:
        window.close()


if __name__ == "__main__":
    main()
