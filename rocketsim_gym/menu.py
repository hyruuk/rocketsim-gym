"""The play window's menus: the start menu (play against bots, host or join a LAN match, the
settings) and the pause menu over a match.

A :class:`Menu` is a stack of :class:`Screen`, each a list of :class:`Row`; it takes
:class:`Command` (from the keyboard, the mouse or a controller, as the play window reads them),
keys and text, and answers with a :class:`Request` when the player asks for a match or to quit.
It draws nothing: :class:`MenuPainter` paints what :meth:`Menu.view` says, in the display's
font. The settings it changes go to the ``save`` it is given at once.
"""

from __future__ import annotations

import dataclasses
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from .camera import CameraSettings  # noqa: E402
from .controls import Button, InputMode  # noqa: E402
from .hud import CLOCK_BOX, FONT_FILE, TEAM_COLORS, WHITE  # noqa: E402
from .lan.connection import find_servers, server_address  # noqa: E402
from .lan.protocol import Announcement  # noqa: E402
from .match import Team  # noqa: E402
from .settings import (BUTTON_LABELS, DEFAULT_BINDINGS, TEAM_SIZES, Bindings,  # noqa: E402
                       PlaySettings, binding_label, seat_names)

#: The match lengths the menu offers, in seconds.
MATCH_LENGTHS = (60.0, 120.0, 180.0, 300.0, 600.0)
#: The frame sizes the menu offers, besides "auto".
RESOLUTIONS = ((1280, 720), (1600, 900), (1920, 1080), (2560, 1440))
#: How often the join screen asks the network for servers, in seconds.
SCAN_SECONDS = 2.0
#: Each camera setting the menu changes: its label, its step, and how it is shown.
CAMERA_ROWS = (("field_of_view", "Field of view", 1.0, "{:.0f}°"),
               ("distance", "Distance", 10.0, "{:.0f}"),
               ("height", "Height", 10.0, "{:.0f}"),
               ("angle", "Angle", 1.0, "{:.0f}°"),
               ("stiffness", "Stiffness", 0.05, "{:.2f}"),
               ("transition_speed", "Transition speed", 0.1, "{:.1f}"))
CONTROLLER_ROWS = (("deadzone", "Deadzone", 0.05, "{:.2f}"),
                   ("steering_sensitivity", "Steering sensitivity", 0.1, "{:.1f}"),
                   ("aerial_sensitivity", "Aerial sensitivity", 0.1, "{:.1f}"))
_RANGES = {"field_of_view": (60, 110), "distance": (100, 400), "height": (40, 200),
           "angle": (-15, 0), "stiffness": (0, 1), "transition_speed": (1, 2),
           "deadzone": (0, 0.95), "steering_sensitivity": (0.1, 10),
           "aerial_sensitivity": (0.1, 10)}


class Command(Enum):
    """What the player does in a menu."""

    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    ACCEPT = auto()
    BACK = auto()
    CLEAR = auto()


@dataclass(frozen=True)
class Request:
    """What the player asks the play window for.

    :ivar kind: ``solo`` (a match against bots on this computer), ``host`` (start a server
        here and join it), ``join`` (join a server's match) or ``quit``.
    :ivar server: the server to join, ``HOST:PORT``.
    :ivar seats: the seats players drive, for a LAN match.
    :ivar seat: the player's own seat, for a LAN match.
    """

    kind: str
    team_size: int = 1
    seconds: float = 300.0
    server: str | None = None
    seats: tuple[str, ...] = ()
    seat: str | None = None
    seed: int | None = None


QUIT = Request("quit")


@dataclass
class Row:
    """A line of a menu.

    :ivar accept: what choosing it does; it may answer with a request.
    :ivar adjust: what left (-1) and right (+1) do to its value.
    :ivar clear: what Delete or Backspace does.
    :ivar selectable: False for a line that only says something.
    """

    label: str
    value: str = ""
    accept: Callable[[], Request | None] | None = None
    adjust: Callable[[int], None] | None = None
    clear: Callable[[], None] | None = None
    selectable: bool = True


@dataclass
class Screen:
    """A page of a menu: its title, its rows (made anew each time, so they show what is now),
    and a line under the title."""

    title: str
    rows: Callable[[], list[Row]]
    subtitle: Callable[[], str] = lambda: ""
    on_close: Callable[[], None] | None = None
    #: Called every frame while the screen is on top: for news from the network.
    tick: Callable[[], None] | None = None
    selected: int = 0
    #: The selected row's label: the selection follows its row when rows come and go above it.
    selected_label: str | None = None


@dataclass(frozen=True)
class MenuView:
    """What a menu shows now."""

    title: str
    subtitle: str
    rows: tuple[Row, ...]
    selected: int
    message: str
    prompt: str
    hint: str


class ServerFinder:
    """Asks the network for servers in the background, every :data:`SCAN_SECONDS` while it
    scans; and looks up one address when asked."""

    def __init__(self, find: Callable[..., list[Announcement]] = find_servers) -> None:
        self._find = find
        self._lock = threading.Lock()
        self._servers: list[Announcement] = []
        self._scanning = False
        self._lookups: dict[str, Announcement | str | None] = {}
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()
        self._closed = False

    @property
    def servers(self) -> list[Announcement]:
        with self._lock:
            return list(self._servers)

    def scan(self, on: bool) -> None:
        """Start or stop scanning."""
        self._scanning = on
        if on:
            self._start()
            self._wake.set()

    def lookup(self, address: str) -> None:
        """Ask the server at ``address`` what it plays; :meth:`answer` says, once it has."""
        with self._lock:
            self._lookups[address] = None
        self._start()
        self._wake.set()

    def answer(self, address: str) -> Announcement | str | None:
        """The server at ``address``, a problem, or None while the lookup runs."""
        with self._lock:
            return self._lookups.get(address)

    def close(self) -> None:
        self._closed = True
        self._wake.set()

    def _start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="rocketsim-gym-server-finder")
            self._thread.start()

    def _run(self) -> None:
        next_scan = 0.0
        while not self._closed:
            with self._lock:
                pending = [address for address, answer in self._lookups.items()
                           if answer is None]
            for address in pending:
                self._answer(address)
            if self._scanning and time.monotonic() >= next_scan:
                try:
                    found = self._find()
                except ConnectionError:
                    found = []
                with self._lock:
                    self._servers = found
                next_scan = time.monotonic() + SCAN_SECONDS
            self._wake.wait(0.2)
            self._wake.clear()

    def _answer(self, address: str) -> None:
        try:
            host, port, _ = server_address(address)
            found = [room for room in self._find(hosts=(host,)) if room.port == port]
            answer: Announcement | str = (found[0] if found else
                                          f"no answer from {host}:{port}: is rocketsim-server "
                                          f"running there, with its ports open?")
        except (ConnectionError, ValueError) as error:
            answer = str(error)
        with self._lock:
            self._lookups[address] = answer


class Menu:
    """A stack of screens, the player's place in them, and the settings they change.

    :param settings: the settings the menu shows and changes.
    :param save: what is done with the settings whenever the player changes one.
    :param finder: where the join screen finds servers.
    :param settings_file: where the settings are kept, to say so; None when they are not.
    """

    def __init__(self, settings: PlaySettings, save: Callable[[PlaySettings], None],
                 finder: ServerFinder | None = None, settings_file: Path | None = None) -> None:
        self.settings = settings
        self._save = save
        self._finder = finder or ServerFinder()
        self._settings_file = settings_file
        self._stack: list[Screen] = []
        self.message = ""
        #: The button whose binding the next key or controller button sets, or None.
        self.capturing: Button | None = None
        #: The text being typed, or None.
        self.typing: str | None = None
        self._on_typed: Callable[[str], Request | None] | None = None
        self._confirm_reset = False
        #: What Esc does on the first screen (the pause menu's closes it); None does nothing.
        self.bottom_back: Callable[[], None] | None = None

    # The stack

    @property
    def screen(self) -> Screen:
        return self._stack[-1]

    def push(self, screen: Screen) -> None:
        self._stack.append(screen)
        rows = screen.rows()
        self._choose(screen, rows, self._first_selectable(rows, 0, 1))

    def pop(self) -> None:
        screen = self._stack.pop()
        if screen.on_close is not None:
            screen.on_close()
        self._confirm_reset = False

    def reset_to(self, screen: Screen) -> None:
        while self._stack:
            self.pop()
        self.push(screen)

    @property
    def depth(self) -> int:
        return len(self._stack)

    # Input

    def command(self, command: Command) -> Request | None:
        """Carry out ``command``; return what the player asked for, if anything."""
        if self.capturing is not None:
            if command is Command.BACK:
                self.capturing = None
            return None
        if self.typing is not None:
            return self._type_command(command)
        rows = self.screen.rows()
        screen = self.screen
        self._place(screen, rows)
        row = rows[screen.selected] if rows and rows[screen.selected].selectable else None
        if command in (Command.UP, Command.DOWN):
            step = -1 if command is Command.UP else 1
            self._choose(screen, rows,
                         self._first_selectable(rows, screen.selected + step, step))
        elif command in (Command.LEFT, Command.RIGHT) and row is not None and row.adjust:
            row.adjust(-1 if command is Command.LEFT else 1)
        elif command is Command.ACCEPT and row is not None:
            if row.accept is not None:
                self.message = ""
                return row.accept()
            if row.adjust is not None:
                row.adjust(1)
        elif command is Command.CLEAR and row is not None and row.clear is not None:
            row.clear()
        elif command is Command.BACK:
            if len(self._stack) > 1:
                self.pop()
                self.message = ""
            else:
                return self._back_from_bottom()
        return None

    def select(self, index: int) -> None:
        """Put the selection on row ``index``, if it can be chosen (the mouse over it)."""
        rows = self.screen.rows()
        if 0 <= index < len(rows) and rows[index].selectable:
            self._choose(self.screen, rows, index)

    def bind(self, binding: str) -> None:
        """While :attr:`capturing`, the key, mouse button or controller button pressed."""
        button, self.capturing = self.capturing, None
        if button is not None:
            self.change(bindings=self.settings.bindings.bind(button, binding))

    def type(self, text: str) -> None:
        """Text typed, while :attr:`typing`."""
        if self.typing is not None:
            self.typing += "".join(ch for ch in text if ch.isprintable())[:64]

    def start_typing(self, initial: str, done: Callable[[str], Request | None]) -> None:
        self.typing, self._on_typed = initial, done

    def _type_command(self, command: Command) -> Request | None:
        if command is Command.CLEAR:
            self.typing = self.typing[:-1]
        elif command is Command.BACK:
            self.typing = None
        elif command is Command.ACCEPT:
            text, self.typing = self.typing.strip(), None
            if text and self._on_typed is not None:
                return self._on_typed(text)
        return None

    def _back_from_bottom(self) -> Request | None:
        if self.bottom_back is not None:
            self.bottom_back()
        return None

    def tick(self) -> None:
        """Once a frame: let the screen on top take what came from the network."""
        if self._stack and self.screen.tick is not None:
            self.screen.tick()

    # What it shows

    def view(self) -> MenuView:
        screen = self.screen
        rows = screen.rows()
        self._place(screen, rows)
        prompt = ""
        if self.capturing is not None:
            prompt = (f"Press a key, a mouse button or a controller button for "
                      f"{BUTTON_LABELS[self.capturing]}  ·  Esc cancels")
        elif self.typing is not None:
            prompt = f"Server address: {self.typing}_   ·   Enter joins, Esc cancels"
        hint = ("Arrows choose and change   ·   Enter selects   ·   Esc goes back"
                + ("   ·   Delete clears" if any(row.clear for row in rows) else ""))
        return MenuView(screen.title, screen.subtitle(), tuple(rows), screen.selected,
                        self.message, prompt, hint)

    @staticmethod
    def _first_selectable(rows: Sequence[Row], start: int, step: int) -> int:
        if not rows:
            return 0
        for offset in range(len(rows)):
            index = (start + offset * step) % len(rows)
            if rows[index].selectable:
                return index
        return 0

    @staticmethod
    def _choose(screen: Screen, rows: Sequence[Row], index: int) -> None:
        screen.selected = index
        screen.selected_label = rows[index].label if 0 <= index < len(rows) else None

    def _place(self, screen: Screen, rows: Sequence[Row]) -> None:
        """Keep the selection on the row it was on, wherever that row is now."""
        index = screen.selected
        if screen.selected_label is not None and not (
                0 <= index < len(rows) and rows[index].label == screen.selected_label):
            index = next((i for i, row in enumerate(rows)
                          if row.selectable and row.label == screen.selected_label), index)
        screen.selected = self._clamp(rows, index)

    def _clamp(self, rows: Sequence[Row], index: int) -> int:
        if not rows:
            return 0
        index = min(max(index, 0), len(rows) - 1)
        return index if rows[index].selectable else self._first_selectable(rows, index, 1)

    # Settings

    def change(self, **changes: Any) -> None:
        """Change settings, and save them."""
        self.settings = self.settings.replace(**changes)
        self._save(self.settings)


# The screens


def start_menu(menu: Menu) -> Screen:
    """The menu's first screen."""
    return Screen("ROCKETSIM-GYM", lambda: [
        Row("Play against bots", accept=lambda: menu.push(solo_screen(menu))),
        Row("Host a LAN match", accept=lambda: menu.push(host_screen(menu))),
        Row("Join a LAN match", accept=lambda: menu.push(join_screen(menu))),
        Row("Settings", accept=lambda: menu.push(settings_screen(menu))),
        Row("Quit", accept=lambda: QUIT),
    ], subtitle=lambda: "Rocket League-style soccar")


def _match_rows(menu: Menu) -> list[Row]:
    settings = menu.settings

    def team_size(step: int) -> None:
        sizes = list(TEAM_SIZES)
        size = sizes[(sizes.index(settings.team_size) + step) % len(sizes)]
        menu.change(team_size=size)

    def length(step: int) -> None:
        lengths = sorted(set(MATCH_LENGTHS) | {settings.seconds})
        index = (lengths.index(settings.seconds) + step) % len(lengths)
        menu.change(seconds=lengths[index])

    return [Row("Team size", f"{settings.team_size}v{settings.team_size}", adjust=team_size),
            Row("Match length", _clock(settings.seconds), adjust=length)]


def solo_screen(menu: Menu) -> Screen:
    def start() -> Request:
        return Request("solo", menu.settings.team_size, menu.settings.seconds)

    return Screen("PLAY AGAINST BOTS", lambda: [
        *_match_rows(menu), Row("Start", accept=start), Row("Back", accept=menu.pop)],
        subtitle=lambda: "You drive blue's first car; bots drive the others")


@dataclass
class _SeatChoice:
    """The players of a LAN match being set up, and the seat of this one."""

    players: int = 2
    seat: str = "blue1"

    def rows(self, menu: Menu) -> list[Row]:
        team_size = menu.settings.team_size
        self.players = min(max(self.players, 1), 2 * team_size)
        seats = seat_names(team_size, self.players)
        if self.seat not in seats:
            self.seat = seats[0]

        def players(step: int) -> None:
            self.players = (self.players - 1 + step) % (2 * team_size) + 1

        def seat(step: int) -> None:
            self.seat = seats[(seats.index(self.seat) + step) % len(seats)]

        return [Row("Players", f"{self.players}: {', '.join(seats)}", adjust=players),
                Row("Your car", self.seat, adjust=seat)]

    def seats(self, menu: Menu) -> tuple[str, ...]:
        return seat_names(menu.settings.team_size, self.players)


def host_screen(menu: Menu) -> Screen:
    choice = _SeatChoice()

    def start() -> Request:
        return Request("host", menu.settings.team_size, menu.settings.seconds,
                       seats=choice.seats(menu), seat=choice.seat)

    return Screen("HOST A LAN MATCH", lambda: [
        *_match_rows(menu), *choice.rows(menu), Row("Start hosting", accept=start),
        Row("Back", accept=menu.pop)],
        subtitle=lambda: "This computer runs the server; bots drive the seats no player has")


def join_screen(menu: Menu) -> Screen:
    menu._finder.scan(True)

    def rows() -> list[Row]:
        found = [Row(_describe_room(room), accept=lambda room=room: _join_room(menu, room))
                 for room in menu._finder.servers]
        if not found:
            found = [Row("Looking for servers on this network…", selectable=False)]
        return [*found,
                Row("Enter an address", accept=lambda: menu.start_typing(
                    "", lambda text: _look_up(menu, text))),
                Row("Back", accept=menu.pop)]

    return Screen("JOIN A LAN MATCH", rows, on_close=lambda: menu._finder.scan(False),
                  subtitle=lambda: "Servers that answer on this network; or type the one "
                                   "rocketsim-server printed (HOST:PORT)")


def _describe_room(room: Announcement) -> str:
    where = f"{room.host} ({room.address or '?'}:{room.port})"
    if room.setup is None:
        return f"{where}: no match yet"
    free = [seat for seat in room.setup.seats if seat not in room.seats]
    return (f"{where}: {room.setup.team_size}v{room.setup.team_size}, "
            f"{_clock(room.setup.match_seconds)}, {len(free)} free "
            f"seat{'s' if len(free) != 1 else ''}")


def _look_up(menu: Menu, address: str) -> None:
    menu._finder.lookup(address)

    def rows() -> list[Row]:
        answer = menu._finder.answer(address)
        line = f"Asking {address}…" if answer is None else str(answer)
        return [Row(line, selectable=False), Row("Back", accept=menu.pop)]

    def tick() -> None:
        answer = menu._finder.answer(address)
        if isinstance(answer, Announcement) and menu.screen is screen:
            menu.pop()
            _join_room(menu, answer)

    screen = Screen("JOIN A LAN MATCH", rows, tick=tick)
    menu.push(screen)


def _join_room(menu: Menu, room: Announcement) -> None:
    server = f"{room.address or room.host}:{room.port}"
    if room.setup is None:
        choice = _SeatChoice()

        def join() -> Request:
            return Request("join", menu.settings.team_size, menu.settings.seconds, server,
                           choice.seats(menu), choice.seat)

        menu.push(Screen("SET UP THE MATCH", lambda: [
            *_match_rows(menu), *choice.rows(menu), Row("Join", accept=join),
            Row("Back", accept=menu.pop)],
            subtitle=lambda: f"{room.host} has no match yet: the first to join sets it"))
        return
    setup = room.setup

    def rows() -> list[Row]:
        free = [seat for seat in setup.seats if seat not in room.seats]
        seats = [Row(f"Drive {seat}", accept=lambda seat=seat: Request(
                     "join", setup.team_size, setup.match_seconds, server, setup.seats, seat))
                 for seat in free]
        if not seats:
            seats = [Row("Every seat is taken", selectable=False)]
        return [*seats, Row("Back", accept=menu.pop)]

    menu.push(Screen("CHOOSE YOUR CAR", rows, subtitle=lambda: _describe_room(room)))


def settings_screen(menu: Menu) -> Screen:
    def reset() -> None:
        if not menu._confirm_reset:
            menu._confirm_reset = True
            return
        menu._confirm_reset = False
        menu.change(**{f.name: f.default if f.default is not dataclasses.MISSING
                        else f.default_factory() for f in dataclasses.fields(PlaySettings)
                        if f.name not in ("team_size", "seconds")})
        menu.message = "Settings reset to their defaults"

    def rows() -> list[Row]:
        where = (f"Saved in {menu._settings_file}" if menu._settings_file is not None
                 else "Not saved: this run's settings only")
        return [Row("Display", accept=lambda: menu.push(display_screen(menu))),
                Row("Camera", accept=lambda: menu.push(camera_screen(menu))),
                Row("Controller", accept=lambda: menu.push(controller_screen(menu))),
                Row("Controls", accept=lambda: menu.push(controls_screen(menu))),
                Row("Press again to reset every setting" if menu._confirm_reset
                    else "Reset to defaults", accept=reset),
                Row("Back", accept=menu.pop),
                Row(where, selectable=False)]

    return Screen("SETTINGS", rows, subtitle=lambda: "They apply from the next match")


def display_screen(menu: Menu) -> Screen:
    def toggle(name: str) -> Callable[[int], None]:
        return lambda step: menu.change(**{name: not getattr(menu.settings, name)})

    def resolution(step: int) -> None:
        options = [None, *RESOLUTIONS]
        current = menu.settings.resolution
        if current not in options:
            options.append(current)
        menu.change(resolution=options[(options.index(current) + step) % len(options)])

    def window(step: int) -> None:
        options = list(RESOLUTIONS)
        current = menu.settings.window
        if current not in options:
            options.append(current)
        menu.change(window=options[(options.index(current) + step) % len(options)])

    def rows() -> list[Row]:
        settings = menu.settings
        size = settings.resolution
        return [Row("Fullscreen", _on_off(settings.fullscreen), adjust=toggle("fullscreen")),
                Row("Resolution", "Auto" if size is None else f"{size[0]}x{size[1]}",
                    adjust=resolution),
                Row("Window size", f"{settings.window[0]}x{settings.window[1]}",
                    adjust=window),
                Row("VSync (next start)", _on_off(settings.vsync), adjust=toggle("vsync")),
                Row("Back", accept=menu.pop)]

    return Screen("DISPLAY", rows,
                  subtitle=lambda: "F11 switches fullscreen at any time. Auto draws frames "
                                   "the screen's shape, at most 1080 tall")


def _number_rows(menu: Menu, group: str, rows: Sequence[tuple[str, str, float, str]]) -> list[Row]:
    current = getattr(menu.settings, group)

    def adjust(name: str, step_size: float) -> Callable[[int], None]:
        def change(step: int) -> None:
            lowest, highest = _RANGES[name]
            value = round(getattr(getattr(menu.settings, group), name) + step * step_size, 4)
            menu.change(**{group: {name: min(max(value, lowest), highest)}})
        return change

    return [Row(label, shown.format(getattr(current, name)), adjust=adjust(name, step))
            for name, label, step, shown in rows]


def camera_screen(menu: Menu) -> Screen:
    def reset() -> None:
        menu.change(camera=CameraSettings())

    return Screen("CAMERA", lambda: [
        *_number_rows(menu, "camera", CAMERA_ROWS), Row("Reset camera", accept=reset),
        Row("Back", accept=menu.pop)],
        subtitle=lambda: "Rocket League's camera settings, in its units")


def controller_screen(menu: Menu) -> Screen:
    def input_mode(step: int) -> None:
        menu.change(input_mode=InputMode.DIGITAL if menu.settings.input_mode
                     is InputMode.ANALOG else InputMode.ANALOG)

    def reset() -> None:
        menu.change(controller=PlaySettings().controller, input_mode=InputMode.ANALOG)

    return Screen("CONTROLLER", lambda: [
        Row("Triggers and stick", "As far as pushed" if menu.settings.input_mode
            is InputMode.ANALOG else "All or nothing", adjust=input_mode),
        *_number_rows(menu, "controller", CONTROLLER_ROWS), Row("Reset controller",
                                                                accept=reset),
        Row("Back", accept=menu.pop)],
        subtitle=lambda: "Rocket League's controller settings")


def controls_screen(menu: Menu) -> Screen:
    def capture(button: Button) -> Callable[[], None]:
        def start() -> None:
            menu.capturing = button
        return start

    def clear(button: Button) -> Callable[[], None]:
        def remove() -> None:
            bindings = menu.settings.bindings
            device = "keyboard" if bindings.on(button, "keyboard") else "controller"
            menu.change(bindings=bindings.clear(button, device))
        return remove

    def reset() -> None:
        menu.change(bindings=Bindings(dict(DEFAULT_BINDINGS)))

    def rows() -> list[Row]:
        bindings = menu.settings.bindings
        result = []
        for button in Button:
            shown = [binding_label(binding) if binding else "—" for binding in
                     (bindings.on(button, "keyboard"), bindings.on(button, "controller"))]
            result.append(Row(BUTTON_LABELS[button], "\t".join(shown), accept=capture(button),
                              clear=clear(button)))
        return [*result, Row("Reset controls", accept=reset), Row("Back", accept=menu.pop)]

    return Screen("CONTROLS", rows,
                  subtitle=lambda: "Enter, then press what should hold it: keyboard and mouse "
                                   "in one column, controller in the other")


def pause_screen(menu: Menu, resume: Callable[[], None], leave: Callable[[], None],
                 host_rows: Callable[[], list[Row]] = list, paused_game: bool = True) -> Screen:
    """The menu over a match: back to it, the host's commands, leave it, or quit."""
    return Screen("PAUSED" if paused_game else "MENU", lambda: [
        Row("Resume", accept=resume), *host_rows(), Row("Leave the match", accept=leave),
        Row("Quit", accept=lambda: QUIT)],
        subtitle=lambda: "" if paused_game else "The match goes on meanwhile")


def _on_off(value: bool) -> str:
    return "On" if value else "Off"


def _clock(seconds: float) -> str:
    whole = round(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


# Painting


ROW_HIGHLIGHT = (32, 112, 245)
ROW_TEXT = (225, 228, 235)
DIM_TEXT = (150, 156, 170)
MESSAGE_TEXT = (255, 200, 40)


class MenuPainter:
    """Paints a :class:`MenuView` on frames of one size, scaled to the frame's height."""

    def __init__(self, width: int, height: int) -> None:
        pygame.font.init()
        self.width, self.height = width, height
        self._scale = height / 720
        self._fonts: dict[int, pygame.font.Font] = {}
        #: Where each row was painted last, in frame pixels: for the mouse.
        self.row_rects: list[pygame.Rect] = []
        #: Where each row's value was painted last, or None: a click on its left half lowers it.
        self.value_rects: list[pygame.Rect | None] = []

    def paint(self, view: MenuView, background: np.ndarray | None = None) -> pygame.Surface:
        """The menu over ``background`` (a frame, darkened), or over the display's dark
        background."""
        if background is not None:
            surface = pygame.image.frombytes(background.tobytes(), (self.width, self.height),
                                             "RGB")
            shade = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            shade.fill((8, 10, 16, 215))
            surface.blit(shade, (0, 0))
        else:
            surface = pygame.Surface((self.width, self.height))
            surface.fill(CLOCK_BOX)
            stripe = pygame.Surface((self.width, self._px(6)))
            stripe.fill(TEAM_COLORS[Team.BLUE])
            surface.blit(stripe, (0, 0))
            stripe.fill(TEAM_COLORS[Team.ORANGE])
            surface.blit(stripe, (0, self.height - self._px(6)))
        centre = self.width // 2
        # Over a match, below its scoreboard.
        y = self._px(60 if background is None else 90)
        title = self._text(view.title, 54, WHITE)
        surface.blit(title, title.get_rect(midtop=(centre, y)))
        y += title.get_height() + self._px(6)
        if view.subtitle:
            subtitle = self._text(view.subtitle, 18, DIM_TEXT)
            surface.blit(subtitle, subtitle.get_rect(midtop=(centre, y)))
        y += self._px(52)
        row_height = self._px(40)
        rows = view.rows
        room = max(1, (self.height - y - self._px(110)) // row_height)
        first = min(max(0, view.selected - room // 2), max(0, len(rows) - room))
        panel_width = min(self.width - self._px(80), self._px(900))
        left = centre - panel_width // 2
        self.row_rects, self.value_rects = [], []
        for index, row in enumerate(rows):
            if not first <= index < first + room:
                self.row_rects.append(pygame.Rect(0, -1000, 0, 0))
                self.value_rects.append(None)
                continue
            rect = pygame.Rect(left, y, panel_width, row_height - self._px(6))
            self.row_rects.append(rect)
            selected = index == view.selected and row.selectable
            if selected:
                pygame.draw.rect(surface, ROW_HIGHLIGHT, rect, border_radius=self._px(8))
            color = WHITE if selected else (ROW_TEXT if row.selectable else DIM_TEXT)
            label = self._text(row.label, 22, color)
            value_rect = None
            if "\t" in row.value:
                # Columns, each right-aligned to its own edge: the keyboard's, the controller's.
                right = rect.right - self._px(18)
                for column in reversed(row.value.split("\t")):
                    text = self._text(column, 22, color)
                    surface.blit(text, text.get_rect(midright=(right, rect.centery)))
                    right -= self._px(190)
                surface.blit(label, label.get_rect(midleft=(rect.left + self._px(18),
                                                            rect.centery)))
            elif row.value or row.adjust:
                value = self._text(f"‹  {row.value}  ›" if row.adjust else row.value, 22, color)
                value_rect = value.get_rect(midright=(rect.right - self._px(18), rect.centery))
                surface.blit(value, value_rect)
                surface.blit(label, label.get_rect(midleft=(rect.left + self._px(18),
                                                            rect.centery)))
            else:
                surface.blit(label, label.get_rect(center=rect.center) if row.selectable
                             and not row.value else label.get_rect(
                                 midleft=(rect.left + self._px(18), rect.centery)))
            self.value_rects.append(value_rect)
            y += row_height
        bottom = self.height - self._px(40)
        if view.prompt:
            prompt = self._text(view.prompt, 22, MESSAGE_TEXT)
            box = prompt.get_rect(midbottom=(centre, bottom - self._px(40))).inflate(
                self._px(30), self._px(16))
            pygame.draw.rect(surface, (14, 17, 26), box, border_radius=self._px(8))
            pygame.draw.rect(surface, MESSAGE_TEXT, box, width=max(1, self._px(2)),
                             border_radius=self._px(8))
            surface.blit(prompt, prompt.get_rect(center=box.center))
        elif view.message:
            message = self._text(view.message, 20, MESSAGE_TEXT)
            surface.blit(message, message.get_rect(midbottom=(centre, bottom - self._px(40))))
        hint = self._text(view.hint, 16, DIM_TEXT)
        surface.blit(hint, hint.get_rect(midbottom=(centre, bottom)))
        return surface

    def footer(self, surface: pygame.Surface, text: str) -> None:
        """``text`` along the bottom of ``surface``, on a dark band: news over a match."""
        line = self._text(text, 18, WHITE)
        band = pygame.Surface((self.width, line.get_height() + self._px(12)), pygame.SRCALPHA)
        band.fill((8, 10, 16, 190))
        surface.blit(band, (0, self.height - band.get_height()))
        surface.blit(line, line.get_rect(midbottom=(self.width // 2,
                                                    self.height - self._px(6))))

    def row_at(self, position: tuple[int, int]) -> int | None:
        """The row at ``position``, in frame pixels, as painted last."""
        for index, rect in enumerate(self.row_rects):
            if rect.collidepoint(position):
                return index
        return None

    def _px(self, size: float) -> int:
        return max(1, round(size * self._scale))

    def _text(self, text: str, size: int, color) -> pygame.Surface:
        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(str(FONT_FILE), self._px(size))
        return self._fonts[size].render(text, True, color)


