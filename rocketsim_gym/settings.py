"""The play window's settings: what a player sets once and keeps, in a settings file.

A player's settings come from three places, each over the one before: the defaults here, the
settings file (:func:`settings_path`, or the one ``--settings`` names), then the flags given
this time. The start menu writes the file whenever the player changes a setting; the flags of
one run are never written to it. In experiment mode the file is read only when ``--settings``
names it, and never written, so a session runs from its flags alone.

The file is TOML, in sections; a setting left out keeps its default::

    [display]
    fullscreen = true          # false for a window of the size below
    window = [1280, 720]       # the window's size, when not fullscreen
    resolution = "auto"        # the frames drawn, [width, height]; "auto" fits the screen
    vsync = true               # each frame shown in step with the screen's refresh

    [controls]
    input = "analog"           # or "digital": the controller's triggers and stick all or nothing
    deadzone = 0.1
    steering_sensitivity = 1.0
    aerial_sensitivity = 1.0

    [camera]                   # Rocket League's camera settings, in its units
    field_of_view = 110.0
    ...

    [bindings]                 # each of the car's buttons, and what holds it
    jump = ["mouse:right", "pad:a"]
    air_roll_left = ["key:q", "pad:leftshoulder"]
    ...

    [match]                    # what the start menu sets up, as it was left
    team_size = 1
    seconds = 300.0

A binding is ``key:NAME``, a key by pygame's name for it (``key:w``, ``key:left shift``,
``key:space``); ``mouse:left``, ``mouse:middle`` or ``mouse:right``; or ``pad:NAME``, a
controller button: ``a``, ``b``, ``x``, ``y``, ``back``, ``start``, ``leftstick``,
``rightstick``, ``leftshoulder``, ``rightshoulder``, ``dpup``, ``dpdown``, ``dpleft``,
``dpright``. The controller's triggers and left stick always drive throttle, reverse and the
stick, as far as they are pushed.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import tomllib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from .camera import CameraSettings  # noqa: E402
from .controls import Button, ControllerSettings, InputMode  # noqa: E402

TEAM_SIZES = (1, 2, 3)
#: The tallest frame "auto" draws in fullscreen: taller ones take longer than a step to draw
#: and read back on a laptop's graphics, and the screen scales them up.
AUTO_MAX_HEIGHT = 1080
#: The mouse buttons a binding can name, by their index in ``pygame.mouse.get_pressed()``.
MOUSE_BUTTONS = {"left": 0, "middle": 1, "right": 2}
#: The controller buttons a binding can name, by SDL's number for them.
PAD_BUTTONS = {
    "a": pygame.CONTROLLER_BUTTON_A, "b": pygame.CONTROLLER_BUTTON_B,
    "x": pygame.CONTROLLER_BUTTON_X, "y": pygame.CONTROLLER_BUTTON_Y,
    "back": pygame.CONTROLLER_BUTTON_BACK, "start": pygame.CONTROLLER_BUTTON_START,
    "leftstick": pygame.CONTROLLER_BUTTON_LEFTSTICK,
    "rightstick": pygame.CONTROLLER_BUTTON_RIGHTSTICK,
    "leftshoulder": pygame.CONTROLLER_BUTTON_LEFTSHOULDER,
    "rightshoulder": pygame.CONTROLLER_BUTTON_RIGHTSHOULDER,
    "dpup": pygame.CONTROLLER_BUTTON_DPAD_UP, "dpdown": pygame.CONTROLLER_BUTTON_DPAD_DOWN,
    "dpleft": pygame.CONTROLLER_BUTTON_DPAD_LEFT, "dpright": pygame.CONTROLLER_BUTTON_DPAD_RIGHT,
}
#: What each of the car's buttons is called, in the menu and in the file.
BUTTON_LABELS = {
    Button.THROTTLE: "Throttle",
    Button.REVERSE: "Reverse",
    Button.STICK_LEFT: "Steer left",
    Button.STICK_RIGHT: "Steer right",
    Button.STICK_UP: "Pitch down (stick up)",
    Button.STICK_DOWN: "Pitch up (stick down)",
    Button.JUMP: "Jump",
    Button.BOOST: "Boost",
    Button.POWERSLIDE: "Powerslide",
    Button.BALL_CAM: "Ball cam",
    Button.AIR_ROLL_LEFT: "Air roll left",
    Button.AIR_ROLL_RIGHT: "Air roll right",
}
#: Rocket League's PC defaults, and its controller defaults for the buttons a controller has.
DEFAULT_BINDINGS: Mapping[Button, tuple[str, ...]] = {
    Button.THROTTLE: ("key:w",),
    Button.REVERSE: ("key:s",),
    Button.STICK_LEFT: ("key:a",),
    Button.STICK_RIGHT: ("key:d",),
    Button.STICK_UP: ("key:w",),
    Button.STICK_DOWN: ("key:s",),
    Button.JUMP: ("mouse:right", "pad:a"),
    Button.BOOST: ("mouse:left", "pad:b"),
    Button.POWERSLIDE: ("key:left shift", "pad:x"),
    Button.BALL_CAM: ("key:space", "pad:y"),
    Button.AIR_ROLL_LEFT: ("key:q",),
    Button.AIR_ROLL_RIGHT: ("key:e",),
}


class SettingsError(ValueError):
    """A settings file that cannot be read, or a setting out of its range."""


def settings_path() -> Path:
    """Where this computer keeps the player's settings: the platform's place for an app's
    settings, in a ``rocketsim-gym`` folder."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "rocketsim-gym" / "settings.toml"


def binding_kind(binding: str) -> str:
    """``key``, ``mouse`` or ``pad``: what a binding names."""
    return binding.partition(":")[0]


def binding_device(binding: str) -> str:
    """``keyboard`` (a key or a mouse button) or ``controller``: what a binding is on. A
    button holds one binding on each."""
    return "controller" if binding_kind(binding) == "pad" else "keyboard"


def binding_label(binding: str) -> str:
    """A binding as the menu shows it: ``W``, ``Left Shift``, ``Mouse right``, ``Pad A``."""
    kind, _, name = binding.partition(":")
    if kind == "mouse":
        return f"Mouse {name}"
    if kind == "pad":
        return "Pad " + {"leftshoulder": "LB", "rightshoulder": "RB", "leftstick": "L3",
                         "rightstick": "R3", "dpup": "D-pad up", "dpdown": "D-pad down",
                         "dpleft": "D-pad left", "dpright": "D-pad right"}.get(name, name.upper())
    return name.title() if len(name) > 1 else name.upper()


def key_code(name: str) -> int:
    """pygame's number for the key it calls ``name``.

    :raises ValueError: pygame has no key of that name.
    """
    with warnings.catch_warnings():
        # Names convert the same before pygame.init(); pygame warns anyway.
        warnings.simplefilter("ignore")
        return pygame.key.key_code(name)


def key_name(code: int) -> str:
    """pygame's name for the key ``code``, as a binding writes it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pygame.key.name(code)


def check_binding(binding: str) -> str:
    """``binding``, if it names a key, mouse button or controller button.

    :raises ValueError: it names none.
    """
    kind, _, name = binding.partition(":") if isinstance(binding, str) else ("", "", "")
    if kind == "key":
        try:
            key_code(name)
            return binding
        except ValueError:
            pass
    elif (kind == "mouse" and name in MOUSE_BUTTONS) or (kind == "pad" and name in PAD_BUTTONS):
        return binding
    raise ValueError(f"binding {binding!r}: key:NAME (such as key:w or key:left shift), "
                     f"mouse:{'/'.join(MOUSE_BUTTONS)}, or pad:NAME, NAME one of "
                     f"{', '.join(PAD_BUTTONS)}")


@dataclass(frozen=True)
class ResolvedBindings:
    """Bindings by the numbers pygame reads: each key, mouse button and controller button, and
    the car's buttons it holds."""

    keys: Mapping[int, tuple[Button, ...]]
    mouse: Mapping[int, tuple[Button, ...]]
    pad: Mapping[int, tuple[Button, ...]]


@dataclass(frozen=True)
class Bindings:
    """What holds each of the car's buttons: keys, mouse buttons and controller buttons, by
    their binding names (see the module's docs).

    :raises ValueError: a binding names nothing there is, or a button is missing.
    """

    buttons: Mapping[Button, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_BINDINGS))

    def __post_init__(self) -> None:
        if set(self.buttons) != set(Button):
            missing = sorted(button.name.lower() for button in set(Button) - set(self.buttons))
            raise ValueError(f"bindings: none for {missing}")
        for bindings in self.buttons.values():
            for binding in bindings:
                check_binding(binding)

    def __getitem__(self, button: Button) -> tuple[str, ...]:
        return tuple(self.buttons[button])

    def bind(self, button: Button, binding: str) -> Bindings:
        """These bindings, with ``binding`` holding ``button`` in place of what held it on the
        same device (:func:`binding_device`)."""
        check_binding(binding)
        kept = tuple(old for old in self.buttons[button]
                     if binding_device(old) != binding_device(binding))
        return Bindings({**self.buttons, button: (binding, *kept)})

    def clear(self, button: Button, device: str) -> Bindings:
        """These bindings, with nothing on ``device`` holding ``button``."""
        kept = tuple(old for old in self.buttons[button] if binding_device(old) != device)
        return Bindings({**self.buttons, button: kept})

    def on(self, button: Button, device: str) -> str | None:
        """The binding on ``device`` that holds ``button``, or None."""
        return next((binding for binding in self.buttons[button]
                     if binding_device(binding) == device), None)

    def resolve(self) -> ResolvedBindings:
        """The bindings by pygame's numbers, for reading inputs."""
        found: dict[str, dict[int, list[Button]]] = {"key": {}, "mouse": {}, "pad": {}}
        for button in Button:
            for binding in self.buttons[button]:
                kind, _, name = binding.partition(":")
                number = {"key": key_code, "mouse": MOUSE_BUTTONS.__getitem__,
                          "pad": PAD_BUTTONS.__getitem__}[kind](name)
                found[kind].setdefault(number, []).append(button)
        return ResolvedBindings(*({number: tuple(buttons) for number, buttons in kind.items()}
                                  for kind in found.values()))


@dataclass(frozen=True)
class PlaySettings:
    """A player's settings for the play window.

    :ivar fullscreen: whether the window fills the screen.
    :ivar window: the window's size, ``(width, height)``, when not fullscreen.
    :ivar resolution: the frames' size, ``(width, height)``; None fits them to the window, and
        in fullscreen to the screen, at most :data:`AUTO_MAX_HEIGHT` tall.
    :ivar vsync: whether each frame is shown in step with the screen's refresh.
    :ivar input_mode: how the controller's triggers and stick count.
    :ivar controller: Rocket League's controller settings.
    :ivar camera: Rocket League's camera settings.
    :ivar bindings: what holds each of the car's buttons.
    :ivar team_size: the cars a team, of the match the start menu sets up.
    :ivar seconds: the regulation time of the match the start menu sets up.
    :raises ValueError: a setting out of its range.
    """

    fullscreen: bool = True
    window: tuple[int, int] = (1280, 720)
    resolution: tuple[int, int] | None = None
    vsync: bool = True
    input_mode: InputMode = InputMode.ANALOG
    controller: ControllerSettings = ControllerSettings(deadzone=0.1)
    camera: CameraSettings = CameraSettings()
    bindings: Bindings = field(default_factory=Bindings)
    team_size: int = 1
    seconds: float = 300.0

    def __post_init__(self) -> None:
        for name in ("fullscreen", "vsync"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} {getattr(self, name)!r}: true or false")
        for name in ("window", "resolution"):
            size = getattr(self, name)
            if size is None and name == "resolution":
                continue
            if (not isinstance(size, tuple) or len(size) != 2
                    or not all(isinstance(side, int) and not isinstance(side, bool)
                               and 16 <= side <= 16384 for side in size)):
                raise ValueError(f"{name} {size!r}: a width and a height, in pixels")
        if self.team_size not in TEAM_SIZES or isinstance(self.team_size, bool):
            raise ValueError(f"team_size {self.team_size!r}: one of {list(TEAM_SIZES)}")
        if (isinstance(self.seconds, bool) or not isinstance(self.seconds, (int, float))
                or not math.isfinite(self.seconds) or self.seconds <= 0):
            raise ValueError(f"seconds {self.seconds!r}: a positive number")

    def frame_size(self, screen: tuple[int, int] | None) -> tuple[int, int]:
        """The size of the frames to draw: :attr:`resolution`, or else the window's; in
        fullscreen, the shape of ``screen`` (the display's size, if known) at most
        :data:`AUTO_MAX_HEIGHT` tall."""
        if self.resolution is not None:
            return self.resolution
        if not self.fullscreen or screen is None:
            return self.window
        width, height = screen
        if height > AUTO_MAX_HEIGHT:
            width, height = round(width * AUTO_MAX_HEIGHT / height), AUTO_MAX_HEIGHT
        return width - width % 2, height - height % 2

    def replace(self, **changes: Any) -> PlaySettings:
        """These settings with ``changes``; ``camera`` and ``controller`` may be given as
        mappings of only the fields that change."""
        for name, kind in (("camera", CameraSettings), ("controller", ControllerSettings)):
            if isinstance(changes.get(name), Mapping):
                changes[name] = dataclasses.replace(getattr(self, name), **changes[name])
        return dataclasses.replace(self, **changes)


def load_settings(path: Path) -> PlaySettings:
    """The settings in the file at ``path``; the defaults where it has none, or if it is
    missing.

    :raises SettingsError: the file cannot be read, or a setting in it is wrong, with where.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return PlaySettings()
    except OSError as error:
        raise SettingsError(f"{path}: {error.strerror or error}") from None
    try:
        return settings_from_record(tomllib.loads(text))
    except tomllib.TOMLDecodeError as error:
        raise SettingsError(f"{path}: not TOML ({error})") from None
    except (ValueError, TypeError) as error:
        raise SettingsError(f"{path}: {error}") from None


def save_settings(settings: PlaySettings, path: Path) -> None:
    """Write ``settings`` to the file at ``path``, whole, making its folder if need be."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(settings_text(settings), encoding="utf-8")
    os.replace(temporary, path)


_SECTIONS = {
    "display": ("fullscreen", "window", "resolution", "vsync"),
    "controls": ("input", "deadzone", "steering_sensitivity", "aerial_sensitivity"),
    "camera": tuple(f.name for f in dataclasses.fields(CameraSettings)),
    "bindings": tuple(button.name.lower() for button in Button),
    "match": ("team_size", "seconds"),
}


def settings_from_record(record: Mapping[str, Any]) -> PlaySettings:
    """Settings from a settings file's contents, read as TOML.

    :raises ValueError: an unknown section or setting, or a setting out of its range.
    """
    unknown = sorted(set(record) - set(_SECTIONS))
    if unknown:
        raise ValueError(f"unknown section {unknown[0]!r}: the sections are "
                         f"{', '.join(_SECTIONS)}")
    sections = {}
    for name, keys in _SECTIONS.items():
        section = record.get(name, {})
        if not isinstance(section, Mapping):
            raise ValueError(f"[{name}]: a section of settings")
        unknown = sorted(set(section) - set(keys))
        if unknown:
            raise ValueError(f"[{name}] {unknown[0]!r}: not a setting there; the settings are "
                             f"{', '.join(keys)}")
        sections[name] = section
    display, controls = sections["display"], sections["controls"]
    changes: dict[str, Any] = {}
    for name in ("fullscreen", "vsync"):
        if name in display:
            changes[name] = display[name]
    if "window" in display:
        changes["window"] = _size(display["window"], "window")
    if "resolution" in display:
        resolution = display["resolution"]
        changes["resolution"] = None if resolution == "auto" else _size(resolution, "resolution")
    if "input" in controls:
        try:
            changes["input_mode"] = InputMode(controls["input"])
        except ValueError:
            raise ValueError(f"[controls] input {controls['input']!r}: \"analog\" or "
                             f"\"digital\"") from None
    controller = {name: value for name, value in controls.items() if name != "input"}
    defaults = PlaySettings()
    changes["controller"] = dataclasses.replace(defaults.controller, **controller)
    changes["camera"] = dataclasses.replace(defaults.camera, **sections["camera"])
    bindings = dict(DEFAULT_BINDINGS)
    for name, value in sections["bindings"].items():
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError(f"[bindings] {name}: a list of bindings, such as [\"key:w\"]")
        bindings[Button[name.upper()]] = tuple(value)
    changes["bindings"] = Bindings(bindings)
    changes.update(sections["match"])
    return dataclasses.replace(defaults, **changes)


def _size(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, list) and len(value) == 2:
        return tuple(value)
    raise ValueError(f"{name} {value!r}: [width, height], in pixels")


def settings_text(settings: PlaySettings) -> str:
    """``settings`` as a settings file writes them."""
    def toml(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(value, (tuple, list)):
            return "[" + ", ".join(toml(item) for item in value) + "]"
        if isinstance(value, float):
            return repr(value)
        return str(value)

    controller, camera = settings.controller, settings.camera
    sections = {
        "display": {"fullscreen": settings.fullscreen, "window": settings.window,
                    "resolution": "auto" if settings.resolution is None else settings.resolution,
                    "vsync": settings.vsync},
        "controls": {"input": settings.input_mode.value,
                     **{f.name: float(getattr(controller, f.name))
                        for f in dataclasses.fields(ControllerSettings)}},
        "camera": {f.name: float(getattr(camera, f.name))
                   for f in dataclasses.fields(CameraSettings)},
        "bindings": {button.name.lower(): settings.bindings[button] for button in Button},
        "match": {"team_size": settings.team_size, "seconds": float(settings.seconds)},
    }
    lines = ["# rocketsim-play's settings, written by its menu. Edit them here or there; a",
             "# setting left out keeps its default. Flags on the command line override them",
             "# for one run. Bindings: key:NAME, mouse:left/middle/right, pad:BUTTON."]
    for name, values in sections.items():
        lines += ["", f"[{name}]"] + [f"{key} = {toml(value)}" for key, value in values.items()]
    return "\n".join(lines) + "\n"


def seat_names(team_size: int, players: int) -> tuple[str, ...]:
    """The seats ``players`` players take in a match of ``team_size`` cars a team, alternating
    blue and orange: ``blue1, orange1, blue2...``."""
    if not 1 <= players <= 2 * team_size:
        raise ValueError(f"players {players!r}: from 1 to {2 * team_size}")
    order = [f"{team}{number}" for number in range(1, team_size + 1)
             for team in ("blue", "orange")]
    return tuple(order[:players])
