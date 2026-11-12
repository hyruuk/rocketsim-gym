"""``RocketSimSoccar-v0``: a whole soccar match, 1v1, 2v2 or 3v3: against scripted bots on this
computer, or with other players over a local network."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import math
from collections import deque
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import gymnasium
import numpy as np

from .arena_geometry import HOMEMADE_ARENA_FOLDER, MANIFEST_FILE_NAME, read_manifest
from .boost_pads import BOOST_PADS
from .camera import CameraSettings
from .controls import (ANALOG_HIGH, ANALOG_LOW, BUTTON_COUNT, ControllerSettings, InputMode, Pad,
                       buttons_from_action, controls_from_pad, pad_from_analog, pad_from_buttons)
from .lan.client import ClientStatus, LanClient
from .lan.protocol import TICKS_PER_STEP, ProtocolError, RoomStatus, ServerFrame, Setup
from .match import TEAMS, TICKS_PER_SECOND, Car, Phase, Team, match_cars
from .snapshot import Snapshot
from .view import Notice, SoccarView

if TYPE_CHECKING:
    from .simulation import MatchSimulation

#: Cars a team, by match: 1v1, 2v2 and 3v3.
TEAM_SIZES = (1, 2, 3)
#: The car the agent drives: blue's first.
PLAYER = Car(Team.BLUE, 1)
#: The values in an observation: the ball's, each car's, each boost pad's and the match's.
BALL_VALUES, CAR_VALUES, BOOST_PAD_COUNT, MATCH_VALUES = 9, 20, 34, 4

_MATCH_SAMPLE_FIELDS = ("phase", "seconds_left", "overtime", "overtime_seconds", "blue_score",
                        "orange_score", "goal", "ball_cam", "ball_x", "ball_y", "ball_z",
                        "ball_vx", "ball_vy", "ball_vz")
_CAR_SAMPLE_FIELDS = ("x", "y", "z", "vx", "vy", "vz", "yaw", "pitch", "roll", "boost",
                      "on_ground", "supersonic", "demolished", "touch", "demolition",
                      "input_throttle", "input_steer", "input_pitch", "input_yaw", "input_roll",
                      "input_jump", "input_boost", "input_handbrake")
#: What :meth:`SoccarEnv.sample` adds in a LAN match: the client's status and the room's, the
#: number of the server frame shown, and the delay of the player's controls, in frames.
LAN_SAMPLE_FIELDS = ("client", "room", "server_frame", "input_delay")
#: What a LAN player sees before the first frame of a match comes, by the client's status.
_NOTICES_WITHOUT_MATCH = {
    ClientStatus.JOINING: Notice("JOINING", "the match"),
    ClientStatus.WAITING: Notice("WAITING", "the room is playing another match"),
    ClientStatus.LIVE: Notice("JOINING", "the match"),
    ClientStatus.LOST: Notice("NO ANSWER", "from the server"),
    ClientStatus.GONE: Notice("DISCONNECTED", "the server is gone"),
}


def observation_size(team_size: int) -> int:
    """The number of values in an observation of a match of ``team_size`` cars a team."""
    return BALL_VALUES + 2 * team_size * CAR_VALUES + BOOST_PAD_COUNT + MATCH_VALUES


def sample_fields(team_size: int) -> tuple[str, ...]:
    """The fields :meth:`SoccarEnv.sample` returns every step, in a match of ``team_size`` cars a
    team: the match's and the ball's, then each car's, after its name (``blue1_x``)."""
    return (*_MATCH_SAMPLE_FIELDS, *(f"{car.name}_{name}" for car in match_cars(team_size)
                                     for name in _CAR_SAMPLE_FIELDS))


def _settings(kind, values: Mapping[str, float] | None, name: str):
    """A player's settings of ``kind`` from ``values`` by name, the rest left at their defaults."""
    values = dict(values or {})
    unknown = sorted(set(values) - {field.name for field in dataclasses.fields(kind)})
    if unknown:
        raise ValueError(f"{name} settings {unknown}: not ones Rocket League has")
    return kind(**values)


class SoccarEnv(gymnasium.Env):
    """A Rocket League-style soccar match, 1v1, 2v2 or 3v3: the agent drives blue's first car,
    scripted bots every other one (:mod:`rocketsim_gym.bot`), on RocketSim physics and the
    homemade arena.

    The match follows Rocket League's rules (:mod:`rocketsim_gym.match`): a countdown before
    each kickoff, a clock that starts at the first touch, a pause after each goal, play to the
    ball's landing at 0:00, overtime on a tie, and demolished cars back after 3 s. The kickoff
    and respawn spots are drawn from the env's own random generator, seeded by ``reset``; the
    same seed and the same actions give the same match.

    **LAN play**: given a ``server``, the env is instead one player's seat, ``seat``, in a match
    that a match server plays for the players on a local network (:mod:`rocketsim_gym.lan`), with
    bots in the seats no player has. Its action, observation, sample and frames are the same as
    on this computer, from its seat's car; ``reset`` joins the server's room, and the match starts
    once every seat's player is ready (from their first step). The reward is the seat's team's,
    and the episode terminates when the match ends, or is truncated when the experimenter ends it
    or the server is gone. The match's seed is the server's. Each step's ``info["server_frame"]``
    holds what the step showed, everything it returns follows from it, and :meth:`replay_from`
    shows such records again in place of the network's.

    **Action**: a game controller, by ``input_mode`` (the tables are in
    :mod:`rocketsim_gym.controls`): ``"digital"``, twelve buttons each held or not
    (``MultiBinary(12)``), the sticks and triggers at full deflection; or ``"analog"``, the
    triggers and the stick as far as they are pushed and six buttons (``Box(10)``), through the
    player's ``controller`` settings. The car obeys it except in a countdown and while it is
    demolished; ``ball_cam`` switches the camera when pressed.

    **Observation**: :func:`observation_size` values, in Rocket League's units (uu, uu/s,
    rad/s), 87 in 1v1, 127 in 2v2 and 167 in 3v3:

    - 9 for the ball: position, velocity, angular velocity;
    - 20 for each car, blue's then orange's, each team's by number (the agent's first on this
      computer): position, velocity, angular velocity, forward direction, up direction, boost
      (0-100), on the ground, can jump or flip, supersonic, demolished;
    - 34 for the boost pads, in RocketSim's order, each 1 while it can be picked up;
    - 4 for the match: seconds left in regulation, blue's goals, orange's goals, 1 in overtime.

    **Reward**: +1 for each blue goal, -1 for each orange goal. The episode terminates when
    the match ends; overtime has no end but a goal.

    **Frames**: with ``render_mode="rgb_array"``, :meth:`render` returns the frame of the
    latest step, as Rocket League's camera behind the agent's car sees it: car cam, or ball cam
    (:mod:`rocketsim_gym.camera`), with Rocket League's on-screen display over it
    (:mod:`rocketsim_gym.hud`). The camera moves in ``step``; drawing changes nothing.

    :param match_seconds: regulation time.
    :param team_size: cars a team: 1, 2 or 3.
    :param ticks_per_step: physics ticks per step, at 120 ticks a second.
    :param render_mode: ``"rgb_array"`` to draw frames, or None.
    :param render_width: a frame's width, in pixels.
    :param render_height: a frame's height, in pixels.
    :param camera: the player's camera settings, by the names and in the units of
        :class:`~rocketsim_gym.camera.CameraSettings`; the ones left out keep their defaults.
    :param input_mode: ``"digital"`` or ``"analog"``.
    :param controller: the player's controller settings, by the names of
        :class:`~rocketsim_gym.controls.ControllerSettings`; the ones left out keep theirs.
    :param server: for a LAN match, its server: ``"auto"`` to find the one on the network, or
        ``"HOST"`` or ``"HOST:PORT"``; None, the default, plays on this computer against bots.
    :param seat: in a LAN match, the player's car, such as ``"orange1"``.
    :param seats: in a LAN match, the cars players drive; bots drive the others.
    :raises ValueError: an argument is out of range.
    :raises RuntimeError: frames are asked for, and no OpenGL 3.3 context opens.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, match_seconds: float = 300.0, team_size: int = 1, ticks_per_step: int = 2,
                 render_mode: str | None = None, render_width: int = 1024,
                 render_height: int = 576, camera: Mapping[str, float] | None = None,
                 input_mode: str = "digital",
                 controller: Mapping[str, float] | None = None, server: str | None = None,
                 seat: str | None = None, seats: Sequence[str] | None = None) -> None:
        if (isinstance(match_seconds, bool) or not isinstance(match_seconds, (int, float))
                or round(match_seconds * TICKS_PER_SECOND) < 1):
            raise ValueError(f"match_seconds {match_seconds!r}: a positive number of seconds, "
                             f"at least one tick")
        if (isinstance(team_size, bool) or not isinstance(team_size, int)
                or team_size not in TEAM_SIZES):
            raise ValueError(f"team_size {team_size!r}: one of {list(TEAM_SIZES)}")
        if isinstance(ticks_per_step, bool) or not isinstance(ticks_per_step, int) \
                or ticks_per_step < 1:
            raise ValueError(f"ticks_per_step {ticks_per_step!r}: a positive whole number")
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"render_mode {render_mode!r}: None or 'rgb_array'")
        for name, size in (("render_width", render_width), ("render_height", render_height)):
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(f"{name} {size!r}: a positive whole number of pixels")
        setup = _lan_setup(server, seat, seats, team_size, match_seconds, ticks_per_step)
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": TICKS_PER_SECOND / ticks_per_step}
        self._match_ticks = round(match_seconds * TICKS_PER_SECOND)
        self._team_size = team_size
        self._cars = match_cars(team_size)
        #: the fields :meth:`sample` returns (:func:`sample_fields`, and in a LAN match
        #: :data:`LAN_SAMPLE_FIELDS`)
        self.sample_fields = sample_fields(team_size) + (() if setup is None
                                                         else LAN_SAMPLE_FIELDS)
        self._ticks_per_step = ticks_per_step
        self._server, self._setup = server, setup
        self._lan: LanClient | None = None
        car = PLAYER
        if setup is None:
            from .simulation import load_homemade_arena
            self._collision_meshes = load_homemade_arena()
        else:
            self._collision_meshes = read_manifest(HOMEMADE_ARENA_FOLDER / MANIFEST_FILE_NAME)
            self._lan, car = LanClient(server, setup, seat, self._collision_meshes), setup.car(seat)
        try:
            self._input_mode = InputMode(input_mode)
        except ValueError:
            raise ValueError(f"input_mode {input_mode!r}: 'digital' or 'analog'") from None
        if self._input_mode is InputMode.DIGITAL:
            self.action_space = gymnasium.spaces.MultiBinary(BUTTON_COUNT)
        else:
            self.action_space = gymnasium.spaces.Box(ANALOG_LOW, ANALOG_HIGH, dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf,
                                                      (observation_size(team_size),), np.float32)
        self._simulation: MatchSimulation | None = None
        self._snapshot: Snapshot | None = None
        self._is_reset = self._is_over = False
        self._replay: deque[bytes] | None = None
        self._shown: ServerFrame | None = None
        self._client_status = ClientStatus.JOINING
        self._camera_settings = _settings(CameraSettings, camera, "camera")
        self._controller_settings = _settings(ControllerSettings, controller, "controller")
        self._view = SoccarView(car, self._camera_settings,
                                (render_width, render_height) if render_mode else None,
                                BOOST_PADS)

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._is_reset, self._is_over = True, False
        if self._lan is not None:
            if self._replay is None:
                self._lan.reset()
            self._snapshot, self._shown = None, None
            self._client_status = ClientStatus.JOINING
            self._view.show_notice(_NOTICES_WITHOUT_MATCH[ClientStatus.JOINING])
            return self._observation(), self._info()
        from .simulation import MatchSimulation
        self._simulation = MatchSimulation(self._team_size, self._match_ticks,
                                           self._ticks_per_step, (PLAYER,), self.np_random)
        self._snapshot = self._simulation.snapshot()
        self._view.start(self._snapshot)
        return self._observation(), self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._lan is not None:
            return self._step_in_lan_match(action)
        if self._simulation is None:
            raise ValueError("call reset before step")
        if self._simulation.has_ended:
            raise ValueError("the match has ended: call reset")
        pad = self._pad(action)
        self._view.press_ball_cam(pad.ball_cam)
        score_before = self._snapshot.match.score
        self._simulation.step({PLAYER: controls_from_pad(pad, self._controller_settings)})
        self._snapshot = self._simulation.snapshot()
        self._view.advance(self._snapshot, self._ticks_per_step / TICKS_PER_SECOND)
        score = self._snapshot.match.score
        reward = (score[Team.BLUE] - score_before[Team.BLUE]
                  - (score[Team.ORANGE] - score_before[Team.ORANGE]))
        return self._observation(), float(reward), self._simulation.has_ended, False, self._info()

    def render(self) -> np.ndarray | None:
        """The frame of the latest ``reset`` or ``step``, ``(render_height, render_width, 3)``
        RGB bytes with the on-screen display, read-only; drawn once, however often it is asked
        for. None without a render mode."""
        if not self._view.draws_frames:
            gymnasium.logger.warn("render_mode is None: this env draws no frame")
            return None
        if not self._is_reset:
            raise ValueError("call reset before render")
        return self._view.frame()

    def sample(self) -> dict[str, Any]:
        """The state to log for analysis, after the latest ``reset`` or ``step``: each of
        :attr:`sample_fields`, with what happened during that step (``goal``, the scoring
        team's name or ""; each car's ``touch`` and ``demolition``) and the controls each car
        obeyed (``input_*``); in a LAN match, before its first frame has come, None for each."""
        if not self._is_reset:
            raise ValueError("call reset before sample")
        values = self._snapshot_values()
        if self._lan is not None:
            values += self._lan_values()
        return dict(zip(self.sample_fields, values, strict=True))

    def replay_from(self, records: Sequence[bytes]) -> None:
        """Have a LAN match's steps show ``records``, in order, in place of what comes over the
        network: a player's records (each step's ``info["server_frame"]``) redraw exactly what
        that player saw, on the same machine. From now on, the env joins no server.

        :raises ValueError: the match is played on this computer, which replays from its seed
            and actions.
        """
        if self._lan is None:
            raise ValueError("a match played on this computer replays from its seed and actions")
        self._replay = deque(bytes(record) for record in records)

    def _snapshot_values(self) -> list[Any]:
        """The values :func:`sample_fields` names, from the latest snapshot."""
        if self._snapshot is None:
            return [None] * len(sample_fields(self._team_size))
        snapshot, match = self._snapshot, self._snapshot.match
        values: list[Any] = [
            match.phase.value, match.clock_ticks_left / TICKS_PER_SECOND, match.overtime,
            match.overtime_ticks / TICKS_PER_SECOND, match.score[Team.BLUE],
            match.score[Team.ORANGE], snapshot.goal.value if snapshot.goal else "",
            self._view.ball_cam, *snapshot.ball.position.tolist(),
            *snapshot.ball.velocity.tolist(),
        ]
        for car in snapshot.cars:
            controls = car.controls
            values += [*car.position.tolist(), *car.velocity.tolist(), car.yaw, car.pitch,
                       car.roll, car.boost, car.is_on_ground, car.is_supersonic,
                       car.is_demolished, car.touched_ball, car.was_demolished,
                       controls.throttle, controls.steer, controls.pitch, controls.yaw,
                       controls.roll, controls.jump, controls.boost, controls.handbrake]
        return values

    def _lan_values(self) -> list[Any]:
        """The values :data:`LAN_SAMPLE_FIELDS` names, from the latest step's record."""
        frame = self._shown
        if frame is None:
            return [self._client_status.name.lower(), None, None, None]
        seat_index = match_cars(frame.team_size).index(self._view.car)
        return [self._client_status.name.lower(), frame.status.value, frame.number,
                frame.input_delays[seat_index]]

    def provenance(self) -> dict[str, Any]:
        """What a match depends on besides its seed and actions: the package versions, the
        collision meshes' SHA-256s, the cars a team and the tick rates; in a LAN match, the
        server, how it was found, the seats, and each match's id and seed."""
        provenance = {
            "rocketsim_gym_version": importlib.metadata.version("rocketsim-gym"),
            "rocketsim_version": _version_of("rocketsim"),
            "collision_meshes": dict(self._collision_meshes),
            "team_size": self._team_size,
            "ticks_per_second": TICKS_PER_SECOND,
            "ticks_per_step": self._ticks_per_step,
            "gl_renderer": self._view.gl_renderer,
            "camera": dataclasses.asdict(self._camera_settings),
            "input_mode": self._input_mode.value,
            "controller": dataclasses.asdict(self._controller_settings),
        }
        if self._lan is not None:
            provenance |= {
                "server": self._server, "server_address": self._lan.address,
                "server_found": self._lan.found, "seat": self._view.car.name,
                "seats": list(self._setup.seats),
                "matches": {f"{match_id:08x}": seed for match_id, seed in self._lan.seeds.items()},
            }
        return provenance

    def close(self) -> None:
        self._view.close()
        if self._lan is not None:
            self._lan.close()
        self._simulation = None
        self._snapshot = None
        self._is_reset = False

    def _pad(self, action: Any) -> Pad:
        """The pad ``action`` holds, as the input mode reads it."""
        if self._input_mode is InputMode.DIGITAL:
            return pad_from_buttons(buttons_from_action(action))
        return pad_from_analog(action)

    def _step_in_lan_match(self, action: Any) -> tuple[np.ndarray, float, bool, bool,
                                                        dict[str, Any]]:
        if not self._is_reset:
            raise ValueError("call reset before step")
        if self._is_over:
            raise ValueError("the match has ended: call reset")
        pad = self._pad(action)
        self._view.press_ball_cam(pad.ball_cam)
        if self._replay is None:
            record = self._lan.step(controls_from_pad(pad, self._controller_settings))
        elif self._replay:
            record = self._replay.popleft()
        else:
            raise ValueError("no recorded frame left to replay")
        return self._show(record)

    def _show(self, record: bytes) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Show a step's record (a LAN client's status, and the server frame it shows, if any)
        and say what it means for the episode."""
        status = ClientStatus(record[0])
        frame = ServerFrame.unpack(record[1:]) if len(record) > 1 else None
        self._client_status, self._shown = status, frame
        reward = 0.0
        if frame is None:
            self._view.show_notice(_NOTICES_WITHOUT_MATCH[status])
        else:
            reward = self._show_frame(frame, status)
        terminated = (frame is not None and frame.status is RoomStatus.ENDED
                      and self._snapshot.match.phase is Phase.ENDED)
        truncated = status is ClientStatus.GONE or (
            frame is not None and frame.status in (RoomStatus.STOPPED, RoomStatus.CLOSING))
        self._is_over = terminated or truncated
        return (self._observation(), reward, terminated, truncated,
                {**self._info(), "server_frame": record})

    def _show_frame(self, frame: ServerFrame, status: ClientStatus) -> float:
        """Show ``frame``'s snapshot from the seat's car; return the reward since the last."""
        snapshot = Snapshot.unpack(frame.snapshot)
        notice = _notice(frame, status)
        previous, self._snapshot = self._snapshot, snapshot
        if previous is None or snapshot.step < previous.step:
            self._view.start(snapshot, notice)
            return 0.0
        seconds = (snapshot.step - previous.step) * TICKS_PER_STEP / TICKS_PER_SECOND
        self._view.advance(snapshot, seconds, notice)
        team = self._view.car.team
        other = next(other for other in TEAMS if other is not team)
        return float(snapshot.match.score[team] - previous.match.score[team]
                     - (snapshot.match.score[other] - previous.match.score[other]))

    def _observation(self) -> np.ndarray:
        if self._snapshot is None:
            return np.zeros(self.observation_space.shape, np.float32)
        snapshot = self._snapshot
        ball = snapshot.ball
        values: list[float] = [*ball.position.tolist(), *ball.velocity.tolist(),
                               *ball.angular_velocity.tolist()]
        for car in snapshot.cars:
            values += [*car.position.tolist(), *car.velocity.tolist(),
                       *car.angular_velocity.tolist(), *car.forward.tolist(), *car.up.tolist(),
                       car.boost, car.is_on_ground, car.has_flip_or_jump, car.is_supersonic,
                       car.is_demolished]
        values += snapshot.pads_active
        match = snapshot.match
        values += [match.clock_ticks_left / TICKS_PER_SECOND, match.score[Team.BLUE],
                   match.score[Team.ORANGE], match.overtime]
        return np.array(values, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        if self._snapshot is None:
            info: dict[str, Any] = {"phase": None, "blue_score": None, "orange_score": None}
        else:
            match = self._snapshot.match
            info = {"phase": match.phase.value, "blue_score": match.score[Team.BLUE],
                    "orange_score": match.score[Team.ORANGE]}
        if self._lan is not None:
            info["seat"] = self._view.car.name
        return info


def _lan_setup(server: str | None, seat: str | None, seats: Sequence[str] | None,
               team_size: int, match_seconds: float, ticks_per_step: int) -> Setup | None:
    """A LAN match's setup, from the env's arguments; None for a match on this computer.

    :raises ValueError: an argument that does not make a LAN match.
    """
    if server is None:
        if seat is not None or seats is not None:
            raise ValueError("seat and seats are for a LAN match: give its server too")
        return None
    if not isinstance(server, str) or not server:
        raise ValueError(f"server {server!r}: 'auto', 'HOST' or 'HOST:PORT'")
    if ticks_per_step != TICKS_PER_STEP:
        raise ValueError(f"ticks_per_step {ticks_per_step!r}: a LAN match plays "
                         f"{TICKS_PER_STEP} a step")
    if isinstance(seats, str) or not isinstance(seats, Sequence):
        raise ValueError(f"seats {seats!r}: the cars players drive, such as ['blue1', 'orange1']")
    try:
        setup = Setup(team_size, tuple(seats), match_seconds)
    except ProtocolError as error:
        raise ValueError(str(error)) from None
    if seat not in setup.seats:
        raise ValueError(f"seat {seat!r}: one of the seats players drive, {list(setup.seats)}")
    return setup


def _notice(frame: ServerFrame, status: ClientStatus) -> Notice | None:
    """What a LAN player's frame says over the match, from the client's status and the room's;
    None leaves the match's own banner."""
    if status is ClientStatus.LOST:
        return Notice("CONNECTION LOST", "waiting for the server")
    if status is ClientStatus.GONE:
        return Notice("DISCONNECTED", "the server is gone")
    names = ", ".join(car.name for car in match_cars(frame.team_size) if car in frame.waiting_for)
    if frame.status is RoomStatus.LOBBY:
        return Notice("WAITING", f"for {names}") if names else None
    if frame.status is RoomStatus.PAUSED:
        return Notice("PAUSED", f"waiting for {names}" if names else "")
    if frame.status is RoomStatus.RESUMING:
        return Notice(str(math.ceil(frame.status_ticks / TICKS_PER_SECOND)), "RESUMING")
    if frame.status is RoomStatus.STOPPED:
        return Notice("MATCH OVER", "ended by the experimenter")
    if frame.status is RoomStatus.CLOSING:
        return Notice("SERVER CLOSED")
    return None


def _version_of(package: str) -> str | None:
    """``package``'s installed version, or None where it is not installed."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
