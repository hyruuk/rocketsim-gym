"""``RocketSimSoccar-v0``: a whole soccar match, 1v1, 2v2 or 3v3, against scripted bots."""

from __future__ import annotations

import dataclasses
import importlib.metadata
from collections.abc import Mapping
from typing import Any

import gymnasium
import numpy as np

from .camera import CameraSettings
from .controls import (ANALOG_HIGH, ANALOG_LOW, BUTTON_COUNT, ControllerSettings, InputMode,
                       buttons_from_action, controls_from_pad, pad_from_analog, pad_from_buttons)
from .match import TICKS_PER_SECOND, Car, Team, match_cars
from .simulation import MatchSimulation, load_homemade_arena, pad_spots
from .snapshot import Snapshot
from .view import SoccarView

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

    **Action**: a game controller, by ``input_mode`` (the tables are in
    :mod:`rocketsim_gym.controls`): ``"digital"``, twelve buttons each held or not
    (``MultiBinary(12)``), the sticks and triggers at full deflection; or ``"analog"``, the
    triggers and the stick as far as they are pushed and six buttons (``Box(10)``), through the
    player's ``controller`` settings. The car obeys it except in a countdown and while it is
    demolished; ``ball_cam`` switches the camera when pressed.

    **Observation**: :func:`observation_size` values, in Rocket League's units (uu, uu/s,
    rad/s), 87 in 1v1, 127 in 2v2 and 167 in 3v3:

    - 9 for the ball: position, velocity, angular velocity;
    - 20 for each car, blue's then orange's, each team's by number, the agent's first: position,
      velocity, angular velocity, forward direction, up direction, boost (0-100), on the ground,
      can jump or flip, supersonic, demolished;
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
    :raises ValueError: an argument is out of range.
    :raises RuntimeError: frames are asked for, and no OpenGL 3.3 context opens.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, match_seconds: float = 300.0, team_size: int = 1, ticks_per_step: int = 2,
                 render_mode: str | None = None, render_width: int = 1024,
                 render_height: int = 576, camera: Mapping[str, float] | None = None,
                 input_mode: str = "digital",
                 controller: Mapping[str, float] | None = None) -> None:
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
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": TICKS_PER_SECOND / ticks_per_step}
        self._match_ticks = round(match_seconds * TICKS_PER_SECOND)
        self._team_size = team_size
        self._cars = match_cars(team_size)
        #: the fields :meth:`sample` returns (:func:`sample_fields`)
        self.sample_fields = sample_fields(team_size)
        self._ticks_per_step = ticks_per_step
        self._collision_meshes = load_homemade_arena()
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
        self._camera_settings = _settings(CameraSettings, camera, "camera")
        self._controller_settings = _settings(ControllerSettings, controller, "controller")
        draws_frames = render_mode == "rgb_array"
        self._view = SoccarView(PLAYER, self._camera_settings,
                                (render_width, render_height) if draws_frames else None,
                                pad_spots() if draws_frames else ())

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._simulation = MatchSimulation(self._team_size, self._match_ticks,
                                           self._ticks_per_step, (PLAYER,), self.np_random)
        self._snapshot = self._simulation.snapshot()
        self._view.start(self._snapshot)
        return self._observation(), self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._simulation is None:
            raise ValueError("call reset before step")
        if self._simulation.has_ended:
            raise ValueError("the match has ended: call reset")
        if self._input_mode is InputMode.DIGITAL:
            pad = pad_from_buttons(buttons_from_action(action))
        else:
            pad = pad_from_analog(action)
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
        if self._snapshot is None:
            raise ValueError("call reset before render")
        return self._view.frame()

    def sample(self) -> dict[str, Any]:
        """The state to log for analysis, after the latest ``reset`` or ``step``: each of
        :attr:`sample_fields`, with what happened during that step (``goal``, the scoring
        team's name or ""; each car's ``touch`` and ``demolition``) and the controls each car
        obeyed (``input_*``)."""
        if self._snapshot is None:
            raise ValueError("call reset before sample")
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
        return dict(zip(self.sample_fields, values, strict=True))

    def provenance(self) -> dict[str, Any]:
        """What a match depends on besides its seed and actions: the package versions, the
        collision meshes' SHA-256s, the cars a team and the tick rates."""
        return {
            "rocketsim_gym_version": importlib.metadata.version("rocketsim-gym"),
            "rocketsim_version": importlib.metadata.version("rocketsim"),
            "collision_meshes": dict(self._collision_meshes),
            "team_size": self._team_size,
            "ticks_per_second": TICKS_PER_SECOND,
            "ticks_per_step": self._ticks_per_step,
            "gl_renderer": self._view.gl_renderer,
            "camera": dataclasses.asdict(self._camera_settings),
            "input_mode": self._input_mode.value,
            "controller": dataclasses.asdict(self._controller_settings),
        }

    def close(self) -> None:
        self._view.close()
        self._simulation = None
        self._snapshot = None

    def _observation(self) -> np.ndarray:
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
        match = self._snapshot.match
        return {"phase": match.phase.value, "blue_score": match.score[Team.BLUE],
                "orange_score": match.score[Team.ORANGE]}
