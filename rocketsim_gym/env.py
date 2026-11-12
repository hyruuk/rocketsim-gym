"""``RocketSimSoccar-v0``: a whole soccar match, 1v1, 2v2 or 3v3, against scripted bots."""

from __future__ import annotations

import dataclasses
import importlib.metadata
from collections.abc import Mapping
from typing import Any

import gymnasium
import numpy as np

from .arena_geometry import (HOMEMADE_ARENA_FOLDER, MANIFEST_FILE_NAME, collision_mesh_files,
                             read_collision_mesh_file, read_manifest)
from .bot import GOAL_LINE_Y, CarReading, StuckWatch, team_controls
from .camera import CameraSettings, advance_rig, camera_view, start_rig
from .controls import (ANALOG_HIGH, ANALOG_LOW, BUTTON_COUNT, NO_CONTROLS, Controls,
                       ControllerSettings, InputMode, buttons_from_action, controls_from_pad,
                       pad_from_analog, pad_from_buttons)
from .engine import KICKOFF_SPOT_COUNT, RESPAWN_SPOT_COUNT, SoccarArena, load_rocketsim
from .hud import HudPainter, hud_for
from .match import (TEAMS, TICKS_PER_SECOND, ArenaChanges, Car, Match, Phase, Team, TickEvents,
                    match_cars)
from .renderer import CarPose, Renderer, Scene

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
        self._collision_meshes = read_manifest(HOMEMADE_ARENA_FOLDER / MANIFEST_FILE_NAME)
        load_rocketsim(HOMEMADE_ARENA_FOLDER, self._collision_meshes)
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
        self._arena: SoccarArena | None = None
        self._camera_settings = _settings(CameraSettings, camera, "camera")
        self._controller_settings = _settings(ControllerSettings, controller, "controller")
        self._renderer: Renderer | None = None
        if render_mode == "rgb_array":
            meshes = [read_collision_mesh_file(path)
                      for path in collision_mesh_files(HOMEMADE_ARENA_FOLDER)]
            self._renderer = Renderer(render_width, render_height, meshes,
                                      SoccarArena().pad_spots())
            self._hud_painter = HudPainter(render_width, render_height)

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._arena = SoccarArena(self._team_size)
        self._match = Match(self._match_ticks)
        self._arena.place_for_kickoff(self._kickoff_spots())
        self._ball_cam = True
        self._ball_cam_held = False
        self._applied = dict.fromkeys(self._cars, NO_CONTROLS)
        self._stuck_watches = {car: StuckWatch() for car in self._cars if car != PLAYER}
        self._clear_step_events()
        car = self._arena.car(PLAYER)
        self._rig = start_rig(car.pos.as_numpy(), car.rot_mat.forward.as_numpy(),
                              car.rot_mat.up.as_numpy(), self._ball_cam)
        self._frame: np.ndarray | None = None
        return self._observation(), self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._arena is None:
            raise ValueError("call reset before step")
        if self._match.phase is Phase.ENDED:
            raise ValueError("the match has ended: call reset")
        if self._input_mode is InputMode.DIGITAL:
            pad = pad_from_buttons(buttons_from_action(action))
        else:
            pad = pad_from_analog(action)
        if pad.ball_cam and not self._ball_cam_held:
            self._ball_cam = not self._ball_cam
        self._ball_cam_held = pad.ball_cam
        wanted = {**self._bot_controls(),
                  PLAYER: controls_from_pad(pad, self._controller_settings)}
        score_before = dict(self._match.score)
        self._clear_step_events()
        for _ in range(self._ticks_per_step):
            if self._match.phase is Phase.ENDED:
                break
            events = TickEvents()
            if self._match.physics_runs:
                for car in self._cars:
                    obeys = self._match.takes_controls(car)
                    self._applied[car] = wanted[car] if obeys else NO_CONTROLS
                    self._arena.set_controls(car, self._applied[car])
                events = self._arena.tick()
            else:
                self._applied = dict.fromkeys(self._cars, NO_CONTROLS)
            scores = dict(self._match.score)
            self._apply(self._match.tick(events))
            self._record(events, scores)
        self._advance_camera(self._ticks_per_step / TICKS_PER_SECOND)
        self._frame = None
        reward = (self._match.score[Team.BLUE] - score_before[Team.BLUE]
                  - (self._match.score[Team.ORANGE] - score_before[Team.ORANGE]))
        terminated = self._match.phase is Phase.ENDED
        return self._observation(), float(reward), terminated, False, self._info()

    def render(self) -> np.ndarray | None:
        """The frame of the latest ``reset`` or ``step``, ``(render_height, render_width, 3)``
        RGB bytes with the on-screen display, read-only; drawn once, however often it is asked
        for. None without a render mode."""
        if self._renderer is None:
            gymnasium.logger.warn("render_mode is None: this env draws no frame")
            return None
        if self._arena is None:
            raise ValueError("call reset before render")
        if self._frame is None:
            hud = hud_for(self._match, self._arena.car(PLAYER).boost, PLAYER)
            self._frame = self._hud_painter.paint(self._renderer.draw(self._scene()), hud)
            self._frame.flags.writeable = False
        return self._frame

    def sample(self) -> dict[str, Any]:
        """The state to log for analysis, after the latest ``reset`` or ``step``: each of
        :attr:`sample_fields`, with what happened during that step (``goal``, the scoring
        team's name or ""; each car's ``touch`` and ``demolition``) and the controls each car
        obeyed (``input_*``)."""
        if self._arena is None:
            raise ValueError("call reset before sample")
        ball = self._arena.ball
        values: list[Any] = [
            self._match.phase.value, self._match.clock_ticks_left / TICKS_PER_SECOND,
            self._match.overtime, self._match.overtime_ticks / TICKS_PER_SECOND,
            self._match.score[Team.BLUE], self._match.score[Team.ORANGE],
            self._step_goal.value if self._step_goal else "", self._ball_cam,
            *ball.pos.as_tuple(), *ball.vel.as_tuple(),
        ]
        for car in self._cars:
            state, applied = self._arena.car(car), self._applied[car]
            angle = state.rot_mat.as_angle()
            values += [*state.pos.as_tuple(), *state.vel.as_tuple(), angle.yaw, angle.pitch,
                       angle.roll, state.boost, state.is_on_ground, state.is_supersonic,
                       state.is_demoed, self._step_touches[car], self._step_demolitions[car],
                       applied.throttle, applied.steer, applied.pitch, applied.yaw, applied.roll,
                       applied.jump, applied.boost, applied.handbrake]
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
            "gl_renderer": None if self._renderer is None else self._renderer.gl_renderer,
            "camera": dataclasses.asdict(self._camera_settings),
            "input_mode": self._input_mode.value,
            "controller": dataclasses.asdict(self._controller_settings),
        }

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._arena = None

    def _ball_in_play(self) -> np.ndarray | None:
        """The ball's position, or None while it is out of play after a goal."""
        return None if self._match.phase is Phase.GOAL else self._arena.ball.pos.as_numpy()

    def _advance_camera(self, seconds: float) -> None:
        car = self._arena.car(PLAYER)
        self._rig = advance_rig(self._rig, car.pos.as_numpy(), car.rot_mat.forward.as_numpy(),
                                car.rot_mat.up.as_numpy(), car.is_on_ground, self._ball_in_play(),
                                self._ball_cam, seconds, self._camera_settings)

    def _scene(self) -> Scene:
        ball = self._ball_in_play()
        cars = []
        for car in self._cars:
            state = self._arena.car(car)
            cars.append(CarPose(car.team, state.pos.as_tuple(), state.rot_mat.forward.as_tuple(),
                                state.rot_mat.right.as_tuple(), state.rot_mat.up.as_tuple(),
                                visible=not state.is_demoed,
                                boosting=self._applied[car].boost and state.boost > 0))
        camera = camera_view(self._rig, self._arena.car(PLAYER).pos.as_numpy(), ball,
                             self._camera_settings)
        axes = self._arena.ball.rot_mat
        return Scene(camera, None if ball is None else tuple(ball), tuple(cars),
                     self._arena.pads_active(),
                     (axes.forward.as_tuple(), axes.right.as_tuple(), axes.up.as_tuple()))

    def _bot_controls(self) -> dict[Car, Controls]:
        """Each bot's controls for the next step; the agent's car is one of its team's."""
        ball = np.array(self._arena.ball.pos.as_tuple())
        seconds = self._ticks_per_step / TICKS_PER_SECOND
        controls: dict[Car, Controls] = {}
        for team in TEAMS:
            cars = [car for car in self._cars if car.team is team]
            readings = [self._reading(car) for car in cars]
            attacked_goal_y = GOAL_LINE_Y if team is Team.BLUE else -GOAL_LINE_Y
            wanted = team_controls(readings, ball, attacked_goal_y)
            for car, reading, car_wanted in zip(cars, readings, wanted, strict=True):
                if car == PLAYER:
                    continue
                if not self._match.takes_controls(car):
                    self._stuck_watches[car] = StuckWatch()
                    controls[car] = car_wanted
                    continue
                controls[car] = self._stuck_watches[car].controls(car_wanted, reading, seconds)
        return controls

    def _reading(self, car: Car) -> CarReading:
        """``car`` as a bot reads it."""
        state = self._arena.car(car)
        axes = state.rot_mat
        return CarReading(np.array(state.pos.as_tuple()), np.array(state.vel.as_tuple()),
                          np.array(axes.forward.as_tuple()), np.array(axes.right.as_tuple()),
                          np.array(axes.up.as_tuple()), state.is_demoed)

    def _kickoff_spots(self) -> list[int]:
        """A different kickoff spot for each car of a team, drawn at random."""
        spots = self.np_random.permutation(KICKOFF_SPOT_COUNT)[:self._team_size]
        return [int(spot) for spot in spots]

    def _apply(self, changes: ArenaChanges) -> None:
        if changes.kickoff:
            self._arena.place_for_kickoff(self._kickoff_spots())
        if changes.remove_ball:
            self._arena.remove_ball()
        for team in TEAMS:
            cars = [car for car in changes.respawn if car.team is team]
            if not cars:
                continue
            # A team's cars that come back together come back at different spots.
            spots = self.np_random.permutation(RESPAWN_SPOT_COUNT)[:len(cars)]
            for car, spot in zip(cars, spots, strict=True):
                self._arena.respawn(car, int(spot))

    def _clear_step_events(self) -> None:
        self._step_goal: Team | None = None
        self._step_touches = dict.fromkeys(self._cars, False)
        self._step_demolitions = dict.fromkeys(self._cars, False)

    def _record(self, events: TickEvents, scores_before_tick: dict[Team, int]) -> None:
        for team in TEAMS:
            if self._match.score[team] > scores_before_tick[team]:
                self._step_goal = team
        for car in events.touched:
            self._step_touches[car] = True
        for car in events.demolished:
            self._step_demolitions[car] = True

    def _observation(self) -> np.ndarray:
        ball = self._arena.ball
        values: list[float] = [*ball.pos.as_tuple(), *ball.vel.as_tuple(), *ball.ang_vel.as_tuple()]
        for car in self._cars:
            state = self._arena.car(car)
            values += [*state.pos.as_tuple(), *state.vel.as_tuple(), *state.ang_vel.as_tuple(),
                       *state.rot_mat.forward.as_tuple(), *state.rot_mat.up.as_tuple(),
                       state.boost, state.is_on_ground, state.has_flip_or_jump(),
                       state.is_supersonic, state.is_demoed]
        values += self._arena.pads_active()
        values += [self._match.clock_ticks_left / TICKS_PER_SECOND, self._match.score[Team.BLUE],
                   self._match.score[Team.ORANGE], self._match.overtime]
        return np.array(values, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        return {"phase": self._match.phase.value, "blue_score": self._match.score[Team.BLUE],
                "orange_score": self._match.score[Team.ORANGE]}
