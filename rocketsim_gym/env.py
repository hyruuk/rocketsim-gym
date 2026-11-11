"""``RocketSimSoccar-v0``: a whole 1v1 soccar match against a scripted ball-chaser."""

from __future__ import annotations

import importlib.metadata
from typing import Any

import gymnasium
import numpy as np

from .arena_geometry import HOMEMADE_ARENA_FOLDER, MANIFEST_FILE_NAME, read_manifest
from .bot import GOAL_LINE_Y, ball_chaser
from .controls import (BUTTON_COUNT, NO_CONTROLS, Button, Controls, buttons_from_action,
                       controls_from_buttons)
from .engine import KICKOFF_SPOT_COUNT, RESPAWN_SPOT_COUNT, SoccarArena, load_rocketsim
from .match import TEAMS, TICKS_PER_SECOND, ArenaChanges, Match, Phase, Team, TickEvents

#: The number of values in an observation.
OBSERVATION_SIZE = 87

_CAR_SAMPLE_FIELDS = ("x", "y", "z", "vx", "vy", "vz", "yaw", "pitch", "roll", "boost",
                      "on_ground", "supersonic", "demolished", "touch", "demolition",
                      "input_throttle", "input_steer", "input_pitch", "input_yaw", "input_roll",
                      "input_jump", "input_boost", "input_handbrake")
#: The fields :meth:`SoccarEnv.sample` returns, every step.
SAMPLE_FIELDS = (
    "phase", "seconds_left", "overtime", "overtime_seconds", "blue_score", "orange_score",
    "goal", "ball_cam", "ball_x", "ball_y", "ball_z", "ball_vx", "ball_vy", "ball_vz",
    *(f"{team.value}_{name}" for team in TEAMS for name in _CAR_SAMPLE_FIELDS),
)


class SoccarEnv(gymnasium.Env):
    """A Rocket League-style 1v1 soccar match: the agent drives the blue car, a scripted
    ball-chaser the orange one, on RocketSim physics and the homemade arena.

    The match follows Rocket League's rules (:mod:`rocketsim_gym.match`): a countdown before
    each kickoff, a clock that starts at the first touch, a pause after each goal, play to the
    ball's landing at 0:00, overtime on a tie, and demolished cars back after 3 s. The kickoff
    and respawn spots are drawn from the env's own random generator, seeded by ``reset``; the
    same seed and the same actions give the same match.

    **Action**: twelve buttons, each held or not (``MultiBinary(12)``; the table is in
    :mod:`rocketsim_gym.controls`). The car obeys them except in a countdown and while it is
    demolished; ``ball_cam`` switches the camera when pressed.

    **Observation**: 87 values, in Rocket League's units (uu, uu/s, rad/s):

    =======  ==============================================================================
    0-8      the ball: position, velocity, angular velocity
    9-28     the blue car: position, velocity, angular velocity, forward direction, up
             direction, boost (0-100), on the ground, can jump or flip, supersonic, demolished
    29-48    the orange car, the same
    49-82    each boost pad, in RocketSim's order: 1 while it can be picked up
    83-86    seconds left in regulation, blue's goals, orange's goals, 1 in overtime
    =======  ==============================================================================

    **Reward**: +1 for each blue goal, -1 for each orange goal. The episode terminates when
    the match ends; overtime has no end but a goal.

    :param match_seconds: regulation time.
    :param ticks_per_step: physics ticks per step, at 120 ticks a second.
    :param render_mode: none: the env does not draw.
    :raises ValueError: an argument is out of range.
    """

    metadata = {"render_modes": []}

    def __init__(self, match_seconds: float = 300.0, ticks_per_step: int = 2,
                 render_mode: str | None = None) -> None:
        if (isinstance(match_seconds, bool) or not isinstance(match_seconds, (int, float))
                or round(match_seconds * TICKS_PER_SECOND) < 1):
            raise ValueError(f"match_seconds {match_seconds!r}: a positive number of seconds, "
                             f"at least one tick")
        if isinstance(ticks_per_step, bool) or not isinstance(ticks_per_step, int) \
                or ticks_per_step < 1:
            raise ValueError(f"ticks_per_step {ticks_per_step!r}: a positive whole number")
        if render_mode is not None:
            raise ValueError(f"render_mode {render_mode!r}: this env does not draw")
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": TICKS_PER_SECOND / ticks_per_step}
        self._match_ticks = round(match_seconds * TICKS_PER_SECOND)
        self._ticks_per_step = ticks_per_step
        self._collision_meshes = read_manifest(HOMEMADE_ARENA_FOLDER / MANIFEST_FILE_NAME)
        load_rocketsim(HOMEMADE_ARENA_FOLDER, self._collision_meshes)
        self.action_space = gymnasium.spaces.MultiBinary(BUTTON_COUNT)
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (OBSERVATION_SIZE,),
                                                      np.float32)
        self._arena: SoccarArena | None = None

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._arena = SoccarArena()
        self._match = Match(self._match_ticks)
        self._arena.place_for_kickoff(int(self.np_random.integers(KICKOFF_SPOT_COUNT)))
        self._ball_cam = True
        self._previous_buttons = (False,) * BUTTON_COUNT
        self._applied = dict.fromkeys(TEAMS, NO_CONTROLS)
        self._clear_step_events()
        return self._observation(), self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._arena is None:
            raise ValueError("call reset before step")
        if self._match.phase is Phase.ENDED:
            raise ValueError("the match has ended: call reset")
        buttons = buttons_from_action(action)
        if buttons[Button.BALL_CAM] and not self._previous_buttons[Button.BALL_CAM]:
            self._ball_cam = not self._ball_cam
        self._previous_buttons = buttons
        wanted = {Team.BLUE: controls_from_buttons(buttons), Team.ORANGE: self._bot_controls()}
        score_before = dict(self._match.score)
        self._clear_step_events()
        for _ in range(self._ticks_per_step):
            if self._match.phase is Phase.ENDED:
                break
            events = TickEvents()
            if self._match.physics_runs:
                for team in TEAMS:
                    obeys = self._match.takes_controls(team)
                    self._applied[team] = wanted[team] if obeys else NO_CONTROLS
                    self._arena.set_controls(team, self._applied[team])
                events = self._arena.tick()
            else:
                self._applied = dict.fromkeys(TEAMS, NO_CONTROLS)
            scores = dict(self._match.score)
            self._apply(self._match.tick(events))
            self._record(events, scores)
        reward = (self._match.score[Team.BLUE] - score_before[Team.BLUE]
                  - (self._match.score[Team.ORANGE] - score_before[Team.ORANGE]))
        terminated = self._match.phase is Phase.ENDED
        return self._observation(), float(reward), terminated, False, self._info()

    def sample(self) -> dict[str, Any]:
        """The state to log for analysis, after the latest ``reset`` or ``step``: each of
        :data:`SAMPLE_FIELDS`, with what happened during that step (``goal``, the scoring
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
        for team in TEAMS:
            car, applied = self._arena.car(team), self._applied[team]
            angle = car.rot_mat.as_angle()
            values += [*car.pos.as_tuple(), *car.vel.as_tuple(), angle.yaw, angle.pitch,
                       angle.roll, car.boost, car.is_on_ground, car.is_supersonic, car.is_demoed,
                       self._step_touches[team], self._step_demolitions[team],
                       applied.throttle, applied.steer, applied.pitch, applied.yaw, applied.roll,
                       applied.jump, applied.boost, applied.handbrake]
        return dict(zip(SAMPLE_FIELDS, values, strict=True))

    def provenance(self) -> dict[str, Any]:
        """What a match depends on besides its seed and actions: the package versions, the
        collision meshes' SHA-256s and the tick rates."""
        return {
            "rocketsim_gym_version": importlib.metadata.version("rocketsim-gym"),
            "rocketsim_version": importlib.metadata.version("rocketsim"),
            "collision_meshes": dict(self._collision_meshes),
            "ticks_per_second": TICKS_PER_SECOND,
            "ticks_per_step": self._ticks_per_step,
        }

    def close(self) -> None:
        self._arena = None

    def _bot_controls(self) -> Controls:
        car = self._arena.car(Team.ORANGE)
        return ball_chaser(np.array(car.pos.as_tuple()), np.array(car.rot_mat.forward.as_tuple()),
                           np.array(car.rot_mat.right.as_tuple()),
                           np.array(self._arena.ball.pos.as_tuple()), -GOAL_LINE_Y)

    def _apply(self, changes: ArenaChanges) -> None:
        if changes.kickoff:
            self._arena.place_for_kickoff(int(self.np_random.integers(KICKOFF_SPOT_COUNT)))
        if changes.remove_ball:
            self._arena.remove_ball()
        for team in changes.respawn:
            self._arena.respawn(team, int(self.np_random.integers(RESPAWN_SPOT_COUNT)))

    def _clear_step_events(self) -> None:
        self._step_goal: Team | None = None
        self._step_touches = dict.fromkeys(TEAMS, False)
        self._step_demolitions = dict.fromkeys(TEAMS, False)

    def _record(self, events: TickEvents, scores_before_tick: dict[Team, int]) -> None:
        for team in TEAMS:
            if self._match.score[team] > scores_before_tick[team]:
                self._step_goal = team
        for team in events.touched:
            self._step_touches[team] = True
        for team in events.demolished:
            self._step_demolitions[team] = True

    def _observation(self) -> np.ndarray:
        ball = self._arena.ball
        values: list[float] = [*ball.pos.as_tuple(), *ball.vel.as_tuple(), *ball.ang_vel.as_tuple()]
        for team in TEAMS:
            car = self._arena.car(team)
            values += [*car.pos.as_tuple(), *car.vel.as_tuple(), *car.ang_vel.as_tuple(),
                       *car.rot_mat.forward.as_tuple(), *car.rot_mat.up.as_tuple(), car.boost,
                       car.is_on_ground, car.has_flip_or_jump(), car.is_supersonic, car.is_demoed]
        values += self._arena.pads_active()
        values += [self._match.clock_ticks_left / TICKS_PER_SECOND, self._match.score[Team.BLUE],
                   self._match.score[Team.ORANGE], self._match.overtime]
        return np.array(values, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        return {"phase": self._match.phase.value, "blue_score": self._match.score[Team.BLUE],
                "orange_score": self._match.score[Team.ORANGE]}
