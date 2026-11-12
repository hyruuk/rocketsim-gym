"""One soccar match on RocketSim, played a step at a time: the cars its clients drive obey the
controls they give, and bots (:mod:`rocketsim_gym.bot`) drive every other car.

A client is whoever gives a car its controls each step: the agent of
:class:`~rocketsim_gym.env.SoccarEnv`, or a player over the network. The match follows
:mod:`rocketsim_gym.match`'s rules; its kickoff and respawn spots are drawn from the random
generator it is given, so the same generator state and the same controls play the same match.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from .bot import GOAL_LINE_Y, CarReading, StuckWatch, team_controls
from .controls import NO_CONTROLS, Controls
from .engine import KICKOFF_SPOT_COUNT, RESPAWN_SPOT_COUNT, SoccarArena
from .match import (TEAMS, TICKS_PER_SECOND, ArenaChanges, Car, Match, Phase, Team, TickEvents)
from .snapshot import BallSnapshot, CarSnapshot, Snapshot


def pad_spots() -> tuple[tuple[tuple[float, float, float], bool], ...]:
    """Each boost pad's position (uu) and whether it is big, in RocketSim's order."""
    return SoccarArena().pad_spots()


class MatchSimulation:
    """A soccar match, from its first kickoff's countdown to its end.

    :param team_size: cars a team.
    :param match_ticks: the regulation time, in ticks.
    :param ticks_per_step: physics ticks a step.
    :param client_cars: the cars clients drive; bots drive the others.
    :param rng: where the kickoff and respawn spots are drawn from.
    :raises ValueError: a client car is not one of the match's.
    :ivar cars: every car, blue's then orange's (:func:`~rocketsim_gym.match.match_cars`).
    """

    def __init__(self, team_size: int, match_ticks: int, ticks_per_step: int,
                 client_cars: Iterable[Car], rng: np.random.Generator) -> None:
        self._team_size = team_size
        self._ticks_per_step = ticks_per_step
        self._rng = rng
        self._arena = SoccarArena(team_size)
        self.cars = self._arena.cars
        self._client_cars = frozenset(client_cars)
        if not self._client_cars <= set(self.cars):
            raise ValueError(f"client cars {sorted(car.name for car in self._client_cars)}: "
                             f"cars of this match, {[car.name for car in self.cars]}")
        self._match = Match(match_ticks)
        self._arena.place_for_kickoff(self._kickoff_spots())
        self._applied = dict.fromkeys(self.cars, NO_CONTROLS)
        self._stuck_watches = {car: StuckWatch() for car in self.cars
                               if car not in self._client_cars}
        self._step = 0
        self._clear_step_events()

    @property
    def has_ended(self) -> bool:
        """Whether the match is over."""
        return self._match.phase is Phase.ENDED

    def step(self, client_controls: Mapping[Car, Controls]) -> None:
        """Play one step: its ticks, each car obeying its controls whenever the match lets it.

        :param client_controls: the controls of each car a client drives.
        :raises ValueError: the match has ended, or ``client_controls`` are not for exactly the
            cars clients drive.
        """
        if self.has_ended:
            raise ValueError("the match has ended")
        if set(client_controls) != self._client_cars:
            raise ValueError(f"controls for {sorted(car.name for car in client_controls)}: "
                             f"clients drive {sorted(car.name for car in self._client_cars)}")
        wanted = {**self._bot_controls(), **client_controls}
        self._clear_step_events()
        for _ in range(self._ticks_per_step):
            if self.has_ended:
                break
            events = TickEvents()
            if self._match.physics_runs:
                for car in self.cars:
                    obeys = self._match.takes_controls(car)
                    self._applied[car] = wanted[car] if obeys else NO_CONTROLS
                    self._arena.set_controls(car, self._applied[car])
                events = self._arena.tick()
            else:
                self._applied = dict.fromkeys(self.cars, NO_CONTROLS)
            scores = dict(self._match.score)
            self._apply(self._match.tick(events))
            self._record(events, scores)
        self._step += 1

    def snapshot(self) -> Snapshot:
        """The match as it is now, at the end of its latest step."""
        ball = self._arena.ball
        ball_axes = ball.rot_mat
        cars = []
        for car in self.cars:
            state = self._arena.car(car)
            axes, angle = state.rot_mat, state.rot_mat.as_angle()
            cars.append(CarSnapshot(
                car, state.pos.as_numpy(), state.vel.as_numpy(), state.ang_vel.as_numpy(),
                axes.forward.as_numpy(), axes.right.as_numpy(), axes.up.as_numpy(),
                angle.yaw, angle.pitch, angle.roll, state.boost, state.is_on_ground,
                state.has_flip_or_jump(), state.is_supersonic, state.is_demoed,
                self._applied[car], self._step_touches[car], self._step_demolitions[car]))
        return Snapshot(
            self._step, self._match.state(), self._step_goal,
            BallSnapshot(ball.pos.as_numpy(), ball.vel.as_numpy(), ball.ang_vel.as_numpy(),
                         ball_axes.forward.as_numpy(), ball_axes.right.as_numpy(),
                         ball_axes.up.as_numpy()),
            tuple(cars), self._arena.pads_active())

    def _bot_controls(self) -> dict[Car, Controls]:
        """Each bot's controls for the next step; the client cars count among their teams'."""
        ball = np.array(self._arena.ball.pos.as_tuple())
        seconds = self._ticks_per_step / TICKS_PER_SECOND
        controls: dict[Car, Controls] = {}
        for team in TEAMS:
            cars = [car for car in self.cars if car.team is team]
            readings = [self._reading(car) for car in cars]
            attacked_goal_y = GOAL_LINE_Y if team is Team.BLUE else -GOAL_LINE_Y
            wanted = team_controls(readings, ball, attacked_goal_y)
            for car, reading, car_wanted in zip(cars, readings, wanted, strict=True):
                if car in self._client_cars:
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
        spots = self._rng.permutation(KICKOFF_SPOT_COUNT)[:self._team_size]
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
            spots = self._rng.permutation(RESPAWN_SPOT_COUNT)[:len(cars)]
            for car, spot in zip(cars, spots, strict=True):
                self._arena.respawn(car, int(spot))

    def _clear_step_events(self) -> None:
        self._step_goal: Team | None = None
        self._step_touches = dict.fromkeys(self.cars, False)
        self._step_demolitions = dict.fromkeys(self.cars, False)

    def _record(self, events: TickEvents, scores_before_tick: dict[Team, int]) -> None:
        for team in TEAMS:
            if self._match.score[team] > scores_before_tick[team]:
                self._step_goal = team
        for car in events.touched:
            self._step_touches[car] = True
        for car in events.demolished:
            self._step_demolitions[car] = True
