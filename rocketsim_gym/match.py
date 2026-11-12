"""The rules of a soccar match, 1v1, 2v2 or 3v3, advanced one physics tick at a time.

A :class:`Match` knows the clock, the score, and what follows a goal, a demolition or the end of
time; it knows nothing of physics. Each tick it is told what happened in the arena
(:class:`TickEvents`) and answers what the arena must do next (:class:`ArenaChanges`). The rules
are Rocket League's:

- before each kickoff, a 3 s countdown holds everything still;
- the clock starts at the kickoff's first touch;
- a goal pauses play for 3 s, the ball gone and the cars free to drive, then comes a kickoff;
- when the clock reaches zero, play goes on until the ball touches the ground;
- a tie then goes to overtime, which the next goal wins;
- a demolished car comes back 3 s later.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

#: Physics ticks per second, as Rocket League runs them.
TICKS_PER_SECOND = 120
#: The countdown before each kickoff.
COUNTDOWN_TICKS = 3 * TICKS_PER_SECOND
#: The pause after a goal, before the next kickoff's countdown.
GOAL_PAUSE_TICKS = 3 * TICKS_PER_SECOND
#: How long a demolished car stays out.
RESPAWN_TICKS = 3 * TICKS_PER_SECOND


class Team(Enum):
    """A side of the field: blue defends the goal at negative y, orange the one at positive y."""

    BLUE = "blue"
    ORANGE = "orange"


#: Both teams, in the order everything goes through them.
TEAMS = (Team.BLUE, Team.ORANGE)


@dataclass(frozen=True)
class Car:
    """One of a match's cars: its team, and its number in the team, from 1."""

    team: Team
    number: int

    @property
    def name(self) -> str:
        """The team and the number, as logs write them: ``blue1``."""
        return f"{self.team.value}{self.number}"


def match_cars(team_size: int) -> tuple[Car, ...]:
    """Every car of a match of ``team_size`` cars a team, in the order everything goes through
    them: blue's, then orange's, each team's by number."""
    return tuple(Car(team, number) for team in TEAMS for number in range(1, team_size + 1))


class Phase(Enum):
    """Where a match is."""

    #: Before a kickoff: the cars and the ball are held at their spots, and physics is paused.
    COUNTDOWN = "countdown"
    #: The kickoff is live; the clock waits for the first touch.
    KICKOFF = "kickoff"
    #: The clock runs; in overtime it counts up.
    PLAY = "play"
    #: Just after a goal: the ball is out of play and the cars drive on.
    GOAL = "goal"
    #: The match is over.
    ENDED = "ended"


@dataclass(frozen=True)
class TickEvents:
    """What happened in the arena during one physics tick.

    :ivar goal: the team that scored, when the ball crossed a goal line.
    :ivar touched: the cars that touched the ball.
    :ivar demolished: the cars that were demolished.
    :ivar ball_on_ground: whether the ball is on the floor.
    """

    goal: Team | None = None
    touched: tuple[Car, ...] = ()
    demolished: tuple[Car, ...] = ()
    ball_on_ground: bool = False


@dataclass(frozen=True)
class ArenaChanges:
    """What the arena must do after a tick.

    :ivar kickoff: set up a kickoff: every car at a kickoff spot, the ball at the centre, every
        boost pad up.
    :ivar remove_ball: take the ball out of play until that kickoff.
    :ivar respawn: the demolished cars that come back now, each at a respawn spot.
    """

    kickoff: bool = False
    remove_ball: bool = False
    respawn: tuple[Car, ...] = ()


class Match:
    """One match: its phase, clock, score, overtime and the demolished cars' wait.

    It starts in the countdown of its first kickoff, which the arena sets up.

    :ivar phase: where the match is.
    :ivar phase_ticks_left: the ticks left in a countdown or in the pause after a goal.
    :ivar clock_ticks_left: the regulation time left, in ticks.
    :ivar overtime: whether the match is in overtime.
    :ivar overtime_ticks: the overtime played, in ticks.
    :ivar score: each team's goals.
    :ivar last_goal: the team that scored last.
    :ivar ticks_since_go: the ticks since the latest kickoff went live.
    :ivar respawn_ticks_left: each demolished car's ticks until it comes back.
    """

    def __init__(self, match_ticks: int) -> None:
        """:param match_ticks: the regulation time, in ticks.
        :raises ValueError: ``match_ticks`` is not a positive whole number.
        """
        if isinstance(match_ticks, bool) or not isinstance(match_ticks, int) or match_ticks < 1:
            raise ValueError(f"match_ticks {match_ticks!r}: a match lasts a positive whole "
                             f"number of ticks")
        self.phase = Phase.COUNTDOWN
        self.phase_ticks_left = COUNTDOWN_TICKS
        self.clock_ticks_left = match_ticks
        self.overtime = False
        self.overtime_ticks = 0
        self.score = dict.fromkeys(TEAMS, 0)
        self.last_goal: Team | None = None
        self.ticks_since_go = 0
        self.respawn_ticks_left: dict[Car, int] = {}

    @property
    def physics_runs(self) -> bool:
        """Whether the arena steps: in every phase but a countdown and the end."""
        return self.phase in (Phase.KICKOFF, Phase.PLAY, Phase.GOAL)

    @property
    def winner(self) -> Team | None:
        """The team that won, once the match has ended."""
        if self.phase is not Phase.ENDED:
            return None
        return max(TEAMS, key=self.score.__getitem__)

    def takes_controls(self, car: Car) -> bool:
        """Whether ``car`` obeys its controls: not in a countdown, not while demolished."""
        return self.physics_runs and car not in self.respawn_ticks_left

    def tick(self, events: TickEvents = TickEvents()) -> ArenaChanges:
        """Advance one tick on what happened in it, and say what the arena must do next.

        :param events: what happened in the arena this tick; nothing while physics is paused.
        :raises ValueError: the match has ended, or events came while physics is paused.
        """
        if self.phase is Phase.ENDED:
            raise ValueError("the match has ended")
        if self.phase is Phase.COUNTDOWN:
            if events != TickEvents():
                raise ValueError(f"{events}: nothing happens in a countdown, physics is paused")
            self.phase_ticks_left -= 1
            if self.phase_ticks_left == 0:
                self.phase, self.ticks_since_go = Phase.KICKOFF, 0
            return ArenaChanges()
        respawn = self._advance_respawns(events.demolished)
        if self.phase is Phase.GOAL:
            self.phase_ticks_left -= 1
            if self.phase_ticks_left > 0:
                return ArenaChanges(respawn=respawn)
            if self.overtime or self.clock_ticks_left == 0:
                return self._finish()
            return self._kickoff()
        self._advance_clock(events.touched)
        self.ticks_since_go += 1
        if events.goal is not None:
            self.score[events.goal] += 1
            self.last_goal = events.goal
            self.phase, self.phase_ticks_left = Phase.GOAL, GOAL_PAUSE_TICKS
            return ArenaChanges(remove_ball=True, respawn=respawn)
        if (self.phase is Phase.PLAY and not self.overtime and self.clock_ticks_left == 0
                and events.ball_on_ground):
            return self._finish()
        return ArenaChanges(respawn=respawn)

    def _advance_clock(self, touched: tuple[Car, ...]) -> None:
        """The kickoff's first touch starts the clock; from the next tick on, it runs."""
        if self.phase is Phase.KICKOFF:
            if touched:
                self.phase = Phase.PLAY
        elif self.overtime:
            self.overtime_ticks += 1
        elif self.clock_ticks_left > 0:
            self.clock_ticks_left -= 1

    def _advance_respawns(self, demolished: tuple[Car, ...]) -> tuple[Car, ...]:
        """Count down the demolished cars' wait, start it for the cars just demolished, and
        return the cars whose wait is over, in the order they were demolished."""
        due = []
        for car in tuple(self.respawn_ticks_left):
            self.respawn_ticks_left[car] -= 1
            if self.respawn_ticks_left[car] == 0:
                del self.respawn_ticks_left[car]
                due.append(car)
        for car in demolished:
            if car in self.respawn_ticks_left:
                raise ValueError(f"{car.name} is demolished already")
            self.respawn_ticks_left[car] = RESPAWN_TICKS
        return tuple(due)

    def _finish(self) -> ArenaChanges:
        """Time is up: the leader wins, and a tie goes to overtime."""
        if self.score[Team.BLUE] != self.score[Team.ORANGE]:
            self.phase = Phase.ENDED
            self.respawn_ticks_left.clear()
            return ArenaChanges()
        self.overtime = True
        return self._kickoff()

    def _kickoff(self) -> ArenaChanges:
        """The countdown of a new kickoff, which puts every car back."""
        self.phase, self.phase_ticks_left = Phase.COUNTDOWN, COUNTDOWN_TICKS
        self.respawn_ticks_left.clear()
        return ArenaChanges(kickoff=True)
