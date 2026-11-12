"""A match at the end of a step, as one value: everything a player's view, the observation and
the logs read of it, and its bytes, which carry it over a network.

Positions, motion and directions are RocketSim's float32 values, kept as float32 arrays, so a
snapshot and its bytes lose nothing of them; the controls each car obeyed are kept as the floats
they were given. A snapshot unpacked from bytes therefore gives the same observation, the same
``sample()`` and the same frame as the snapshot it was packed from.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .controls import Controls
from .match import Car, MatchState, Phase, Team, match_cars

#: The version of :meth:`Snapshot.pack`'s bytes; :meth:`Snapshot.unpack` reads only its own.
FORMAT_VERSION = 1

_PHASES = tuple(Phase)
_TEAM_CODES = {None: 0, Team.BLUE: 1, Team.ORANGE: 2}
_TEAMS_BY_CODE = {code: team for team, code in _TEAM_CODES.items()}
#: The format version, the cars a team, and the step.
_HEADER = struct.Struct("<BBI")
#: The phase, the ticks left in it and on the clock, overtime and its ticks, blue's and orange's
#: goals, the team that scored last, the ticks since the kickoff went live, and the step's goal.
_MATCH = struct.Struct("<BHI?IHHBIB")
#: The ball's position, velocity, angular velocity, forward, right and up, as float32.
_BALL_FLOATS = 18
#: A car's position, velocity, angular velocity, forward, right, up, yaw, pitch, roll and
#: boost, as float32.
_CAR_FLOATS = 22
#: A car's throttle, steer, pitch, yaw and roll (float64), one bit each for whether it is on the
#: ground, can jump or flip, is supersonic, is demolished, jumps, boosts, powerslides, touched
#: the ball and was demolished, and its ticks until it comes back from a demolition.
_CAR_SCALARS = struct.Struct("<5dHH")
_CAR_FLAG_COUNT = 9
#: The boost pads, and which of them are active, one bit each.
_PADS = struct.Struct("<BQ")


@dataclass(frozen=True, eq=False)
class BallSnapshot:
    """The ball: its position (uu), velocity (uu/s), angular velocity (rad/s), and its forward,
    right and up directions, each a float32 array of 3."""

    position: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray


@dataclass(frozen=True, eq=False)
class CarSnapshot:
    """One car, as RocketSim gives it: its position (uu), velocity (uu/s), angular velocity
    (rad/s) and its forward, right and up directions, each a float32 array of 3.

    :ivar yaw, pitch, roll: its rotation (rad).
    :ivar boost: 0 to 100.
    :ivar controls: the controls it obeyed on the step's last tick.
    :ivar touched_ball: whether it touched the ball during the step.
    :ivar was_demolished: whether it was demolished during the step.
    """

    car: Car
    position: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    yaw: float
    pitch: float
    roll: float
    boost: float
    is_on_ground: bool
    has_flip_or_jump: bool
    is_supersonic: bool
    is_demolished: bool
    controls: Controls
    touched_ball: bool
    was_demolished: bool


@dataclass(frozen=True, eq=False)
class Snapshot:
    """A match at the end of a step.

    :ivar step: the steps played since the match began.
    :ivar goal: the team that scored during the step.
    :ivar cars: every car, blue's then orange's (:func:`~rocketsim_gym.match.match_cars`).
    :ivar pads_active: each boost pad, in RocketSim's order: whether it can be picked up.
    """

    step: int
    match: MatchState
    goal: Team | None
    ball: BallSnapshot
    cars: tuple[CarSnapshot, ...]
    pads_active: tuple[bool, ...]

    @property
    def team_size(self) -> int:
        """Cars a team."""
        return len(self.cars) // 2

    def car(self, car: Car) -> CarSnapshot:
        """``car``'s snapshot."""
        return self.cars[(0 if car.team is Team.BLUE else self.team_size) + car.number - 1]

    @property
    def ball_in_play(self) -> np.ndarray | None:
        """The ball's position, or None while it is out of play after a goal."""
        return None if self.match.phase is Phase.GOAL else self.ball.position

    def pack(self) -> bytes:
        """The snapshot as bytes, which :meth:`unpack` reads back."""
        match = self.match
        parts = [
            _HEADER.pack(FORMAT_VERSION, self.team_size, self.step),
            _MATCH.pack(_PHASES.index(match.phase), match.phase_ticks_left,
                        match.clock_ticks_left, match.overtime, match.overtime_ticks,
                        match.score[Team.BLUE], match.score[Team.ORANGE],
                        _TEAM_CODES[match.last_goal], match.ticks_since_go,
                        _TEAM_CODES[self.goal]),
            _float32_bytes(self.ball.position, self.ball.velocity, self.ball.angular_velocity,
                           self.ball.forward, self.ball.right, self.ball.up),
        ]
        for car in self.cars:
            controls = car.controls
            flags = _bits((car.is_on_ground, car.has_flip_or_jump, car.is_supersonic,
                           car.is_demolished, controls.jump, controls.boost, controls.handbrake,
                           car.touched_ball, car.was_demolished))
            parts += [
                _float32_bytes(car.position, car.velocity, car.angular_velocity, car.forward,
                               car.right, car.up, (car.yaw, car.pitch, car.roll, car.boost)),
                _CAR_SCALARS.pack(controls.throttle, controls.steer, controls.pitch,
                                  controls.yaw, controls.roll, flags,
                                  match.respawn_ticks_left.get(car.car, 0)),
            ]
        parts.append(_PADS.pack(len(self.pads_active), _bits(self.pads_active)))
        return b"".join(parts)

    @classmethod
    def unpack(cls, data: bytes) -> Snapshot:
        """The snapshot :meth:`pack` made ``data`` of.

        :raises ValueError: ``data`` is not a snapshot of this format.
        """
        reader = _Reader(data)
        version, team_size, step = reader.take(_HEADER)
        if version != FORMAT_VERSION:
            raise ValueError(f"a snapshot of format {version}; this one reads {FORMAT_VERSION}")
        (phase, phase_ticks_left, clock_ticks_left, overtime, overtime_ticks, blue_score,
         orange_score, last_goal, ticks_since_go, goal) = reader.take(_MATCH)
        ball = BallSnapshot(*_vectors(reader.floats(_BALL_FLOATS), 6))
        cars, respawn_ticks_left = [], {}
        for car in match_cars(team_size):
            floats = reader.floats(_CAR_FLOATS)
            throttle, steer, pitch, yaw, roll, flags, respawn_ticks = reader.take(_CAR_SCALARS)
            (is_on_ground, has_flip_or_jump, is_supersonic, is_demolished, jump, boost, handbrake,
             touched_ball, was_demolished) = _flags(flags, _CAR_FLAG_COUNT)
            cars.append(CarSnapshot(
                car, *_vectors(floats, 6), *(float(value) for value in floats[18:]),
                is_on_ground, has_flip_or_jump, is_supersonic, is_demolished,
                Controls(throttle, steer, pitch, yaw, roll, jump, boost, handbrake),
                touched_ball, was_demolished))
            if respawn_ticks:
                respawn_ticks_left[car] = respawn_ticks
        pad_count, pad_bits = reader.take(_PADS)
        reader.finish()
        match = MatchState(_PHASES[phase], phase_ticks_left, clock_ticks_left, overtime,
                           overtime_ticks,
                           MappingProxyType({Team.BLUE: blue_score, Team.ORANGE: orange_score}),
                           _TEAMS_BY_CODE[last_goal], ticks_since_go,
                           MappingProxyType(respawn_ticks_left))
        return cls(step, match, _TEAMS_BY_CODE[goal], ball, tuple(cars),
                   _flags(pad_bits, pad_count))


class _Reader:
    """Reads a snapshot's bytes in order, and refuses ones that run short or long."""

    def __init__(self, data: bytes) -> None:
        self._data, self._offset = bytes(data), 0

    def take(self, layout: struct.Struct) -> tuple:
        values = self._unpack(layout.unpack_from, layout.size)
        self._offset += layout.size
        return values

    def floats(self, count: int) -> np.ndarray:
        values = self._unpack(lambda data, offset: np.frombuffer(data, "<f4", count, offset),
                              4 * count)
        self._offset += 4 * count
        return values.astype(np.float32)

    def finish(self) -> None:
        if self._offset != len(self._data):
            raise ValueError(f"a snapshot with {len(self._data) - self._offset} bytes too many")

    def _unpack(self, read, size: int):
        if self._offset + size > len(self._data):
            raise ValueError(f"a snapshot cut short at {len(self._data)} bytes")
        return read(self._data, self._offset)


def _bits(flags) -> int:
    """``flags`` as the bits of an integer, the first the lowest."""
    return sum(1 << bit for bit, flag in enumerate(flags) if flag)


def _flags(bits: int, count: int) -> tuple[bool, ...]:
    """The first ``count`` bits of ``bits``, the lowest first."""
    return tuple(bool(bits >> bit & 1) for bit in range(count))


def _float32_bytes(*values) -> bytes:
    return np.concatenate([np.asarray(value, np.float32) for value in values]).astype("<f4").tobytes()


def _vectors(floats: np.ndarray, count: int) -> list[np.ndarray]:
    """The first ``count`` vectors of 3 in ``floats``, each its own read-only array."""
    vectors = []
    for index in range(count):
        vector = floats[3 * index:3 * index + 3].copy()
        vector.flags.writeable = False
        vectors.append(vector)
    return vectors
