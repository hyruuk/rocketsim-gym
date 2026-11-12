"""What a match server and its clients say to each other.

- **Finding a server:** a client broadcasts a query to :data:`DISCOVERY_PORT`, and every server
  that hears it answers the client directly with an announcement. Both are JSON.
- **Joining, over TCP** on the server's port: JSON messages, each after its length. A client
  asks to join (:class:`JoinRequest`), and the server welcomes it (:class:`Welcome`) or refuses
  (:class:`Refusal`). Then the client says when it is ready for a match, when it no longer is,
  and when it leaves; the server says when it closes.
- **Playing, over UDP** on the same port number: every step, the client sends its newest
  controls and the few before them (:class:`ControlsMessage`), so that a lost datagram loses no
  press, and the server sends every client a :class:`ServerFrame`: the room's state and the
  match's snapshot. The server answers each client from the socket it sends to, so firewalls see
  the frames as replies.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..controls import Controls
from ..match import TICKS_PER_SECOND, Car, match_cars

#: Changes whenever a message changes; a server and a client of different versions refuse to
#: play together.
PROTOCOL_VERSION = 1
#: The server's TCP and UDP port.
DEFAULT_PORT = 47800
#: The UDP port servers answer queries on.
DISCOVERY_PORT = 47801
#: The longest message either side sends over TCP.
MESSAGE_LIMIT = 1 << 16
#: The most controls a client repeats in one datagram: its newest and the ones before.
CONTROLS_REPEATED = 4
#: Physics ticks a step: the server plays 60 steps a second, as the env does by default.
TICKS_PER_STEP = 2
#: Cars a team, by match.
TEAM_SIZES = (1, 2, 3)

_LENGTH = struct.Struct("<I")
_CONTROLS_KIND, _FRAME_KIND = b"C", b"F"
#: A controls datagram: its kind, the client's token and how many entries follow.
_CONTROLS_HEADER = struct.Struct("<c16sB")
#: An entry: its sequence number, the server frame the client showed when it chose it, the
#: throttle, steer, pitch, yaw and roll, and one bit each for jump, boost and powerslide.
_CONTROLS_ENTRY = struct.Struct("<II5dB")
#: A server frame: its kind and number, the room's status and its ticks, cars a team, the cars
#: the room waits for and the ones bots drive (a bit each, in match order), the match's id and
#: seed, and each car's input delay in frames (-1: none yet).
_FRAME_HEADER = struct.Struct("<cIBHBBBII6b")
_FIND = {"find": "rocketsim-gym"}


class RoomStatus(Enum):
    """Where a server's room is."""

    #: Before a match: the seats' clients join and get ready.
    LOBBY = "lobby"
    #: The match is on.
    PLAYING = "playing"
    #: The match is stopped, for a player missing or for the experimenter.
    PAUSED = "paused"
    #: The match counts down to going on again.
    RESUMING = "resuming"
    #: The match was played to its end.
    ENDED = "ended"
    #: The experimenter ended the match.
    STOPPED = "stopped"
    #: The server is shutting down.
    CLOSING = "closing"


_STATUSES = tuple(RoomStatus)


class ProtocolError(ValueError):
    """A message that is not one of this protocol's."""


@dataclass(frozen=True)
class Setup:
    """What a match is: the cars a team, the seats clients drive (bots drive the others), and
    the regulation time.

    :raises ProtocolError: a value out of range.
    """

    team_size: int
    seats: tuple[str, ...]
    match_seconds: float

    def __post_init__(self) -> None:
        if (isinstance(self.team_size, bool) or not isinstance(self.team_size, int)
                or self.team_size not in TEAM_SIZES):
            raise ProtocolError(f"team_size {self.team_size!r}: one of {list(TEAM_SIZES)}")
        names = [car.name for car in match_cars(self.team_size)]
        if (not self.seats or len(set(self.seats)) != len(self.seats)
                or not set(self.seats) <= set(names)):
            raise ProtocolError(f"seats {list(self.seats)}: different cars of a "
                                f"{self.team_size}v{self.team_size} match, from {names}")
        if (isinstance(self.match_seconds, bool)
                or not isinstance(self.match_seconds, (int, float))
                or not math.isfinite(self.match_seconds)
                or round(self.match_seconds * TICKS_PER_SECOND) < 1):
            raise ProtocolError(f"match_seconds {self.match_seconds!r}: a positive number of "
                                f"seconds, at least one tick")
        object.__setattr__(self, "seats", tuple(name for name in names if name in self.seats))

    @property
    def match_ticks(self) -> int:
        """The regulation time, in ticks."""
        return round(self.match_seconds * TICKS_PER_SECOND)

    @property
    def cars(self) -> tuple[Car, ...]:
        """Every car of the match, blue's then orange's."""
        return match_cars(self.team_size)

    @property
    def seat_cars(self) -> tuple[Car, ...]:
        """The cars clients drive."""
        return tuple(car for car in self.cars if car.name in self.seats)

    def car(self, seat: str) -> Car:
        """The car of seat ``seat``.

        :raises ProtocolError: ``seat`` is not one of the match's seats.
        """
        for car in self.seat_cars:
            if car.name == seat:
                return car
        raise ProtocolError(f"seat {seat!r}: one of {list(self.seats)}")

    def record(self) -> dict[str, Any]:
        return {"team_size": self.team_size, "seats": list(self.seats),
                "match_seconds": self.match_seconds}

    @classmethod
    def from_record(cls, record: Any) -> Setup:
        """:raises ProtocolError: ``record`` is not a setup's."""
        fields = _object(record, "a setup", ("team_size", "seats", "match_seconds"))
        seats = fields["seats"]
        if not isinstance(seats, list) or not all(isinstance(seat, str) for seat in seats):
            raise ProtocolError(f"seats {seats!r}: a list of car names")
        return cls(fields["team_size"], tuple(seats), fields["match_seconds"])


@dataclass(frozen=True)
class JoinRequest:
    """A client asking for a seat.

    :ivar rocketsim_gym_version: the client's package version, which must be the server's.
    :ivar collision_meshes: the arena the client draws: each mesh file and its SHA-256.
    :ivar setup: the match the client wants, or None to take the room's.
    :ivar seat: the seat it wants, or None for any free one.
    :ivar host: the client's computer, as the server's console names it.
    """

    rocketsim_gym_version: str
    collision_meshes: Mapping[str, str]
    setup: Setup | None
    seat: str | None
    host: str
    protocol: int = PROTOCOL_VERSION

    def record(self) -> dict[str, Any]:
        return {"type": "join", "protocol": self.protocol,
                "rocketsim_gym_version": self.rocketsim_gym_version,
                "collision_meshes": dict(self.collision_meshes),
                "setup": None if self.setup is None else self.setup.record(),
                "seat": self.seat, "host": self.host}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> JoinRequest:
        protocol = _object(record, "a join request", ("type", "protocol"))["protocol"]
        if protocol != PROTOCOL_VERSION:
            raise ProtocolError(f"a client of protocol {protocol!r}; this server speaks "
                                f"{PROTOCOL_VERSION}: install the same rocketsim-gym on both")
        fields = _object(record, "a join request",
                         ("type", "protocol", "rocketsim_gym_version", "collision_meshes",
                          "setup", "seat", "host"))
        for name in ("rocketsim_gym_version", "host"):
            if not isinstance(fields[name], str):
                raise ProtocolError(f"{name} {fields[name]!r}: text")
        meshes = fields["collision_meshes"]
        if not isinstance(meshes, dict) or not all(isinstance(key, str) and isinstance(value, str)
                                                   for key, value in meshes.items()):
            raise ProtocolError(f"collision_meshes {meshes!r}: each file's SHA-256")
        if fields["seat"] is not None and not isinstance(fields["seat"], str):
            raise ProtocolError(f"seat {fields['seat']!r}: a car's name, or null")
        setup = None if fields["setup"] is None else Setup.from_record(fields["setup"])
        return cls(fields["rocketsim_gym_version"], meshes, setup, fields["seat"],
                   fields["host"], fields["protocol"])


@dataclass(frozen=True)
class Welcome:
    """The server's yes: the client's seat and the match, the token its datagrams carry, and
    the server's UDP port."""

    token: bytes
    udp_port: int
    seat: str
    setup: Setup
    server_host: str

    def record(self) -> dict[str, Any]:
        return {"type": "welcome", "token": self.token.hex(), "udp_port": self.udp_port,
                "seat": self.seat, "setup": self.setup.record(), "server_host": self.server_host}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Welcome:
        fields = _object(record, "a welcome", ("type", "token", "udp_port", "seat", "setup",
                                               "server_host"))
        try:
            return cls(bytes.fromhex(fields["token"]), int(fields["udp_port"]), fields["seat"],
                       Setup.from_record(fields["setup"]), fields["server_host"])
        except (TypeError, ValueError) as error:
            raise ProtocolError(f"a welcome that does not read: {error}") from None


@dataclass(frozen=True)
class Refusal:
    """The server's no, and why. A refusal that is not final (the room busy with another
    match) can be asked again."""

    reason: str
    is_final: bool = True

    def record(self) -> dict[str, Any]:
        return {"type": "refused", "reason": self.reason, "is_final": self.is_final}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Refusal:
        fields = _object(record, "a refusal", ("type", "reason", "is_final"))
        return cls(str(fields["reason"]), bool(fields["is_final"]))


@dataclass(frozen=True)
class ControlsEntry:
    """A client's controls for one step: its sequence number, and the server frame it showed
    when it chose them."""

    sequence: int
    shown_frame: int
    controls: Controls


@dataclass(frozen=True)
class ControlsMessage:
    """A client's datagram: its token, then its newest controls and the ones before, newest
    first."""

    token: bytes
    entries: tuple[ControlsEntry, ...]

    def pack(self) -> bytes:
        parts = [_CONTROLS_HEADER.pack(_CONTROLS_KIND, self.token, len(self.entries))]
        for entry in self.entries:
            controls = entry.controls
            parts.append(_CONTROLS_ENTRY.pack(
                entry.sequence, entry.shown_frame, controls.throttle, controls.steer,
                controls.pitch, controls.yaw, controls.roll,
                controls.jump | controls.boost << 1 | controls.handbrake << 2))
        return b"".join(parts)

    @classmethod
    def unpack(cls, data: bytes) -> ControlsMessage:
        """:raises ProtocolError: ``data`` is not a controls datagram."""
        if len(data) < _CONTROLS_HEADER.size or data[:1] != _CONTROLS_KIND:
            raise ProtocolError("not a controls datagram")
        _, token, count = _CONTROLS_HEADER.unpack_from(data)
        if not 1 <= count <= CONTROLS_REPEATED or len(data) != (
                _CONTROLS_HEADER.size + count * _CONTROLS_ENTRY.size):
            raise ProtocolError("a controls datagram of the wrong length")
        entries = []
        for index in range(count):
            sequence, shown_frame, *axes, buttons = _CONTROLS_ENTRY.unpack_from(
                data, _CONTROLS_HEADER.size + index * _CONTROLS_ENTRY.size)
            if not all(math.isfinite(axis) and -1.0 <= axis <= 1.0 for axis in axes):
                raise ProtocolError(f"controls {axes}: each from -1 to 1")
            entries.append(ControlsEntry(sequence, shown_frame, Controls(
                *axes, bool(buttons & 1), bool(buttons & 2), bool(buttons & 4))))
        return cls(token, tuple(entries))


@dataclass(frozen=True)
class ServerFrame:
    """What the server sends every client every step.

    :ivar number: the frame's number; the server counts every step, played or not.
    :ivar status_ticks: while resuming, the ticks until play goes on.
    :ivar waiting_for: in the lobby, the seats not yet ready; paused, the seats missing.
    :ivar bots: the seats bots drive for the rest of the match.
    :ivar match_id, seed: the match's, as its log has them.
    :ivar input_delays: each car's latest controls' delay, in frames, or None.
    :ivar snapshot: the match's snapshot (:meth:`~rocketsim_gym.snapshot.Snapshot.pack`).
    """

    number: int
    status: RoomStatus
    team_size: int
    snapshot: bytes
    status_ticks: int = 0
    waiting_for: frozenset[Car] = frozenset()
    bots: frozenset[Car] = frozenset()
    match_id: int = 0
    seed: int = 0
    input_delays: tuple[int | None, ...] = field(default=())

    def pack(self) -> bytes:
        cars = match_cars(self.team_size)
        delays = [-1 if delay is None else min(delay, 127) for delay in self.input_delays]
        delays += [-1] * (6 - len(delays))
        return _FRAME_HEADER.pack(
            _FRAME_KIND, self.number, _STATUSES.index(self.status), self.status_ticks,
            self.team_size, _bits(car in self.waiting_for for car in cars),
            _bits(car in self.bots for car in cars), self.match_id, self.seed,
            *delays) + self.snapshot

    @classmethod
    def unpack(cls, data: bytes) -> ServerFrame:
        """:raises ProtocolError: ``data`` is not a server frame."""
        if len(data) < _FRAME_HEADER.size or data[:1] != _FRAME_KIND:
            raise ProtocolError("not a server frame")
        (_, number, status, status_ticks, team_size, waiting_for, bots, match_id, seed,
         *delays) = _FRAME_HEADER.unpack_from(data)
        if status >= len(_STATUSES) or team_size not in TEAM_SIZES:
            raise ProtocolError("a server frame of an unknown status or team size")
        cars = match_cars(team_size)
        return cls(number, _STATUSES[status], team_size, bytes(data[_FRAME_HEADER.size:]),
                   status_ticks, _cars(cars, waiting_for), _cars(cars, bots), match_id, seed,
                   tuple(None if delay < 0 else delay for delay in delays[:len(cars)]))


def encode_message(message: Mapping[str, Any]) -> bytes:
    """``message`` as TCP bytes: its length, then its JSON."""
    data = json.dumps(message).encode()
    if len(data) > MESSAGE_LIMIT:
        raise ProtocolError(f"a message of {len(data)} bytes, over {MESSAGE_LIMIT}")
    return _LENGTH.pack(len(data)) + data


class MessageReader:
    """Collects TCP bytes as they come, and gives back the whole messages among them."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> Iterator[dict[str, Any]]:
        """The messages ``data`` completes, in order.

        :raises ProtocolError: a message too long, or not a JSON object with a type.
        """
        self._buffer += data
        while len(self._buffer) >= _LENGTH.size:
            (length,) = _LENGTH.unpack_from(self._buffer)
            if length > MESSAGE_LIMIT:
                raise ProtocolError(f"a message of {length} bytes, over {MESSAGE_LIMIT}")
            if len(self._buffer) < _LENGTH.size + length:
                return
            data = bytes(self._buffer[_LENGTH.size:_LENGTH.size + length])
            del self._buffer[:_LENGTH.size + length]
            try:
                message = json.loads(data)
            except ValueError:
                raise ProtocolError("a message that is not JSON") from None
            if not isinstance(message, dict) or not isinstance(message.get("type"), str):
                raise ProtocolError("a message without a type")
            yield message


def find_query() -> bytes:
    """The datagram a client broadcasts to find servers."""
    return json.dumps({**_FIND, "protocol": PROTOCOL_VERSION}).encode()


def is_find_query(data: bytes) -> bool:
    """Whether ``data`` is a client's query, of any protocol version."""
    try:
        message = json.loads(data)
    except ValueError:
        return False
    return isinstance(message, dict) and message.get("find") == _FIND["find"]


@dataclass(frozen=True)
class Announcement:
    """A server's answer to a query: where it is, and what its room is playing.

    :ivar address: the server's address, as the answer came from it.
    :ivar setup: its room's match, or None while the room is empty.
    :ivar seats: each seat taken, and the computer in it.
    """

    host: str
    port: int
    protocol: int
    status: RoomStatus | None
    setup: Setup | None
    seats: Mapping[str, str]
    address: str = ""

    def record(self) -> dict[str, Any]:
        return {"server": _FIND["find"], "protocol": self.protocol, "host": self.host,
                "port": self.port, "status": None if self.status is None else self.status.value,
                "setup": None if self.setup is None else self.setup.record(),
                "seats": dict(self.seats)}

    @classmethod
    def from_datagram(cls, data: bytes, address: str) -> Announcement:
        """:raises ProtocolError: ``data`` is not an announcement."""
        try:
            record = json.loads(data)
            if record.get("server") != _FIND["find"]:
                raise ProtocolError("not an announcement")
            status = None if record["status"] is None else RoomStatus(record["status"])
            setup = None if record["setup"] is None else Setup.from_record(record["setup"])
            return cls(record["host"], int(record["port"]), int(record["protocol"]), status,
                       setup, dict(record["seats"]), address)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ProtocolError(f"an announcement that does not read: {error}") from None

    @property
    def join_address(self) -> str:
        """Where to join it: ``address:port``."""
        return f"{self.address}:{self.port}"


def _object(record: Any, what: str, keys: Sequence[str]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ProtocolError(f"{what}: a JSON object")
    missing = sorted(set(keys) - set(record))
    if missing:
        raise ProtocolError(f"{what} without {missing}")
    return record


def _bits(flags) -> int:
    return sum(1 << bit for bit, flag in enumerate(flags) if flag)


def _cars(cars: Sequence[Car], bits: int) -> frozenset[Car]:
    return frozenset(car for bit, car in enumerate(cars) if bits >> bit & 1)
