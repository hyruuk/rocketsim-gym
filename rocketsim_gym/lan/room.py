"""A match server's room: its seats, the clients in them, and the match they play.

The rules:

- The room takes its setup (:class:`~.protocol.Setup`) from the first client to join. While a
  client is in it or a match is under way, it refuses a client with another setup as busy, and
  that client may ask again; it forgets its setup once it is empty with no match under way.
- A match starts when every seat's client is ready, which a client is from its first step after
  a reset. A client that joins and never steps (a check that the server is there) starts
  nothing.
- While a match runs, a seat whose client goes silent for :data:`SILENCE_SECONDS`, closes its
  connection, leaves or resets pauses it for everyone. In the lobby and after a match, only a
  closed connection or a leave frees a seat.
- The experimenter pauses the match, resumes it once every seat's client is back and playing,
  lets a bot drive a seat to the match's end, or ends it. A resume counts down
  :data:`RESUME_SECONDS`, except after a pause that began in a kickoff's countdown.
- A seat's controls queue up in order and one is applied each step (:class:`ControlsQueue`), so a
  datagram lost or late loses no press.
- Each match gets its own seed: the server's seed plus the match's number, or one drawn at
  random.

The room keeps no socket and reads no clock: the server hands it each message with the time,
plays one step on :meth:`Room.step`, and sends :meth:`Room.frame` to :meth:`Room.recipients`.
"""

from __future__ import annotations

import math
import secrets
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..controls import NO_CONTROLS, Controls
from ..match import TICKS_PER_SECOND, Car, Phase
from ..simulation import MatchSimulation
from ..snapshot import Snapshot
from .match_log import MatchLog, controls_record, score_record, snapshot_digest
from .protocol import (PROTOCOL_VERSION, TICKS_PER_STEP, Announcement, ControlsEntry,
                       ControlsMessage, JoinRequest, ProtocolError, Refusal, RoomStatus,
                       ServerFrame, Setup, Welcome)

#: A seat's client unheard this long while a match runs is gone, and the match pauses.
SILENCE_SECONDS = 2.0
#: A resume needs every seat's client heard within this long.
RECENT_SECONDS = 0.5
#: How long a resumed match counts down before play goes on.
RESUME_SECONDS = 3
#: The entries a seat's queue holds at most for long; more is a client running ahead.
QUEUE_TARGET = 2
#: Steps in a row above the target after which the queue drops its oldest entry.
QUEUE_DEEP_STEPS = 30
#: Above this many entries at once, the queue drops back to its target straight away.
QUEUE_BURST = 6

_MATCH_ON = (RoomStatus.PLAYING, RoomStatus.PAUSED, RoomStatus.RESUMING)
_MATCH_RUNNING = (RoomStatus.PLAYING, RoomStatus.RESUMING)
_MATCH_OVER = (RoomStatus.ENDED, RoomStatus.STOPPED)


@dataclass(frozen=True)
class QueueTake:
    """A step's controls from a queue: the entry they came from (None: the last ones, held),
    the entries skipped as lost before it, and the ones dropped after it."""

    controls: Controls
    entry: ControlsEntry | None
    lost: int = 0
    dropped: int = 0


class ControlsQueue:
    """One seat's controls, in the order its client chose them; one is taken each step.

    The queue starts at the newest entry of the first datagram it gets. An entry whose turn
    comes before it does is waited for, the last controls held meanwhile, while nothing later has
    come, and skipped as lost once something has. When more than :data:`QUEUE_TARGET` entries
    wait for :data:`QUEUE_DEEP_STEPS` steps in a row (a client's display running a little faster
    than the server), the oldest is dropped; after a burst of more than :data:`QUEUE_BURST`, the
    queue drops back to its target at once.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        """Forget every entry, and hold no controls."""
        self._entries: dict[int, ControlsEntry] = {}
        self._next: int | None = None
        self._last = NO_CONTROLS
        self._deep_steps = 0

    def add(self, entries: Iterable[ControlsEntry]) -> None:
        entries = list(entries)
        if self._next is None and entries:
            self._next = max(entry.sequence for entry in entries)
        for entry in entries:
            if entry.sequence >= self._next:
                self._entries.setdefault(entry.sequence, entry)

    def take(self) -> QueueTake:
        if not self._entries:
            return QueueTake(self._last, None)
        lost = 0
        if self._next not in self._entries:
            lost = min(self._entries) - self._next
            self._next = min(self._entries)
        entry = self._entries.pop(self._next)
        self._next += 1
        self._last = entry.controls
        return QueueTake(entry.controls, entry, lost, self._shorten())

    def _shorten(self) -> int:
        """Drop what the rules above drop; return how many."""
        depth = len(self._entries)
        if depth <= QUEUE_TARGET:
            self._deep_steps = 0
            return 0
        self._deep_steps += 1
        if depth <= QUEUE_BURST and self._deep_steps < QUEUE_DEEP_STEPS:
            return 0
        dropped = depth - QUEUE_TARGET if depth > QUEUE_BURST else 1
        for sequence in sorted(self._entries)[:dropped]:
            del self._entries[sequence]
        self._next = min(self._entries)
        self._deep_steps = 0
        return dropped


@dataclass
class Seat:
    """A seat of the room, and the client in it, if any.

    :ivar token: the client's, which its datagrams carry; None while the seat is free.
    :ivar is_ready: whether the client has stepped since its last reset.
    :ivar is_silent: whether the client went unheard for :data:`SILENCE_SECONDS` while a match
        ran, until it is heard again.
    :ivar is_bot: whether a bot drives the seat to the match's end.
    :ivar heard_at: when the client's latest datagram came.
    :ivar address: where its latest datagram came from, which its frames go to.
    :ivar input_delay: its latest controls' delay, in frames.
    """

    car: Car
    token: bytes | None = None
    host: str = ""
    is_ready: bool = False
    is_silent: bool = False
    is_bot: bool = False
    heard_at: float = -math.inf
    address: tuple[str, int] | None = None
    queue: ControlsQueue = field(default_factory=ControlsQueue)
    input_delay: int | None = None


class Room:
    """A match server's room, by the rules above.

    :param rocketsim_gym_version: the server's package version, which clients must have.
    :param collision_meshes: the server's arena (each mesh file and its SHA-256), which clients
        must draw.
    :param server_host: the server's computer, as clients name it.
    :param udp_port: the port clients send their controls to.
    :param log_folder: where match logs go.
    :param seed: the first match's seed, the next ones counting up from it; None draws each one.
    :param report: where the room's news goes.
    :param clock: the server's clock, for the logs' flushes.
    """

    def __init__(self, rocketsim_gym_version: str, collision_meshes: Mapping[str, str],
                 server_host: str, udp_port: int, log_folder: Path,
                 seed: int | None, report: Callable[[str], None],
                 clock: Callable[[], float]) -> None:
        self._version = rocketsim_gym_version
        self._collision_meshes = dict(collision_meshes)
        self._server_host = server_host
        self._udp_port = udp_port
        self._log_folder = log_folder
        self._first_seed = seed
        self._report = report
        self._clock = clock
        self.status = RoomStatus.LOBBY
        self.setup: Setup | None = None
        self.seats: dict[Car, Seat] = {}
        self.frame_number = 0
        self.matches_started = 0
        self.match_id = 0
        self.seed = 0
        #: Tokens a newcomer took the seat of; the server closes their connections.
        self.voided_tokens: list[bytes] = []
        self._setup_host = ""
        self._simulation: MatchSimulation | None = None
        self._snapshot_bytes = b""
        self._snapshot: Snapshot | None = None
        self._log: MatchLog | None = None
        self._resume_ticks = 0
        self._paused_in_countdown = False

    # Clients

    def join(self, request: JoinRequest, now: float) -> Welcome | Refusal:
        """A client's request for a seat: a welcome to it, or a refusal."""
        if request.rocketsim_gym_version != self._version:
            return Refusal(f"rocketsim-gym {request.rocketsim_gym_version} on {request.host}, "
                           f"{self._version} on the server: install the same version on both")
        if dict(request.collision_meshes) != self._collision_meshes:
            return Refusal(f"{request.host} draws another arena than the server plays: install "
                           f"the same collision meshes on both")
        if self.setup is None:
            if request.setup is None:
                return Refusal("the room has no match yet: the first client to join says what it "
                               "is (team size, seats, length)")
            self._take_setup(request.setup, request.host)
        elif request.setup is not None and request.setup != self.setup:
            return Refusal(f"the room plays {_describe(self.setup)}, set by {self._setup_host}; "
                           f"{request.host} asks for {_describe(request.setup)}. It can join "
                           f"once the room is empty", is_final=False)
        seat = self._seat_for(request.seat)
        if isinstance(seat, Refusal):
            return seat
        if seat.token is not None:
            self.voided_tokens.append(seat.token)
            self._report(f"{seat.car.name}: {request.host} takes the seat of {seat.host}, "
                         f"unheard since {now - seat.heard_at:.1f} s")
        rejoins = self.status is RoomStatus.PAUSED
        seat.token, seat.host, seat.address = secrets.token_bytes(16), request.host, None
        seat.is_ready, seat.is_silent, seat.heard_at = False, False, now
        seat.queue.clear()
        self._report(f"{seat.car.name}: {request.host} joins"
                     + (" the paused match" if rejoins else ""))
        if rejoins:
            self._event(now, "rejoin", seat=seat.car.name, host=request.host)
        return Welcome(seat.token, self._udp_port, seat.car.name, self.setup, self._server_host)

    def ready(self, token: bytes, now: float) -> None:
        """A client stepping since its reset: ready for a match, or back in a paused one."""
        seat = self._seat_of(token)
        if seat is None or seat.is_ready:
            return
        if self.status in _MATCH_OVER:
            self._open_lobby()
        seat.is_ready = True
        if self.status is RoomStatus.PAUSED:
            self._report(f"{seat.car.name}: {seat.host} is back and playing")
            self._event(now, "ready", seat=seat.car.name)

    def not_ready(self, token: bytes, now: float) -> None:
        """A client reset: not ready for a match; in one, it is missing from it."""
        seat = self._seat_of(token)
        if seat is None:
            return
        seat.is_ready = False
        if self.status in _MATCH_RUNNING and not seat.is_bot:
            self._pause(now, f"{seat.car.name} reset its game")

    def leave(self, token: bytes, reason: str, now: float) -> None:
        """A client gone: it left, or its connection closed."""
        seat = self._seat_of(token)
        if seat is None:
            return
        self._report(f"{seat.car.name}: {seat.host} leaves ({reason})")
        seat.token, seat.address, seat.is_ready, seat.is_silent = None, None, False, False
        seat.queue.clear()
        if self.status in _MATCH_RUNNING and not seat.is_bot:
            self._pause(now, f"{seat.car.name} left ({reason})")
        elif self.status in _MATCH_ON:
            self._event(now, "leave", seat=seat.car.name, reason=reason)
        if self.status not in _MATCH_ON and not any(seat.token for seat in self.seats.values()):
            self._forget_setup()

    def controls(self, message: ControlsMessage, address: tuple[str, int], now: float) -> bool:
        """A client's controls; False when no seat holds its token."""
        seat = self._seat_of(message.token)
        if seat is None:
            return False
        if seat.is_silent:
            self._report(f"{seat.car.name}: {seat.host} heard again after "
                         f"{now - seat.heard_at:.1f} s")
            self._event(now, "heard", seat=seat.car.name, after=now - seat.heard_at)
            seat.is_silent = False
        seat.address, seat.heard_at = address, now
        if not seat.is_bot:
            seat.queue.add(message.entries)
        return True

    # The experimenter

    def pause(self, now: float) -> str:
        if self.status not in _MATCH_RUNNING:
            return "no match is running"
        self._pause(now, "the experimenter paused it")
        return "paused"

    def resume(self, now: float) -> str:
        if self.status is not RoomStatus.PAUSED:
            return "the match is not paused"
        missing = [seat for seat in self.seats.values() if not seat.is_bot and (
            seat.token is None or seat.is_silent or not seat.is_ready
            or now - seat.heard_at > RECENT_SECONDS)]
        if missing:
            return (f"waiting for {', '.join(seat.car.name for seat in missing)}: each must be "
                    f"back and playing, or driven by a bot (bot SEAT)")
        if self._paused_in_countdown:
            self.status = RoomStatus.PLAYING
        else:
            self.status, self._resume_ticks = RoomStatus.RESUMING, RESUME_SECONDS * TICKS_PER_SECOND
        self._event(now, "resume")
        return "resuming" if self.status is RoomStatus.RESUMING else "playing"

    def hand_to_bot(self, seat_name: str, now: float) -> str:
        if self.status not in _MATCH_ON:
            return "no match is on"
        try:
            car = self.setup.car(seat_name)
        except ProtocolError as error:
            return str(error)
        seat = self.seats[car]
        if seat.is_bot:
            return f"a bot drives {seat_name} already"
        self._simulation.hand_to_bot(car)
        seat.is_bot = True
        seat.queue.clear()
        self._event(now, "bot", seat=seat_name)
        return f"a bot drives {seat_name} to the end of this match"

    def end(self, now: float) -> str:
        if self.status not in _MATCH_ON:
            return "no match is on"
        self._finish(now, RoomStatus.STOPPED, "ended by the experimenter")
        return "ended"

    def close(self, now: float) -> None:
        """The server shuts down: a match under way ends, and the frames say so."""
        if self._log is not None:
            self._finish(now, RoomStatus.STOPPED, "ended as the server closed")
        self.status = RoomStatus.CLOSING

    # Steps

    def step(self, now: float) -> None:
        """One step of the room: a silent client noticed, a match started or played on, or a
        resume counted down."""
        self.frame_number += 1
        if self.setup is None or self.status is RoomStatus.CLOSING:
            return
        if self.status in _MATCH_RUNNING:
            self._notice_silence(now)
        if self.status is RoomStatus.LOBBY and all(
                seat.token is not None and seat.is_ready for seat in self.seats.values()):
            self._start_match(now)
        if self.status is RoomStatus.PLAYING:
            self._play(now)
            return
        for seat in self.seats.values():
            seat.queue.clear()
        if self.status is RoomStatus.RESUMING:
            self._resume_ticks -= TICKS_PER_STEP
            if self._resume_ticks <= 0:
                self.status = RoomStatus.PLAYING

    def frame(self) -> ServerFrame:
        """This step's frame, for every client."""
        team_size = self.setup.team_size if self.setup else 1
        if self.status is RoomStatus.LOBBY:
            waiting_for = {car for car, seat in self.seats.items()
                           if seat.token is None or not seat.is_ready}
        elif self.status is RoomStatus.PAUSED:
            waiting_for = {car for car, seat in self.seats.items() if not seat.is_bot and (
                seat.token is None or seat.is_silent or not seat.is_ready)}
        else:
            waiting_for = set()
        cars = self.setup.cars if self.setup else ()
        return ServerFrame(
            self.frame_number, self.status, team_size, self._snapshot_bytes,
            self._resume_ticks if self.status is RoomStatus.RESUMING else 0,
            frozenset(waiting_for),
            frozenset(car for car, seat in self.seats.items() if seat.is_bot),
            self.match_id, self.seed,
            tuple(self.seats[car].input_delay if car in self.seats else None for car in cars))

    def recipients(self) -> list[tuple[str, int]]:
        """Where this step's frame goes: every seated client heard from."""
        return [seat.address for seat in self.seats.values()
                if seat.token is not None and seat.address is not None]

    def announcement(self, host: str, port: int) -> Announcement:
        return Announcement(host, port, PROTOCOL_VERSION,
                            None if self.setup is None else self.status, self.setup,
                            {seat.car.name: seat.host for seat in self.seats.values()
                             if seat.token is not None})

    def describe(self, now: float) -> list[str]:
        """The room, a line for it and one for each seat, for the console."""
        if self.setup is None:
            return ["room empty: the first client to join sets the match"]
        lines = [f"room {self.status.value}: {_describe(self.setup)}"
                 + (f"; match {self.match_id:08x}, seed {self.seed}" if self.match_id else "")]
        for seat in self.seats.values():
            if seat.is_bot:
                state = "a bot drives it"
            elif seat.token is None:
                state = "free"
            else:
                state = (f"{seat.host}, " + ("silent" if seat.is_silent else
                                              "ready" if seat.is_ready else "not ready")
                         + (f", heard {now - seat.heard_at:.1f} s ago"
                            if math.isfinite(seat.heard_at) else "")
                         + ("" if seat.input_delay is None
                            else f", input delay {seat.input_delay} frames"))
            lines.append(f"  {seat.car.name}: {state}")
        return lines

    # The rules' workings

    def _take_setup(self, setup: Setup, host: str) -> None:
        self.setup, self._setup_host = setup, host
        self.seats = {car: Seat(car) for car in setup.seat_cars}
        self._report(f"room set by {host}: {_describe(setup)}")
        self._open_lobby()

    def _forget_setup(self) -> None:
        self._report("room empty")
        self.setup, self.seats, self._simulation = None, {}, None
        self._snapshot, self._snapshot_bytes = None, b""
        self.status, self.match_id = RoomStatus.LOBBY, 0

    def _open_lobby(self) -> None:
        """A new match's arena, set for its kickoff, while its clients get ready."""
        self.seed = (secrets.randbits(32) if self._first_seed is None
                     else (self._first_seed + self.matches_started) % 2 ** 32)
        self._simulation = MatchSimulation(self.setup.team_size, self.setup.match_ticks,
                                           TICKS_PER_STEP, self.setup.seat_cars,
                                           np.random.default_rng(self.seed))
        self._set_snapshot(self._simulation.snapshot())
        self.status, self.match_id = RoomStatus.LOBBY, 0
        for seat in self.seats.values():
            seat.is_ready = seat.is_bot = False
            seat.input_delay = None

    def _start_match(self, now: float) -> None:
        self.matches_started += 1
        self.match_id = secrets.randbits(32) or 1
        self._log = MatchLog(self._log_folder, self.match_id, self.setup, self.seed,
                             {seat.car.name: seat.host for seat in self.seats.values()},
                             self._collision_meshes, self._clock)
        for seat in self.seats.values():
            seat.queue.clear()
            seat.heard_at, seat.is_silent = now, False
        self.status = RoomStatus.PLAYING
        self._report(f"match {self.match_id:08x} begins: {_describe(self.setup)}, seed "
                     f"{self.seed}; its log: {self._log.path}")

    def _play(self, now: float) -> None:
        controls, sequences, held, lost, dropped = {}, {}, [], {}, {}
        for car, seat in self.seats.items():
            if seat.is_bot:
                continue
            take = seat.queue.take()
            controls[car] = take.controls
            if take.entry is None:
                held.append(car.name)
            else:
                sequences[car.name] = take.entry.sequence
                seat.input_delay = self.frame_number - take.entry.shown_frame
            if take.lost:
                lost[car.name] = take.lost
            if take.dropped:
                dropped[car.name] = take.dropped
        self._simulation.step(controls)
        self._set_snapshot(self._simulation.snapshot())
        self._log.write({
            "type": "step", "step": self._snapshot.step, "frame": self.frame_number,
            "controls": {car.name: controls_record(value) for car, value in controls.items()},
            "sequences": sequences, "held": held, "lost": lost, "dropped": dropped,
            "bots": [car.name for car, seat in self.seats.items() if seat.is_bot],
            "input_delays": {car.name: seat.input_delay for car, seat in self.seats.items()
                             if seat.input_delay is not None},
            "digest": snapshot_digest(self._snapshot)})
        if self._simulation.has_ended:
            self._finish(now, RoomStatus.ENDED, "played to its end")

    def _pause(self, now: float, reason: str) -> None:
        if self.status is not RoomStatus.PAUSED:
            self._paused_in_countdown = (self.status is RoomStatus.PLAYING
                                         and self._snapshot.match.phase is Phase.COUNTDOWN)
            self.status = RoomStatus.PAUSED
        self._report(f"match paused: {reason}. Commands: resume, bot SEAT, end")
        self._event(now, "pause", reason=reason)

    def _notice_silence(self, now: float) -> None:
        for seat in self.seats.values():
            if (seat.token is not None and not seat.is_bot and not seat.is_silent
                    and now - seat.heard_at > SILENCE_SECONDS):
                seat.is_silent = True
                self._pause(now, f"{seat.car.name} ({seat.host}) unheard for "
                                 f"{SILENCE_SECONDS:g} s")

    def _finish(self, now: float, status: RoomStatus, how: str) -> None:
        score = score_record(self._snapshot.match.score)
        self._log.write({"type": "end", "frame": self.frame_number, "step": self._snapshot.step,
                         "how": how, "score": score})
        self._log.close()
        self._log = None
        self.status = status
        for seat in self.seats.values():
            seat.is_ready = False
        self._report(f"match {self.match_id:08x} {how}: blue {score['blue']}, orange "
                     f"{score['orange']}")

    def _event(self, now: float, event: str, **details) -> None:
        if self._log is not None:
            self._log.write({"type": "event", "frame": self.frame_number,
                             "step": self._snapshot.step, "event": event, **details})

    def _seat_for(self, name: str | None) -> Seat | Refusal:
        """The seat a joining client asks for, or the first free one, or why not."""
        if name is None:
            free = [seat for seat in self.seats.values()
                    if seat.token is None and not seat.is_bot]
            return free[0] if free else Refusal("every seat is taken", is_final=False)
        try:
            seat = self.seats[self.setup.car(name)]
        except ProtocolError as error:
            return Refusal(str(error))
        if seat.is_bot:
            return Refusal(f"a bot drives {name} until this match ends", is_final=False)
        if seat.token is not None and not seat.is_silent:
            return Refusal(f"{name} is taken by {seat.host}")
        return seat

    def _seat_of(self, token: bytes) -> Seat | None:
        for seat in self.seats.values():
            if seat.token is not None and seat.token == token:
                return seat
        return None

    def _set_snapshot(self, snapshot: Snapshot) -> None:
        self._snapshot, self._snapshot_bytes = snapshot, snapshot.pack()


def _describe(setup: Setup) -> str:
    return (f"{setup.team_size}v{setup.team_size}, clients in {', '.join(setup.seats)}, "
            f"{setup.match_seconds:g} s")
