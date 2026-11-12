"""A player's side of a LAN match: its seat in a server's room, the controls it sends every step,
and the frame it shows, one a step.

Every step, :meth:`LanClient.step` gives a record of what the client shows: its status
(:class:`ClientStatus`), then the server frame it shows, if one has come. Everything a client
shows and logs follows from its records, so a client's play is redrawn from them alone.

Frames are shown in order (:class:`FrameBuffer`): the next one each step, or the last again
while the next has not come. A buffer of a frame or more absorbs the network's jitter, so a late
datagram never shows as a stutter; only the drift between the server's clock and the client's
display skips or repeats a frame, now and then.
"""

from __future__ import annotations

import importlib.metadata
import socket
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from enum import IntEnum

from ..controls import Controls
from .connection import JoinRefused, MatchConnection, server_address
from .protocol import (CONTROLS_REPEATED, ControlsEntry, JoinRequest, RoomStatus, ServerFrame,
                       Setup)

#: Without frames this long, the client shows that the connection is lost...
LOST_SECONDS = 2.0
#: ...and this long, it gives up: the server is gone.
GIVE_UP_SECONDS = 30.0
#: A client the room is too busy for asks again this often...
RETRY_SECONDS = 2.0
#: ...for this long, then gives up.
WAITING_GIVE_UP_SECONDS = 120.0
#: The frames a buffer keeps ahead of the one it shows, at first and at most.
FIRST_TARGET, MOST_TARGET = 1, 4
#: Frames ahead beyond the target for this many steps in a row (the server's clock running
#: faster than the client's display) skip one.
SKIP_AFTER_STEPS = 60
#: More frames ahead than this at once (after a stall) jump to the target straight away.
BURST = 6
#: Repeats this many times within this many steps raise the target by one frame.
REPEATS_TO_RAISE, REPEAT_WINDOW_STEPS = 3, 600

_FINISHED = (RoomStatus.ENDED, RoomStatus.STOPPED)


class ClientStatus(IntEnum):
    """Where a client is, the first byte of each of its records."""

    #: Joined, with no frame yet.
    JOINING = 0
    #: The room plays another match; the client asks again from time to time.
    WAITING = 1
    #: Frames come.
    LIVE = 2
    #: No frame for :data:`LOST_SECONDS`.
    LOST = 3
    #: The server closed, or sent nothing for :data:`GIVE_UP_SECONDS`, or the room stayed busy
    #: for :data:`WAITING_GIVE_UP_SECONDS`.
    GONE = 4


class FrameBuffer:
    """The frames a client has, and the one it shows each step.

    It shows the next frame in order, or the last again while that has not come (a repeat). A
    frame lost on the way is skipped once a later one has come. When more frames than its target
    wait ahead for :data:`SKIP_AFTER_STEPS` steps in a row, it skips one; after a burst of more
    than :data:`BURST`, it jumps to the target at once. Repeats that recur raise the target.

    A new buffer ignores the frames of a match that has finished, which a server sends until its
    clients are ready for the next.
    """

    def __init__(self) -> None:
        self._frames: dict[int, ServerFrame] = {}
        self._shown: ServerFrame | None = None
        self._is_accepting = False
        self.target = FIRST_TARGET
        self._steps = 0
        self._steps_ahead = 0
        self._repeat_steps: deque[int] = deque()

    @property
    def shown_number(self) -> int:
        """The number of the frame shown last, or 0."""
        return 0 if self._shown is None else self._shown.number

    def add(self, frames: Iterable[ServerFrame]) -> None:
        for frame in frames:
            if not self._is_accepting:
                if frame.status in _FINISHED:
                    continue
                self._is_accepting = True
            if frame.number > self.shown_number:
                self._frames[frame.number] = frame

    def next(self) -> ServerFrame | None:
        """The frame to show this step: None until one has come."""
        self._steps += 1
        if not self._frames:
            if self._shown is not None:
                self._note_repeat()
            return self._shown
        if self._shown is None:
            number = max(min(self._frames), max(self._frames) - self.target)
        else:
            number = (self._shown.number + 1 if self._shown.number + 1 in self._frames
                      else min(self._frames))
            number = self._skip(number)
        self._shown = self._frames[number]
        self._frames = {key: frame for key, frame in self._frames.items() if key > number}
        return self._shown

    def _skip(self, number: int) -> int:
        """``number``, or a later frame when the rules above skip."""
        ahead = sum(1 for key in self._frames if key > number)
        if ahead > BURST:
            self._steps_ahead = 0
            return max(self._frames) - self.target
        if ahead <= self.target:
            self._steps_ahead = 0
            return number
        self._steps_ahead += 1
        if self._steps_ahead >= SKIP_AFTER_STEPS and number + 1 in self._frames:
            self._steps_ahead = 0
            return number + 1
        return number

    def _note_repeat(self) -> None:
        self._repeat_steps.append(self._steps)
        while self._repeat_steps[0] <= self._steps - REPEAT_WINDOW_STEPS:
            self._repeat_steps.popleft()
        if len(self._repeat_steps) >= REPEATS_TO_RAISE and self.target < MOST_TARGET:
            self.target += 1
            self._repeat_steps.clear()


class LanClient:
    """A player's seat in a server's room, and what the player sees of the match there.

    :param server: ``"auto"`` to find the one server on the network, or ``"HOST"`` or
        ``"HOST:PORT"``.
    :param setup: the match the player asks for.
    :param seat: the seat it asks for.
    :param collision_meshes: the arena it draws: each mesh file and its SHA-256.
    :param clock: the clock the timeouts read.
    :ivar address: the server's address, once joined.
    :ivar found: how the server was found.
    :ivar seeds: each match's id seen so far, and its seed.
    """

    def __init__(self, server: str, setup: Setup, seat: str,
                 collision_meshes: Mapping[str, str],
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self._server = server
        self._request = JoinRequest(importlib.metadata.version("rocketsim-gym"),
                                    dict(collision_meshes), setup, seat, socket.gethostname())
        self._clock = clock
        self._connection: MatchConnection | None = None
        self._buffer = FrameBuffer()
        self._sent: deque[ControlsEntry] = deque(maxlen=CONTROLS_REPEATED)
        self._sequence = 0
        self._is_ready = False
        self._heard_at = self._waiting_since = self._retry_at = 0.0
        self.status = ClientStatus.JOINING
        self.address: str | None = None
        self.found = ""
        self.seeds: dict[int, int] = {}

    def reset(self) -> None:
        """Start an episode: join the room, or tell it the client is not ready for now, and
        forget the frames so far.

        :raises ConnectionError: the server cannot be reached, with what to do.
        :raises JoinRefused: the server refused for good, with why.
        """
        self._buffer = FrameBuffer()
        self._sent.clear()
        self._is_ready = False
        if self._connection is not None and not self._connection.is_server_gone:
            self._connection.not_ready()
            self.status, self._heard_at = ClientStatus.JOINING, self._clock()
            return
        self._waiting_since = self._clock()
        self._join()

    def step(self, controls: Controls) -> bytes:
        """Send this step's controls, take the frames that came, and give this step's record:
        the client's status, then the frame to show, if one has come."""
        now = self._clock()
        if self.status is ClientStatus.WAITING and now >= self._retry_at:
            self._join()
            if (self.status is ClientStatus.WAITING
                    and now - self._waiting_since > WAITING_GIVE_UP_SECONDS):
                self.status = ClientStatus.GONE
        if self._connection is not None and self.status is not ClientStatus.GONE:
            self._exchange(controls, now)
        frame = self._buffer.next()
        return bytes([self.status]) + (b"" if frame is None else frame.pack())

    def close(self, reason: str = "quit") -> None:
        """Leave the room, saying why."""
        if self._connection is not None:
            self._connection.leave(reason)
            self._connection = None

    def _join(self) -> None:
        host, port, found = server_address(self._server)
        try:
            self._connection = MatchConnection(host, port, self._request)
        except JoinRefused as refused:
            if refused.refusal.is_final:
                raise
            self.status, self._retry_at = ClientStatus.WAITING, self._clock() + RETRY_SECONDS
            return
        self.address, self.found = f"{host}:{port}", found
        self.status, self._heard_at = ClientStatus.JOINING, self._clock()

    def _exchange(self, controls: Controls, now: float) -> None:
        self._sequence += 1
        self._sent.appendleft(ControlsEntry(self._sequence, self._buffer.shown_number, controls))
        self._connection.send(self._sent)
        if not self._is_ready:
            self._connection.ready()
            self._is_ready = True
        frames = self._connection.receive()
        self._buffer.add(frames)
        for frame in frames:
            if frame.match_id:
                self.seeds[frame.match_id] = frame.seed
        if self._connection.is_server_gone:
            self.status = ClientStatus.GONE
        elif frames:
            self.status, self._heard_at = ClientStatus.LIVE, now
        elif now - self._heard_at > GIVE_UP_SECONDS:
            self.status = ClientStatus.GONE
        elif now - self._heard_at > LOST_SECONDS and self.status is ClientStatus.LIVE:
            self.status = ClientStatus.LOST
