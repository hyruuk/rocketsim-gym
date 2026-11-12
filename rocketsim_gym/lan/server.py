"""``rocketsim-server``: a match server for players on a local network.

    rocketsim-server [--port 47800] [--log-dir matches] [--seed N]

It answers queries on the discovery port, takes clients' joins over TCP and their controls over
UDP, plays its room's match (:mod:`.room`) in real time at 60 steps a second, sends every client
each step's frame, and writes each match's log. Its console prints what happens and takes the
experimenter's commands:

==========  ================================================
status      the room and each seat
pause       pause the match
resume      go on, once every seat's client is back
bot SEAT    a bot drives SEAT to the end of the match
end         end the match now
quit        end the match, tell the clients, and stop
==========  ================================================

Its console takes the keyboard, so it belongs on a computer no participant uses. On a Mac, run
it under ``caffeinate -i``, so that the computer does not doze.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import queue
import socket
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..match import TICKS_PER_SECOND
from ..pacing import GIVE_UP_SECONDS, wait_for
from ..simulation import load_homemade_arena
from .protocol import (DEFAULT_PORT, DISCOVERY_PORT, TICKS_PER_STEP, ControlsMessage,
                       JoinRequest, MessageReader, ProtocolError, Refusal, encode_message,
                       is_find_query)
from .room import Room

#: A step's length: the server plays this often.
STEP_SECONDS = TICKS_PER_STEP / TICKS_PER_SECOND
#: How long the server keeps telling its clients it is closing before it stops.
CLOSING_SECONDS = 0.5
_RECEIVE_BYTES = 1 << 16
_COMMANDS = "status, pause, resume, bot SEAT, end, quit"


@dataclass
class _Connection:
    """A client's TCP connection: its messages as they come, and its seat's token once it has
    one."""

    sock: socket.socket
    host: str
    reader: MessageReader = field(default_factory=MessageReader)
    token: bytes | None = None


class MatchServer:
    """A match server: its sockets, its room, and the experimenter's commands.

    :param port: its TCP and UDP port; 0 for any free one.
    :param log_folder: where match logs go.
    :param seed: the first match's seed, the next ones counting up from it; None draws each.
    :param discovery_port: where it answers clients' queries; None answers none.
    :param report: where its news goes.
    :param clock: its clock.
    :raises OSError: a port is taken; on the discovery port, by another server on this computer.
    """

    def __init__(self, port: int = DEFAULT_PORT, log_folder: Path = Path("matches"),
                 seed: int | None = None, discovery_port: int | None = DISCOVERY_PORT,
                 report: Callable[[str], None] = print,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self._report = report
        self._clock = clock
        collision_meshes = load_homemade_arena()
        self._tcp, self._udp = _bind_tcp_and_udp(port)
        self.port = self._tcp.getsockname()[1]
        self._discovery: socket.socket | None = None
        #: Where it answers clients' queries, or None.
        self.discovery_port = discovery_port
        if discovery_port is not None:
            self._discovery = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._discovery.bind(("", discovery_port))
            self._discovery.setblocking(False)
            self.discovery_port = self._discovery.getsockname()[1]
        self.host = socket.gethostname()
        self.room = Room(importlib.metadata.version("rocketsim-gym"), collision_meshes,
                         self.host, self.port, log_folder, seed, report, clock)
        self._connections: list[_Connection] = []
        self._commands: queue.Queue[str] = queue.Queue()
        #: False once the quit command came.
        self.is_running = True
        #: Datagrams that were not this protocol's, or carried a token no seat holds.
        self.ignored_datagrams = 0

    def submit(self, command: str) -> None:
        """A command for the server to carry out at its next step; safe from any thread."""
        self._commands.put(command)

    def tick(self) -> None:
        """One step: take what came, carry out the commands, step the room, send its frame."""
        now = self._clock()
        self._accept()
        self._read_connections(now)
        self._read_controls(now)
        self._answer_queries()
        while not self._commands.empty():
            reply = self.command(self._commands.get(), now)
            if reply:
                self._report(reply)
        self.room.step(now)
        self._send_frame()
        self._close_voided_connections()

    def command(self, text: str, now: float) -> str:
        """Carry out one of the experimenter's commands; return the reply to print."""
        words = text.split()
        if not words:
            return ""
        name, arguments = words[0].lower(), words[1:]
        if name == "status":
            return "\n".join(self.room.describe(now))
        if name in ("pause", "resume", "end") and not arguments:
            return getattr(self.room, name)(now)
        if name == "bot" and len(arguments) == 1:
            return self.room.hand_to_bot(arguments[0], now)
        if name == "quit" and not arguments:
            self.is_running = False
            return "closing"
        return f"{text.strip()!r}: one of {_COMMANDS}"

    def run(self) -> None:
        """Play in real time until the quit command; then tell the clients, and stop."""
        gc.freeze()
        moment = self._clock()
        while self.is_running:
            self.tick()
            moment += STEP_SECONDS
            late = self._clock() - moment
            if late > GIVE_UP_SECONDS:
                self._report(f"the server fell {late:.2f} s behind; it goes on from now")
                moment = self._clock()
            else:
                wait_for(moment)
        self.shut_down()

    def shut_down(self) -> None:
        """End the match under way, tell the clients the server is closing, and close."""
        self.room.close(self._clock())
        for connection in self._connections:
            self._send(connection, {"type": "closing"})
        end = self._clock() + CLOSING_SECONDS
        while self._clock() < end:
            self.tick()
            wait_for(self._clock() + STEP_SECONDS)
        self.close()

    def close(self) -> None:
        """Close every socket."""
        for connection in self._connections:
            connection.sock.close()
        self._connections.clear()
        for sock in (self._tcp, self._udp, self._discovery):
            if sock is not None:
                sock.close()

    def _accept(self) -> None:
        while True:
            try:
                sock, (host, _) = self._tcp.accept()
            except BlockingIOError:
                return
            sock.setblocking(False)
            self._connections.append(_Connection(sock, host))

    def _read_connections(self, now: float) -> None:
        for connection in list(self._connections):
            try:
                data = connection.sock.recv(_RECEIVE_BYTES)
            except BlockingIOError:
                continue
            except OSError:
                data = b""
            if not data:
                self._drop(connection, "its connection closed", now)
                continue
            try:
                for message in connection.reader.feed(data):
                    self._handle(connection, message, now)
            except ProtocolError as error:
                self._report(f"{connection.host}: {error}; its connection is closed")
                self._drop(connection, "it sent what the server cannot read", now)

    def _handle(self, connection: _Connection, message: dict[str, Any], now: float) -> None:
        kind = message["type"]
        if kind == "join" and connection.token is None:
            try:
                answer = self.room.join(JoinRequest.from_record(message), now)
            except ProtocolError as error:
                answer = Refusal(str(error))
            self._send(connection, answer.record())
            if isinstance(answer, Refusal):
                self._report(f"{message.get('host', connection.host)} refused: {answer.reason}")
                self._drop(connection, "refused", now)
            else:
                connection.token = answer.token
        elif kind == "ready" and connection.token is not None:
            self.room.ready(connection.token, now)
        elif kind == "not_ready" and connection.token is not None:
            self.room.not_ready(connection.token, now)
        elif kind == "leave" and connection.token is not None:
            self._drop(connection, str(message.get("reason", "it left")), now)
        else:
            raise ProtocolError(f"a {kind!r} message out of turn")

    def _drop(self, connection: _Connection, reason: str, now: float) -> None:
        if connection.token is not None:
            self.room.leave(connection.token, reason, now)
        connection.sock.close()
        if connection in self._connections:
            self._connections.remove(connection)

    def _read_controls(self, now: float) -> None:
        while True:
            try:
                data, address = self._udp.recvfrom(_RECEIVE_BYTES)
            except BlockingIOError:
                return
            except ConnectionResetError:
                # Windows reports a datagram sent earlier to a closed port as an error here.
                continue
            try:
                if not self.room.controls(ControlsMessage.unpack(data), address, now):
                    self.ignored_datagrams += 1
            except ProtocolError:
                self.ignored_datagrams += 1

    def _answer_queries(self) -> None:
        if self._discovery is None:
            return
        while True:
            try:
                data, address = self._discovery.recvfrom(_RECEIVE_BYTES)
            except BlockingIOError:
                return
            except ConnectionResetError:
                continue
            if is_find_query(data):
                answer = json.dumps(self.room.announcement(self.host, self.port).record())
                try:
                    self._discovery.sendto(answer.encode(), address)
                except OSError:
                    pass

    def _send_frame(self) -> None:
        recipients = self.room.recipients()
        if not recipients:
            return
        data = self.room.frame().pack()
        for address in recipients:
            try:
                self._udp.sendto(data, address)
            except OSError:
                pass  # the room notices a client it cannot reach by its silence

    def _send(self, connection: _Connection, message: dict[str, Any]) -> None:
        try:
            connection.sock.sendall(encode_message(message))
        except OSError:
            pass  # a connection that cannot take a message is read as closed at the next step

    def _close_voided_connections(self) -> None:
        for token in self.room.voided_tokens:
            for connection in list(self._connections):
                if connection.token == token:
                    connection.token = None
                    self._drop(connection, "its seat was taken over", self._clock())
        self.room.voided_tokens.clear()


def _bind_tcp_and_udp(port: int) -> tuple[socket.socket, socket.socket]:
    """A TCP listener and a UDP socket on the same port number, both non-blocking."""
    for _ in range(20):
        tcp = socket.create_server(("", port))
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            udp.bind(("", tcp.getsockname()[1]))
        except OSError:
            tcp.close()
            udp.close()
            if port:
                raise
            continue
        tcp.setblocking(False)
        udp.setblocking(False)
        return tcp, udp
    raise OSError("no port number free for both TCP and UDP")


def _lan_addresses() -> list[str]:
    """This computer's addresses on its networks, as far as the standard library tells."""
    addresses = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))  # sends nothing: it only picks a route
            addresses.append(probe.getsockname()[0])
    except OSError:
        pass
    try:
        addresses += socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        pass
    return [address for index, address in enumerate(addresses)
            if not address.startswith("127.") and address not in addresses[:index]]


def _print_news(text: str) -> None:
    print(time.strftime("%H:%M:%S"), text, flush=True)


def _read_commands(server: MatchServer) -> None:
    for line in sys.stdin:
        server.submit(line)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Host matches for players on this network.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"the TCP and UDP port clients join on (default {DEFAULT_PORT})")
    parser.add_argument("--log-dir", type=Path, default=Path("matches"),
                        help="where each match's log goes (default: matches)")
    parser.add_argument("--seed", type=int,
                        help="the first match's seed, the next ones counting up from it; "
                             "without it, each match's is drawn at random")
    args = parser.parse_args(argv)
    if args.seed is not None and not 0 <= args.seed < 2 ** 32:
        parser.error("--seed: a whole number from 0 to 2**32 - 1")
    try:
        server = MatchServer(args.port, args.log_dir, args.seed, report=_print_news)
    except OSError as error:
        parser.exit(1, f"rocketsim-server: {error}. Another server may be running on this "
                       f"computer already.\n")
    print(f"rocketsim-server on {server.host}, port {server.port}; match logs go to "
          f"{args.log_dir.resolve()}")
    for address in _lan_addresses():
        print(f"  players join with: rocketsim-play --join {address}:{server.port}")
    print(f"commands: {_COMMANDS}", flush=True)
    threading.Thread(target=_read_commands, args=(server,), daemon=True).start()
    try:
        server.run()
    except KeyboardInterrupt:
        server.shut_down()


if __name__ == "__main__":
    main()
