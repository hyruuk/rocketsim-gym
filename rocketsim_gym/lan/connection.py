"""A client's connection to a match server, and finding servers on the local network.

A client finds servers by broadcasting a query (:func:`find_servers`), or is given an address
(:func:`server_address`). Its connection (:class:`MatchConnection`) joins over TCP, then sends
its controls over UDP every step and takes the frames the server sends back.
"""

from __future__ import annotations

import errno
import socket
import sys
import time
from collections.abc import Iterable
from typing import Any

from .protocol import (DEFAULT_PORT, DISCOVERY_PORT, Announcement, ControlsEntry,
                       ControlsMessage, JoinRequest, MessageReader, ProtocolError, Refusal,
                       ServerFrame, Welcome, encode_message, find_query)

#: How long :func:`find_servers` waits for answers.
FIND_SECONDS = 1.0
#: How long joining waits for the server.
JOIN_SECONDS = 3.0
_RECEIVE_BYTES = 1 << 16
_UNREACHABLE = {errno.EHOSTUNREACH, errno.ENETUNREACH}
_LOCAL_NETWORK_FIX = (" macOS may be keeping this app off the local network: in System Settings "
                      "› Privacy & Security › Local Network, allow the terminal app this runs in, "
                      "then try again.")


class JoinRefused(RuntimeError):
    """The server's refusal; one that is not final can be asked again."""

    def __init__(self, refusal: Refusal) -> None:
        super().__init__(refusal.reason)
        self.refusal = refusal


def find_servers(timeout: float = FIND_SECONDS, discovery_port: int = DISCOVERY_PORT,
                 hosts: Iterable[str] = ("<broadcast>", "127.0.0.1")) -> list[Announcement]:
    """The servers that answer a query within ``timeout`` seconds, each once: by default, the
    ones on this network and on this computer.

    :param hosts: where the query goes.
    :raises ConnectionError: the query could not be sent anywhere, with what to do.
    """
    found: dict[tuple[str, int], Announcement] = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        problems = []
        for host in hosts:
            try:
                sock.sendto(find_query(), (host, discovery_port))
            except OSError as error:
                problems.append(error)
        deadline = time.perf_counter() + timeout
        while (left := deadline - time.perf_counter()) > 0:
            sock.settimeout(left)
            try:
                data, (address, _) = sock.recvfrom(_RECEIVE_BYTES)
            except (TimeoutError, socket.timeout):
                break
            except ConnectionResetError:
                continue
            try:
                announcement = Announcement.from_datagram(data, address)
            except ProtocolError:
                continue
            found.setdefault((announcement.host, announcement.port), announcement)
    if not found and problems:
        raise ConnectionError(_problem("the local network", problems[0]))
    return list(found.values())


def server_address(server: str, timeout: float = FIND_SECONDS) -> tuple[str, int, str]:
    """``server`` as the address to join and how it was found: ``"auto"`` finds the one server
    on the network, and ``"HOST"`` or ``"HOST:PORT"`` is taken as given.

    :raises ConnectionError: no server answers, or more than one does, with what to do.
    """
    if server == "auto":
        found = find_servers(timeout)
        if not found:
            raise ConnectionError("no rocketsim-server answered on this network: start one, or "
                                  "join by its address (HOST:PORT), which it prints")
        if len(found) > 1:
            listed = ", ".join(f"{one.host} at {one.join_address}" for one in found)
            raise ConnectionError(f"{len(found)} servers answered ({listed}): join one by its "
                                  f"address")
        return found[0].address, found[0].port, f"found {found[0].host} on the network"
    host, separator, port = server.rpartition(":")
    if not separator:
        return server, DEFAULT_PORT, "given"
    if not host or not port.isdigit():
        raise ConnectionError(f"server {server!r}: HOST or HOST:PORT")
    return host, int(port), "given"


class MatchConnection:
    """A client's seat in a server's room: its controls going out, the frames coming in.

    :param host, port: the server's address.
    :param request: what the client asks for.
    :param timeout: how long joining waits for the server.
    :raises ConnectionError: the server cannot be reached, with what to do.
    :raises JoinRefused: the server refused.
    :ivar welcome: the server's welcome: the seat, the match, the boost pads.
    """

    def __init__(self, host: str, port: int, request: JoinRequest,
                 timeout: float = JOIN_SECONDS) -> None:
        try:
            self._tcp = socket.create_connection((host, port), timeout=timeout)
        except OSError as error:
            raise ConnectionError(_problem(f"the server at {host}:{port}", error)) from None
        self._reader = MessageReader()
        try:
            self._tcp.sendall(encode_message(request.record()))
            reply = self._read_reply()
        except OSError as error:
            self._tcp.close()
            raise ConnectionError(_problem(f"the server at {host}:{port}", error)) from None
        if reply["type"] == "refused":
            self._tcp.close()
            raise JoinRefused(Refusal.from_record(reply))
        self.welcome = Welcome.from_record(reply)
        self._tcp.setblocking(False)
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp.connect((self._tcp.getpeername()[0], self.welcome.udp_port))
        self._udp.setblocking(False)
        #: Whether the server said it is closing, or its connection closed.
        self.is_server_gone = False

    def send(self, entries: Iterable[ControlsEntry]) -> None:
        """This step's controls and the ones before, newest first."""
        try:
            self._udp.send(ControlsMessage(self.welcome.token, tuple(entries)).pack())
        except OSError:
            pass  # the frames stop coming, which the client notices

    def receive(self) -> list[ServerFrame]:
        """The frames that came since the last call, in the order they came."""
        frames = []
        while True:
            try:
                data = self._udp.recv(_RECEIVE_BYTES)
            except BlockingIOError:
                break
            except (ConnectionRefusedError, ConnectionResetError):
                continue  # an earlier datagram found no server listening
            try:
                frames.append(ServerFrame.unpack(data))
            except ProtocolError:
                continue
        self._read_messages()
        return frames

    def ready(self) -> None:
        """The client steps since its reset: ready for a match."""
        self._send_message({"type": "ready"})

    def not_ready(self) -> None:
        """The client reset its game."""
        self._send_message({"type": "not_ready"})

    def leave(self, reason: str) -> None:
        """Leave the room, saying why, and close."""
        self._send_message({"type": "leave", "reason": reason})
        self.close()

    def close(self) -> None:
        self._tcp.close()
        self._udp.close()

    def _read_reply(self) -> dict[str, Any]:
        while True:
            data = self._tcp.recv(_RECEIVE_BYTES)
            if not data:
                raise ConnectionError("the server closed the connection before answering")
            for message in self._reader.feed(data):
                return message

    def _read_messages(self) -> None:
        while not self.is_server_gone:
            try:
                data = self._tcp.recv(_RECEIVE_BYTES)
            except BlockingIOError:
                return
            except OSError:
                data = b""
            if not data:
                self.is_server_gone = True
                return
            try:
                messages = list(self._reader.feed(data))
            except ProtocolError:
                messages = [{"type": "closing"}]
            if any(message["type"] == "closing" for message in messages):
                self.is_server_gone = True

    def _send_message(self, message: dict[str, Any]) -> None:
        try:
            self._tcp.sendall(encode_message(message))
        except OSError:
            self.is_server_gone = True


def _problem(what: str, error: OSError) -> str:
    """Why ``what`` cannot be reached, and what to do."""
    if isinstance(error, ConnectionRefusedError):
        return (f"nothing answers at {what.removeprefix('the server at ')}: is rocketsim-server "
                f"running there, with its port open in that computer's firewall?")
    if isinstance(error, (TimeoutError, socket.timeout)):
        return (f"no answer from {what} within {JOIN_SECONDS:g} s: check the address, and that "
                f"both computers are on the same network")
    message = f"cannot reach {what}: {error}."
    if error.errno in _UNREACHABLE and sys.platform == "darwin":
        message += _LOCAL_NETWORK_FIX
    return message
