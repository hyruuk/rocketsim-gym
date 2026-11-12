"""The server over real sockets on this computer: clients find it, join, play and get its frames,
a client gone pauses the match, and refusals and unreachable servers say what to do."""

import importlib.metadata
import json
import socket
import threading
import time

import pytest

from rocketsim_gym.controls import Controls
from rocketsim_gym.lan.connection import (JoinRefused, MatchConnection, find_servers,
                                          server_address)
from rocketsim_gym.lan.protocol import (DEFAULT_PORT, ControlsEntry, JoinRequest,
                                        MessageReader, RoomStatus, Setup, encode_message)
from rocketsim_gym.lan.server import MatchServer
from rocketsim_gym.match import Car, Team
from rocketsim_gym.simulation import load_homemade_arena

SETUP = Setup(1, ("blue1", "orange1"), 20)
ORANGE1 = Car(Team.ORANGE, 1)


@pytest.fixture
def server(tmp_path):
    server = MatchServer(port=0, log_folder=tmp_path, seed=5, discovery_port=0,
                         report=lambda text: None)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    yield server
    server.submit("quit")
    thread.join(timeout=5)


def _request(seat, version=None):
    return JoinRequest(version or importlib.metadata.version("rocketsim-gym"),
                       load_homemade_arena(), SETUP, seat, f"test-{seat}")


class Player:
    """A headless client: it sends its controls every step and keeps the frames that come."""

    def __init__(self, port, seat):
        self.connection = MatchConnection("127.0.0.1", port, _request(seat))
        self.sequence, self.shown, self.frames = 0, 0, []

    def step(self):
        self.sequence += 1
        self.connection.send([ControlsEntry(self.sequence, self.shown, Controls(throttle=1.0))])
        for frame in self.connection.receive():
            self.frames.append(frame)
            self.shown = frame.number


def _play_until(players, condition, seconds=10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for player in players:
            player.step()
        if condition():
            return
        time.sleep(1 / 60)
    raise AssertionError("not within the time")


def test_a_client_finds_the_server_on_this_computer(server):
    found = find_servers(timeout=0.5, discovery_port=server.discovery_port)
    assert [(one.host, one.port) for one in found] == [(server.host, server.port)]
    assert server_address("127.0.0.1:1234") == ("127.0.0.1", 1234, "given")
    assert server_address("lab-pc") == ("lab-pc", DEFAULT_PORT, "given")


def test_two_clients_play_one_match_and_one_gone_pauses_it(server, tmp_path):
    blue, orange = Player(server.port, "blue1"), Player(server.port, "orange1")
    for player in (blue, orange):
        player.step()
        player.connection.ready()
    _play_until([blue, orange], lambda: all(
        player.frames and player.frames[-1].status is RoomStatus.PLAYING
        for player in (blue, orange)))
    _play_until([blue, orange], lambda: len(blue.frames) > 120 and len(orange.frames) > 120)
    blue_snapshots = {frame.number: frame.snapshot for frame in blue.frames}
    shared = [frame for frame in orange.frames if frame.number in blue_snapshots]
    assert len(shared) > 60
    assert all(frame.snapshot == blue_snapshots[frame.number] for frame in shared)
    orange.connection.close()
    _play_until([blue], lambda: blue.frames[-1].status is RoomStatus.PAUSED)
    assert blue.frames[-1].waiting_for == {ORANGE1}
    server.submit("end")
    _play_until([blue], lambda: blue.frames[-1].status is RoomStatus.STOPPED)
    (log,) = tmp_path.glob("match-*.jsonl")
    assert json.loads(log.read_text().splitlines()[-1])["how"] == "ended by the experimenter"


def test_a_refused_join_says_why(server):
    with pytest.raises(JoinRefused, match="same version"):
        MatchConnection("127.0.0.1", server.port, _request("blue1", version="0.0.0"))


def test_a_join_of_another_protocol_is_refused_with_why(server):
    with socket.create_connection(("127.0.0.1", server.port), timeout=3) as sock:
        sock.sendall(encode_message({**_request("blue1").record(), "protocol": 99}))
        reply = next(MessageReader().feed(sock.recv(1 << 16)))
    assert reply["type"] == "refused" and "protocol 99" in reply["reason"]


def test_an_unreachable_server_says_what_to_do():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    with pytest.raises(ConnectionError, match="nothing answers"):
        MatchConnection("127.0.0.1", closed_port, _request("blue1"))


def test_a_server_that_quits_tells_its_clients(server):
    blue = Player(server.port, "blue1")
    blue.step()
    server.submit("quit")
    _play_until([blue], lambda: blue.connection.is_server_gone, seconds=5)
