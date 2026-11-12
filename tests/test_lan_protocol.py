"""The LAN protocol's messages: each reads back as it was written, and what is not one is
refused."""

import json
import struct

import pytest

from rocketsim_gym.controls import Controls
from rocketsim_gym.lan.protocol import (CONTROLS_REPEATED, MESSAGE_LIMIT, Announcement,
                                        ControlsEntry, ControlsMessage, JoinRequest,
                                        MessageReader, ProtocolError, Refusal, RoomStatus,
                                        ServerFrame, Setup, Welcome, encode_message, find_query,
                                        is_find_query)
from rocketsim_gym.match import Car, Team

TOKEN = bytes(range(16))
SETUP = Setup(1, ("blue1", "orange1"), 60)


def test_a_setup_lists_its_seats_in_match_order_and_compares_so():
    setup = Setup(2, ("orange1", "blue2"), 300)
    assert setup.seats == ("blue2", "orange1")
    assert setup == Setup(2, ("blue2", "orange1"), 300.0)
    assert setup.seat_cars == (Car(Team.BLUE, 2), Car(Team.ORANGE, 1))
    assert setup.car("orange1") == Car(Team.ORANGE, 1)
    assert Setup.from_record(setup.record()) == setup


@pytest.mark.parametrize("team_size, seats, seconds", [
    (4, ("blue1",), 300), (1, ("blue2",), 300), (1, (), 300), (1, ("blue1", "blue1"), 300),
    (1, ("blue1",), 0), (1, ("blue1",), float("nan")), (True, ("blue1",), 300)])
def test_a_setup_out_of_range_is_refused(team_size, seats, seconds):
    with pytest.raises(ProtocolError):
        Setup(team_size, seats, seconds)


def test_joins_welcomes_and_refusals_read_back():
    request = JoinRequest("0.3.0", {"soccar/arena.cmf": "ab12"}, SETUP, "blue1", "mac")
    assert JoinRequest.from_record(request.record()) == request
    welcome = Welcome(TOKEN, 47800, "blue1", SETUP, "server")
    assert Welcome.from_record(welcome.record()) == welcome
    refusal = Refusal("busy", is_final=False)
    assert Refusal.from_record(refusal.record()) == refusal


def test_a_join_of_another_protocol_or_shape_is_refused():
    record = JoinRequest("0.3.0", {}, None, None, "mac").record()
    with pytest.raises(ProtocolError, match="protocol"):
        JoinRequest.from_record({**record, "protocol": 99})
    with pytest.raises(ProtocolError, match="host"):
        JoinRequest.from_record({**record, "host": 3})
    with pytest.raises(ProtocolError, match="without"):
        JoinRequest.from_record({"type": "join"})


def test_controls_read_back_to_the_bit():
    entries = tuple(ControlsEntry(10 - index, 500 - index, Controls(
        -0.0, 0.123456789012345, -1.0, 1.0, 0.5, index % 2 == 0, True, index == 3))
        for index in range(CONTROLS_REPEATED))
    message = ControlsMessage(TOKEN, entries)
    unpacked = ControlsMessage.unpack(message.pack())
    assert unpacked == message
    assert repr(unpacked.entries[0].controls.throttle) == "-0.0"


def _controls_datagram(count, throttle=0.0):
    header = struct.pack("<c16sB", b"C", TOKEN, count)
    entry = struct.pack("<II5dB", 1, 1, throttle, 0.0, 0.0, 0.0, 0.0, 0)
    return header + entry * count


@pytest.mark.parametrize("data", [
    b"", b"F" + bytes(60), _controls_datagram(0), _controls_datagram(CONTROLS_REPEATED + 1),
    _controls_datagram(2)[:-1], _controls_datagram(1, throttle=1.5),
    _controls_datagram(1, throttle=float("inf"))])
def test_a_datagram_that_is_not_controls_is_refused(data):
    with pytest.raises(ProtocolError):
        ControlsMessage.unpack(data)


def test_a_server_frame_reads_back():
    frame = ServerFrame(7, RoomStatus.PAUSED, 2, b"the snapshot", 120,
                        frozenset({Car(Team.BLUE, 2)}), frozenset({Car(Team.ORANGE, 1)}),
                        0xDEADBEEF, 12345, (3, None, 200, None))
    unpacked = ServerFrame.unpack(frame.pack())
    assert unpacked == ServerFrame(7, RoomStatus.PAUSED, 2, b"the snapshot", 120,
                                   frozenset({Car(Team.BLUE, 2)}),
                                   frozenset({Car(Team.ORANGE, 1)}), 0xDEADBEEF, 12345,
                                   (3, None, 127, None))
    with pytest.raises(ProtocolError):
        ServerFrame.unpack(b"C" + frame.pack()[1:])


def test_messages_come_whole_however_the_bytes_arrive():
    data = encode_message({"type": "ready"}) + encode_message({"type": "leave", "reason": "x"})
    reader, messages = MessageReader(), []
    for byte in data:
        messages += reader.feed(bytes([byte]))
    assert messages == [{"type": "ready"}, {"type": "leave", "reason": "x"}]


@pytest.mark.parametrize("data", [
    struct.pack("<I", MESSAGE_LIMIT + 1), struct.pack("<I", 3) + b"{no",
    struct.pack("<I", 2) + b"[]"])
def test_a_message_too_long_or_without_a_type_is_refused(data):
    with pytest.raises(ProtocolError):
        list(MessageReader().feed(data))


def test_a_query_is_answered_with_where_the_server_is():
    assert is_find_query(find_query()) and not is_find_query(b"hello")
    announcement = Announcement("lab-pc", 47800, 1, RoomStatus.LOBBY, SETUP, {"blue1": "mac"})
    found = Announcement.from_datagram(json.dumps(announcement.record()).encode(), "10.0.0.5")
    assert found.setup == SETUP and found.seats == {"blue1": "mac"}
    assert found.join_address == "10.0.0.5:47800"
