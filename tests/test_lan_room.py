"""A server's room: who it lets in, when a match starts, pauses and goes on, what the experimenter
can do, how each seat's controls queue up, and the log a match leaves, which replays."""

import json
import subprocess
import sys

import numpy as np
import pytest

from rocketsim_gym.controls import NO_CONTROLS, Controls
from rocketsim_gym.lan.protocol import (TICKS_PER_STEP, ControlsEntry, ControlsMessage,
                                        JoinRequest, Refusal, RoomStatus, Setup, Welcome)
from rocketsim_gym.lan.room import (QUEUE_BURST, QUEUE_DEEP_STEPS, QUEUE_TARGET,
                                    RESUME_SECONDS, SILENCE_SECONDS, ControlsQueue, Room)
from rocketsim_gym.match import TICKS_PER_SECOND, Car, Phase, Team
from rocketsim_gym.simulation import load_homemade_arena

VERSION, MESHES = "test", load_homemade_arena()
SETUP = Setup(1, ("blue1", "orange1"), 20)
BLUE1, ORANGE1 = Car(Team.BLUE, 1), Car(Team.ORANGE, 1)
STEP = TICKS_PER_STEP / TICKS_PER_SECOND


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Lab:
    """A room, its clock, its news, and the clients in it, stepped by hand."""

    def __init__(self, log_folder, seed=7):
        self.clock, self.news = Clock(), []
        self.room = Room(VERSION, MESHES, "server", 47800, log_folder, seed, self.news.append,
                         self.clock)
        self.sequences = {}

    def join(self, seat, setup=SETUP, version=VERSION, meshes=MESHES):
        return self.room.join(JoinRequest(version, meshes, setup, seat, f"pc-{seat}"),
                              self.clock.now)

    def step(self, *talking, controls=NO_CONTROLS, steps=1):
        """Step the room, each client in ``talking`` sending its controls first."""
        for _ in range(steps):
            self.clock.now += STEP
            for welcome in talking:
                sequence = self.sequences.get(welcome.seat, 0) + 1
                self.sequences[welcome.seat] = sequence
                entries = tuple(ControlsEntry(sequence - back, self.room.frame_number, controls)
                                for back in range(min(sequence, 4)))
                self.room.controls(ControlsMessage(welcome.token, entries),
                                   ("127.0.0.1", 5000), self.clock.now)
            self.room.step(self.clock.now)

    def start_match(self):
        blue, orange = self.join("blue1"), self.join("orange1")
        self.room.ready(blue.token, self.clock.now)
        self.room.ready(orange.token, self.clock.now)
        self.step(blue, orange)
        assert self.room.status is RoomStatus.PLAYING
        return blue, orange


def test_the_first_client_sets_the_room_and_another_setup_waits_until_it_empties(tmp_path):
    lab = Lab(tmp_path)
    blue = lab.join("blue1")
    assert isinstance(blue, Welcome) and lab.room.setup == SETUP
    other_setup = Setup(2, ("blue1", "blue2"), 20)
    refusal = lab.join("blue2", setup=other_setup)
    assert isinstance(refusal, Refusal) and not refusal.is_final and "plays" in refusal.reason
    lab.room.leave(blue.token, "quit", lab.clock.now)
    assert lab.room.setup is None
    assert isinstance(lab.join("blue2", setup=other_setup), Welcome)
    assert lab.room.setup == other_setup


def test_a_client_of_another_version_or_arena_is_refused_for_good(tmp_path):
    lab = Lab(tmp_path)
    for refusal in (lab.join("blue1", version="other"),
                    lab.join("blue1", meshes={"soccar/real.cmf": "11"})):
        assert isinstance(refusal, Refusal) and refusal.is_final
    assert "same version" in lab.join("blue1", version="other").reason


def test_a_client_takes_the_seat_it_asks_for_or_any_free_one(tmp_path):
    lab = Lab(tmp_path)
    assert lab.join("blue1").seat == "blue1"
    assert "taken" in lab.join("blue1").reason
    assert lab.join(None).seat == "orange1"
    full = lab.join(None)
    assert isinstance(full, Refusal) and not full.is_final
    assert lab.join("orange2").is_final


def test_a_match_starts_once_every_seat_is_ready_and_never_for_a_client_that_only_joined(
        tmp_path):
    lab = Lab(tmp_path)
    blue, orange = lab.join("blue1"), lab.join("orange1")
    lab.room.ready(blue.token, lab.clock.now)
    lab.step(blue, steps=120)
    assert lab.room.status is RoomStatus.LOBBY
    assert lab.room.frame().waiting_for == {ORANGE1}
    lab.room.ready(orange.token, lab.clock.now)
    lab.step(blue, orange)
    assert lab.room.status is RoomStatus.PLAYING and lab.room.match_id
    assert len(list(tmp_path.glob("match-*.jsonl"))) == 1


def test_silence_pauses_a_running_match_but_never_the_lobby(tmp_path):
    lab = Lab(tmp_path)
    blue, orange = lab.join("blue1"), lab.join("orange1")
    lab.step(steps=round(10 * SILENCE_SECONDS / STEP))
    assert lab.room.status is RoomStatus.LOBBY and lab.room.seats[BLUE1].token == blue.token
    lab.room.ready(blue.token, lab.clock.now)
    lab.room.ready(orange.token, lab.clock.now)
    lab.step(blue, orange, steps=240)  # past the kickoff's countdown
    lab.step(blue, steps=round(SILENCE_SECONDS / STEP) + 2)
    assert lab.room.status is RoomStatus.PAUSED
    assert lab.room.frame().waiting_for == {ORANGE1}
    assert lab.room.resume(lab.clock.now).startswith("waiting for orange1")
    lab.step(blue, orange)
    assert any("heard again" in line for line in lab.news)
    assert lab.room.resume(lab.clock.now) == "resuming"
    lab.step(blue, orange, steps=RESUME_SECONDS * TICKS_PER_SECOND // TICKS_PER_STEP)
    assert lab.room.status is RoomStatus.PLAYING


def test_a_client_that_leaves_pauses_the_match_and_one_that_takes_its_seat_brings_it_back(
        tmp_path):
    lab = Lab(tmp_path)
    blue, orange = lab.start_match()
    lab.step(blue, orange, steps=200)
    lab.room.leave(orange.token, "quit", lab.clock.now)
    assert lab.room.status is RoomStatus.PAUSED
    step_paused = lab.room._snapshot.step
    lab.step(blue, steps=30)
    assert lab.room._snapshot.step == step_paused
    assert lab.room.resume(lab.clock.now).startswith("waiting for orange1")
    back = lab.join("orange1")
    assert isinstance(back, Welcome)
    lab.room.ready(back.token, lab.clock.now)
    lab.step(blue, back)
    assert lab.room.resume(lab.clock.now) == "resuming"


def test_a_bot_drives_a_seat_to_the_end_of_the_match(tmp_path):
    lab = Lab(tmp_path)
    blue, orange = lab.start_match()
    lab.step(blue, orange, steps=200)  # past the kickoff's countdown
    lab.room.leave(orange.token, "error", lab.clock.now)
    assert lab.room.hand_to_bot("orange1", lab.clock.now).startswith("a bot drives orange1")
    assert lab.room.resume(lab.clock.now) == "resuming"
    refusal = lab.join("orange1")
    assert isinstance(refusal, Refusal) and not refusal.is_final
    assert lab.room.frame().bots == {ORANGE1}
    lab.step(blue, steps=RESUME_SECONDS * TICKS_PER_SECOND // TICKS_PER_STEP + 60)
    assert lab.room.status is RoomStatus.PLAYING
    assert lab.room._snapshot.car(ORANGE1).controls != NO_CONTROLS


def test_the_experimenter_ends_a_match_and_the_next_waits_for_ready_with_the_next_seed(
        tmp_path):
    lab = Lab(tmp_path, seed=7)
    blue, orange = lab.start_match()
    assert lab.room.seed == 7
    lab.step(blue, orange, steps=100)
    assert lab.room.pause(lab.clock.now) == "paused"
    assert lab.room.end(lab.clock.now) == "ended"
    assert lab.room.status is RoomStatus.STOPPED
    lab.step(blue, orange, steps=10)
    lab.room.ready(blue.token, lab.clock.now)
    assert lab.room.status is RoomStatus.LOBBY and lab.room.seed == 8
    lab.room.ready(orange.token, lab.clock.now)
    lab.step(blue, orange)
    assert lab.room.status is RoomStatus.PLAYING and lab.room.matches_started == 2
    (log,) = [path for path in tmp_path.glob("match-*.jsonl")
              if json.loads(path.read_text().splitlines()[0])["seed"] == 7]
    assert json.loads(log.read_text().splitlines()[-1])["how"] == "ended by the experimenter"


def test_a_pause_in_a_kickoff_countdown_goes_on_without_another_countdown(tmp_path):
    lab = Lab(tmp_path)
    blue, orange = lab.start_match()
    assert lab.room._snapshot.match.phase is Phase.COUNTDOWN
    lab.room.pause(lab.clock.now)
    assert lab.room.resume(lab.clock.now) == "playing"


def test_the_input_delay_is_the_frames_from_showing_to_applying(tmp_path):
    lab = Lab(tmp_path)
    blue, orange = lab.start_match()
    lab.step(blue, orange, steps=30)
    # Each client sent its controls with the frame before this step's as the one it showed.
    assert lab.room.frame().input_delays == (1, 1)


def test_a_match_log_replays_to_its_digests_in_another_process(tmp_path):
    lab = Lab(tmp_path, seed=11)
    blue, orange = lab.start_match()
    rng = np.random.default_rng(3)
    for step in range(900):
        throttle, steer = rng.uniform(-1, 1, 2)
        lab.step(blue, orange, controls=Controls(throttle=throttle, steer=steer,
                                                 jump=bool(rng.integers(2)),
                                                 boost=bool(rng.integers(2))))
        if step == 500:
            lab.room.leave(orange.token, "quit", lab.clock.now)
            lab.room.hand_to_bot("orange1", lab.clock.now)
            lab.room.resume(lab.clock.now)
            orange = None
            break
    lab.step(blue, steps=400)
    lab.room.end(lab.clock.now)
    (log,) = tmp_path.glob("match-*.jsonl")
    code = ("import sys; from rocketsim_gym.lan.match_log import replay_match; "
            "replay = replay_match(sys.argv[1]); print(replay.steps, len(replay.mismatched_steps))")
    result = subprocess.run([sys.executable, "-c", code, str(log)], capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr
    output = result.stdout.split()
    steps = sum(1 for line in log.read_text().splitlines() if '"type": "step"' in line)
    assert output == [str(steps), "0"] and steps > 700


def _entries(sequences, controls=NO_CONTROLS):
    return [ControlsEntry(sequence, 0, controls) for sequence in sequences]


def test_the_queue_takes_each_entry_once_in_order_and_holds_the_last_while_one_is_late():
    queue = ControlsQueue()
    jump = Controls(jump=True)
    queue.add(_entries([1]))
    queue.add(_entries([2, 1], jump))
    assert [queue.take().entry.sequence, queue.take().entry.sequence] == [1, 2]
    late = queue.take()
    assert late.entry is None and late.controls == jump
    queue.add(_entries([5, 4, 3]))
    assert queue.take().entry.sequence == 3


def test_a_jittery_lossy_client_loses_no_one_step_jump():
    rng = np.random.default_rng(0)
    queue, in_flight, jumps_sent, jumps_applied = ControlsQueue(), [], [], []
    history = []
    for sequence in range(1, 3001):
        jump = bool(rng.random() < 0.1)
        history.append(ControlsEntry(sequence, 0, Controls(jump=jump)))
        if jump:
            jumps_sent.append(sequence)
        datagram = list(reversed(history[-4:]))
        if rng.random() > 0.05:  # 5% of datagrams lost
            in_flight.append((sequence + rng.integers(0, 3), datagram))  # 0 to 2 steps late
        for arrived in [flight for flight in in_flight if flight[0] <= sequence]:
            queue.add(arrived[1])
            in_flight.remove(arrived)
        take = queue.take()
        if take.entry is not None and take.controls.jump:
            jumps_applied.append(take.entry.sequence)
    # Each jump once and in order; all of them but those before the queue started (its first
    # datagram's newest entry) and those still in flight at the end.
    assert jumps_applied == sorted(set(jumps_applied))
    assert {jump for jump in jumps_sent if 10 < jump <= 2990} <= set(jumps_applied)


def test_a_client_running_ahead_has_its_oldest_entry_dropped_after_a_while():
    queue = ControlsQueue()
    queue.add(_entries([1]))
    sequence, dropped = 1, 0
    for step in range(QUEUE_DEEP_STEPS + QUEUE_TARGET + 2):
        sequence += 2 if step < QUEUE_TARGET + 1 else 1
        queue.add(_entries(range(sequence, sequence - 4, -1)))
        dropped += queue.take().dropped
    assert dropped == 1


def test_a_burst_drops_back_to_the_target_at_once():
    queue = ControlsQueue()
    queue.add(_entries([1]))
    queue.add(_entries(range(QUEUE_BURST + 5, 1, -1)))
    take = queue.take()
    assert take.dropped == QUEUE_BURST + 4 - QUEUE_TARGET
    assert take.entry.sequence == 1


@pytest.mark.parametrize("seat", [None, "blue1"])
def test_a_room_without_a_setup_refuses_a_client_without_one(tmp_path, seat):
    refusal = Lab(tmp_path).join(seat, setup=None)
    assert isinstance(refusal, Refusal) and "no match yet" in refusal.reason
