"""A match server's log of one match, and replaying it.

The log is JSON lines in a file of its own (``match-<time>-<id>.jsonl``): a header (the versions,
the arena, the setup, who sat where, the seed), a line for every step (the controls each seat's
client gave, what came late or was dropped, the seats bots drive, the input delays, and a digest
of the step's snapshot), a line for everything that happened in the room (a player gone or back,
a pause, a command), and a last line for how the match ended. It is flushed every second.

:func:`replay_match` plays a log's match again from its seed and controls and checks every
step's digest. RocketSim's physics is deterministic on one machine, so that is the server's.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..controls import Controls
from ..match import Team
from ..simulation import MatchSimulation, load_homemade_arena
from ..snapshot import Snapshot
from .protocol import TICKS_PER_STEP, Setup

#: The version of the log's lines.
LOG_FORMAT = 1
#: How often the log reaches the disk.
FLUSH_SECONDS = 1.0


def snapshot_digest(snapshot: Snapshot) -> str:
    """A short digest of ``snapshot``'s bytes, which a replay must reproduce."""
    return hashlib.sha256(snapshot.pack()).hexdigest()[:16]


def controls_record(controls: Controls) -> list:
    """``controls`` as a log writes them: the five axes, then jump, boost and powerslide."""
    return [controls.throttle, controls.steer, controls.pitch, controls.yaw, controls.roll,
            controls.jump, controls.boost, controls.handbrake]


class MatchLog:
    """The log of the match a room plays; written as the match goes.

    :param folder: where the log goes; made if missing.
    :param match_id: the match's id, which names the file.
    :param setup: the match.
    :param seed: its random generator's seed.
    :param seats: each seat and the computer in it.
    :param collision_meshes: the arena: each mesh file and its SHA-256.
    :param clock: the server's clock, for flushing.
    """

    def __init__(self, folder: Path, match_id: int, setup: Setup, seed: int,
                 seats: Mapping[str, str], collision_meshes: Mapping[str, str], clock) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"match-{time.strftime('%Y%m%d-%H%M%S')}-{match_id:08x}.jsonl"
        self._file = self.path.open("x", encoding="utf-8")
        self._clock = clock
        self._flushed_at = clock()
        self.write({"type": "match", "format": LOG_FORMAT, "match_id": f"{match_id:08x}",
                    "seed": seed, "setup": setup.record(), "seats": dict(seats),
                    "ticks_per_step": TICKS_PER_STEP,
                    "rocketsim_gym_version": importlib.metadata.version("rocketsim-gym"),
                    "rocketsim_version": importlib.metadata.version("rocketsim"),
                    "collision_meshes": dict(collision_meshes),
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        self._file.flush()

    def write(self, record: Mapping[str, Any]) -> None:
        """Add a line; flush once a second has passed since the last flush."""
        self._file.write(json.dumps(record) + "\n")
        if self._clock() - self._flushed_at >= FLUSH_SECONDS:
            self._file.flush()
            self._flushed_at = self._clock()

    def close(self) -> None:
        self._file.close()


@dataclass(frozen=True)
class MatchReplay:
    """What replaying a log gave: the steps played again, the ones whose digest differs from
    the log's, and the last snapshot."""

    steps: int
    mismatched_steps: tuple[int, ...]
    last_snapshot: Snapshot


def read_match_log(path: Path) -> Iterator[dict[str, Any]]:
    """A log's lines, in order; a last line cut short by a crash is left out."""
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            try:
                yield json.loads(line)
            except ValueError:
                return


def replay_match(path: Path) -> MatchReplay:
    """Play a log's match again, from its seed and the controls it logged, and compare every
    step's snapshot with the log's digest.

    :raises ValueError: the log has no header, or one of another format, or its match was played
        in another arena than this package's.
    """
    records = read_match_log(path)
    header = next(records, None)
    if header is None or header.get("type") != "match" or header.get("format") != LOG_FORMAT:
        raise ValueError(f"{path}: not a match log of format {LOG_FORMAT}")
    if header["collision_meshes"] != load_homemade_arena():
        raise ValueError(f"{path}: a match played in another arena than this package's")
    setup = Setup.from_record(header["setup"])
    simulation = MatchSimulation(setup.team_size, setup.match_ticks, header["ticks_per_step"],
                                 setup.seat_cars, np.random.default_rng(header["seed"]))
    cars = {car.name: car for car in setup.cars}
    steps, mismatched = 0, []
    snapshot = simulation.snapshot()
    for record in records:
        if record["type"] == "event" and record["event"] == "bot":
            simulation.hand_to_bot(cars[record["seat"]])
        if record["type"] != "step":
            continue
        simulation.step({cars[seat]: Controls(*values)
                         for seat, values in record["controls"].items()})
        snapshot = simulation.snapshot()
        steps += 1
        if snapshot_digest(snapshot) != record["digest"]:
            mismatched.append(record["step"])
    return MatchReplay(steps, tuple(mismatched), snapshot)


def score_record(score: Mapping[Team, int]) -> dict[str, int]:
    """A score as a log writes it: each team's goals, by its name."""
    return {team.value: goals for team, goals in score.items()}
