"""RocketSim: loaded with one folder of soccar collision meshes per process, and a 1v1 arena.

RocketSim reads its collision meshes once per process and keeps them for every arena after.
A file it cannot parse leaves it unusable for the rest of the process, and an arena made
before it is loaded reads whatever ``collision_meshes`` folder the working directory has; so
the meshes are checked here, in full, before RocketSim ever sees them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import RocketSim

from .arena_geometry import collision_mesh_manifest, read_collision_mesh_file
from .controls import Controls
from .match import TEAMS, TICKS_PER_SECOND, Team, TickEvents

#: The folder RocketSim was loaded from in this process, once it has been.
_loaded_collision_meshes_folder: Path | None = None

#: The kickoff spots, each with its mirror image for the orange car.
KICKOFF_SPOT_COUNT = len(RocketSim.RLConst.CAR_SPAWN_LOCATIONS_SOCCAR)
#: The spots a demolished car comes back at, each with its mirror image for the orange car.
RESPAWN_SPOT_COUNT = len(RocketSim.RLConst.CAR_RESPAWN_LOCATIONS_SOCCAR)
#: Where the ball waits out the pause after a goal: in the void behind a corner wall, short of
#: the goal lines and out of every car's reach. A ball without velocity stays where it is set.
BALL_PARKING_POSITION = (3800.0, 4800.0, 1000.0)
#: The ball is on the ground while its centre is at most this high (it rests at 93.15).
BALL_ON_GROUND_MAXIMUM_Z = 98.0
#: A demolished car's wait before RocketSim would respawn it, so long that it never does.
_NEVER_SECONDS = 1e9
_ROCKETSIM_TEAMS = {Team.BLUE: RocketSim.Team.BLUE, Team.ORANGE: RocketSim.Team.ORANGE}


def check_collision_meshes_folder(folder: Path, expected_manifest: Mapping[str, str]) -> None:
    """Check that ``folder`` holds exactly the soccar mesh files ``expected_manifest`` lists.

    :param folder: a collision-mesh folder, its soccar meshes under ``soccar/``.
    :param expected_manifest: each mesh file's path, relative to ``folder``, and its SHA-256.
    :raises ValueError: a file is missing, extra, changed, or not a mesh RocketSim reads.
    """
    manifest = collision_mesh_manifest(folder)
    if not manifest:
        raise ValueError(f"{folder}: no soccar collision mesh (soccar/*.cmf)")
    missing = sorted(set(expected_manifest) - set(manifest))
    extra = sorted(set(manifest) - set(expected_manifest))
    changed = sorted(p for p in set(manifest) & set(expected_manifest)
                     if manifest[p] != expected_manifest[p])
    if missing or extra or changed:
        raise ValueError(f"{folder}: its mesh files are not the ones its manifest lists "
                         f"(missing {missing}, not listed {extra}, changed {changed})")
    for relative_path in manifest:
        read_collision_mesh_file(Path(folder) / relative_path)


def load_rocketsim(collision_meshes_folder: Path, expected_manifest: Mapping[str, str]) -> None:
    """Load RocketSim with the soccar meshes of ``collision_meshes_folder``, once per process.

    Asking again for the folder already loaded does nothing.

    :param collision_meshes_folder: the folder to load, checked first in full.
    :param expected_manifest: each mesh file's path, relative to the folder, and its SHA-256.
    :raises ValueError: the folder fails its check, or this process loaded another folder.
    """
    global _loaded_collision_meshes_folder
    folder = Path(collision_meshes_folder).resolve()
    if _loaded_collision_meshes_folder is not None:
        if folder != _loaded_collision_meshes_folder:
            raise ValueError(f"RocketSim is loaded with the collision meshes of "
                             f"{_loaded_collision_meshes_folder}, and one process takes one "
                             f"folder: {folder} cannot be loaded as well")
        return
    check_collision_meshes_folder(folder, expected_manifest)
    RocketSim.init(str(folder))
    _loaded_collision_meshes_folder = folder


def new_soccar_arena(tick_rate: float = 120.0) -> RocketSim.Arena:
    """A new, empty soccar arena of the loaded meshes, stepping ``tick_rate`` ticks a second.

    It uses Bullet's own broadphase. RocketSim's grid broadphase spends about a second
    building, first stepping and deleting each arena over a mesh that spans the whole field,
    which every reset would pay; Bullet's costs nothing there, and a tick still takes
    microseconds.
    """
    if _loaded_collision_meshes_folder is None:
        raise ValueError("RocketSim has no collision meshes loaded yet")
    config = RocketSim.ArenaConfig()
    config.use_custom_broadphase = False
    return RocketSim.Arena(RocketSim.GameMode.SOCCAR, tick_rate=tick_rate, config=config)


class SoccarArena:
    """A 1v1 soccar arena: the ball, a blue Octane and an orange Octane, stepped a tick at a time.

    The cars and the ball are only ever placed by setting their state. RocketSim's own kickoff
    and respawn draw from a clock-seeded random generator, so this arena never lets it run
    either; whoever drives it picks the spots.
    """

    def __init__(self) -> None:
        self._arena = new_soccar_arena(TICKS_PER_SECOND)
        mutator = self._arena.get_mutator_config()
        mutator.respawn_delay = _NEVER_SECONDS
        self._arena.set_mutator_config(mutator)
        self._spawn_boost = mutator.car_spawn_boost_amount
        self._goal_line_y = mutator.goal_base_threshold_y + mutator.ball_radius
        self._cars = {team: self._arena.add_car(_ROCKETSIM_TEAMS[team], RocketSim.CarConfig.OCTANE)
                      for team in TEAMS}
        self._pads = self._arena.get_boost_pads()
        self._demolished = dict.fromkeys(TEAMS, False)

    @property
    def ball(self) -> RocketSim.BallState:
        """The ball's state."""
        return self._arena.ball.get_state()

    def car(self, team: Team) -> RocketSim.CarState:
        """``team``'s car's state."""
        return self._cars[team].get_state()

    def pads_active(self) -> tuple[bool, ...]:
        """For each boost pad, in RocketSim's order, whether it can be picked up."""
        return tuple(pad.get_state().is_active for pad in self._pads)

    def pad_spots(self) -> tuple[tuple[tuple[float, float, float], bool], ...]:
        """Each boost pad's position (uu) and whether it is big, in RocketSim's order."""
        return tuple((pad.get_pos().as_tuple(), pad.is_big) for pad in self._pads)

    def place_for_kickoff(self, spot: int) -> None:
        """Set up a kickoff: both cars at kickoff spot ``spot`` (mirrored for orange) with the
        spawn boost, the ball at rest at the centre, and every boost pad up."""
        x, y, yaw = RocketSim.RLConst.CAR_SPAWN_LOCATIONS_SOCCAR[spot]
        for team in TEAMS:
            self._place_car(team, x, y, RocketSim.RLConst.CAR_SPAWN_REST_Z, yaw, on_ground=True)
        self._arena.ball.set_state(RocketSim.BallState())
        for pad in self._pads:
            pad.set_state(RocketSim.BoostPadState(is_active=True, cooldown=0.0))

    def remove_ball(self) -> None:
        """Take the ball out of play: at rest behind a corner wall, until the next kickoff."""
        self._arena.ball.set_state(RocketSim.BallState(pos=RocketSim.Vec(*BALL_PARKING_POSITION)))

    def respawn(self, team: Team, spot: int) -> None:
        """Bring ``team``'s car back at respawn spot ``spot`` (mirrored for orange), with the
        spawn boost."""
        x, y, yaw = RocketSim.RLConst.CAR_RESPAWN_LOCATIONS_SOCCAR[spot]
        self._place_car(team, x, y, RocketSim.RLConst.CAR_RESPAWN_Z, yaw, on_ground=False)

    def set_controls(self, team: Team, controls: Controls) -> None:
        """The controls ``team``'s car takes from the next tick on."""
        self._cars[team].set_controls(RocketSim.CarControls(
            throttle=controls.throttle, steer=controls.steer, pitch=controls.pitch,
            yaw=controls.yaw, roll=controls.roll, jump=controls.jump, boost=controls.boost,
            handbrake=controls.handbrake))

    def tick(self) -> TickEvents:
        """Step one tick, and say what happened in it."""
        self._arena.step(1)
        touched, demolished = [], []
        for team in TEAMS:
            state = self._cars[team].get_state()
            hit = state.ball_hit_info
            if hit.is_valid and hit.tick_count_when_hit == self._arena.tick_count - 1:
                touched.append(team)
            if state.is_demoed and not self._demolished[team]:
                demolished.append(team)
            self._demolished[team] = state.is_demoed
        position = self._arena.ball.get_state().pos
        goal = None
        if abs(position.y) > self._goal_line_y:
            goal = Team.BLUE if position.y > 0 else Team.ORANGE
        return TickEvents(goal=goal, touched=tuple(touched), demolished=tuple(demolished),
                          ball_on_ground=position.z <= BALL_ON_GROUND_MAXIMUM_Z)

    def _place_car(self, team: Team, x: float, y: float, z: float, yaw: float,
                   on_ground: bool) -> None:
        if team is Team.ORANGE:
            x, y, yaw = -x, -y, yaw + math.pi
        self._cars[team].set_state(RocketSim.CarState(
            pos=RocketSim.Vec(x, y, z), rot_mat=RocketSim.Angle(yaw, 0, 0).as_rot_mat(),
            boost=self._spawn_boost, is_on_ground=on_ground))
        self._demolished[team] = False
