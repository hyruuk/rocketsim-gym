# rocketsim-gym

Rocket League-style soccar as a [Gymnasium](https://gymnasium.farama.org) env, on [RocketSim](https://github.com/ZealanL/RocketSim) physics, built so that a person and a model play the same game. It is made for [fmri-gym](https://github.com/tomov/fmri-gym), which runs games for people to play in an MRI scanner, but it knows nothing about scanners: it is a plain RL env.

RocketSim simulates Rocket League's physics but needs the arena's collision meshes, which come from a Rocket League install. This package ships a homemade arena instead, built from Rocket League's published field dimensions, so it runs anywhere and holds nothing from the game. Its curves, corners and goal depths are estimates, to be calibrated against the real meshes. Those can be dumped on a Windows PC that has the game ([DUMPING_COLLISION_MESHES.md](DUMPING_COLLISION_MESHES.md)) and checked with this package's tool; they stay with the game's owner and are never redistributed.

## Status

Early. What works today: the homemade arena, loading RocketSim with it, and the checker for collision meshes. Next, in this order:

1. The match (kickoffs, the clock, goals, overtime, demolitions) and a scripted ball-chasing opponent, as the Gymnasium env `RocketSimSoccar-v0`.
2. A 3D renderer with Rocket League's car cam and ball cam, a scoreboard and a boost meter.
3. The real arena, from dumped meshes.

## Install

Python 3.10 or later. RocketSim ships wheels only, for macOS 14 or later on Apple Silicon, Linux (x86_64, aarch64) and Windows (x86_64).

```bash
pip install git+https://github.com/hyruuk/rocketsim-gym
```

## Use

```python
import math

import RocketSim as rs

from rocketsim_gym.arena_geometry import HOMEMADE_ARENA_FOLDER, MANIFEST_FILE_NAME, read_manifest
from rocketsim_gym.engine import load_rocketsim, new_soccar_arena

# Once per process: check the homemade arena's files against their manifest, then load them.
load_rocketsim(HOMEMADE_ARENA_FOLDER, read_manifest(HOMEMADE_ARENA_FOLDER / MANIFEST_FILE_NAME))

arena = new_soccar_arena()  # 120 ticks per second
arena.ball.set_state(rs.BallState(pos=rs.Vec(0, 0, 93.15)))
car = arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
car.set_state(rs.CarState(pos=rs.Vec(0, -2000, 17), rot_mat=rs.Angle(math.pi / 2, 0, 0).as_rot_mat()))
car.set_controls(rs.CarControls(throttle=1, boost=True))  # straight at the ball, boosting
arena.step(120 * 4)  # four seconds
print(arena.blue_score)  # 1: the car has pushed the ball into the orange goal
```

The rest is RocketSim's own Python API, from [its bindings](https://github.com/mtheall/RocketSim).

## What is here

- `rocketsim_gym.arena_geometry`: the homemade arena as one triangle mesh (`build_homemade_soccar_arena`), and RocketSim's collision-mesh files: read, write, their SHA-256 manifest, and RocketSim's own mesh hash. RocketSim adds the floor, a ceiling plane and the side walls itself; the mesh is everything else (the curves into the walls and the ceiling, corner and back walls, the ceiling, the goals). `python -m rocketsim_gym.arena_geometry FOLDER` writes it.
- `rocketsim_gym/collision_meshes/homemade/`: the homemade arena's mesh file and its manifest, as that command writes them.
- `rocketsim_gym.engine`: `load_rocketsim(folder, manifest)` loads RocketSim with one folder of meshes per process, after checking every file against its manifest; `new_soccar_arena()` makes an arena from them.
- `python -m rocketsim_gym.check_collision_meshes FOLDER`: describes each mesh file, loads the folder, measures how the arena plays (rest heights, a shot at each goal, a drive up a wall), and writes the folder's manifest and a report.
- [DUMPING_COLLISION_MESHES.md](DUMPING_COLLISION_MESHES.md): the brief for dumping Rocket League's own meshes on a Windows PC with the game, for a person or an agent such as Claude Code.

## Traps

- RocketSim reads its meshes once per process. A file it cannot parse leaves it unusable for the rest of the process, and an arena made before it is loaded reads a `collision_meshes` folder from the working directory; `load_rocketsim` checks first and loads explicitly.
- RocketSim's grid broadphase spends about a second building, first stepping and deleting each arena over a mesh that spans the whole field; `new_soccar_arena` uses Bullet's own broadphase, which costs nothing there.
- A ball whose velocity is exactly zero does not move until something touches it.
- Kickoffs and demolition respawns in RocketSim (`reset_kickoff`, `Car.respawn`, the automatic respawn) draw from a clock-seeded random generator. Set cars and the ball by state instead; done that way, a run is identical in every process.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
```

The tests check the arena's mesh (closed where it must be, every triangle facing the field, the shipped file the same as the builder's) and how RocketSim plays it: rest heights, drives up every kind of wall onto the ceiling, goals, the goal frame, and that nothing leaves the arena. CI runs them on Linux, macOS and Windows.

## License

MIT, in [LICENSE](LICENSE). RocketSim, by ZealanL, is MIT too; its Python bindings are by mtheall.

Rocket League is a trademark of Psyonix. This project is not affiliated with or endorsed by Psyonix or Epic Games.
