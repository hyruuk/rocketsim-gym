# rocketsim-gym

Rocket League-style soccar on [RocketSim](https://github.com/ZealanL/RocketSim) physics (the pip package `rocketsim`, mtheall's Python bindings), built to become a Gymnasium env that a person and a model play alike.

RocketSim simulates the game but needs the arena's collision meshes, which come from a Rocket League install. This package ships a homemade arena instead, built from Rocket League's published field dimensions, so it plays anywhere; the real meshes can be dumped on a Windows PC with the game ([DUMPING_COLLISION_MESHES.md](DUMPING_COLLISION_MESHES.md)) and checked with the same tool.

## What is here

- `rocketsim_gym.arena_geometry`: the homemade arena as one triangle mesh (`build_homemade_soccar_arena`), and RocketSim's collision-mesh files: read, write, their SHA-256 manifest, and RocketSim's own mesh hash. RocketSim adds the floor, a ceiling plane and the side walls itself; the mesh is everything else (the curves into the walls and the ceiling, corner and back walls, the ceiling, the goals). `python -m rocketsim_gym.arena_geometry FOLDER` writes it.
- `rocketsim_gym/collision_meshes/homemade/`: the homemade arena's mesh file and its manifest, as that command writes them.
- `rocketsim_gym.engine`: `load_rocketsim(folder, manifest)` loads RocketSim with one folder of meshes per process, after checking every file against its manifest; `new_soccar_arena()` makes an arena from them.
- `python -m rocketsim_gym.check_collision_meshes FOLDER`: describes each mesh file, loads the folder, measures how the arena plays (rest heights, a shot at each goal, a drive up a wall), and writes the folder's manifest and a report.

## Traps

- RocketSim reads its meshes once per process. A file it cannot parse leaves it unusable for the rest of the process, and an arena made before it is loaded reads a `collision_meshes` folder from the working directory; `load_rocketsim` checks first and loads explicitly.
- RocketSim's grid broadphase spends about a second building, first stepping and deleting each arena over a mesh that spans the whole field; `new_soccar_arena` uses Bullet's own broadphase, which costs nothing there.
- A ball whose velocity is exactly zero does not move until something touches it.
- Kickoffs and demolition respawns in RocketSim (`reset_kickoff`, `Car.respawn`, the automatic respawn) draw from a clock-seeded random generator. Set cars and the ball by state instead; done that way, a run is identical in every process.

## Tests

```bash
uv sync
uv run pytest
```
