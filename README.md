# rocketsim-gym

Rocket League-style soccar as a [Gymnasium](https://gymnasium.farama.org) env, on [RocketSim](https://github.com/ZealanL/RocketSim) physics, built so that a person and a model play the same game. It is made for [fmri-gym](https://github.com/tomov/fmri-gym), which runs games for people to play in an MRI scanner, but it knows nothing about scanners: it is a plain RL env.

RocketSim simulates Rocket League's physics but needs the arena's collision meshes, which come from a Rocket League install. This package ships a homemade arena instead, built from Rocket League's published field dimensions, so it runs anywhere and holds nothing from the game. Its curves, corners and goal depths are estimates, to be calibrated against the real meshes. Those can be dumped on a Windows PC that has the game ([DUMPING_COLLISION_MESHES.md](DUMPING_COLLISION_MESHES.md)) and checked with this package's tool; they stay with the game's owner and are never redistributed.

## Status

Early. What works today: the env `RocketSimSoccar-v0`, a whole 1v1 match on the homemade arena against a scripted ball-chasing bot, with Rocket League's match rules. It does not draw yet. Next, in this order:

1. A 3D renderer with Rocket League's car cam and ball cam, a scoreboard and a boost meter.
2. The real arena, from dumped meshes.

## Install

Python 3.10 or later. RocketSim ships wheels only, for macOS 14 or later on Apple Silicon, Linux (x86_64, aarch64) and Windows (x86_64).

```bash
pip install git+https://github.com/hyruuk/rocketsim-gym
```

## Use

```python
import gymnasium
import rocketsim_gym  # registers RocketSimSoccar-v0

env = gymnasium.make("RocketSimSoccar-v0", match_seconds=60)
observation, info = env.reset(seed=0)
terminated = False
while not terminated:
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(info)  # {'phase': 'ended', 'blue_score': ..., 'orange_score': ...}
```

## The env

The agent drives the blue car; a scripted ball-chaser drives the orange one. The bot drives at the ball, aiming a little behind it so that its touch sends the ball toward blue's goal, boosts while it faces the ball and powerslides through sharp turns; it never jumps.

The match follows Rocket League's rules: a 3 s countdown holds everything still before each kickoff; the clock starts at the kickoff's first touch; a goal pauses play for 3 s, the ball gone and the cars free to drive, before the next kickoff; when the clock reaches zero, play goes on until the ball touches the ground; a tie goes to overtime, which the next goal wins; a demolished car comes back 3 s later.

**Arguments**: `match_seconds`, the regulation time (300); `ticks_per_step`, physics ticks per step at 120 ticks a second (2, so 60 steps a second).

**Action**: twelve buttons, each held or not (`MultiBinary(12)`), the way a controller's digital inputs are, sticks and triggers at full deflection. The car obeys them except in a countdown and while it is demolished.

| Index | Button | Car control |
|---|---|---|
| 0 | throttle | throttle +1 |
| 1 | reverse | throttle -1 (with throttle: none) |
| 2 | stick_left | steer -1, and in the air yaw -1 (roll -1 with powerslide) |
| 3 | stick_right | steer +1, and in the air yaw +1 (roll +1 with powerslide) |
| 4 | stick_up | pitch -1: nose down |
| 5 | stick_down | pitch +1: nose up |
| 6 | jump | jump |
| 7 | boost | boost |
| 8 | powerslide | handbrake on the ground; in the air, the stick rolls |
| 9 | ball_cam | no control: switches the camera between car and ball when pressed |
| 10 | air_roll_left | roll -1 |
| 11 | air_roll_right | roll +1 |

**Observation**: 87 values, in Rocket League's units (uu, uu/s, rad/s).

| Index | Values |
|---|---|
| 0-8 | the ball: position, velocity, angular velocity |
| 9-28 | the blue car: position, velocity, angular velocity, forward direction, up direction, boost (0-100), on the ground, can jump or flip, supersonic, demolished |
| 29-48 | the orange car, the same |
| 49-82 | each boost pad, in RocketSim's order: 1 while it can be picked up |
| 83-86 | seconds left in regulation, blue's goals, orange's goals, 1 in overtime |

**Reward**: +1 for each blue goal, -1 for each orange goal. The episode terminates when the match ends; `info` holds the phase and the score.

**Logging**: `env.unwrapped.sample()` returns the same fields after every step (`rocketsim_gym.env.SAMPLE_FIELDS`): the phase, clock and score, the ball, each car's state, what happened during the step (a goal, each car's touch and demolition) and the controls each car obeyed. `env.unwrapped.provenance()` names what a match depends on besides its seed and actions: the package versions, the collision meshes' SHA-256s and the tick rates.

**Replay**: the kickoff and respawn spots come from the env's random generator, seeded by `reset`, and RocketSim's own random kickoffs and respawns are never used. So the same seed and the same actions give the same match, step for step, in any process on the same machine. Floating-point physics is not promised to match across machines.

## What is here

- `rocketsim_gym.env`: `SoccarEnv`, registered as `RocketSimSoccar-v0`.
- `rocketsim_gym.match`: the match rules, a state machine told what happened each tick and answering what the arena must do.
- `rocketsim_gym.controls`: the buttons and the car controls they make.
- `rocketsim_gym.bot`: the ball-chaser.
- `rocketsim_gym.engine`: `load_rocketsim(folder, manifest)` loads RocketSim with one folder of meshes per process, after checking every file against its manifest; `SoccarArena` is the 1v1 arena, placed only by setting state and stepped a tick at a time, each tick's goal, touches and demolitions reported.
- `rocketsim_gym.arena_geometry`: the homemade arena as one triangle mesh (`build_homemade_soccar_arena`), and RocketSim's collision-mesh files: read, write, their SHA-256 manifest, and RocketSim's own mesh hash. RocketSim adds the floor, a ceiling plane and the side walls itself; the mesh is everything else (the curves into the walls and the ceiling, corner and back walls, the ceiling, the goals). `python -m rocketsim_gym.arena_geometry FOLDER` writes it, as `rocketsim_gym/collision_meshes/homemade/` holds it.
- `python -m rocketsim_gym.check_collision_meshes FOLDER`: describes each mesh file, loads the folder, measures how the arena plays (rest heights, a shot at each goal, a drive up a wall), and writes the folder's manifest and a report.
- [DUMPING_COLLISION_MESHES.md](DUMPING_COLLISION_MESHES.md): the brief for dumping Rocket League's own meshes on a Windows PC with the game, for a person or an agent such as Claude Code.

## Traps

- RocketSim reads its meshes once per process. A file it cannot parse leaves it unusable for the rest of the process, and an arena made before it is loaded reads a `collision_meshes` folder from the working directory; `load_rocketsim` checks first and loads explicitly.
- RocketSim's grid broadphase spends about a second building, first stepping and deleting each arena over a mesh that spans the whole field; `new_soccar_arena` uses Bullet's own broadphase, which costs nothing there.
- A ball whose velocity is exactly zero does not move until something touches it: the kickoff ball waits for its first touch, and the ball out of play after a goal waits behind a corner wall.
- Kickoffs and demolition respawns in RocketSim (`reset_kickoff`, `Car.respawn`, the automatic respawn) draw from a clock-seeded random generator. `SoccarArena` sets cars and the ball by state instead and turns the automatic respawn off.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
```

The tests check the arena's mesh and how RocketSim plays it (rest heights, drives up every kind of wall onto the ceiling, goals, the goal frame, nothing leaving the arena); the match rules, event by event; each button's controls and which way RocketSim turns on them; the bot; the arena's kickoffs, demolitions and respawns; and the env: Gymnasium's own checker, whole matches, the logged fields, and a match replayed identically in a second process. CI runs them on Linux, macOS and Windows.

## License

MIT, in [LICENSE](LICENSE). RocketSim, by ZealanL, is MIT too; its Python bindings are by mtheall.

Rocket League is a trademark of Psyonix. This project is not affiliated with or endorsed by Psyonix or Epic Games.
