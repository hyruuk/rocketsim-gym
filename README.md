# rocketsim-gym

Rocket League-style soccar to play against a bot on your own computer, and the same game as a [Gymnasium](https://gymnasium.farama.org) env for AI agents, so that a person and a model play exactly the same game. It runs on [RocketSim](https://github.com/ZealanL/RocketSim), a simulation of Rocket League's physics, on a homemade arena: nothing from Rocket League itself is needed. It is made for [fmri-gym](https://github.com/tomov/fmri-gym), which runs games for people to play in an MRI scanner.

![The kickoff, on ball cam](docs/screenshots/kickoff.png)
![Chasing the ball, the orange bot on the right](docs/screenshots/chasing-the-ball.png)
![Ball cam looking up at a ball in the air](docs/screenshots/ball-in-the-air.png)

## Get started

Two commands: the first installs [uv](https://docs.astral.sh/uv/), which fetches the game and the Python it runs on, apart from everything else on your computer; the second starts a match.

**1. Install uv**, once. On macOS or Linux, in a terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then close the terminal and open a new one, so that it finds `uv`.

**2. Play**, whenever you like, the same command everywhere:

```bash
uvx --from https://github.com/hyruuk/rocketsim-gym/archive/main.zip rocketsim-play
```

A window opens on a 5-minute match against the bot. The first start downloads what the game needs and takes a minute or so; later ones take seconds. If the window opens behind the terminal, click it. Add `--seconds 120` at the end for a shorter match, or `--width 1920 --height 1080` for a bigger window.

It runs on a Mac with Apple silicon (M1 or later, macOS 14 or later), on 64-bit Windows 10 or 11, and on Linux (x86_64 or ARM64), with graphics that support OpenGL 3.3, as any computer from the last ten years does. Intel Macs cannot run it: RocketSim publishes no version for them.

### Controls

The keyboard and mouse follow Rocket League's PC defaults. An Xbox-style controller (Xbox, PlayStation, Switch Pro) works too, plugged in before or during the match; for now its triggers and stick act as full presses.

| Action | Keyboard and mouse | Controller |
|---|---|---|
| Drive / reverse | W / S | right trigger / left trigger |
| Steer | A / D | left stick |
| Tilt the nose down / up, in the air | W / S | left stick |
| Jump (again in the air to flip) | right mouse button | A |
| Boost | left mouse button | B |
| Powerslide, and air roll with the stick | left Shift | X |
| Ball cam on or off | Space | Y |
| Air roll left / right | Q / E | |

The window's title shows the score, the clock and your boost. Esc quits; when the match is over, Enter plays another.

### Updating

`uvx` keeps the version it downloaded. To play the latest one, run once:

```bash
uvx --refresh --from https://github.com/hyruuk/rocketsim-gym/archive/main.zip rocketsim-play
```

### If it does not start

- **RocketSim cannot be installed** (an error naming `rocketsim`): the computer is not one RocketSim supports; see above.
- **"no OpenGL 3.3 context opens to draw frames"**: update the graphics drivers. On Linux, install EGL too; on Debian or Ubuntu: `sudo apt install libegl1 libgl1-mesa-dri`.
- **The controller does nothing**: check that other games see it, then start the game again with it plugged in.

## Status

Early. What works today: a whole 1v1 match on the homemade arena against a scripted ball-chasing bot, with Rocket League's match rules, seen from Rocket League's car cam and ball cam; the play window; and the Gymnasium env behind both. The look is homemade and low-poly. Next, in this order:

1. Rocket League's look and feel: the scoreboard, boost meter and banners on screen, and the camera tuned with Rocket League players.
2. The real arena, from collision meshes dumped from a Rocket League install.
3. Analog steering and throttle.

## Use it from Python

```bash
pip install https://github.com/hyruuk/rocketsim-gym/archive/main.zip
```

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

**Arguments**: `match_seconds`, the regulation time (300); `ticks_per_step`, physics ticks per step at 120 ticks a second (2, so 60 steps a second); `render_mode`, `"rgb_array"` to draw frames; `render_width` and `render_height`, a frame's size (1024x576).

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

**Frames**: with `render_mode="rgb_array"`, `render()` returns the frame of the latest step, as Rocket League's camera behind the blue car sees it: car cam, or ball cam, which the `ball_cam` button switches. The camera hangs 270 uu behind the car and 100 above, looking 3 degrees down, with a 110 degree field of view, as many players set theirs; on a wall, the wall becomes its floor. The camera moves in `step`, and a frame is drawn once however often it is asked for, so drawing changes nothing in the match. Drawing needs OpenGL 3.3, without a window: EGL on Linux, so no display is needed there.

**Logging**: `env.unwrapped.sample()` returns the same fields after every step (`rocketsim_gym.env.SAMPLE_FIELDS`): the phase, clock and score, the ball, each car's state, what happened during the step (a goal, each car's touch and demolition) and the controls each car obeyed. `env.unwrapped.provenance()` names what a match depends on besides its seed and actions: the package versions, the collision meshes' SHA-256s, the tick rates and the OpenGL renderer.

**Replay**: the kickoff and respawn spots come from the env's random generator, seeded by `reset`, and RocketSim's own random kickoffs and respawns are never used. So the same seed and the same actions give the same match, step for step, in any process on the same machine. Floating-point physics is not promised to match across machines.

## The arena

RocketSim simulates Rocket League's physics but needs the arena's collision meshes, which come from a Rocket League install. This package ships a homemade arena instead, built from Rocket League's published field dimensions, so it runs anywhere and holds nothing from the game. Its curves, corners and goal depths are estimates, to be calibrated against the real meshes. Those can be dumped on a Windows PC that has the game ([DUMPING_COLLISION_MESHES.md](DUMPING_COLLISION_MESHES.md)) and checked with this package's tool; they stay with the game's owner and are never redistributed.

## What is here

- `rocketsim_gym.env`: `SoccarEnv`, registered as `RocketSimSoccar-v0`.
- `rocketsim_gym.match`: the match rules, a state machine told what happened each tick and answering what the arena must do.
- `rocketsim_gym.controls`: the buttons and the car controls they make.
- `rocketsim_gym.bot`: the ball-chaser.
- `rocketsim_gym.camera`: car cam and ball cam, the rig behind the car that follows it.
- `rocketsim_gym.renderer`: draws the frames offscreen with OpenGL (moderngl): the arena the physics uses, the ball, the cars and the boost pads.
- `rocketsim-play` (`python -m rocketsim_gym.play`): the play window.
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
git clone https://github.com/hyruuk/rocketsim-gym
cd rocketsim-gym
uv sync
uv run rocketsim-play
uv run pytest
uv run ruff check .
```

The tests check the arena's mesh and how RocketSim plays it (rest heights, drives up every kind of wall onto the ceiling, goals, the goal frame, nothing leaving the arena); the match rules, event by event; each button's controls and which way RocketSim turns on them; the bot; the arena's kickoffs, demolitions and respawns; the env: Gymnasium's own checker, whole matches, the logged fields, and a match replayed identically in a second process; the camera; the frames: where the ball and the cars land on the screen, that drawing changes nothing, and the same pixels in a second process; and the play window's inputs. CI runs them on Linux, macOS and Windows; the frames' tests skip where no OpenGL 3.3 context opens.

## License

MIT, in [LICENSE](LICENSE). RocketSim, by ZealanL, is MIT too; its Python bindings are by mtheall.

Rocket League is a trademark of Psyonix. This project is not affiliated with or endorsed by Psyonix or Epic Games.
