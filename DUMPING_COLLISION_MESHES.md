# Dumping Rocket League's soccar collision meshes (Windows)

This is a brief for a person, or an agent such as Claude Code, on a Windows PC where Rocket League is installed under the owner's own account. The goal is one folder: the soccar arena's collision meshes as RocketSim reads them, `collision_meshes/soccar/*.cmf`, checked, with a manifest of each file's SHA-256. Nothing else is extracted: no models, textures, sounds or other game files.

## Rules

- Use only a Rocket League install that the owner of this PC licensed. Rocket League's EULA restricts copying and extracting its data; the owner decides whether this dump is allowed for their research, and checks with their institution when unsure.
- The meshes never leave the owner's hands: never commit, push, upload or post them anywhere. They are carried to the research machine privately (USB drive or private storage). Only the manifest (hashes, not data) is committed to the repository.
- Never connect a modified game, a dumping tool or an injected process to Epic's or Psyonix's online services.
- Ask the owner before each step that installs software, and stop and ask before the fallback tool (step 4).

## 1. Prepare

- Find the game's folder, for example `C:\Program Files\Epic Games\rocketleague\` (Epic) or `C:\Program Files (x86)\Steam\steamapps\common\rocketleague\` (Steam). It holds `TAGame\CookedPCConsole\`.
- Note the game's version (the launcher shows it, or the file properties of `Binaries\Win64\RocketLeague.exe`).
- Install Python 3.12 (python.org or `winget install Python.Python.3.12`) and Git (`winget install Git.Git`).
- Clone this repository and install it, which brings the checker and `rocketsim` 2.2.1 (a Windows wheel exists):
  ```powershell
  git clone https://github.com/hyruuk/rocketsim-gym.git
  cd rocketsim-gym
  py -3.12 -m venv .venv
  .venv\Scripts\python -m pip install -e .
  ```

## 2. Dump with the game closed (preferred)

Use [VirxEC/rl_arena_collision_dumper](https://github.com/VirxEC/rl_arena_collision_dumper): it reads the game's files, with the game closed, through [UModel](https://www.gildor.org/en/projects/umodel), and never touches a running game.

- Install what its README asks for (the Rust toolchain from rustup.rs, and UModel), then follow its README to point it at `TAGame\CookedPCConsole` and run it.
- Its README warns that RocketSim may report the meshes' hashes as unknown; that alone is not a failure, the check below says how the meshes play.
- Collect its output so that the soccar meshes sit in `collision_meshes\soccar\*.cmf`. Any `hoops` or `dropshot` folders it writes can stay, but only `soccar` is used.

## 3. Check

```powershell
.venv\Scripts\python -m rocketsim_gym.check_collision_meshes C:\path\to\collision_meshes
```

It prints a report and writes two files into the folder: `manifest.json` (each mesh file's SHA-256) and `check_report.json`. What a good dump looks like:

- 16 files under `soccar`, and `known_rocket_league_meshes` 16: each file's RocketSim hash is one RocketSim lists for Rocket League's soccar arena. Fewer known hashes with sensible bounds and arena numbers can still be a good dump made by a different tool; report it as it is.
- Bounds within about x ±4100, y ±6000 (the goals), z 0 to 2050.
- `arena`: `ball_rest_z` near 93.15, `car_rest_z` near 17 with 4 wheels in contact, each shot scoring once (the shot at the positive-y goal for blue, the other for orange), and `side_wall_drive_highest_z` above 2000 (the car reaches the ceiling).

If the check raises an error, or the numbers are far off, stop and report the output.

## 4. Fallback: dump from the running game (only with the owner's explicit go-ahead)

[ZealanL/RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper) injects into a running `RocketLeague.exe` in free play and writes `collision-meshes\soccar\mesh_0..15.cmf`, whose hashes RocketSim knows. It patches the game's process, so: ask the owner first, run the game offline only, with Easy Anti-Cheat off (Rocket League runs it online), and never go online with the tool or the modified process running. Then run the check of step 3 on its output.

## 5. Hand over

- Zip the folder (`collision_meshes\`, with `manifest.json` and `check_report.json` in it) and give it to the owner for private transfer.
- Report back, in the conversation or a short note: the game version and store (Epic or Steam), the tool used and its commit, and the whole `check_report.json`.
- Do not commit anything to the repository from this machine.
