"""RocketSim, loaded with one folder of soccar collision meshes per process.

RocketSim reads its collision meshes once per process and keeps them for every arena after.
A file it cannot parse leaves it unusable for the rest of the process, and an arena made
before it is loaded reads whatever ``collision_meshes`` folder the working directory has; so
the meshes are checked here, in full, before RocketSim ever sees them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .arena_geometry import collision_mesh_manifest, read_collision_mesh_file

#: The folder RocketSim was loaded from in this process, once it has been.
_loaded_collision_meshes_folder: Path | None = None


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
    import RocketSim

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


def new_soccar_arena(tick_rate: float = 120.0):
    """A new, empty soccar arena of the loaded meshes, stepping ``tick_rate`` ticks a second.

    It uses Bullet's own broadphase. RocketSim's grid broadphase spends about a second
    building, first stepping and deleting each arena over a mesh that spans the whole field,
    which every reset would pay; Bullet's costs nothing there, and a tick still takes
    microseconds.
    """
    import RocketSim

    if _loaded_collision_meshes_folder is None:
        raise ValueError("RocketSim has no collision meshes loaded yet")
    config = RocketSim.ArenaConfig()
    config.use_custom_broadphase = False
    return RocketSim.Arena(RocketSim.GameMode.SOCCAR, tick_rate=tick_rate, config=config)
