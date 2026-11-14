"""Every test process has RocketSim loaded with the homemade arena."""

import pytest

from rocketsim_gym.arena_geometry import HOMEMADE_ARENA_FOLDER, MANIFEST_FILE_NAME, read_manifest
from rocketsim_gym.engine import load_rocketsim


@pytest.fixture(scope="session", autouse=True)
def homemade_arena_loaded():
    load_rocketsim(HOMEMADE_ARENA_FOLDER, read_manifest(HOMEMADE_ARENA_FOLDER / MANIFEST_FILE_NAME))


@pytest.fixture(autouse=True)
def settings_kept_apart(tmp_path, monkeypatch):
    """No test reads or writes the player's own settings file."""
    for name in ("XDG_CONFIG_HOME", "APPDATA"):
        monkeypatch.setenv(name, str(tmp_path / "config"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
