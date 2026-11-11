"""Every test process has RocketSim loaded with the homemade arena."""

import pytest

from rocketsim_gym.arena_geometry import HOMEMADE_ARENA_FOLDER, MANIFEST_FILE_NAME, read_manifest
from rocketsim_gym.engine import load_rocketsim


@pytest.fixture(scope="session", autouse=True)
def homemade_arena_loaded():
    load_rocketsim(HOMEMADE_ARENA_FOLDER, read_manifest(HOMEMADE_ARENA_FOLDER / MANIFEST_FILE_NAME))
