"""The match simulation: the cars clients drive obey the controls they give, bots drive the
others, and the same generator state and controls play the same match."""

import numpy as np
import pytest

from rocketsim_gym.controls import NO_CONTROLS, Controls
from rocketsim_gym.match import COUNTDOWN_TICKS, TICKS_PER_SECOND, Car, Team
from rocketsim_gym.simulation import MatchSimulation

BLUE1, BLUE2, ORANGE1, ORANGE2 = (Car(Team.BLUE, 1), Car(Team.BLUE, 2), Car(Team.ORANGE, 1),
                                  Car(Team.ORANGE, 2))
TICKS_PER_STEP = 2
COUNTDOWN_STEPS = COUNTDOWN_TICKS // TICKS_PER_STEP


def _simulation(team_size=2, client_cars=(BLUE1, ORANGE1), seed=3):
    return MatchSimulation(team_size, 300 * TICKS_PER_SECOND, TICKS_PER_STEP, client_cars,
                           np.random.default_rng(seed))


def test_client_cars_are_cars_of_the_match():
    with pytest.raises(ValueError, match="client cars"):
        _simulation(team_size=1, client_cars=(BLUE2,))


def test_a_step_takes_controls_for_exactly_the_client_cars():
    simulation = _simulation()
    with pytest.raises(ValueError, match="clients drive"):
        simulation.step({BLUE1: NO_CONTROLS})
    with pytest.raises(ValueError, match="clients drive"):
        simulation.step({BLUE1: NO_CONTROLS, ORANGE1: NO_CONTROLS, BLUE2: NO_CONTROLS})


def test_client_cars_obey_their_controls_and_bots_drive_the_others():
    simulation = _simulation()
    full_throttle = Controls(throttle=1.0)
    for _ in range(COUNTDOWN_STEPS + 30):
        simulation.step({BLUE1: full_throttle, ORANGE1: NO_CONTROLS})
    snapshot = simulation.snapshot()
    assert snapshot.step == COUNTDOWN_STEPS + 30
    assert snapshot.car(BLUE1).controls == full_throttle
    assert snapshot.car(ORANGE1).controls == NO_CONTROLS
    assert snapshot.car(BLUE2).controls != NO_CONTROLS
    assert snapshot.car(ORANGE2).controls != NO_CONTROLS


def test_the_same_generator_state_and_controls_play_the_same_match():
    def play(seed):
        simulation, controls = _simulation(seed=seed), np.random.default_rng(1)
        packed = [simulation.snapshot().pack()]
        for _ in range(600):
            throttle, steer = controls.uniform(-1, 1, 2)
            simulation.step({BLUE1: Controls(throttle=throttle, steer=steer),
                             ORANGE1: Controls(throttle=1.0, boost=bool(controls.integers(2)))})
            packed.append(simulation.snapshot().pack())
        return packed

    assert play(5) == play(5)
    assert len({play(seed)[0] for seed in range(6)}) > 1
