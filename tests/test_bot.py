"""The bots: which way the ball-chaser steers, whose turn it is at the ball, where the others wait,
and that a stuck car comes free."""

import math

import numpy as np
import pytest
import RocketSim as rs

from rocketsim_gym.bot import (BACKING_OFF_SECONDS, GOAL_LINE_Y, ON_THE_SPOT, PAST_THE_BALL,
                               RIGHTING_SECONDS, STUCK_SECONDS, WAITING_GAP, WAITING_MAXIMUM_Y,
                               CarReading, StuckWatch, ball_chaser, drive_to, team_controls,
                               turn_order, waiting_spot)
from rocketsim_gym.controls import Controls
from rocketsim_gym.engine import KICKOFF_SPOT_COUNT, SoccarArena
from rocketsim_gym.match import Car, Team

STEP_SECONDS = 1 / 60
BLUE, ORANGE = Car(Team.BLUE, 1), Car(Team.ORANGE, 1)


def reading(position, forward=(0.0, 1.0, 0.0), velocity=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0),
            is_demolished=False):
    """A car as a bot reads it. Rocket League's coordinates are left-handed: facing +y, a car's
    right is -x."""
    forward, up = np.array(forward, float), np.array(up, float)
    return CarReading(np.array(position, float), np.array(velocity, float), forward,
                      np.cross(up, forward), up, is_demolished)


def arena_reading(arena, car):
    state = arena.car(car)
    axes = state.rot_mat
    return CarReading(state.pos.as_numpy(), state.vel.as_numpy(), axes.forward.as_numpy(),
                      axes.right.as_numpy(), axes.up.as_numpy(), state.is_demoed)


CENTRE = reading((0.0, 0.0, 17.0))


def _chase(ball_position):
    return ball_chaser(CENTRE, np.array(ball_position, float), -GOAL_LINE_Y)


def test_a_cars_right_is_minus_x_when_it_faces_plus_y():
    arena = SoccarArena()
    arena.place_for_kickoff((4,))
    assert arena.car(BLUE).rot_mat.right.as_numpy() == pytest.approx(CENTRE.right, abs=1e-6)


def test_the_chaser_drives_straight_at_a_ball_ahead_boosting():
    controls = _chase([0, 1500, 93])
    assert controls.throttle == 1 and controls.steer == pytest.approx(0)
    assert controls.boost and not controls.handbrake


@pytest.mark.parametrize("ball_x, side", [(-800, 1), (800, -1)], ids=["right", "left"])
def test_the_chaser_steers_toward_a_ball_to_one_side(ball_x, side):
    controls = _chase([ball_x, 800, 93])
    assert np.sign(controls.steer) == side and not controls.boost


def test_the_chaser_powerslides_toward_a_ball_behind_it():
    controls = _chase([300, -1500, 93])
    assert controls.handbrake and not controls.boost and abs(controls.steer) == 1


@pytest.mark.parametrize("spot", range(KICKOFF_SPOT_COUNT))
def test_the_chaser_reaches_the_ball_from_every_kickoff_spot(spot):
    arena = SoccarArena()
    arena.place_for_kickoff((spot,))
    for _ in range(4 * 120):
        arena.set_controls(ORANGE, ball_chaser(arena_reading(arena, ORANGE),
                                               arena.ball.pos.as_numpy(), -GOAL_LINE_Y))
        if ORANGE in arena.tick().touched:
            return
    pytest.fail("the bot did not touch the ball within 4 s")


BALL = np.array([0.0, 0.0, 93.0])


def test_the_car_nearest_the_ball_goes_and_the_others_line_up_by_distance():
    team = [reading((0, -3000, 17)), reading((0, -1000, 17)), reading((0, -2000, 17))]
    assert turn_order(team, BALL, GOAL_LINE_Y) == [2, 0, 1]


def test_a_car_past_the_ball_counts_as_farther():
    past = reading((0, 1000, 17))
    behind = reading((0, -1000 - PAST_THE_BALL / 2, 17))
    assert turn_order([past, behind], BALL, GOAL_LINE_Y) == [1, 0]
    assert turn_order([past, behind], BALL, -GOAL_LINE_Y) == [0, 1]


def test_a_demolished_car_comes_last_and_a_tie_goes_to_the_car_listed_first():
    left, right = reading((-2048, -2560, 17)), reading((2048, -2560, 17))
    assert turn_order([left, right], BALL, GOAL_LINE_Y) == [0, 1]
    assert turn_order([right, left], BALL, GOAL_LINE_Y) == [0, 1]
    demolished = reading((0, -100, 17), is_demolished=True)
    assert turn_order([demolished, left], BALL, GOAL_LINE_Y) == [1, 0]


@pytest.mark.parametrize("attacked_goal_y", [GOAL_LINE_Y, -GOAL_LINE_Y], ids=["blue", "orange"])
def test_the_cars_waiting_line_up_between_the_ball_and_their_own_goal(attacked_goal_y):
    ball = np.array([1000.0, 0.0, 93.0])
    own_goal = np.array([0.0, -attacked_goal_y])
    first, second = (waiting_spot(place, ball, attacked_goal_y) for place in (1, 2))
    for place, spot in ((1, first), (2, second)):
        assert np.hypot(*(spot - ball[:2])) == pytest.approx(place * WAITING_GAP)
        toward_goal = (own_goal - ball[:2]) / np.hypot(*(own_goal - ball[:2]))
        assert (spot - ball[:2]) @ toward_goal == pytest.approx(place * WAITING_GAP)


def test_a_car_waits_no_farther_than_its_own_goal_and_inside_the_field():
    ball = np.array([3000.0, -4000.0, 93.0])
    spot = waiting_spot(2, ball, GOAL_LINE_Y)
    assert spot == pytest.approx((0.0, -WAITING_MAXIMUM_Y))


def test_one_car_a_team_chases_the_ball_and_the_others_drive_to_their_spots():
    team = [reading((0, -3000, 17)), reading((500, -1000, 17)), reading((0, -2000, 17))]
    controls = team_controls(team, BALL, GOAL_LINE_Y)
    assert controls[1] == ball_chaser(team[1], BALL, GOAL_LINE_Y)
    assert controls[2] == drive_to(team[2], waiting_spot(1, BALL, GOAL_LINE_Y))
    assert controls[0] == drive_to(team[0], waiting_spot(2, BALL, GOAL_LINE_Y))


def test_a_lone_bot_always_chases():
    past = reading((0, 3000, 17))
    assert team_controls([past], BALL, GOAL_LINE_Y) == [ball_chaser(past, BALL, GOAL_LINE_Y)]


def test_a_waiting_car_drives_to_its_spot_and_stops_there():
    arena = SoccarArena()
    arena.place_for_kickoff((4,))
    spot = np.array([1500.0, -1000.0])
    for _ in range(6 * 120):
        arena.set_controls(BLUE, drive_to(arena_reading(arena, BLUE), spot))
        arena.tick()
    car = arena.car(BLUE)
    assert np.hypot(*(car.pos.as_numpy()[:2] - spot)) < ON_THE_SPOT
    assert np.linalg.norm(car.vel.as_numpy()) < 150


def _watched(watch, wanted, car, steps):
    return [watch.controls(wanted, car, STEP_SECONDS) for _ in range(steps)]


def test_an_upright_car_that_pushes_without_moving_backs_off_then_tries_again():
    watch, still, wanted = StuckWatch(), reading((0, 0, 17)), Controls(throttle=1, steer=0.5)
    pushing = round(STUCK_SECONDS / STEP_SECONDS)
    backing_off = round(BACKING_OFF_SECONDS / STEP_SECONDS)
    controls = _watched(watch, wanted, still, pushing - 1 + backing_off + 1)
    assert controls[:pushing - 1] == [wanted] * (pushing - 1)
    assert controls[pushing - 1:-1] == [Controls(throttle=-1, steer=-1)] * backing_off
    assert controls[-1] == wanted


def test_a_car_that_moves_or_does_not_push_is_never_stuck():
    moving = reading((0, 0, 17), velocity=(0, 400, 0))
    still = reading((0, 0, 17))
    for car, wanted in ((moving, Controls(throttle=1)), (still, Controls())):
        assert set(_watched(StuckWatch(), wanted, car, 300)) == {wanted}


def test_a_car_stuck_on_its_side_jumps_and_rolls_its_wheels_down():
    on_its_left_side = reading((0, 0, 44), up=(1, 0, 0))
    assert on_its_left_side.right[2] > 0
    controls = _watched(StuckWatch(), Controls(throttle=1), on_its_left_side,
                        round((STUCK_SECONDS + RIGHTING_SECONDS) / STEP_SECONDS))
    assert controls[-1] == Controls(roll=1, jump=True)


def _drive_under_watch(arena, ball, seconds):
    """Drive the blue car at ``ball`` under a stuck watch for ``seconds``: its state each step."""
    watch, states = StuckWatch(), []
    for _ in range(round(seconds / STEP_SECONDS)):
        car = arena_reading(arena, BLUE)
        arena.set_controls(BLUE, watch.controls(ball_chaser(car, ball, GOAL_LINE_Y), car,
                                                STEP_SECONDS))
        arena.tick()
        arena.tick()
        states.append(arena.car(BLUE))
    return states


def test_a_car_pinned_against_a_goal_post_backs_off():
    # The post's side of the curve into the back wall is upright: a car driving at it from the
    # goal mouth pushes against it.
    arena = SoccarArena()
    arena.place_for_kickoff((4,))
    arena._rocketsim_cars[BLUE].set_state(rs.CarState(pos=rs.Vec(819, -5100, 17),
                                                      rot_mat=rs.Angle(0, 0, 0).as_rot_mat()))
    states = _drive_under_watch(arena, np.array([2500.0, -4000.0, 93.0]), 2)
    assert max(np.hypot(state.pos.x - 819, state.pos.y + 5100) for state in states) > 200


def test_a_car_on_its_side_in_a_goal_corner_comes_back_onto_its_wheels():
    arena = SoccarArena()
    arena.place_for_kickoff((4,))
    on_its_side = rs.Angle(-math.pi / 2, 0, math.pi / 2).as_rot_mat()
    arena._rocketsim_cars[BLUE].set_state(rs.CarState(pos=rs.Vec(-875, -5925, 44),
                                                      rot_mat=on_its_side))
    car = _drive_under_watch(arena, np.array([0.0, 3000.0, 93.0]), 3)[-1]
    assert car.rot_mat.up.z > 0.9 and np.hypot(car.pos.x + 875, car.pos.y + 5925) > 150
