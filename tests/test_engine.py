"""The 1v1 arena: kickoffs, the ball out of play, demolitions and respawns, and each tick's
events."""

import math

import pytest
import RocketSim as rs

from rocketsim_gym.controls import Controls
from rocketsim_gym.engine import (BALL_PARKING_POSITION, KICKOFF_SPOT_COUNT, RESPAWN_SPOT_COUNT,
                                  SoccarArena)
from rocketsim_gym.match import TEAMS, Team

FULL_SPEED = Controls(throttle=1, boost=True)


def _ticks(arena, count):
    return [arena.tick() for _ in range(count)]


@pytest.mark.parametrize("spot", range(KICKOFF_SPOT_COUNT))
def test_a_kickoff_faces_mirrored_cars_at_a_ball_that_waits_for_a_touch(spot):
    arena = SoccarArena()
    arena.place_for_kickoff(spot)
    blue, orange = arena.car(Team.BLUE), arena.car(Team.ORANGE)
    assert (orange.pos.x, orange.pos.y) == pytest.approx((-blue.pos.x, -blue.pos.y))
    assert blue.pos.y < 0 and blue.boost == pytest.approx(100 / 3, abs=0.01)
    for car in (blue, orange):
        assert car.rot_mat.forward.x * -car.pos.x + car.rot_mat.forward.y * -car.pos.y > 0
    events = _ticks(arena, 240)
    assert not any(event.touched or event.goal or event.demolished for event in events)
    assert arena.ball.pos.as_tuple() == pytest.approx((0, 0, rs.RLConst.BALL_REST_Z))
    assert all(arena.car(team).wheels_with_contact == (True,) * 4 for team in TEAMS)
    assert all(arena.pads_active()) and len(arena.pads_active()) == 34


def test_a_touch_is_an_event_of_the_tick_it_happens_in():
    arena = SoccarArena()
    arena.place_for_kickoff(4)
    arena.set_controls(Team.BLUE, FULL_SPEED)
    for _ in range(360):
        events = arena.tick()
        if events.touched:
            break
    assert events.touched == (Team.BLUE,)
    assert arena.ball.vel.y > 500


@pytest.mark.parametrize("direction, team", [(1, Team.BLUE), (-1, Team.ORANGE)])
def test_a_goal_is_an_event_for_the_team_that_scored(direction, team):
    arena = SoccarArena()
    arena.place_for_kickoff(0)
    arena._arena.ball.set_state(rs.BallState(pos=rs.Vec(0, direction * 4500, 93.15),
                                             vel=rs.Vec(0, direction * 2000, 0)))
    goals = [event.goal for event in _ticks(arena, 120) if event.goal]
    assert goals and set(goals) == {team}


def test_a_removed_ball_stays_out_of_every_cars_reach():
    arena = SoccarArena()
    arena.place_for_kickoff(0)
    arena.remove_ball()
    arena.set_controls(Team.BLUE, Controls(throttle=1, boost=True, steer=0.2))
    arena.set_controls(Team.ORANGE, Controls(throttle=1, steer=-0.4, jump=True))
    events = _ticks(arena, 3 * 120)
    assert not any(event.touched or event.goal or event.ball_on_ground for event in events)
    assert arena.ball.pos.as_tuple() == pytest.approx(BALL_PARKING_POSITION)


def _demolish_orange(arena):
    arena._cars[Team.BLUE].set_state(rs.CarState(
        pos=rs.Vec(0, -1500, 17), vel=rs.Vec(0, 2290, 0),
        rot_mat=rs.Angle(math.pi / 2, 0, 0).as_rot_mat(), boost=100))
    arena._cars[Team.ORANGE].set_state(rs.CarState(pos=rs.Vec(0, 0, 17)))
    arena.set_controls(Team.BLUE, FULL_SPEED)
    return [event.demolished for event in _ticks(arena, 120) if event.demolished]


@pytest.mark.parametrize("spot", range(RESPAWN_SPOT_COUNT))
def test_a_demolished_car_stays_out_until_respawned_at_a_spot(spot):
    arena = SoccarArena()
    arena.place_for_kickoff(0)
    assert _demolish_orange(arena) == [(Team.ORANGE,)]
    _ticks(arena, 5 * 120)
    assert arena.car(Team.ORANGE).is_demoed
    arena.respawn(Team.ORANGE, spot)
    x, y, _ = rs.RLConst.CAR_RESPAWN_LOCATIONS_SOCCAR[spot]
    _ticks(arena, 120)
    car = arena.car(Team.ORANGE)
    assert not car.is_demoed and car.wheels_with_contact == (True,) * 4
    assert (car.pos.x, car.pos.y) == pytest.approx((-x, -y), abs=5)
    assert _demolish_orange(arena) == [(Team.ORANGE,)]
