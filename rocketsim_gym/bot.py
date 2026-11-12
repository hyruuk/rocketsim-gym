"""The bots: a scripted player at the wheel of each car the agent does not drive.

A team's cars take turns at the ball, as Rocket League players rotate (:func:`turn_order`): the car
nearest the ball goes for it, and the others wait in line behind it, between the ball and their
own goal (:func:`waiting_spot`). The agent's car counts as one of its team's, so its teammates
leave the ball to it while it is nearest; a bot alone on its team always goes.

The car whose turn it is chases the ball: it drives at the ball, aiming a little behind it so that
the touch sends the ball toward the goal it attacks, boosts while it faces its target and
powerslides through sharp turns. A waiting car drives to its spot and stops there. Bots jump only
to come free. Their controls follow from the state, and from one thing a bot remembers
between steps (:class:`StuckWatch`): how long its car has pushed without moving, so that a car
stuck against a goal post backs off, and one stuck on its side jumps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .controls import Controls

#: The back walls' y, where the goal mouths open.
GOAL_LINE_Y = 5120.0
#: Full steering lock for a heading error of 1 / STEERING_GAIN radians or more.
STEERING_GAIN = 5.0
#: Boost while the target is within this many radians of straight ahead.
BOOST_MAXIMUM_HEADING = 0.3
#: Powerslide while the target is more than this many radians off.
POWERSLIDE_MINIMUM_HEADING = 1.5
#: How far behind the ball, from the goal it attacks, the bot aims (uu).
AIM_BEHIND_BALL = 150.0
#: How much farther from the ball a car past it counts, when its team takes turns (uu): past the
#: ball, nearer the goal it attacks than the ball is, it would have to turn round to play it.
PAST_THE_BALL = 1000.0
#: The gap between the ball and each car waiting in line behind it (uu).
WAITING_GAP = 2000.0
#: How far from the centre a waiting spot can be, across and along the field (uu): inside the side
#: walls, and in front of the goal lines.
WAITING_MAXIMUM_X, WAITING_MAXIMUM_Y = 3600.0, 4900.0
#: A waiting car is on its spot within this distance (uu).
ON_THE_SPOT = 150.0
#: A waiting car drives at most this fast per uu still to go (1/s), so that braking stops it on
#: its spot.
ARRIVING_SPEED_PER_DISTANCE = 2.0
#: A waiting car brakes once it goes this much faster than it would arrive at (uu/s).
ARRIVING_SPEED_MARGIN = 200.0
#: A waiting car boosts toward its spot only from this far (uu).
BOOST_MINIMUM_DISTANCE = 2500.0
#: A car slower than this is stopped (uu/s).
STOPPED_SPEED = 100.0
#: A car pushing on with its throttle that long without moving is stuck (s): an upright one backs
#: off, reversing with its wheels turned the other way...
STUCK_SECONDS = 0.75
BACKING_OFF_SECONDS = 0.75
#: ...and one on a wall, its side or its roof jumps and rolls its wheels down, which takes it off
#: the wall and back onto its wheels (on its roof, the jump flips it over).
RIGHTING_SECONDS = 0.5
#: A car is upright while its up direction points at least this far up (its z).
UPRIGHT_MINIMUM_UP_Z = 0.7


@dataclass(frozen=True)
class CarReading:
    """What a bot reads of a car: its position (uu) and velocity (uu/s), which way it faces
    (RocketSim's ``rot_mat.forward``, ``rot_mat.right`` and ``rot_mat.up``), and whether it is
    demolished."""

    position: np.ndarray
    velocity: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    is_demolished: bool = False


@dataclass
class StuckWatch:
    """What a bot remembers of its car between steps: how long it has pushed without moving, and
    what it does to come free, until when."""

    pushing_seconds: float = 0.0
    freeing_seconds: float = 0.0
    freeing_controls: Controls = Controls()

    def controls(self, wanted: Controls, car: CarReading, seconds: float) -> Controls:
        """``wanted``, unless the car is stuck. Then an upright car reverses for
        :data:`BACKING_OFF_SECONDS`, its wheels turned away from where it wanted to go, which
        turns its nose toward there; any other car jumps and rolls its wheels down for
        :data:`RIGHTING_SECONDS`.

        :param wanted: the controls the bot would give the car now.
        :param car: the car, now.
        :param seconds: how long these controls last.
        """
        if self.freeing_seconds > 0:
            self.freeing_seconds -= seconds
            return self.freeing_controls
        if wanted.throttle > 0 and float(np.linalg.norm(car.velocity)) < STOPPED_SPEED:
            self.pushing_seconds += seconds
        else:
            self.pushing_seconds = 0.0
        if self.pushing_seconds < STUCK_SECONDS:
            return wanted
        self.pushing_seconds = 0.0
        if car.up[2] >= UPRIGHT_MINIMUM_UP_Z:
            self.freeing_controls = Controls(throttle=-1.0, steer=-1.0 if wanted.steer > 0 else 1.0)
            self.freeing_seconds = BACKING_OFF_SECONDS - seconds
        else:
            # Rolling right lowers the car's right side.
            self.freeing_controls = Controls(roll=1.0 if car.right[2] > 0 else -1.0, jump=True)
            self.freeing_seconds = RIGHTING_SECONDS - seconds
        return self.freeing_controls


def team_controls(team: Sequence[CarReading], ball_position: np.ndarray,
                  attacked_goal_y: float) -> list[Controls]:
    """The controls a bot would give each of a team's cars.

    :param team: the team's cars; of two as far from the ball, the one listed first goes for it.
    :param ball_position: the ball's position (uu).
    :param attacked_goal_y: the y of the goal line the team attacks.
    """
    controls = []
    for car, place in zip(team, turn_order(team, ball_position, attacked_goal_y), strict=True):
        if place == 0:
            controls.append(ball_chaser(car, ball_position, attacked_goal_y))
        else:
            controls.append(drive_to(car, waiting_spot(place, ball_position, attacked_goal_y)))
    return controls


def turn_order(team: Sequence[CarReading], ball_position: np.ndarray,
               attacked_goal_y: float) -> list[int]:
    """Each car's place in its team's line for the ball: 0 for the car whose turn it is, then 1,
    2... for the cars waiting behind it, nearest the ball first.

    A car past the ball counts as :data:`PAST_THE_BALL` farther, a demolished car comes last, and
    of two cars as far, the one listed first goes first.
    """
    def distance(car: CarReading) -> float:
        if car.is_demolished:
            return math.inf
        is_past = (car.position[1] - ball_position[1]) * math.copysign(1.0, attacked_goal_y) > 0
        return float(np.hypot(*(car.position[:2] - ball_position[:2]))) + PAST_THE_BALL * is_past

    places = [0] * len(team)
    in_line = sorted(range(len(team)), key=lambda index: (distance(team[index]), index))
    for place, index in enumerate(in_line):
        places[index] = place
    return places


def waiting_spot(place: int, ball_position: np.ndarray, attacked_goal_y: float) -> np.ndarray:
    """Where the car ``place`` in line waits (x, y in uu): ``place`` gaps from the ball toward the
    centre of its own goal line, no farther than that, and inside the field."""
    own_goal = np.array([0.0, -attacked_goal_y])
    to_own_goal = own_goal - ball_position[:2]
    distance = float(np.hypot(*to_own_goal))
    spot = own_goal
    if distance > place * WAITING_GAP:
        spot = ball_position[:2] + to_own_goal * (place * WAITING_GAP / distance)
    return np.clip(spot, (-WAITING_MAXIMUM_X, -WAITING_MAXIMUM_Y),
                   (WAITING_MAXIMUM_X, WAITING_MAXIMUM_Y))


def ball_chaser(car: CarReading, ball_position: np.ndarray, attacked_goal_y: float) -> Controls:
    """The controls that chase the ball.

    :param car: the car.
    :param ball_position: the ball's position (uu).
    :param attacked_goal_y: the y of the goal line the car attacks.
    """
    to_goal = np.array([0.0, attacked_goal_y]) - ball_position[:2]
    distance_to_goal = float(np.hypot(*to_goal))
    target = ball_position[:2]
    if distance_to_goal > 1.0:
        target = target - AIM_BEHIND_BALL * to_goal / distance_to_goal
    heading = _heading(target - car.position[:2], car)
    return Controls(throttle=1.0,
                    steer=float(np.clip(STEERING_GAIN * heading, -1.0, 1.0)),
                    boost=abs(heading) < BOOST_MAXIMUM_HEADING,
                    handbrake=abs(heading) > POWERSLIDE_MINIMUM_HEADING)


def drive_to(car: CarReading, spot: np.ndarray) -> Controls:
    """The controls that drive to ``spot`` (x, y in uu), slowing on the way in, and stop there."""
    to_spot = spot - car.position[:2]
    distance = float(np.hypot(*to_spot))
    speed = float(car.velocity @ car.forward)
    if distance < ON_THE_SPOT:
        return Controls(throttle=_braking(speed))
    heading = _heading(to_spot, car)
    steer = float(np.clip(STEERING_GAIN * heading, -1.0, 1.0))
    if abs(heading) > math.pi / 2:
        return Controls(throttle=1.0, steer=steer,
                        handbrake=abs(heading) > POWERSLIDE_MINIMUM_HEADING)
    arriving_speed = ARRIVING_SPEED_PER_DISTANCE * distance
    if speed > arriving_speed + ARRIVING_SPEED_MARGIN:
        return Controls(throttle=-1.0, steer=steer)
    if speed > arriving_speed:
        return Controls(steer=steer)
    is_far_and_ahead = (distance > BOOST_MINIMUM_DISTANCE
                        and abs(heading) < BOOST_MAXIMUM_HEADING)
    return Controls(throttle=1.0, steer=steer, boost=is_far_and_ahead)


def _braking(speed: float) -> float:
    """The throttle against a car going ``speed`` forward (uu/s), which brakes it; none once it
    has stopped."""
    return 0.0 if abs(speed) < STOPPED_SPEED else -math.copysign(1.0, speed)


def _heading(direction: np.ndarray, car: CarReading) -> float:
    """The angle on the ground from where ``car`` faces to ``direction``, positive to its right."""
    return math.atan2(float(direction @ car.right[:2]), float(direction @ car.forward[:2]))
