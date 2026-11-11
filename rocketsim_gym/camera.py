"""The camera behind the agent's car: Rocket League's car cam and ball cam.

The camera hangs behind the car and above it, on a rig that follows the car's heading and the
surface it drives on: on a wall, the wall becomes the camera's floor; in the air, the rig eases
back to the world's up. Ball cam turns the rig to face the ball instead of the car's heading.
The rig moves only in :func:`advance_rig`, once per env step, so drawing a frame changes
nothing.

Rocket League's world is left-handed: a car facing +y has its right toward -x. Directions here
keep RocketSim's ``forward``, ``right`` and ``up``, and the screen's right is the car's right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

WORLD_UP = (0.0, 0.0, 1.0)
#: How fast the rig follows the car's heading, and the up of the surface under it, or the
#: world's up in the air: the seconds to cover about two thirds of the way.
TURN_SECONDS = 0.08
SURFACE_UP_SECONDS = 0.12
AIR_UP_SECONDS = 0.6
#: At stiffness 0 the camera trails the car by this many seconds of its motion, so that it pulls
#: back at speed; at stiffness 1 it does not trail at all.
MOST_TRAIL_SECONDS = 0.05
#: A car that moves further than this in one step was placed (a kickoff, a respawn): the
#: camera jumps with it.
PLACED_DISTANCE = 500.0
#: A switch between car cam and ball cam takes this long at transition speed 1.
TRANSITION_SECONDS = 0.25
#: Ball cam looks up at a high ball by swinging the camera around the car, which keeps the car
#: at the bottom of the screen, up to this many degrees (more would take the camera under the
#: car), then by tilting it, up to this many more.
BALL_CAM_SWING_LIMIT = 20.0
BALL_CAM_TILT_LIMIT = 30.0


@dataclass(frozen=True)
class CameraSettings:
    """Rocket League's camera settings, in its units and within the ranges its settings menu
    allows; a player can bring their own. The defaults are values many players use, with the
    camera held rigidly behind the car.

    :ivar field_of_view: horizontal, in degrees (60 to 110).
    :ivar distance: behind the car (100 to 400 uu).
    :ivar height: above the car (40 to 200 uu).
    :ivar angle: the car cam's pitch, in degrees (-15 to 0); negative looks down.
    :ivar stiffness: how rigidly the camera follows the car (0 to 1); below 1 it trails the
        car, pulling back at speed.
    :ivar transition_speed: how fast it swings between car cam and ball cam (1 to 2).
    :raises ValueError: a setting is not a number in its range.
    """

    field_of_view: float = 110.0
    distance: float = 270.0
    height: float = 100.0
    angle: float = -3.0
    stiffness: float = 1.0
    transition_speed: float = 1.0

    def __post_init__(self) -> None:
        for name, (lowest, highest) in _SETTING_RANGES.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not lowest <= value <= highest:
                raise ValueError(f"camera {name} {value!r}: a number from {lowest} to {highest}")


_SETTING_RANGES = {"field_of_view": (60, 110), "distance": (100, 400), "height": (40, 200),
                   "angle": (-15, 0), "stiffness": (0, 1), "transition_speed": (1, 2)}


@dataclass(frozen=True)
class CameraRig:
    """The camera's moving state.

    :ivar anchor: the point the camera hangs behind: the car, or trailing it below full
        stiffness.
    :ivar forward: the rig's heading, square to ``up``.
    :ivar up: the rig's up.
    :ivar ball_cam: how far the rig has swung to ball cam: 0 is car cam, 1 ball cam.
    """

    anchor: tuple[float, float, float]
    forward: tuple[float, float, float]
    up: tuple[float, float, float]
    ball_cam: float


@dataclass(frozen=True)
class Camera:
    """Where a frame is drawn from.

    :ivar position: the eye (uu).
    :ivar forward: the direction it looks.
    :ivar up: the screen's up, square to ``forward``.
    :ivar field_of_view: horizontal, in degrees.
    """

    position: tuple[float, float, float]
    forward: tuple[float, float, float]
    up: tuple[float, float, float]
    field_of_view: float


def start_rig(car_position: np.ndarray, car_forward: np.ndarray, car_up: np.ndarray,
              ball_cam: bool) -> CameraRig:
    """A rig settled behind a car on the ground: heading where it heads."""
    up = np.array(WORLD_UP)
    forward = _flatten(car_forward, up)
    if forward is None:
        forward = _flatten(car_up, up)
    return CameraRig(tuple(np.asarray(car_position, float)), tuple(forward), WORLD_UP,
                     1.0 if ball_cam else 0.0)


def advance_rig(rig: CameraRig, car_position: np.ndarray, car_forward: np.ndarray,
                car_up: np.ndarray, car_on_ground: bool, ball_position: np.ndarray | None,
                ball_cam: bool, seconds: float, settings: CameraSettings) -> CameraRig:
    """The rig ``seconds`` later, following the car and, on ball cam, turning to the ball.

    :param ball_position: the ball, or None while it is out of play.
    """
    up = np.array(rig.up)
    target_up = np.asarray(car_up) if car_on_ground else np.array(WORLD_UP)
    up = _normalized(up + (target_up - up) * _approach(seconds, SURFACE_UP_SECONDS
                                                        if car_on_ground else AIR_UP_SECONDS))
    forward = _flatten(np.array(rig.forward), up)
    heading = _flatten(car_forward, up)
    if forward is None:
        forward = heading if heading is not None else _flatten(np.eye(3)[np.argmin(abs(up))], up)
    if heading is None:
        heading = forward
    step = seconds * settings.transition_speed / TRANSITION_SECONDS
    amount = rig.ball_cam + step if ball_cam else rig.ball_cam - step
    amount = 1.0 if amount > 1.0 - 1e-9 else 0.0 if amount < 1e-9 else amount  # no rounding residue
    to_ball = None if ball_position is None else _flatten(ball_position - car_position, up)
    if to_ball is not None and amount > 0.0:
        blended = _flatten(heading * (1.0 - amount) + to_ball * amount, up)
        heading = to_ball if blended is None else blended
    turned = _flatten(forward + (heading - forward) * _approach(seconds, TURN_SECONDS), up)
    if turned is None:
        turned = heading
    anchor = np.array(rig.anchor)
    trail_seconds = (1.0 - settings.stiffness) * MOST_TRAIL_SECONDS
    if trail_seconds == 0.0 or np.linalg.norm(car_position - anchor) > PLACED_DISTANCE:
        anchor = np.asarray(car_position, float)
    else:
        anchor = anchor + (car_position - anchor) * _approach(seconds, trail_seconds)
    return CameraRig(tuple(anchor), tuple(turned), tuple(up), amount)


def camera_view(rig: CameraRig, car_position: np.ndarray, ball_position: np.ndarray | None,
                settings: CameraSettings) -> Camera:
    """The camera on ``rig`` behind the car at ``car_position``; on ball cam, looking up at a
    ball above the rig's horizon."""
    forward, up = np.array(rig.forward), np.array(rig.up)
    swing = tilt = 0.0
    if ball_position is not None and rig.ball_cam > 0.0:
        offset = ball_position - car_position
        elevation = math.atan2(float(offset @ up), max(float(offset @ forward), 1.0))
        elevation = max(0.0, elevation) * rig.ball_cam
        swing = min(elevation, math.radians(BALL_CAM_SWING_LIMIT))
        tilt = min(elevation - swing, math.radians(BALL_CAM_TILT_LIMIT))
    forward, up = _pitched(forward, up, swing)
    position = np.array(rig.anchor) - forward * settings.distance + up * settings.height
    look, screen_up = _pitched(forward, up, math.radians(settings.angle) + tilt)
    return Camera(tuple(position), tuple(look), tuple(screen_up), settings.field_of_view)


def _pitched(forward: np.ndarray, up: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """``forward`` and ``up`` turned ``angle`` radians nose-up about their right."""
    return (forward * math.cos(angle) + up * math.sin(angle),
            up * math.cos(angle) - forward * math.sin(angle))


def _approach(seconds: float, time_constant: float) -> float:
    """The share of the way an exponential follower covers in ``seconds``."""
    return 1.0 - math.exp(-seconds / time_constant)


def _normalized(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _flatten(vector: np.ndarray, up: np.ndarray) -> np.ndarray | None:
    """``vector`` laid flat on the plane square to ``up``, as a direction; None when it points
    along ``up``."""
    flat = np.asarray(vector, float) - (np.asarray(vector, float) @ up) * up
    length = float(np.linalg.norm(flat))
    return None if length < 1e-6 else flat / length
