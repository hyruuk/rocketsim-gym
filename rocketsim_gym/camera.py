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
#: Ball cam looks up at a high ball by swinging the camera around the car, which keeps the car
#: at the bottom of the screen, up to this many degrees (more would take the camera under the
#: car), then by tilting it, up to this many more.
BALL_CAM_SWING_LIMIT = 20.0
BALL_CAM_TILT_LIMIT = 30.0


@dataclass(frozen=True)
class CameraSettings:
    """Rocket League's camera settings, in its units: the ones many players use.

    :ivar field_of_view: horizontal, in degrees.
    :ivar distance: behind the car (uu).
    :ivar height: above the car (uu).
    :ivar angle: the car cam's pitch, in degrees; negative looks down.
    :ivar transition_seconds: the time to swing between car cam and ball cam.
    """

    field_of_view: float = 110.0
    distance: float = 270.0
    height: float = 100.0
    angle: float = -3.0
    transition_seconds: float = 0.25


@dataclass(frozen=True)
class CameraRig:
    """The camera's moving state.

    :ivar forward: the rig's heading, square to ``up``.
    :ivar up: the rig's up.
    :ivar ball_cam: how far the rig has swung to ball cam: 0 is car cam, 1 ball cam.
    """

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


def start_rig(car_forward: np.ndarray, car_up: np.ndarray, ball_cam: bool) -> CameraRig:
    """A rig settled behind a car on the ground: heading where it heads."""
    up = np.array(WORLD_UP)
    forward = _flatten(car_forward, up)
    if forward is None:
        forward = _flatten(car_up, up)
    return CameraRig(tuple(forward), WORLD_UP, 1.0 if ball_cam else 0.0)


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
    step = seconds / settings.transition_seconds
    amount = rig.ball_cam + step if ball_cam else rig.ball_cam - step
    amount = 1.0 if amount > 1.0 - 1e-9 else 0.0 if amount < 1e-9 else amount  # no rounding residue
    to_ball = None if ball_position is None else _flatten(ball_position - car_position, up)
    if to_ball is not None and amount > 0.0:
        blended = _flatten(heading * (1.0 - amount) + to_ball * amount, up)
        heading = to_ball if blended is None else blended
    turned = _flatten(forward + (heading - forward) * _approach(seconds, TURN_SECONDS), up)
    if turned is None:
        turned = heading
    return CameraRig(tuple(turned), tuple(up), amount)


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
    position = car_position - forward * settings.distance + up * settings.height
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
