"""The camera's rig: where car cam and ball cam put the eye, and how the rig follows the car."""

import math

import numpy as np
import pytest

from rocketsim_gym.camera import CameraSettings, advance_rig, camera_view, start_rig

SETTINGS = CameraSettings()
STEP = 1 / 60
CAR = np.array([0.0, -3000.0, 17.0])
FACING_Y, FACING_X = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])


def follow(rig, seconds, car_forward=FACING_Y, car_up=UP, on_ground=True, ball=None,
           ball_cam=False, car=CAR):
    for _ in range(round(seconds / STEP)):
        rig = advance_rig(rig, car, car_forward, car_up, on_ground, ball, ball_cam, STEP, SETTINGS)
    return rig


def test_car_cam_hangs_behind_and_above_the_car_looking_slightly_down():
    camera = camera_view(start_rig(FACING_Y, UP, ball_cam=False), CAR, None, SETTINGS)
    assert camera.position == pytest.approx(tuple(CAR + (0, -270, 100)))
    assert math.degrees(math.asin(camera.forward[2])) == pytest.approx(-3)
    assert np.dot(camera.forward, camera.up) == pytest.approx(0, abs=1e-12)


def test_the_rig_turns_with_the_car():
    rig = follow(start_rig(FACING_Y, UP, ball_cam=False), 1.0, car_forward=FACING_X)
    assert rig.forward == pytest.approx(tuple(FACING_X), abs=1e-3)


def test_ball_cam_turns_the_rig_to_the_ball_and_back_in_a_quarter_second():
    ball = CAR + (2000, 0, 75)
    rig = follow(start_rig(FACING_Y, UP, ball_cam=False), 1.0, ball=ball, ball_cam=True)
    assert rig.ball_cam == 1 and rig.forward == pytest.approx(tuple(FACING_X), abs=1e-3)
    rig = follow(rig, SETTINGS.transition_seconds, ball=ball, ball_cam=False)
    assert rig.ball_cam == 0


def test_ball_cam_looks_up_at_a_ball_above_the_horizon_and_keeps_the_car_in_sight():
    rig = start_rig(FACING_Y, UP, ball_cam=True)
    ball = CAR + (0, 1500, 700)
    camera = camera_view(rig, CAR, ball, SETTINGS)
    car_cam = camera_view(rig, CAR, None, SETTINGS)
    assert camera.forward[2] > car_cam.forward[2] + 0.3
    to_car = CAR - np.array(camera.position)
    below = math.degrees(math.atan2(-to_car @ np.array(camera.up), to_car @ np.array(camera.forward)))
    half_height = math.degrees(math.atan(math.tan(math.radians(SETTINGS.field_of_view) / 2) * 9 / 16))
    assert 0 < below < half_height
    assert camera.position[2] > CAR[2]


def test_a_ball_out_of_play_leaves_ball_cam_on_the_car():
    rig = start_rig(FACING_Y, UP, ball_cam=True)
    assert camera_view(rig, CAR, None, SETTINGS) == camera_view(
        start_rig(FACING_Y, UP, ball_cam=False), CAR, None, SETTINGS)


def test_on_a_wall_the_rig_stands_on_the_wall_and_in_the_air_it_rights_itself():
    wall_up, wall_forward = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    rig = follow(start_rig(FACING_Y, UP, ball_cam=False), 1.0, car_forward=wall_forward,
                 car_up=wall_up)
    assert rig.up == pytest.approx(tuple(wall_up), abs=0.01)
    assert rig.forward == pytest.approx(tuple(wall_forward), abs=0.01)
    rig = follow(rig, 4.0, car_forward=FACING_Y, car_up=wall_up, on_ground=False)
    assert rig.up == pytest.approx(tuple(UP), abs=0.01)
