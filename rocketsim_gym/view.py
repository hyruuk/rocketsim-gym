"""What one player sees of a match: Rocket League's camera behind their car, on car cam or
ball cam, and the frame it shows with the on-screen display over it, drawn from snapshots
(:mod:`rocketsim_gym.snapshot`).

The camera moves only when the view is given a new snapshot, as far as the match moved on;
drawing changes nothing. A notice (a LAN match paused, say) takes the place of the match's own
banner, and with no snapshot to show, a frame shows a notice alone.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .arena_geometry import HOMEMADE_ARENA_FOLDER, collision_mesh_files, read_collision_mesh_file
from .camera import CameraSettings, advance_rig, camera_view, start_rig
from .hud import Hud, HudPainter, hud_for
from .match import Car
from .renderer import CarPose, Renderer, Scene
from .snapshot import Snapshot


@dataclass(frozen=True)
class Notice:
    """What a frame says in place of the match's own banner: a word, and a line under it."""

    banner: str
    caption: str = ""


class SoccarView:
    """One player's view of a match, behind ``car``.

    :param car: the car the camera follows.
    :param camera_settings: the player's camera settings.
    :param frame_size: the frames' width and height in pixels, or None to draw none.
    :param pad_spots: each boost pad's position and whether it is big, in RocketSim's order;
        needed only to draw.
    :raises RuntimeError: frames are asked for, and no OpenGL 3.3 context opens.
    """

    def __init__(self, car: Car, camera_settings: CameraSettings,
                 frame_size: tuple[int, int] | None = None,
                 pad_spots: Sequence[tuple[tuple[float, float, float], bool]] = ()) -> None:
        self.car = car
        self._camera_settings = camera_settings
        self._renderer: Renderer | None = None
        self._snapshot: Snapshot | None = None
        self._notice: Notice | None = None
        self._ball_cam, self._ball_cam_held = True, False
        self._frame: np.ndarray | None = None
        if frame_size is not None:
            meshes = [read_collision_mesh_file(path)
                      for path in collision_mesh_files(HOMEMADE_ARENA_FOLDER)]
            self._renderer = Renderer(*frame_size, meshes, pad_spots)
            self._hud_painter = HudPainter(*frame_size)

    @property
    def draws_frames(self) -> bool:
        """Whether the view draws frames."""
        return self._renderer is not None

    @property
    def gl_renderer(self) -> str | None:
        """The OpenGL renderer that draws the frames, or None without frames."""
        return None if self._renderer is None else self._renderer.gl_renderer

    @property
    def ball_cam(self) -> bool:
        """Whether the camera is on ball cam."""
        return self._ball_cam

    def start(self, snapshot: Snapshot, notice: Notice | None = None) -> None:
        """Settle the camera behind the car, on ball cam, at a match's first snapshot."""
        self._ball_cam = True
        self._ball_cam_held = False
        car = snapshot.car(self.car)
        self._rig = start_rig(car.position, car.forward, car.up, self._ball_cam)
        self._snapshot, self._notice, self._frame = snapshot, notice, None

    def show_notice(self, notice: Notice) -> None:
        """Show ``notice`` alone, with no match to show."""
        self._snapshot, self._notice, self._frame = None, notice, None

    def press_ball_cam(self, is_held: bool) -> None:
        """The ball cam button, held or not: a press switches between car cam and ball cam."""
        if is_held and not self._ball_cam_held:
            self._ball_cam = not self._ball_cam
        self._ball_cam_held = is_held

    def advance(self, snapshot: Snapshot, seconds: float, notice: Notice | None = None) -> None:
        """Move the camera on to ``snapshot``, ``seconds`` of the match after the last one; with
        none, the camera stays where it is."""
        if seconds > 0:
            car = snapshot.car(self.car)
            self._rig = advance_rig(self._rig, car.position, car.forward, car.up,
                                    car.is_on_ground, snapshot.ball_in_play, self._ball_cam,
                                    seconds, self._camera_settings)
        self._snapshot, self._notice, self._frame = snapshot, notice, None

    def contents(self) -> tuple[Scene, Hud] | Notice:
        """What the frame of the latest snapshot or notice shows, before it is drawn: the scene
        and the on-screen display over it, or a notice alone.

        :raises ValueError: the view has nothing to show yet.
        """
        if self._snapshot is not None:
            car = self._snapshot.car(self.car)
            hud = hud_for(self._snapshot.match, car.boost, self.car)
            if self._notice is not None:
                hud = dataclasses.replace(hud, banner=self._notice.banner, banner_team=None,
                                          caption=self._notice.caption)
            return self._scene(), hud
        if self._notice is not None:
            return self._notice
        raise ValueError("this view has nothing to show yet")

    def frame(self) -> np.ndarray:
        """The frame of the latest snapshot or notice, ``(height, width, 3)`` RGB bytes with the
        on-screen display, read-only; drawn once, however often it is asked for.

        :raises ValueError: the view draws no frames, or has nothing to show yet.
        """
        if self._renderer is None:
            raise ValueError("this view draws no frames")
        if self._frame is not None:
            return self._frame
        contents = self.contents()
        if isinstance(contents, Notice):
            self._frame = self._hud_painter.message(contents.banner, contents.caption)
        else:
            scene, hud = contents
            self._frame = self._hud_painter.paint(self._renderer.draw(scene), hud)
        self._frame.flags.writeable = False
        return self._frame

    def close(self) -> None:
        """Let go of the OpenGL context; the view draws no more frames."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _scene(self) -> Scene:
        snapshot = self._snapshot
        ball = snapshot.ball_in_play
        cars = tuple(CarPose(car.car.team, tuple(car.position.tolist()),
                             tuple(car.forward.tolist()), tuple(car.right.tolist()),
                             tuple(car.up.tolist()), visible=not car.is_demolished,
                             boosting=car.controls.boost and car.boost > 0)
                     for car in snapshot.cars)
        camera = camera_view(self._rig, snapshot.car(self.car).position, ball,
                             self._camera_settings)
        axes = snapshot.ball
        return Scene(camera, None if ball is None else tuple(ball), cars, snapshot.pads_active,
                     (tuple(axes.forward.tolist()), tuple(axes.right.tolist()),
                      tuple(axes.up.tolist())))
