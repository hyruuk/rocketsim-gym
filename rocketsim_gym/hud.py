"""Rocket League's on-screen display, painted over a frame: the scoreboard and clock at the top,
the boost meter at the bottom right, and the banners in the middle (the kickoff countdown, GO!,
GOAL!, OVERTIME, the winner, a demolition).

:func:`hud_for` says what the display shows from the state of a match; :class:`HudPainter`
paints it. The text is set in Russo One (SIL Open Font License, ``fonts/OFL.txt``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from .match import TICKS_PER_SECOND, Match, Phase, Team  # noqa: E402

FONT_FILE = Path(__file__).resolve().parent / "fonts" / "RussoOne-Regular.ttf"
#: GO! stays up this long after a kickoff goes live.
GO_TICKS = 90

TEAM_COLORS = {Team.BLUE: (32, 112, 245), Team.ORANGE: (245, 132, 26)}
CLOCK_BOX = (22, 26, 36)
WHITE = (255, 255, 255)
OUTLINE = (10, 12, 18)
BOOST_FILL = (255, 200, 40, 255)
BOOST_TRACK = (72, 76, 90, 255)
BOOST_BACKGROUND = (14, 17, 26, 190)
#: The boost ring runs clockwise from the bottom left (225 degrees) over the top to the bottom
#: right, as Rocket League's does.
_BOOST_START, _BOOST_SWEEP = 225.0, 270.0
_SUPERSAMPLING = 3


@dataclass(frozen=True)
class Hud:
    """What the display shows.

    :ivar clock: as the clock reads: "4:59", or "+0:12" in overtime.
    :ivar boost: the player's boost, 0 to 100.
    :ivar banner: the big text in the middle of the screen, or "".
    :ivar banner_team: the team whose color the banner takes; None for white.
    :ivar caption: the smaller text under the banner, or "".
    """

    blue_score: int
    orange_score: int
    clock: str
    boost: float
    banner: str = ""
    banner_team: Team | None = None
    caption: str = ""


def hud_for(match: Match, boost: float, team: Team = Team.BLUE) -> Hud:
    """The display for the player of ``team``'s car in ``match``, which has ``boost``."""
    if match.overtime:
        clock = "+" + _minutes(match.overtime_ticks // TICKS_PER_SECOND)
    else:
        clock = _minutes(_whole_seconds(match.clock_ticks_left))
    banner, banner_team, caption = "", None, ""
    if match.phase is Phase.ENDED:
        banner, banner_team = f"{match.winner.value.upper()} WINS", match.winner
        caption = f"{match.score[Team.BLUE]} - {match.score[Team.ORANGE]}"
    elif match.phase is Phase.GOAL:
        banner, banner_team = "GOAL!", match.last_goal
    elif match.phase is Phase.COUNTDOWN:
        banner = str(_whole_seconds(match.phase_ticks_left))
        caption = "OVERTIME" if match.overtime else ""
    elif match.ticks_since_go < GO_TICKS:
        banner = "GO!"
    elif team in match.respawn_ticks_left:
        banner = "DEMOLISHED"
        caption = f"Back in {_whole_seconds(match.respawn_ticks_left[team])}"
    return Hud(match.score[Team.BLUE], match.score[Team.ORANGE], clock, boost, banner,
               banner_team, caption)


def _whole_seconds(ticks: int) -> int:
    """``ticks`` in seconds, rounded up, as a countdown shows them."""
    return -(-ticks // TICKS_PER_SECOND)


def _minutes(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


class HudPainter:
    """Paints a :class:`Hud` over frames of one size, its parts scaled to the frame's height."""

    def __init__(self, width: int, height: int) -> None:
        pygame.font.init()
        self.width, self.height = width, height
        self._scale = height / 720
        self._fonts: dict[int, pygame.font.Font] = {}
        self._texts: dict[tuple[str, int, tuple[int, ...], bool], pygame.Surface] = {}
        self._meters: dict[int, pygame.Surface] = {}

    def paint(self, frame: np.ndarray, hud: Hud) -> np.ndarray:
        """``frame`` with ``hud`` over it, as a new read-only array."""
        surface = pygame.image.frombytes(frame.tobytes(), (self.width, self.height), "RGB")
        self._scoreboard(surface, hud)
        self._boost_meter(surface, hud.boost)
        if hud.banner:
            self._banner(surface, hud)
        painted = np.frombuffer(pygame.image.tobytes(surface, "RGB"), np.uint8)
        return painted.reshape(self.height, self.width, 3)

    def _pixels(self, size: float) -> int:
        return max(1, round(size * self._scale))

    def _text(self, text: str, size: int, color, outlined: bool = False) -> pygame.Surface:
        key = (text, size, tuple(color), outlined)
        if key not in self._texts:
            if size not in self._fonts:
                self._fonts[size] = pygame.font.Font(str(FONT_FILE), self._pixels(size))
            font = self._fonts[size]
            glyphs = font.render(text, True, color)
            if outlined:
                edge = self._pixels(size / 28)
                shadow = font.render(text, True, OUTLINE)
                outlined_surface = pygame.Surface((glyphs.get_width() + 2 * edge,
                                                   glyphs.get_height() + 2 * edge), pygame.SRCALPHA)
                for dx in (-edge, 0, edge):
                    for dy in (-edge, 0, edge):
                        outlined_surface.blit(shadow, (edge + dx, edge + dy))
                outlined_surface.blit(glyphs, (edge, edge))
                glyphs = outlined_surface
            self._texts[key] = glyphs
        return self._texts[key]

    def _blit_centred(self, surface: pygame.Surface, image: pygame.Surface, centre) -> None:
        surface.blit(image, image.get_rect(center=(round(centre[0]), round(centre[1]))))

    def _scoreboard(self, surface: pygame.Surface, hud: Hud) -> None:
        top, height = self._pixels(12), self._pixels(52)
        clock_width, score_width, radius = self._pixels(140), self._pixels(78), self._pixels(10)
        left = self.width // 2 - clock_width // 2
        boxes = ((Team.BLUE, left - score_width, score_width, str(hud.blue_score)),
                 (Team.ORANGE, left + clock_width, score_width, str(hud.orange_score)))
        pygame.draw.rect(surface, CLOCK_BOX, (left, top, clock_width, height))
        for team, x, width, score in boxes:
            corners = ({"border_top_left_radius": radius, "border_bottom_left_radius": radius}
                       if team is Team.BLUE else
                       {"border_top_right_radius": radius, "border_bottom_right_radius": radius})
            pygame.draw.rect(surface, TEAM_COLORS[team], (x, top, width, height), **corners)
            self._blit_centred(surface, self._text(score, 40, WHITE), (x + width / 2,
                                                                       top + height / 2))
        self._blit_centred(surface, self._text(hud.clock, 34, WHITE),
                           (self.width / 2, top + height / 2))

    def _boost_meter(self, surface: pygame.Surface, boost: float) -> None:
        amount = int(round(max(0.0, min(100.0, boost))))
        if amount not in self._meters:
            self._meters[amount] = self._meter(amount)
        meter = self._meters[amount]
        margin = self._pixels(28)
        centre = (self.width - margin - meter.get_width() / 2,
                  self.height - margin - meter.get_height() / 2)
        self._blit_centred(surface, meter, centre)
        self._blit_centred(surface, self._text(str(amount), 46, WHITE), centre)

    def _meter(self, amount: int) -> pygame.Surface:
        """The boost meter at ``amount``: drawn large, then smoothly scaled down."""
        size = self._pixels(168)
        large = size * _SUPERSAMPLING
        meter = pygame.Surface((large, large), pygame.SRCALPHA)
        centre, radius = large / 2, large / 2
        pygame.draw.circle(meter, BOOST_BACKGROUND, (centre, centre), radius)
        ring_radius, ring_width = radius * 0.80, radius * 0.17
        for sweep, color in ((_BOOST_SWEEP, BOOST_TRACK), (_BOOST_SWEEP * amount / 100, BOOST_FILL)):
            if sweep > 0:
                pygame.draw.polygon(meter, color, _arc(centre, ring_radius, ring_width, sweep))
        return pygame.transform.smoothscale(meter, (size, size))

    def _banner(self, surface: pygame.Surface, hud: Hud) -> None:
        color = WHITE if hud.banner_team is None else TEAM_COLORS[hud.banner_team]
        banner = self._text(hud.banner, 110, color, outlined=True)
        centre_y = self.height * 0.33
        self._blit_centred(surface, banner, (self.width / 2, centre_y))
        if hud.caption:
            caption = self._text(hud.caption, 38, WHITE, outlined=True)
            self._blit_centred(surface, caption, (self.width / 2, centre_y + banner.get_height() / 2
                                                  + caption.get_height() / 2))


def _arc(centre: float, radius: float, width: float, sweep: float) -> list[tuple[float, float]]:
    """The outline of the ring's arc of ``sweep`` degrees, clockwise from its start."""
    steps = max(2, math.ceil(sweep / 3))
    angles = np.radians(_BOOST_START - np.linspace(0.0, sweep, steps + 1))
    outer, inner = radius + width / 2, radius - width / 2
    points = [(centre + outer * math.cos(a), centre - outer * math.sin(a)) for a in angles]
    return points + [(centre + inner * math.cos(a), centre - inner * math.sin(a))
                     for a in angles[::-1]]
