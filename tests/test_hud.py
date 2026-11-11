"""The on-screen display: what it shows at each moment of a match, and how it is painted."""

import numpy as np

from rocketsim_gym.hud import BOOST_FILL, GO_TICKS, TEAM_COLORS, Hud, HudPainter, hud_for
from rocketsim_gym.match import (COUNTDOWN_TICKS, GOAL_PAUSE_TICKS, TICKS_PER_SECOND, Match,
                                 Team, TickEvents)

ON_THE_GROUND = TickEvents(ball_on_ground=True)
TOUCH = TickEvents(touched=(Team.BLUE,))
WIDTH, HEIGHT = 640, 360


def advance(match, ticks, events=TickEvents()):
    for _ in range(ticks):
        match.tick(events)


def kicked_off(seconds):
    match = Match(seconds * TICKS_PER_SECOND)
    advance(match, COUNTDOWN_TICKS)
    match.tick(TOUCH)
    return match


def test_the_countdown_reads_3_2_1_then_go():
    match, banners = Match(300 * TICKS_PER_SECOND), []
    for _ in range(COUNTDOWN_TICKS + GO_TICKS + 1):
        banners.append(hud_for(match, 33).banner)
        match.tick()
    assert [banners[0], banners[120], banners[240], banners[359]] == ["3", "2", "1", "1"]
    assert banners[COUNTDOWN_TICKS] == "GO!" and banners[-1] == ""


def test_the_clock_reads_minutes_and_seconds_and_counts_up_in_overtime():
    assert hud_for(Match(61 * TICKS_PER_SECOND), 0).clock == "1:01"
    match = kicked_off(1)
    advance(match, TICKS_PER_SECOND, ON_THE_GROUND)
    overtime = hud_for(match, 0)
    assert match.overtime and (overtime.clock, overtime.caption) == ("+0:00", "OVERTIME")
    advance(match, COUNTDOWN_TICKS)
    match.tick(TOUCH)
    advance(match, 75 * TICKS_PER_SECOND)
    assert hud_for(match, 0).clock == "+1:15"


def test_a_goal_and_the_winner_take_their_teams_color():
    match = kicked_off(2)
    match.tick(TickEvents(goal=Team.ORANGE))
    goal = hud_for(match, 0)
    assert (goal.banner, goal.banner_team, goal.orange_score) == ("GOAL!", Team.ORANGE, 1)
    advance(match, GOAL_PAUSE_TICKS + COUNTDOWN_TICKS)
    match.tick(TOUCH)
    advance(match, match.clock_ticks_left, ON_THE_GROUND)
    end = hud_for(match, 0)
    assert (end.banner, end.banner_team, end.caption) == ("ORANGE WINS", Team.ORANGE, "0 - 1")


def test_a_demolished_player_sees_when_their_car_comes_back():
    match = kicked_off(300)
    advance(match, GO_TICKS)
    match.tick(TickEvents(demolished=(Team.BLUE,)))
    demolished = hud_for(match, 0)
    assert (demolished.banner, demolished.caption) == ("DEMOLISHED", "Back in 3")
    assert hud_for(match, 0, team=Team.ORANGE).banner == ""


def _frame():
    return np.full((HEIGHT, WIDTH, 3), 40, np.uint8)


def _near(image, color, tolerance=40):
    return np.abs(image.astype(int) - np.array(color[:3])).sum(axis=2) < tolerance


def test_painting_the_same_display_gives_the_same_pixels():
    hud = Hud(2, 1, "3:07", 64, "GOAL!", Team.BLUE, "")
    painted = HudPainter(WIDTH, HEIGHT).paint(_frame(), hud)
    assert np.array_equal(painted, HudPainter(WIDTH, HEIGHT).paint(_frame(), hud))
    assert painted.shape == (HEIGHT, WIDTH, 3) and painted.dtype == np.uint8


def test_the_scoreboard_puts_blue_left_and_orange_right():
    top = HudPainter(WIDTH, HEIGHT).paint(_frame(), Hud(0, 0, "5:00", 0))[:HEIGHT // 8]
    blue = np.nonzero(_near(top, TEAM_COLORS[Team.BLUE]))[1]
    orange = np.nonzero(_near(top, TEAM_COLORS[Team.ORANGE]))[1]
    assert len(blue) > 100 and len(orange) > 100
    assert blue.mean() < WIDTH / 2 < orange.mean()


def test_the_boost_meter_fills_with_boost():
    painter = HudPainter(WIDTH, HEIGHT)
    filled = [_near(painter.paint(_frame(), Hud(0, 0, "5:00", boost))[HEIGHT // 2:, WIDTH // 2:],
                    BOOST_FILL).sum() for boost in (0, 50, 100)]
    assert filled[0] == 0 < filled[1] < filled[2]


def test_a_banner_is_painted_above_the_middle():
    painter = HudPainter(WIDTH, HEIGHT)
    plain = painter.paint(_frame(), Hud(0, 0, "5:00", 0))
    rows, columns = np.nonzero((painter.paint(_frame(), Hud(0, 0, "5:00", 0, "3")) != plain).any(axis=2))
    assert len(rows) > 200
    assert abs(columns.mean() - WIDTH / 2) < WIDTH * 0.05 and rows.mean() < HEIGHT / 2
