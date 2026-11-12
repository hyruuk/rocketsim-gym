"""The match rules, driven by made-up events: no physics here."""

import pytest

from rocketsim_gym.match import (COUNTDOWN_TICKS, GOAL_PAUSE_TICKS, RESPAWN_TICKS, ArenaChanges,
                                 Car, Match, Phase, Team, TickEvents, match_cars)

BLUE1, BLUE2, ORANGE1 = Car(Team.BLUE, 1), Car(Team.BLUE, 2), Car(Team.ORANGE, 1)
IN_THE_AIR = TickEvents()
ON_THE_GROUND = TickEvents(ball_on_ground=True)
TOUCH = TickEvents(touched=(BLUE1,))


def advance(match, ticks, events=IN_THE_AIR):
    """Tick ``match`` ``ticks`` times on the same events; the arena changes it asked for."""
    return [match.tick(events) for _ in range(ticks)]


def kicked_off(match_ticks=1200):
    """A match whose first kickoff was just touched."""
    match = Match(match_ticks)
    advance(match, COUNTDOWN_TICKS)
    match.tick(TOUCH)
    return match


def score_and_kick_off(match, team):
    """``team`` scores, and the next kickoff is touched."""
    match.tick(TickEvents(goal=team))
    advance(match, GOAL_PAUSE_TICKS + COUNTDOWN_TICKS)
    match.tick(TOUCH)


def test_a_countdown_holds_everything_for_three_seconds():
    match = Match(1200)
    assert match.phase is Phase.COUNTDOWN and not match.physics_runs
    assert not match.takes_controls(BLUE1)
    assert set(advance(match, COUNTDOWN_TICKS - 1, TickEvents())) == {ArenaChanges()}
    assert match.phase is Phase.COUNTDOWN
    match.tick()
    assert match.phase is Phase.KICKOFF and match.physics_runs and match.takes_controls(BLUE1)


def test_events_in_a_countdown_are_refused():
    with pytest.raises(ValueError, match="countdown"):
        Match(1200).tick(TOUCH)


def test_the_clock_starts_at_the_first_touch():
    match = Match(1200)
    advance(match, COUNTDOWN_TICKS + 500)
    assert match.phase is Phase.KICKOFF and match.clock_ticks_left == 1200
    match.tick(TOUCH)
    assert match.phase is Phase.PLAY and match.clock_ticks_left == 1200
    advance(match, 10)
    assert match.clock_ticks_left == 1190


def test_a_goal_scores_then_pauses_with_the_ball_out_then_brings_a_kickoff():
    match = kicked_off()
    assert match.tick(TickEvents(goal=Team.ORANGE)) == ArenaChanges(remove_ball=True)
    assert match.score == {Team.BLUE: 0, Team.ORANGE: 1} and match.phase is Phase.GOAL
    clock = match.clock_ticks_left
    advance(match, GOAL_PAUSE_TICKS - 1)
    assert match.phase is Phase.GOAL and match.clock_ticks_left == clock
    assert match.tick() == ArenaChanges(kickoff=True)
    assert match.phase is Phase.COUNTDOWN and match.phase_ticks_left == COUNTDOWN_TICKS


def test_at_zero_with_the_ball_on_the_ground_the_leader_wins():
    match = kicked_off(100)
    score_and_kick_off(match, Team.BLUE)
    advance(match, match.clock_ticks_left - 1, ON_THE_GROUND)
    assert match.phase is Phase.PLAY and match.clock_ticks_left == 1
    assert match.tick(ON_THE_GROUND) == ArenaChanges()
    assert match.phase is Phase.ENDED and match.winner is Team.BLUE
    with pytest.raises(ValueError, match="ended"):
        match.tick()


def test_at_zero_play_goes_on_until_the_ball_lands():
    match = kicked_off(100)
    score_and_kick_off(match, Team.BLUE)
    advance(match, 300)
    assert match.phase is Phase.PLAY and match.clock_ticks_left == 0
    match.tick(ON_THE_GROUND)
    assert match.phase is Phase.ENDED


def test_a_goal_before_the_ball_lands_counts_and_a_tie_goes_to_overtime():
    match = kicked_off(100)
    score_and_kick_off(match, Team.BLUE)
    advance(match, 150)
    match.tick(TickEvents(goal=Team.ORANGE))
    assert match.score == {Team.BLUE: 1, Team.ORANGE: 1}
    advance(match, GOAL_PAUSE_TICKS - 1)
    assert match.tick() == ArenaChanges(kickoff=True)
    assert match.overtime and match.phase is Phase.COUNTDOWN


def test_overtime_counts_up_from_its_first_touch_and_the_next_goal_wins():
    match = kicked_off(100)
    advance(match, 99, ON_THE_GROUND)
    assert match.tick(ON_THE_GROUND) == ArenaChanges(kickoff=True)
    assert match.overtime
    advance(match, COUNTDOWN_TICKS + 50)
    assert match.overtime_ticks == 0
    match.tick(TOUCH)
    advance(match, 240, ON_THE_GROUND)
    assert match.phase is Phase.PLAY and match.overtime_ticks == 240
    match.tick(TickEvents(goal=Team.ORANGE))
    advance(match, GOAL_PAUSE_TICKS)
    assert match.phase is Phase.ENDED and match.winner is Team.ORANGE


def test_a_demolished_car_comes_back_three_seconds_later_and_obeys_no_one_meanwhile():
    match = kicked_off()
    match.tick(TickEvents(demolished=(ORANGE1,)))
    assert not match.takes_controls(ORANGE1) and match.takes_controls(BLUE1)
    changes = advance(match, RESPAWN_TICKS)
    assert [change.respawn for change in changes if change.respawn] == [(ORANGE1,)]
    assert changes[-1].respawn == (ORANGE1,) and match.takes_controls(ORANGE1)


def test_each_car_of_a_team_comes_back_three_seconds_after_its_own_demolition():
    match = kicked_off()
    match.tick(TickEvents(demolished=(BLUE2,)))
    advance(match, 60)
    match.tick(TickEvents(demolished=(BLUE1, ORANGE1)))
    assert not any(match.takes_controls(car) for car in (BLUE1, BLUE2, ORANGE1))
    changes = advance(match, RESPAWN_TICKS)
    assert [change.respawn for change in changes if change.respawn] == [(BLUE2,), (BLUE1, ORANGE1)]
    assert changes.index(ArenaChanges(respawn=(BLUE2,))) == RESPAWN_TICKS - 62


def test_a_car_demolished_twice_at_once_is_refused():
    match = kicked_off()
    match.tick(TickEvents(demolished=(BLUE2,)))
    with pytest.raises(ValueError, match="blue2 is demolished already"):
        match.tick(TickEvents(demolished=(BLUE2,)))


def test_the_cars_of_a_match_are_blues_then_oranges_each_by_number():
    assert [car.name for car in match_cars(3)] == ["blue1", "blue2", "blue3",
                                                   "orange1", "orange2", "orange3"]
    assert match_cars(1) == (BLUE1, ORANGE1)


def test_a_kickoff_brings_every_demolished_car_back_with_it():
    match = kicked_off()
    match.tick(TickEvents(goal=Team.ORANGE, demolished=(BLUE1,)))
    changes = advance(match, GOAL_PAUSE_TICKS)
    assert changes[-1] == ArenaChanges(kickoff=True)
    assert not any(change.respawn for change in changes) and match.respawn_ticks_left == {}


@pytest.mark.parametrize("match_ticks", [0, -1, 1.5, True])
def test_a_match_lasts_a_positive_whole_number_of_ticks(match_ticks):
    with pytest.raises(ValueError, match="match_ticks"):
        Match(match_ticks)


def test_the_match_knows_who_scored_last_and_how_long_since_go():
    match = Match(1200)
    advance(match, COUNTDOWN_TICKS)
    assert match.ticks_since_go == 0 and match.last_goal is None
    advance(match, 30)
    assert match.ticks_since_go == 30
    match.tick(TickEvents(goal=Team.ORANGE))
    assert match.last_goal is Team.ORANGE
    advance(match, GOAL_PAUSE_TICKS + COUNTDOWN_TICKS)
    assert match.ticks_since_go == 0
