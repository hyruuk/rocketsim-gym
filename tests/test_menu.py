"""The menus, driven as the play window drives them: no window here."""

import numpy as np
import pytest

from rocketsim_gym.controls import Button, InputMode
from rocketsim_gym.lan.protocol import Announcement, RoomStatus, Setup
from rocketsim_gym.menu import (Command, Menu, MenuPainter, Request, ServerFinder,
                                pause_screen, start_menu)
from rocketsim_gym.settings import PlaySettings


class FakeFinder(ServerFinder):
    def __init__(self, servers=(), answers=None):
        super().__init__(find=lambda **_: [])
        self._servers, self.answers, self.scanning = list(servers), dict(answers or {}), None

    def scan(self, on):
        self.scanning = on

    def lookup(self, address):
        pass

    def answer(self, address):
        return self.answers.get(address)


@pytest.fixture
def saved():
    return []


def make_menu(saved, settings=PlaySettings(), finder=None):
    menu = Menu(settings, saved.append, finder or FakeFinder())
    menu.push(start_menu(menu))
    return menu


def choose(menu, label):
    """Select the row whose label starts with ``label`` and accept it."""
    rows = menu.view().rows
    index = next(i for i, row in enumerate(rows) if row.label.startswith(label))
    menu.select(index)
    return menu.command(Command.ACCEPT)


def value(menu, label):
    return next(row.value for row in menu.view().rows if row.label.startswith(label))


def test_the_start_menu_offers_play_host_join_settings_and_quit(saved):
    labels = [row.label for row in make_menu(saved).view().rows]
    assert labels == ["Play against bots", "Host a LAN match", "Join a LAN match", "Settings",
                      "Quit"]


def test_quit_asks_to_quit(saved):
    assert choose(make_menu(saved), "Quit").kind == "quit"


def test_the_selection_skips_lines_that_only_say_something_and_wraps(saved):
    menu = make_menu(saved)
    menu.command(Command.UP)
    assert menu.view().selected == 4
    menu.command(Command.DOWN)
    assert menu.view().selected == 0


def test_a_match_against_bots_takes_the_team_size_and_length_chosen_and_saves_them(saved):
    menu = make_menu(saved)
    choose(menu, "Play against bots")
    menu.command(Command.RIGHT)  # 2v2
    menu.command(Command.DOWN)
    menu.command(Command.LEFT)  # 3:00
    assert (value(menu, "Team size"), value(menu, "Match length")) == ("2v2", "3:00")
    assert choose(menu, "Start") == Request("solo", 2, 180.0)
    assert (saved[-1].team_size, saved[-1].seconds) == (2, 180.0)


def test_hosting_takes_the_players_and_the_hosts_car(saved):
    menu = make_menu(saved)
    choose(menu, "Host a LAN match")
    assert value(menu, "Players") == "2: blue1, orange1"
    index = next(i for i, row in enumerate(menu.view().rows) if row.label == "Your car")
    menu.select(index)
    menu.command(Command.RIGHT)
    assert choose(menu, "Start hosting") == Request("host", 1, 300.0,
                                                    seats=("blue1", "orange1"), seat="orange1")


def test_joining_a_room_offers_its_free_seats():
    room = Announcement("hostbox", 47800, 1, RoomStatus.LOBBY,
                        Setup(1, ("blue1", "orange1"), 300.0), {"blue1": "hostbox"},
                        address="192.168.4.100")
    finder = FakeFinder([room])
    menu = make_menu([], finder=finder)
    choose(menu, "Join a LAN match")
    assert finder.scanning is True
    assert menu.view().rows[0].label == ("hostbox (192.168.4.100:47800): 1v1, 5:00, "
                                         "1 free seat")
    menu.command(Command.ACCEPT)
    assert [row.label for row in menu.view().rows] == ["Drive orange1", "Back"]
    assert menu.command(Command.ACCEPT) == Request(
        "join", 1, 300.0, "192.168.4.100:47800", ("blue1", "orange1"), "orange1")
    menu.command(Command.BACK)
    menu.command(Command.BACK)
    assert finder.scanning is False


def test_joining_an_empty_room_sets_up_its_match():
    room = Announcement("hostbox", 47800, 1, RoomStatus.LOBBY, None, {}, address="10.0.0.2")
    menu = make_menu([], finder=FakeFinder([room]))
    choose(menu, "Join a LAN match")
    menu.command(Command.ACCEPT)
    assert menu.view().title == "SET UP THE MATCH"
    assert choose(menu, "Join") == Request("join", 1, 300.0, "10.0.0.2:47800",
                                           ("blue1", "orange1"), "blue1")


def test_an_address_typed_is_looked_up_then_joined():
    room = Announcement("far", 47800, 1, RoomStatus.LOBBY, Setup(1, ("blue1",), 60.0), {},
                        address="10.1.1.1")
    finder = FakeFinder()
    menu = make_menu([], finder=finder)
    choose(menu, "Join a LAN match")
    choose(menu, "Enter an address")
    menu.type("10.1.1.1")
    menu.command(Command.CLEAR)
    menu.type("1")
    assert "10.1.1.1_" in menu.view().prompt
    menu.command(Command.ACCEPT)
    assert menu.view().rows[0].label == "Asking 10.1.1.1…"
    finder.answers["10.1.1.1"] = room
    menu.tick()
    assert menu.view().title == "CHOOSE YOUR CAR"


def test_a_failed_lookup_says_why():
    finder = FakeFinder(answers={"nowhere": "no answer from nowhere:47800"})
    menu = make_menu([], finder=finder)
    choose(menu, "Join a LAN match")
    choose(menu, "Enter an address")
    menu.type("nowhere")
    menu.command(Command.ACCEPT)
    menu.tick()
    assert menu.view().rows[0].label == "no answer from nowhere:47800"


def test_camera_settings_change_by_steps_within_their_range(saved):
    menu = make_menu(saved)
    choose(menu, "Settings")
    choose(menu, "Camera")
    for _ in range(3):
        menu.command(Command.RIGHT)  # field of view, at most 110
    menu.command(Command.LEFT)
    assert saved[-1].camera.field_of_view == 109
    assert value(menu, "Field of view") == "109°"


def test_the_controller_screen_switches_the_input_mode(saved):
    menu = make_menu(saved)
    choose(menu, "Settings")
    choose(menu, "Controller")
    menu.command(Command.RIGHT)
    assert saved[-1].input_mode is InputMode.DIGITAL


def test_display_settings_switch_fullscreen_and_resolution(saved):
    menu = make_menu(saved)
    choose(menu, "Settings")
    choose(menu, "Display")
    menu.command(Command.ACCEPT)
    menu.command(Command.DOWN)
    menu.command(Command.RIGHT)
    assert not saved[-1].fullscreen and saved[-1].resolution == (1280, 720)


def test_a_button_is_bound_to_what_is_pressed_next_and_cleared(saved):
    menu = make_menu(saved)
    choose(menu, "Settings")
    choose(menu, "Controls")
    choose(menu, "Air roll left")
    assert menu.capturing is Button.AIR_ROLL_LEFT and "Air roll left" in menu.view().prompt
    menu.command(Command.ACCEPT)  # ignored while waiting for an input
    menu.bind("pad:leftshoulder")
    assert menu.capturing is None
    assert saved[-1].bindings[Button.AIR_ROLL_LEFT] == ("pad:leftshoulder", "key:q")
    assert value(menu, "Air roll left") == "Q\tPad LB"
    menu.command(Command.CLEAR)
    assert saved[-1].bindings[Button.AIR_ROLL_LEFT] == ("pad:leftshoulder",)


def test_esc_cancels_a_binding(saved):
    menu = make_menu(saved)
    choose(menu, "Settings")
    choose(menu, "Controls")
    choose(menu, "Jump")
    menu.command(Command.BACK)
    assert menu.capturing is None and not saved and menu.view().title == "CONTROLS"


def test_reset_asks_twice(saved):
    settings = PlaySettings(fullscreen=False, team_size=3)
    menu = make_menu(saved, settings)
    choose(menu, "Settings")
    choose(menu, "Reset")
    assert not saved
    choose(menu, "Press again")
    assert saved[-1].fullscreen and saved[-1].team_size == 3


def test_esc_goes_back_a_screen_and_stays_on_the_first(saved):
    menu = make_menu(saved)
    choose(menu, "Settings")
    choose(menu, "Camera")
    menu.command(Command.BACK)
    assert menu.view().title == "SETTINGS"
    menu.command(Command.BACK)
    menu.command(Command.BACK)
    assert menu.view().title == "ROCKETSIM-GYM"


def test_the_pause_menu_resumes_on_esc_and_offers_the_hosts_commands():
    resumed, left = [], []
    menu = Menu(PlaySettings(), lambda settings: None, FakeFinder())
    menu.push(pause_screen(menu, lambda: resumed.append(1), lambda: left.append(1),
                           lambda: [], paused_game=True))
    menu.bottom_back = lambda: resumed.append(1)
    assert [row.label for row in menu.view().rows] == ["Resume", "Leave the match", "Quit"]
    menu.command(Command.BACK)
    assert resumed == [1]
    assert choose(menu, "Quit").kind == "quit"


def test_the_painter_paints_a_frame_and_finds_rows_under_the_mouse():
    menu = make_menu([])
    painter = MenuPainter(640, 360)
    surface = painter.paint(menu.view())
    assert surface.get_size() == (640, 360)
    assert painter.row_at(painter.row_rects[3].center) == 3
    over = painter.paint(menu.view(), background=np.zeros((360, 640, 3), np.uint8))
    assert over.get_size() == (640, 360)


def test_the_selection_stays_on_its_row_when_servers_appear_above_it():
    finder = FakeFinder()
    menu = make_menu([], finder=finder)
    choose(menu, "Join a LAN match")
    assert menu.view().rows[menu.view().selected].label == "Enter an address"
    finder._servers = [Announcement("a", 47800, 1, RoomStatus.LOBBY, None, {}, address="10.0.0.1"),
                       Announcement("b", 47800, 1, RoomStatus.LOBBY, None, {}, address="10.0.0.2")]
    assert menu.view().rows[menu.view().selected].label == "Enter an address"
