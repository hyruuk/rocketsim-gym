"""The play window's settings: the file, its errors, the bindings, and the frames' size."""

import pygame
import pytest

from rocketsim_gym.camera import CameraSettings
from rocketsim_gym.controls import Button, InputMode
from rocketsim_gym.settings import (AUTO_MAX_HEIGHT, Bindings, PlaySettings, SettingsError,
                                    binding_label, load_settings, save_settings, seat_names,
                                    settings_text)


def test_a_missing_file_gives_the_defaults(tmp_path):
    assert load_settings(tmp_path / "none.toml") == PlaySettings()


def test_the_defaults_are_fullscreen_with_rocket_leagues_settings():
    settings = PlaySettings()
    assert settings.fullscreen and settings.vsync and settings.resolution is None
    assert settings.camera == CameraSettings() and settings.controller.deadzone == 0.1
    assert settings.bindings[Button.JUMP] == ("mouse:right", "pad:a")


def test_settings_written_read_back_the_same(tmp_path):
    settings = PlaySettings(
        fullscreen=False, window=(1600, 900), resolution=(1920, 1080), vsync=False,
        input_mode=InputMode.DIGITAL, team_size=2, seconds=120.0,
        bindings=Bindings().bind(Button.AIR_ROLL_LEFT, "pad:leftshoulder"),
    ).replace(camera={"field_of_view": 100.0, "stiffness": 0.45},
              controller={"steering_sensitivity": 1.3})
    path = tmp_path / "deep" / "settings.toml"
    save_settings(settings, path)
    assert load_settings(path) == settings
    assert not path.with_name("settings.toml.tmp").exists()


def test_a_file_sets_only_what_it_names(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('[camera]\nfield_of_view = 95\n[bindings]\njump = "key:space"\n')
    settings = load_settings(path)
    assert settings.camera.field_of_view == 95 and settings.camera.distance == 270
    assert settings.bindings[Button.JUMP] == ("key:space",)
    assert settings.bindings[Button.BOOST] == ("mouse:left", "pad:b")
    assert settings.fullscreen


@pytest.mark.parametrize("text, said", [
    ("[camera\n", "not TOML"),
    ("[sound]\nvolume = 1\n", "unknown section 'sound'"),
    ("[camera]\nzoom = 2\n", "'zoom': not a setting there"),
    ("[camera]\nfield_of_view = 130\n", "field_of_view"),
    ("[display]\nwindow = [1280]\n", "window"),
    ("[controls]\ninput = \"loud\"\n", "\"analog\" or \"digital\""),
    ("[bindings]\njump = [\"key:nonsense\"]\n", "key:nonsense"),
    ("[bindings]\njump = [\"pad:z\"]\n", "pad:z"),
    ("[match]\nteam_size = 4\n", "team_size"),
])
def test_a_wrong_file_says_where_and_what(tmp_path, text, said):
    path = tmp_path / "settings.toml"
    path.write_text(text)
    with pytest.raises(SettingsError) as error:
        load_settings(path)
    assert str(path) in str(error.value) and said in str(error.value)


def test_the_file_written_explains_itself():
    text = settings_text(PlaySettings())
    assert text.startswith("# rocketsim-play's settings")
    assert 'resolution = "auto"' in text and 'jump = ["mouse:right", "pad:a"]' in text


def test_a_binding_replaces_only_the_one_on_its_device():
    bindings = Bindings()
    keyboard = bindings.bind(Button.JUMP, "key:space")
    assert keyboard[Button.JUMP] == ("key:space", "pad:a")
    controller = bindings.bind(Button.JUMP, "pad:rightshoulder")
    assert controller[Button.JUMP] == ("pad:rightshoulder", "mouse:right")
    assert bindings.clear(Button.JUMP, "keyboard")[Button.JUMP] == ("pad:a",)
    assert bindings.on(Button.JUMP, "controller") == "pad:a"
    assert bindings.on(Button.AIR_ROLL_LEFT, "controller") is None


def test_bindings_resolve_to_pygames_numbers():
    resolved = Bindings().resolve()
    assert set(resolved.keys[pygame.K_w]) == {Button.THROTTLE, Button.STICK_UP}
    assert resolved.keys[pygame.K_LSHIFT] == (Button.POWERSLIDE,)
    assert resolved.mouse[2] == (Button.JUMP,)
    assert resolved.pad[pygame.CONTROLLER_BUTTON_Y] == (Button.BALL_CAM,)


@pytest.mark.parametrize("binding, label", [("key:w", "W"), ("key:left shift", "Left Shift"),
                                            ("mouse:right", "Mouse right"), ("pad:a", "Pad A"),
                                            ("pad:leftshoulder", "Pad LB")])
def test_bindings_read_as_the_menu_shows_them(binding, label):
    assert binding_label(binding) == label


def test_auto_frames_fit_the_screen_at_most_1080_tall():
    settings = PlaySettings()
    assert settings.frame_size((2880, 1920)) == (1620, AUTO_MAX_HEIGHT)
    assert settings.frame_size((1366, 768)) == (1366, 768)
    assert settings.frame_size(None) == settings.window
    assert PlaySettings(fullscreen=False).frame_size((2880, 1920)) == (1280, 720)
    assert PlaySettings(resolution=(800, 600)).frame_size((2880, 1920)) == (800, 600)


def test_seats_alternate_blue_and_orange():
    assert seat_names(1, 2) == ("blue1", "orange1")
    assert seat_names(2, 3) == ("blue1", "orange1", "blue2")
    with pytest.raises(ValueError):
        seat_names(1, 3)
