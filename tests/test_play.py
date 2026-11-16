# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Tests for the play window. Uses a hidden window so it runs anywhere.

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame

from roguelike.env import RoguelikeEnv
from roguelike.game import Events
from roguelike.play import ASSET_NAMES, TILE, describe, draw, load_assets, make_window


def test_draw_runs_without_a_screen():
    """drawing a frame works with the art and with plain shapes"""
    env = RoguelikeEnv()
    env.reset(seed=3, options={"run": 0})
    screen, fonts, images = make_window(env.game)
    draw(screen, fonts, env.game, "hello", "REC (0 saved)", images)
    draw(screen, fonts, env.game, "hello")
    pygame.quit()


def test_all_art_loads_at_tile_size():
    """every asset in the repo loads and is scaled to one tile"""
    env = RoguelikeEnv()
    env.reset(seed=3)
    make_window(env.game)
    images = load_assets()
    for name in ASSET_NAMES:
        assert images[name].get_size() == (TILE, TILE)
    pygame.quit()


def test_missing_art_falls_back(tmp_path):
    """an empty assets folder still works and just uses plain shapes"""
    env = RoguelikeEnv()
    env.reset(seed=3)
    make_window(env.game)
    images = load_assets(tmp_path)
    assert set(images) == {"shade"}
    pygame.quit()


def test_describe_messages():
    """the turn message mentions what happened"""
    events = Events(attacked=True, killed_enemy=True, damage_taken=2).as_dict()
    message = describe(events)
    assert "killed" in message and "2 damage" in message
    assert describe(Events().as_dict()) == ""
