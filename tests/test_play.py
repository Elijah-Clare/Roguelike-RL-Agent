# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Tests for the play window. Uses a hidden window so it runs anywhere.

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame

from roguelike.env import RoguelikeEnv
from roguelike.game import Events
from roguelike.play import describe, draw, make_window


def test_draw_runs_without_a_screen():
    """drawing a frame works with a hidden window"""
    env = RoguelikeEnv()
    env.reset(seed=3, options={"run": 0})
    screen, fonts = make_window(env.game)
    draw(screen, fonts, env.game, "hello")
    pygame.quit()


def test_describe_messages():
    """the turn message mentions what happened"""
    events = Events(attacked=True, killed_enemy=True, damage_taken=2).as_dict()
    message = describe(events)
    assert "killed" in message and "2 damage" in message
    assert describe(Events().as_dict()) == ""
