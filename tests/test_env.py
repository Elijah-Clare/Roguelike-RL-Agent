# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Tests for the Gymnasium environment.

import numpy as np
from gymnasium.utils.env_checker import check_env

from roguelike.dungeon import WALL
from roguelike.env import STAIRS_CH, VIEW_SIZE, VISITED_CH, WALL_CH, RoguelikeEnv
from roguelike.game import WAIT


def test_passes_gymnasium_checks():
    """gymnasium's own checker finds no problems"""
    check_env(RoguelikeEnv(), skip_render_check=True)


def test_same_seed_same_start():
    """reset with the same seed gives the same map"""
    env = RoguelikeEnv()
    obs_a, info_a = env.reset(seed=42)
    obs_b, info_b = env.reset(seed=42)
    assert info_a["map_seed"] == info_b["map_seed"] == 42
    assert np.array_equal(obs_a["view"], obs_b["view"])


def test_view_matches_map():
    """every tile in the view matches the real map"""
    env = RoguelikeEnv()
    obs, _ = env.reset(seed=4)
    game = env.game
    half = VIEW_SIZE // 2
    for d_row in range(-half, half + 1):
        for d_col in range(-half, half + 1):
            row, col = game.pos[0] + d_row, game.pos[1] + d_col
            inside = 0 <= row < game.height and 0 <= col < game.width
            expect_wall = (not inside) or game.grid[row, col] == WALL
            assert obs["view"][WALL_CH, d_row + half, d_col + half] == expect_wall
    assert obs["view"][VISITED_CH, half, half] == 1.0
    assert obs["view"][STAIRS_CH].sum() <= 1


def test_seed_pool_only_uses_pool():
    """without a seed, reset only picks maps from the pool"""
    env = RoguelikeEnv(seed_pool=[5, 6, 7])
    env.reset(seed=0)
    seen = {env.reset()[1]["map_seed"] for _ in range(30)}
    assert seen <= {5, 6, 7}


def test_timeout_is_truncation():
    """hitting the turn limit counts as truncated, not terminated"""
    env = RoguelikeEnv(turns_per_floor=5)
    env.reset(seed=1)
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(WAIT)
    assert truncated and not terminated
