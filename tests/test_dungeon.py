# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Tests for the dungeon generator.

import numpy as np

from roguelike.dungeon import FLOOR, STAIRS, WALL, distances_from, generate_dungeon


def make(seed):
    """dungeon for a seed"""
    return generate_dungeon(np.random.default_rng(seed))


def test_same_seed_same_map():
    """the same seed always gives the same map"""
    a, b = make(7), make(7)
    assert np.array_equal(a.grid, b.grid)
    assert a.start == b.start and a.stairs == b.stairs


def test_different_seeds_differ():
    """different seeds give different maps"""
    grids = {make(seed).grid.tobytes() for seed in range(20)}
    assert len(grids) > 15


def test_stairs_reachable_and_not_on_start():
    """the stairs can always be reached and are never on the start tile"""
    for seed in range(300):
        dungeon = make(seed)
        assert dungeon.start != dungeon.stairs
        assert dungeon.grid[dungeon.start] == FLOOR
        assert dungeon.grid[dungeon.stairs] == STAIRS
        assert dungeon.stairs in distances_from(dungeon.grid, dungeon.start)


def test_border_is_wall():
    """the outer edge of the map is always wall"""
    for seed in range(50):
        grid = make(seed).grid
        assert (grid[0] == WALL).all() and (grid[-1] == WALL).all()
        assert (grid[:, 0] == WALL).all() and (grid[:, -1] == WALL).all()
