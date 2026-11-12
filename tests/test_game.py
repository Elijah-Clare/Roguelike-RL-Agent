# PROGRAMMER: Dashawn Williams
# DUE DATE: Monday, 11/30/26
# Tests for the game rules.

from roguelike.dungeon import WALL, distances_from
from roguelike.game import DOWN, LEFT, MOVES, RIGHT, UP, WAIT, Game


def path_to_stairs(game):
    """actions that walk the player to the stairs on the current floor"""
    dist = distances_from(game.grid, game.dungeon.stairs)
    actions, pos = [], game.pos
    while pos != game.dungeon.stairs:
        for action, (d_row, d_col) in MOVES.items():
            next_pos = (pos[0] + d_row, pos[1] + d_col)
            if dist.get(next_pos, 10**9) < dist[pos]:
                actions.append(action)
                pos = next_pos
                break
    return actions


def test_walls_block_movement():
    """walking into a wall leaves the player where they were"""
    game = Game(seed=0)
    while game.grid[game.pos[0] - 1, game.pos[1]] != WALL:  # walk up to a wall
        game.step(UP)
    start = game.pos
    events = game.step(UP)
    assert events.hit_wall and not events.moved
    assert game.pos == start


def test_new_tile_only_counted_once():
    """going back and forth only counts each tile as new once"""
    game = Game(seed=3)
    seen = []
    for action in (RIGHT, LEFT, RIGHT, LEFT, DOWN, UP, DOWN, UP):
        events = game.step(action)
        if events.moved:
            seen.append(events.new_tile)
    assert seen.count(True) <= 2


def test_reaching_stairs_wins_single_floor():
    """with one floor, reaching the stairs wins the game"""
    game = Game(seed=5)
    for action in path_to_stairs(game):
        events = game.step(action)
    assert events.reached_stairs and events.won and game.done


def test_multiple_floors():
    """the stairs lead to the next floor until the last one"""
    game = Game(seed=11, num_floors=3)
    for floor in (1, 2, 3):
        assert game.floor == floor
        for action in path_to_stairs(game):
            game.step(action)
    assert game.won


def test_turn_limit_times_out():
    """the run ends when the turn limit is hit"""
    game = Game(seed=1, turns_per_floor=10)
    for _ in range(10):
        events = game.step(WAIT)
    assert events.timed_out and game.done and not game.won
